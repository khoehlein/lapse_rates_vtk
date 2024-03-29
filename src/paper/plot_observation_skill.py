import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from itertools import product
from tqdm import tqdm

from src.observations.compute_score_analysis import load_predictions
from src.paper.volume_visualization.lapse_rates.clipping import RampMinClip, RampMaxClip, DEFAULT_CLIP_MAX, \
    DEFAULT_CLIP_MIN
from src.paper.volume_visualization.lapse_rates.lapse_rate_visualization import LapseRateProperties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    args = vars(parser.parse_args())
    plot_scores(args['input_file'], train=True)
    plot_scores(args['input_file'], train=False)


def plot_scores(input_file: str, train=False):
    print('Loading')
    obs, pred = load_predictions(input_file, train)

    print('Computing')
    scores = pred['score'].values
    score_bin = np.digitize(np.fmax(scores, 0.), np.linspace(0, 1, 11))
    dz = pred['elevation_difference'].values
    dz_bin = np.digitize(dz, np.array([np.min(dz) - 1, -50., 50., np.max(dz) + 1]))

    res_default = np.abs(obs['value_0'].values - (pred['hres'].values - 0.0065 * dz))

    scores_per_bin = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'score': scores,
        'dz': dz,
    }).groupby(['score_bin', 'dz_bin']).agg(['mean', 'min', 'max', 'count'])
    scores_per_bin.columns = ['_'.join(c) for c in scores_per_bin.columns]

    lapsrate_settings = LapseRateProperties()
    min_clip = RampMinClip(lapsrate_settings.min_clip)
    max_clip = RampMaxClip(lapsrate_settings.max_clip)

    raw_lapse = pred['lapse_rate'].values
    lapse_rates = min_clip.clip(raw_lapse, scores, return_thresholds=False)
    lapse_rates = max_clip.clip(lapse_rates, scores, return_thresholds=False)
    lapse_rates[np.abs(dz) < lapsrate_settings.min_elevation] = lapsrate_settings.default_lapse_rate
    lapse_rates[pred['neighbor_count'].values < lapsrate_settings.min_samples] = lapsrate_settings.default_lapse_rate

    pred_adaptive = pred['hres'].values + lapse_rates / 1000 * dz

    res_adaptive = np.abs(obs['value_0'].values - pred_adaptive)

    # for key in [1, 2, 3]:
    #     fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    #     ax[0].hist(dz[dz_bin == key])
    #     ax[0].set(title='dz')
    #     ax[1].hist(scores[dz_bin == key])
    #     ax[1].set(title='score')
    #     ax[2].hist(raw_lapse[dz_bin == key])
    #     ax[2].hist(lapse_rates[dz_bin == key])
    #     ax[2].set(title='lapse rate')
    #     ax[3].hist(pred['neighbor_count'].values[dz_bin == key])
    #     ax[3].set(title='neighbor count')
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
    #
    #     fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    #     mask = dz_bin == key
    #     ax[0].scatter(raw_lapse[mask], dz[mask], alpha=0.05)
    #     ax[0].axhline(lapsrate_settings.min_elevation)
    #     ax[0].axhline(-lapsrate_settings.min_elevation)
    #     ax[0].set(title='dz')
    #     ax[1].scatter(raw_lapse[mask], scores[mask], alpha=0.05)
    #     ax[1].set(title='score')
    #     ax[2].scatter(raw_lapse[mask], pred['neighbor_count'].values[mask], alpha=0.05)
    #     ax[2].axhline(lapsrate_settings.min_samples)
    #     ax[2].set(title='neighbor count')
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()



    plt.figure()
    plt.plot()
    df_mse = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'adaptive': res_adaptive**2,
        'default': res_default**2,
    }).groupby(['score_bin', 'dz_bin']).mean()
    df_mse.columns = [f'{x}_mse' for x in df_mse.columns]
    df_max = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'adaptive': res_adaptive,
        'default': res_default,
    }).groupby(['score_bin', 'dz_bin']).max()
    df_max.columns = [f'{x}_max' for x in df_max.columns]
    metrics = pd.concat([df_mse, df_max, scores_per_bin], axis='columns')
    metrics = metrics.reset_index()

    groups = metrics.groupby('dz_bin')
    print()

    fig, ax = plt.subplots()
    labels = {
        1: 'valley stations',
        2: 'neutral stations',
        3: 'mountain stations',
    }
    for key, group in groups:
        group_label = labels[key]
        group = group.set_index('score_bin').sort_index()
        mse_score = 1. - group['adaptive_mse'].values / group['default_mse'].values
        max_score = 1. - group['adaptive_max'].values / group['default_max'].values
        lines = ax.plot((group.index.values - 0.5) * 10, mse_score * 100, label=f'{group_label} (MSE)')
        ax.plot((group.index.values - 0.5) * 10, max_score * 100, label=f'{group_label} (MAX)', linestyle='--', color=lines[0].get_color())

    ax.legend()
    ax.set(xlabel='R2 score (%)', ylabel='Skill improvement (%)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

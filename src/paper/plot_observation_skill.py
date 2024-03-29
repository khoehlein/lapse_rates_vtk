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
    min_clip = RampMinClip(lapsrate_settings)
    max_clip = RampMaxClip(lapsrate_settings)

    lapse_rates = min_clip.clip(pred['lapse_rate'].values, scores)
    lapse_rates = max_clip.clip(lapse_rates, scores)
    lapse_rates[dz < lapsrate_settings.min_elevation] = lapsrate_settings.default_lapse_rate
    lapse_rates[pred['neighbor_count'].values < lapsrate_settings.min_samples] = lapsrate_settings.default_lapse_rate

    pred_adaptive = pred['hres'].values + lapse_rates / 1000 * dz

    res_adaptive = np.abs(obs['value_0'].values - pred_adaptive)
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
        1: 'valley',
        2: 'neutral',
        3: 'mountain',
    }
    for key, group in groups:
        group_label = labels[key]
        group = group.set_index('score_bin').sort_index()
        mse_score = 1. - group['adaptive_mse'].values / group['default_mse'].values
        max_score = 1. - group['adaptive_max'].values / group['default_max'].values
        lines = ax.plot((group.index.values - 0.5) * 10, mse_score * 100, label=f'{group_label} (MSE)')
        ax.plot((group.index.values - 0.5) * 10, max_score * 100, label=f'{group_label} (MAX)', linestyle='--', color=lines[0].get_color())

    ax.set(xlabel='R2 score (%)', ylabel='Relative Skill (%)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

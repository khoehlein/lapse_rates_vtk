import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
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
    bin_bounds = np.linspace(0, 1, 5)
    score_bin = np.digitize(np.fmax(scores, 0.), bin_bounds)
    dz = pred['elevation_difference'].values
    dz_bin = np.digitize(dz, np.array([np.min(dz) - 1, -50., 50., np.max(dz) + 1]))

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

    gamma_min = -15
    gamma_max = 50

    bins = np.linspace(gamma_min, gamma_max, 27)

    fig, axs = plt.subplots(4, 1, sharex='col', figsize=(8, 5), gridspec_kw={'hspace': 0})
    for i, ax in enumerate(axs):

        ax.hist(lapse_rates[score_bin == (i + 1)], bins, log=True)
        ax.set(ylabel='{}% - {}%'.format(int(bin_bounds[i] * 100), int(bin_bounds[i + 1] * 100)))
        ax.axvline(-6.5, linestyle='--', color='k')

    axs[-1].set(xlabel='Lapse rate (K/km)')
    plt.tight_layout()
    label = 'train' if train else 'eval'
    plt.savefig(os.path.join(os.path.dirname(input_file), f'histograms_final_{label}.pdf'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

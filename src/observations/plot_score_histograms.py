import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.observations.compute_score_analysis import load_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    args = vars(parser.parse_args())
    export_histograms(args['input_file'])


def export_histograms(input_file, train=False):
    print('Loading')
    label = 'train' if train else 'eval'
    obs, pred = load_predictions(input_file, train=train)

    scores = pred['score'].values
    score_bin = np.digitize(np.fmax(scores, 0.), np.linspace(0, 1, 11))

    dz = pred['elevation_difference'].values
    dz_bin = np.digitize(dz, np.array([np.min(dz) - 1, -100, 100, np.max(dz) + 1]))

    lr_per_bin = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'lapse_rate': pred['lapse_rate'].values
    }).groupby(['score_bin', 'dz_bin'])

    plot_path = os.path.join(os.path.dirname(input_file), f'score_histograms_{label}')
    os.makedirs(plot_path, exist_ok=True)

    for bin, group in lr_per_bin:
        if len(group) < 1000:
            continue
        fig, ax = plt.subplots(1, 1)
        ax.hist(group['lapse_rate'].values, bins=np.arange(-20, 50), log=True)
        ax.set(xlabel='lapse rate', ylabel='count', title='Bin: {}'.format(bin))
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'hist_bin-{:02d}-{:02d}.png'.format(*bin)))
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()

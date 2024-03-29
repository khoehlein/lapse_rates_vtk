import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    args = vars(parser.parse_args())
    export_plots(args['input_file'], train=True)
    export_plots(args['input_file'], train=False)


def export_plots(input_file: str, train=False):
    label = 'train' if train else 'eval'
    plot_path_mse = os.path.join(os.path.dirname(input_file), f'mse_plots_{label}')
    os.makedirs(plot_path_mse, exist_ok=True)
    plot_path_max = os.path.join(os.path.dirname(input_file), f'max_plots_{label}')
    os.makedirs(plot_path_max, exist_ok=True)

    paths = {
        'mse': plot_path_mse,
        'max': plot_path_max
    }

    def plot_scores(data: xr.Dataset, score_name):
        sbin_ = int(data['score_bin'].values[0, 0])
        dbin_ = int(data['dz_bin'].values[0, 0])
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Bins (score, dz): ({}, {}) (score = {:.2f} - {:.2f}, dz = {:.2f} - {:.2f})'.format(sbin_, dbin_, data['score_min'].values[0, 0], data['score_max'].values[0, 0], data['dz_min'].values[0, 0], data['dz_max'].values[0, 0]))
        score = 1. - data[f'adaptive_{score_name}'].values / data[f'default_{score_name}'].values
        best = np.argmax(score.ravel())
        X, Y = np.meshgrid(data['max_cutoff'].values, data['min_cutoff'].values, indexing='xy')
        best_x = X.ravel()[best]
        best_y = Y.ravel()[best]
        ax.scatter([best_x], [best_y], color='k', zorder=40)
        p = ax.pcolor(X, Y, score, cmap='hot')
        ax.set(xlabel='Max cutoff', ylabel='Min cutoff')
        plt.colorbar(p, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(paths[score_name], 'scores_bin-{:02d}-{:02d}.png'.format(int(sbin_), int(dbin_))))
        plt.show()
        plt.close()

    data = pd.read_csv(input_file)

    grouped = data.groupby(['score_bin', 'dz_bin'])

    for bin, group in grouped:
        if group['score_count'].iloc[0] < 1000:
            continue
        bin_data = group.set_index(['min_cutoff', 'max_cutoff'])
        bin_data_xr = bin_data.to_xarray()
        plot_scores(bin_data_xr, 'mse')
        plot_scores(bin_data_xr, 'max')


if __name__ == '__main__':
    main()

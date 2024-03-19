import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import time




parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True)
args = vars(parser.parse_args())

plot_path = os.path.join(os.path.dirname(args['input_file']), 'score_plots')
os.makedirs(plot_path, exist_ok=True)

def plot_scores(data: xr.Dataset):
    data = data.sel(max_cutoff=(data['max_cutoff'].values >= 0.) )
    sbin_ = int(data['score_bin'].values[0, 0])
    dbin_ = int(data['dz_bin'].values[0, 0])
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Bins (score, dz): ({}, {}) (score = {:.2f} - {:.2f}, dz = {:.2f} - {:.2f})'.format(sbin_, dbin_, data['score_min'].values[0, 0], data['score_max'].values[0, 0], data['dz_min'].values[0, 0], data['dz_max'].values[0, 0]))
    score = 1. - data['adaptive'].values / data['default'].values
    best = np.argmax(score.ravel())
    X, Y = np.meshgrid(data['max_cutoff'].values, data['min_cutoff'].values, indexing='xy')
    best_x = X.ravel()[best]
    best_y = Y.ravel()[best]
    ax.scatter([best_x], [best_y], color='k', zorder=40)
    p = ax.pcolor(X, Y, score)
    ax.set(xlabel='Max cutoff', ylabel='Min cutoff')
    plt.colorbar(p, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'scores_bin-{:02d}-{:02d}.png'.format(int(sbin_), int(dbin_))))
    plt.show()
    plt.close()


data = pd.read_csv(args['input_file'])
num_j = data['score_bin'].values.max()
num_i = data['dz_bin'].values.max()

grouped = data.groupby(['score_bin', 'dz_bin'])

for bin, group in grouped:
    if group['score_count'].iloc[0] < 1000:
        continue
    bin_data = group.set_index(['min_cutoff', 'max_cutoff'])
    bin_data_xr = bin_data.to_xarray()
    plot_scores(bin_data_xr)


# fig, axs = plt.subplots(num_i, num_j, figsize=(15,10), sharex='all', sharey='all', gridspec_kw={'hspace': 0, 'wspace':0})
#
# grouped = data.groupby(['score_bin', 'dz_bin'])
#
# for bin, group in grouped:
#     j, i = bin
#     i = i - 1
#     j = j - 1
#     if group['score_count'].iloc[0] < 1000:
#         continue
#     bin_data = group.set_index(['min_cutoff', 'max_cutoff'])
#     bin_data_xr = bin_data.to_xarray()
#     plot_scores(bin_data_xr, axs[i, j])
#
# plt.tight_layout()
# plt.show()
# plt.close()
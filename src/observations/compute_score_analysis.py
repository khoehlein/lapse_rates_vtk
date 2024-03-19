import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import product
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True)
args = vars(parser.parse_args())

print('Loading')
train_split = pd.read_csv('/mnt/data2/ECMWF/Obs/train_stations.csv')
obs = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet', columns=['value_0', 'valid', 'elevation', 'stnid'])
pred = pd.read_parquet(args['input_file'])

print('Computing')
mask = np.all(np.stack([
    obs['stnid'].isin(train_split['stnid'].values),
    obs['valid'].values,
    # obs['elevation'].values <= pred['elevation_max'].values,
    ~np.isnan(pred['score'].values)
],axis=0), axis=0)

obs = obs.loc[mask]
pred = pred.loc[mask]


scores = pred['score'].values
score_rank = rankdata(scores)
# score_bin = np.floor(score_rank * (100 / len(score_rank)))
score_bin = np.digitize(np.fmax(scores, 0.), np.linspace(0, 1, 11))

dz = pred['elevation_difference'].values
dz_bin = np.digitize(dz, np.linspace(np.min(dz), np.max(dz), 11))
dz_abs = np.abs(dz)
order = np.argsort(dz_abs)
dz_max = np.max(dz_abs)

res_default = np.abs(obs['value_0'].values - (pred['hres'].values - 0.0065 * dz))

scores_per_bin = pd.DataFrame({
    'score_bin': score_bin,
    'dz_bin': dz_bin,
    'score': scores,
    'dz': dz,
}).groupby(['score_bin', 'dz_bin']).agg(['mean', 'min', 'max', 'count'])
scores_per_bin.columns = ['_'.join(c) for c in scores_per_bin.columns]

cutoffs_lower = np.arange(-15, -6)
cutoffs_upper = np.arange(0, 21)

all = []
for cutoff_min, cutoff_max in tqdm(product(cutoffs_lower, cutoffs_upper)):
    pred_adaptive = pred['hres'].values + np.clip(pred['lapse_rate'].values, cutoff_min, cutoff_max) / 1000 * dz
    res_adaptive = np.abs(obs['value_0'].values - pred_adaptive)
    df = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'adaptive': res_adaptive**2,
        'default': res_default**2,
    })
    metrics = pd.concat([df.groupby(['score_bin', 'dz_bin']).mean(), scores_per_bin], axis='columns')
    metrics = metrics.reset_index()
    metrics['min_cutoff'] = [cutoff_min] * len(metrics)
    metrics['max_cutoff'] = [cutoff_max] * len(metrics)

    all.append(metrics)

all = pd.concat(all, axis=0, ignore_index=True)
all.to_csv(os.path.join(os.path.dirname(args['input_file']), 'score_analysis.csv'))

# print('Plotting')
# fig, axs = plt.subplots(2, 1, figsize=(10, 5))
# axs[0].hist(scores, )
# ax.plot(metrics['score'].values, 1 - np.sqrt(metrics['adaptive'].values/metrics['default'].values))#, c=dz[order], vmin=-dz_max, vmax=dz_max)
# ax.axvline(1., color='k', linestyle='--')
# ax.axvline(0.98, color='k', linestyle='--')
# ax.axvline(0.95, color='k', linestyle='--')
# ax.axvline(0.9, color='k', linestyle='--')
# ax.axvline(0.8, color='k', linestyle='--')
# ax.axvline(0., color='k', linestyle='--')
# ax.axhline(0., color='k', linestyle='--')
# ax.set(xlabel='score', ylabel='residual')
# plt.tight_layout()
# plt.show()
# plt.close()

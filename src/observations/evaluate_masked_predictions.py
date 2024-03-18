import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True)
args = vars(parser.parse_args())

observations = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet', columns=['value_0', 'valid', 'stnid'])
mask = observations['valid'].values
stnids = observations['stnid'].values
predictions = pd.read_parquet(args['input_file'], columns=['value_0', 'stnid', 'hres', 'elevation_difference', 'lapse_rate'])

use_mask = True

difference = observations['value_0'].values - (predictions['hres'].values + np.clip(predictions['lapse_rate'].values, -13, 20) / 1000 * predictions['elevation_difference'])
if use_mask:
    difference = difference[mask]
    stnids = stnids[mask]

abs_difference = np.abs(difference)

df_adaptive = pd.DataFrame({
    'bias': difference,
    'mae': abs_difference,
    'mse': abs_difference ** 2.,
    'stnid': stnids,
    'dz': predictions['elevation_difference'].values[mask]
}).sort_index()

metrics_adaptive = df_adaptive.groupby('stnid').mean()

print(metrics_adaptive.describe())

difference = observations['value_0'].values - (predictions['hres'].values - 0.0065 * predictions['elevation_difference'].values)
if use_mask:
    difference = difference[mask]

abs_difference = np.abs(difference)

df_default = pd.DataFrame({
    'bias': difference,
    'mae': abs_difference,
    'mse': abs_difference ** 2.,
    'stnid': stnids,
    'dz': predictions['elevation_difference'].values[mask]
}).sort_index()

metrics_default = df_default.groupby('stnid').mean()

print(metrics_default.describe())

assert np.all(metrics_default.index.values == metrics_adaptive.index.values)

scores = 1 - (metrics_adaptive.mse.values / metrics_default.mse.values)

plt.figure()
plt.scatter(metrics_default.dz.values, scores, alpha=0.05)
plt.tight_layout()
plt.show()
plt.close()

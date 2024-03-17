import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

observations = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet', columns=['value_0', 'valid', 'stnid'])
mask = observations['valid'].values
stnids = observations['stnid'].values[mask]
obs = observations['value_0'].values[mask]
predictions = pd.read_parquet('/mnt/data2/ECMWF/Predictions/20240317212043/predictions.parquet', columns=['value_0', 'stnid', 'hres', 'elevation_difference', 'lapse_rate'])

t0 = predictions['hres'].values[mask]
dz = predictions['elevation_difference'].values[mask] / 1000.
lr = predictions['lapse_rate'].values[mask]

plt.figure()
plt.hist(lr + 13, bins=np.linspace(-20, 20, 100), log=True)
plt.tight_layout()
plt.show()
plt.close()

min_cutoffs = np.arange(-14, -6., 0.5)
max_cutoffs = np.arange(-6.5, 20.5, 0.5)

results = np.zeros((len(min_cutoffs), len(max_cutoffs)))

for i, min_cutoff in tqdm(enumerate(min_cutoffs), total=len(min_cutoffs)):
    for j, max_cutoff in enumerate(max_cutoffs):
        pred = t0 + np.clip(lr, min_cutoff, max_cutoff) * dz
        mse = np.mean((obs - pred) ** 2.)
        results[i, j] = mse

results = np.sqrt(results)

X, Y = np.meshgrid(max_cutoffs, min_cutoffs, indexing='xy')

fig, ax = plt.subplots(1, 1)
p = ax.pcolor(X, Y, results, vmin=0.)
plt.colorbar(p, ax=ax)
ax.set(xlabel='max cutoff', ylabel='min cutoff')
plt.tight_layout()
plt.show()
plt.close()


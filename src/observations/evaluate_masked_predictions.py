import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True)
args = vars(parser.parse_args())

observations = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet', columns=['value_0', 'valid', 'stnid'])
mask = observations['valid'].values
stnids = observations['stnid'].values
predictions = pd.read_parquet(args['input_file'], columns=['value_0', 'stnid', 'hres', 'elevation_difference'])

use_mask = True

difference = observations['value_0'].values - predictions['value_0'].values
if use_mask:
    difference = difference[mask]
    stnids = stnids[mask]

abs_difference = np.abs(difference)

df = pd.DataFrame({
    'bias': difference,
    'mae': abs_difference,
    'mse': abs_difference ** 2.,
    'stnid': stnids
})

metrics = df.groupby('stnid').mean()

print(metrics.describe())


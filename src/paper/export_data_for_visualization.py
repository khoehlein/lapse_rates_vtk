import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import DomainLimits, DomainBoundingBox, LocationBatch, Coordinates


DOMAIN_LIMITS = DomainLimits(34.5, 71.5, -11, 40)
MAX_RADIUS = 200


parser = argparse.ArgumentParser()
parser.add_argument('--obs-path', type=str)
parser.add_argument('--pred-path', type=str)
parser.add_argument('--out-path', type=str)
args = vars(parser.parse_args())

obs_path = args['obs_path']
pred_path = args['pred_path']
out_path = args['out_path']
meta_path = "/mnt/ssd4tb/ECMWF/Obs/station_locations_nearest.csv"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

metadata = pd.read_csv(meta_path, index_col=0).set_index('stnid').sort_index()
locations = LocationBatch(Coordinates.from_xarray(metadata))
bounding_box = DomainBoundingBox(DOMAIN_LIMITS)
in_bounding_box = bounding_box.contains(locations)
print(np.mean(in_bounding_box))

coords = locations.coords.as_xyz().values
lookup = NearestNeighbors()
lookup.fit(coords)
indices = lookup.radius_neighbors(coords[in_bounding_box], radius=MAX_RADIUS * 1000, return_distance=False)
indices = np.unique(np.concatenate(indices))
in_surrounding = metadata.index.isin(metadata.index.values[indices])

is_valid = np.logical_or(in_bounding_box, in_surrounding)
print(np.mean(is_valid) * 100)

valid_stnids = metadata.index.values[is_valid]

obs_data = pd.read_parquet(obs_path, columns=['date', 'time', 'value_0', 'stnid', 'valid'])
mask = np.logical_and(
    obs_data['valid'].values,
    obs_data['stnid'].isin(valid_stnids)
)
print(np.mean(mask) * 100)

obs_data = obs_data.loc[mask]
pred_data = pd.read_parquet(pred_path, columns=['value_0']).loc[mask]

stnids = obs_data['stnid'].values
dates = pd.to_datetime(obs_data['date'], format='%Y%m%d')
hours = pd.to_timedelta(obs_data['time'].values.astype(int) // 100, unit='hours')
timestamps = dates + hours
observations = obs_data['value_0'].values
predictions = pred_data['value_0'].values
difference = observations - predictions

out_data = pd.DataFrame({
    # 'date': dates,
    # 'hour': obs_data['time'].values.astype(int) // 100,
    'timestamp': timestamps,
    'stnid': stnids,
    'observation': observations,
    'prediction': predictions,
    'difference': difference,
    'abs_difference': np.abs(difference)
})

out_data.to_parquet(out_path)
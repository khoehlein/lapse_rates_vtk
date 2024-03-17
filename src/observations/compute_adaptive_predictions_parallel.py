import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

print(os.getcwd())
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import Coordinates


class LapseRateEstimator(object):

    def __init__(
            self,
            radius_km=60.,
            lsm_threshold=0.5,
            min_lr = -1000.,
            max_lr = 1000.,
            default_lr = -6.5,
            weight_scale = 30.,
            min_samples = 20,
            method='linreg',
            fit_intercept=True,
    ):
        self.radius = radius_km
        self.lsm_threshold = lsm_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.default_lr = default_lr
        self.weight_scale = weight_scale
        self.min_samples = min_samples
        self.fit_intercept = fit_intercept
        self.method = method

    def get_config_dict(self):
        return {
            'radius_km': self.radius,
            'lsm_threshold': self.lsm_threshold,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'default_lr': self.default_lr,
            'weight_scale': self.weight_scale,
            'min_samples': self.min_samples,
            'fit_intercept': self.fit_intercept,
            'method': self.method
        }

    def compute(self, df: pd.DataFrame):
        df = df.loc[df['lsm'].values >= self.lsm_threshold]
        if len(df) < self.min_samples:
            return self.default_lr
        t2m = df['t2m'].values
        z = df['z'].values
        weights = None
        if self.weight_scale is not None:
            try:
                distance = df['distance'].values
            except KeyError:
                warnings.warn('No distance available')
                distance = None
            if distance is not None:
                weights = np.exp(- (distance / self.weight_scale) ** 2. / 2.)
        if self.method == 'linreg':
            model = LinearRegression(fit_intercept=self.fit_intercept)
        elif self.method == 'ridge':
            model = Ridge(fit_intercept=self.fit_intercept, alpha=0.01)
        elif self.method == 'ransac':
            model = RANSACRegressor(
                Ridge(fit_intercept=self.fit_intercept, alpha=0.01),
                min_samples=4, max_trials=200, random_state=42
            )
        else:
            raise NotImplementedError()
        dt2m = t2m + 0.0065 * z
        model.fit(z[:, None], dt2m, sample_weight=weights)
        if self.method == 'ransac':
            model = model.estimator_
        lr_raw = (model.coef_[0] * 1000.) - 6.5
        lr = max(min(lr_raw, self.max_lr), self.min_lr)
        return lr


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='linreg', choices=['linreg', 'ridge', 'ransac'])
parser.add_argument('--radius', type=float, default=80.)
parser.add_argument('--weight-scale', type=float, default=40.)
args = vars(parser.parse_args())


temp_data_root_path = '/mnt/data2/ECMWF/Temp_Data'
temp_file_pattern = 'HRES_2m_temp_{}.grib'
raw_elevation_path = '/mnt/data2/ECMWF/Orog_Data/HRES_orog_o1279_2021-2022.grib'
raw_lsm_path = '/mnt/data2/ECMWF/LSM_Data/LSM_HRES_Sep2022.grib'
output_path = '/mnt/data2/ECMWF/Predictions'
observation_path = '/mnt/data2/ECMWF/Obs/observations_masked.parquet'


radius_km = args['radius']
method = args['method']
weight_scale = args['weight_scale']


estimator = LapseRateEstimator(
    radius_km=radius_km,
    method=method,
    weight_scale=weight_scale
)

observation_data = pd.read_parquet(observation_path, columns=['stnid', 'date', 'time'])
metadata = pd.read_csv(os.path.join(os.path.dirname(observation_path), 'station_locations_nearest.csv')).set_index('stnid')
elevation_data = xr.open_dataset(raw_elevation_path)['z']

def build_neighbor_data():
    lsm_data = xr.open_dataset(raw_lsm_path)['lsm']
    neighbor_lookup = NearestNeighbors()
    neighbor_lookup.fit(Coordinates.from_xarray(elevation_data).as_xyz().values)
    distances, neighbors = neighbor_lookup.radius_neighbors(
        Coordinates.from_xarray(metadata).as_xyz().values,
        radius=radius_km * 1000
    )
    neighbor_data = [
        pd.DataFrame({
            'neighbor': n,
            'distance': d / 1000.,
            'site': np.full_like(n, i)
        })
        for i, (d, n) in enumerate(zip(distances, neighbors))
        if len(n) > 0
    ]
    neighbor_data = pd.concat(neighbor_data, axis=0, ignore_index=True)
    neighbor_data['z'] = elevation_data.values[neighbor_data.neighbor.values]
    neighbor_data['lsm'] = lsm_data.values[neighbor_data.neighbor.values]
    neighbor_data['stnid'] = metadata.index.values[neighbor_data.site.values]
    return neighbor_data

neighbor_data = build_neighbor_data()
neighbor_data = neighbor_data.set_index(['stnid', 'neighbor'])

def process_day(x):
    date, observations_per_date = x
    t1 = time.time()

    path_to_predictions = os.path.join(temp_data_root_path, temp_file_pattern.format(date))
    predictions_per_date = xr.open_dataset(path_to_predictions)['t2m'].stack(hour=('time', 'step'))

    output_per_day = []
    for hour, observations_per_hour in observations_per_date.groupby('time'):
        station_ids = observations_per_hour['stnid'].values
        if not len(station_ids):
            continue
        site_metadata = metadata.loc[station_ids]
        nearest_node = site_metadata['nearest_node_o1280'].values
        predictions_per_site = predictions_per_date.isel(hour=int(hour) // 100, values=nearest_node)
        elevation_per_site = elevation_data.isel(values=nearest_node)
        dz = site_metadata['elevation'].values - elevation_per_site

        site_neighbor_data = neighbor_data.loc[(station_ids, slice(None))]
        predictions_in_surrounding = predictions_per_date.isel(
            hour=int(hour) // 100,
            values=site_neighbor_data.index.get_level_values(1).values
        ).values
        site_neighbor_data['t2m'] = predictions_in_surrounding
        neighbor_groups = site_neighbor_data.groupby('stnid')
        lapse_rates = neighbor_groups.apply(estimator.compute)
        lapse_rates = lapse_rates.loc[station_ids].values
        aggregates = neighbor_groups['z'].agg(['min', 'max', 'count'])
        aggregates = aggregates.loc[station_ids].values

        t_pred = predictions_per_site.values + lapse_rates / 1000. * dz

        data = pd.DataFrame({
            'stnid': station_ids,
            'latitude': site_metadata['latitude'].values,
            'longitude': site_metadata['longitude'].values,
            'date': [date] * len(station_ids),
            'time': [hour] * len(station_ids),
            'elevation': site_metadata['elevation'].values,
            'elevation_min': aggregates[:, 0],
            'elevation_max': aggregates[:, 1],
            'neighbor_count': aggregates[:, 2],
            'lapse_rate': lapse_rates,
            'hres': predictions_per_site.values,
            'elevation_difference': dz,
            'value_0': t_pred
        }, index=observations_per_hour.index)
        output_per_day.append(data)

    output_per_day = pd.concat(output_per_day, axis=0)
    output_per_day.sort_index(inplace=True)
    t2 = time.time()
    print(f'Date {date} completed in {t2 - t1} sec')
    return output_per_day

def model_predictions_adaptive_lapse(use_3d_data=False):
    dates = list(observation_data.groupby('date'))
    print('Running predictions')
    with Pool(6) as p:
        site_predictions = p.map(process_day, dates)
    # site_predictions = [process_day(k) for k in dates]
    site_predictions = pd.concat(site_predictions, axis=0)
    output_path_ = os.path.join(output_path, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    os.makedirs(output_path_, exist_ok=True)
    with open(os.path.join(output_path_, 'parameters.json'), 'w') as f:
        configs = estimator.get_config_dict()
        configs['r_km'] = radius_km
        configs['use_3d'] = use_3d_data
        json.dump(configs, f, indent=4, sort_keys=True)
    site_predictions.to_parquet(os.path.join(output_path_, 'predictions.parquet'))

if __name__ == '__main__':
    model_predictions_adaptive_lapse()

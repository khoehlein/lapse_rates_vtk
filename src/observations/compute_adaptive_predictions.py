import datetime
import json
import os
print(os.getcwd())
import warnings

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import Coordinates


class LapseRateEstimator(object):

    def __init__(
            self,
            min_lr = -13., max_lr = 13.,
            default_lr = -6.5,
            weight_scale = None,
            min_samples = 20,
            fit_intercept=True
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.default_lr = default_lr
        self.weight_scale = weight_scale
        self.min_samples = min_samples
        self.fit_intercept = fit_intercept

    def get_config_dict(self):
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'default_lr': self.default_lr,
            'weight_scale': self.weight_scale,
            'min_samples': self.min_samples,
            'fit_intercept': self.fit_intercept,
        }

    def compute(self, df: pd.DataFrame):
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

        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(z[:, None], t2m, sample_weight=weights)
        lr_raw = model.coef_[0] * 1000.
        lr = max(min(lr_raw, self.max_lr), self.min_lr)
        return lr


temp_data_root_path = '/path/to/data/Temp_Data'
temp_file_pattern = 'HRES_2m_temp_{}.grib'
raw_elevation_path = '/path/to/data/Orog_Data/HRES_orog_o1279_2021-2022.grib'
cache_path = '/path/to/data/Cache/predictions_by_day'
output_path = '/path/to/data/Predictions'
observation_path = '/path/to/data/Obs/observations_masked.parquet'


if cache_path is not None:
    os.makedirs(cache_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

print()
observation_data = pd.read_parquet(observation_path)
metadata = pd.read_csv(os.path.join(os.path.dirname(observation_path), 'station_locations_nearest.csv')).set_index('stnid')
elevation_data = xr.open_dataset(raw_elevation_path)['z']
neighbor_lookup = NearestNeighbors()
neighbor_lookup.fit(Coordinates.from_xarray(elevation_data).as_xyz().values)


def build_neighbor_data(radius_km):
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
    neighbor_data['stnid'] = metadata.index.values[neighbor_data.site.values]
    return neighbor_data


def model_predictions_adaptive_lapse(radius_km: float, use_3d_data=False):

    estimator = LapseRateEstimator()

    neighbor_data = build_neighbor_data(radius_km)
    neighbor_data = neighbor_data.set_index(['stnid', 'neighbor'])

    site_predictions = []

    for date, observations_per_date in observation_data.groupby('date'):

        path_to_predictions = os.path.join(temp_data_root_path, temp_file_pattern.format(date))
        predictions_per_date = xr.open_dataset(path_to_predictions)['t2m'].stack(hour=('time', 'step'))

        output_per_day = []
        with tqdm.tqdm(total=24, desc=date) as pbar:
            for hour, observations_per_hour in observations_per_date.groupby('time'):
                station_ids = observations_per_hour['stnid'].values
                if len(station_ids):
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
                    lr = neighbor_groups.apply(estimator.compute)

                    aggregates = neighbor_groups['z'].agg(['min', 'max', 'count'])

                    t_pred = predictions_per_site.values + lr.values * dz
                    data = pd.DataFrame({
                        'stnid': station_ids,
                        'latitude': site_metadata['latitude'].values,
                        'longitude': site_metadata['longitude'].values,
                        'date': [date] * len(station_ids),
                        'time': [hour] * len(station_ids),
                        'elevation': site_metadata['elevation'].values,
                        'elevation_min': aggregates['min'].values,
                        'elevation_max': aggregates['max'].values,
                        'neighbor_count': aggregates['count'].values,
                        'lapse_rate': lr.values,
                        'value_0': t_pred
                    }, index=observations_per_hour.index)
                    output_per_day.append(data)
                pbar.update(1)

        output_per_day = pd.concat(output_per_day, axis=0)
        output_per_day.sort_index(inplace=True)
        if cache_path is not None:
            output_per_day.to_parquet(os.path.join(cache_path, f'predictions_adaptive_{date}.parquet'))
        site_predictions.append(output_per_day)

    site_predictions = pd.concat(site_predictions, axis=0)
    output_path_ = os.path.join(output_path, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    os.makedirs(output_path_, exist_ok=True)
    with open(os.path.join(output_path_, 'config.json')) as f:
        configs = estimator.get_config_dict()
        configs['r_km'] = radius_km
        configs['use_3d'] = use_3d_data
        json.dump(configs, f)
    site_predictions.to_parquet(os.path.join(output_path_, 'predictions.parquet'))


if __name__ == '__main__':
    model_predictions_adaptive_lapse(30.)

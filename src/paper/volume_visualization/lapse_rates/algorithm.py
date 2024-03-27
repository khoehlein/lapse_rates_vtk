import warnings
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import Coordinates


class LapseRateEstimator(object):

    def __init__(self, weight_scale_km = None, min_samples = 3, fit_intercept=True):
        self.weight_scale_km = weight_scale_km
        self.min_samples = min_samples
        self.fit_intercept = fit_intercept

    def compute(self, df: pd.DataFrame):
        if len(df) < self.min_samples:
            return pd.Series({'lapse_rate': np.nan, 'score': np.nan, 'intercept': np.nan})
        t2m = df['t'].values
        z = df['z'].values
        weights = None
        if self.weight_scale_km is not None:
            try:
                distance = df['distance_km'].values
            except KeyError:
                warnings.warn('No distance available')
                distance = None
            if distance is not None:
                weights = np.exp(- (distance / self.weight_scale_km) ** 2. / 2.)

        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(z[:, None], t2m, sample_weight=weights)
        lr_raw = model.coef_[0] * 1000.
        score = model.score(z[:, None], t2m, sample_weight=weights)
        return pd.Series({'lapse_rate': lr_raw, 'score': score, 'intercept': model.intercept_})


class LapseRateData(object):

    def __init__(
            self,
            model_data: xr.Dataset, terrain_data: xr.Dataset,
            model_lookup=None, terrain_level_key: str = 'z_surf', value_key: str = 't2m'
    ):
        self.model_data = model_data
        self.terrain_data = terrain_data
        self._model_coords = Coordinates.from_xarray(terrain_data).as_xyz().values

        if model_lookup is None:
            self._model_lookup = NearestNeighbors()
            self._model_lookup.fit(self._model_coords)
        else:
            self._model_lookup = model_lookup
        self.num_sites = len(self._model_coords)

        self.terrain_level_key = terrain_level_key
        self.value_key = value_key

        self.properties = None
        self.nearest_sites = None
        self.neighbor_graph = None
        self._neighbor_groups = None
        self.data = None

    def update(self, properties):
        self.properties = properties
        self._query_neighbor_graph()
        self._augment_neighbor_graph()
        self._filter_non_land_stations()
        self._neighbor_groups = self.neighbor_graph.groupby('site')
        self._compute_lapse_rates()
        return self

    def get_data_at_closest_site(self, coords: np.ndarray):
        if self.properties is None:
            return None
        neighbors = self._model_lookup.kneighbors(coords, n_neighbors=1, return_distance=False).ravel()
        return self.data.loc[neighbors]

    def _query_neighbor_graph(self):
        distances, neighbors = self._model_lookup.radius_neighbors(self._model_coords, radius=self.properties.radius_km * 1000)
        distances = np.concatenate(distances, axis=0)
        neighbor_ids = np.concatenate(neighbors, axis=0)
        num_neighbors = np.asarray([len(x) for x in neighbors])
        indptr = np.cumsum(num_neighbors)
        site_index = np.zeros(indptr[-1], dtype=int)
        site_index[indptr[:-1]] = 1
        site_ids = np.cumsum(site_index)
        self.neighbor_graph = pd.DataFrame({
            'neighbor': neighbor_ids,
            'site': site_ids,
            'distance_km': distances / 1000.
        })
        return self

    def _augment_neighbor_graph(self):
        neighbor_id = self.neighbor_graph['neighbor'].values
        self.neighbor_graph['z'] = self.terrain_data[self.terrain_level_key][neighbor_id]
        self.neighbor_graph['lsm'] = self.terrain_data['lsm'][neighbor_id]
        self.neighbor_graph['t'] = self.model_data[self.value_key][neighbor_id]
        return self

    def _filter_non_land_stations(self):
        valid = self.neighbor_graph['lsm'].values > self.properties.lsm_threshold
        self.neighbor_graph = self.neighbor_graph.loc[valid]

    def _compute_lapse_rates(self):
        estimator = LapseRateEstimator(weight_scale_km=self.properties.weight_scale_km)
        lapse_rates = self._neighbor_groups.apply(estimator.compute)

        z_summary = self._neighbor_groups['z'].agg(['count', 'mean', 'min', 'max'])
        z_summary.columns = ['neighbor_count', 'z_mean', 'z_min', 'z_max']
        z_summary['z_range'] = z_summary['z_max'].values - z_summary['z_min'].values

        t_summary = self._neighbor_groups['t'].agg(['mean', 'min', 'max'])
        t_summary.columns = ['t2m_mean', 't2m_min', 't2m_max']
        t_summary['t2m_range'] = t_summary['t2m_max'].values - t_summary['t2m_min'].values

        data = pd.concat([lapse_rates, z_summary, t_summary], axis=1)
        site_ids = np.arange(self.num_sites)
        valid = np.isin(site_ids, data.index.values)
        data = data.reindex(site_ids)
        data.sort_index(inplace=True)
        data['lapse_rate'].fillna(self.properties.default_lapse_rate, inplace=True)
        data['score'].fillna(0., inplace=True)
        data['neighbor_count'].fillna(0, inplace=True)
        data['z_range'].fillna(0., inplace=True)
        data['t2m_range'].fillna(0., inplace=True)
        z_nearest = self.terrain_data[self.terrain_level_key].values
        for key in ['z_mean', 'z_min', 'z_max']:
            values = data[key].values
            data[key] = np.where(np.isnan(values), z_nearest, values)
        t_nearest = self.model_data[self.value_key].values
        for key in ['t2m_mean', 't2m_min', 't2m_max']:
            values = data[key].values
            data[key] = np.where(np.isnan(values), t_nearest, values)
        data['z_nearest'] = z_nearest
        data['t2m_nearest'] = t_nearest
        data['valid'] = valid.astype(int)
        self.data = data

import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from PyQt5.QtCore import QObject
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree

from src.model.data_store.world_data import SampleBatch
from src.model.geometry import SurfaceDataset, LocationBatch


class DownscalerModel(QObject):

    def __init__(self, properties_class, parent=None):
        super().__init__(parent)
        self._properties_class = properties_class

    def set_downscaler_properties(self, properties):
        raise NotImplementedError()

    def validate_downscaler_properties(self, properties):
        assert isinstance(properties, self._properties_class)

    def compute_temperatures(self, target: SurfaceDataset, source: SurfaceDataset, samples: SampleBatch) -> np.ndarray:
        raise NotImplementedError()


@dataclass(init=True, repr=True)
class LapseRateDownscalerProperties():
    use_volume: bool
    use_weights: bool
    weight_scale_km: float
    min_num_neighbors: int
    fit_intercept: bool
    default_lapse_rate: float


class LapseRateDownscaler(DownscalerModel):

    def __init__(self, parent=None):
        super().__init__(LapseRateDownscalerProperties, parent)
        self.use_volume = None
        self.use_weights = None
        self.weight_scale_km = None
        self.min_num_neighbors = None
        self.fit_intercept = False
        self.default_lapse_rate = - 0.0065

    def set_downscaler_properties(self, properties: LapseRateDownscalerProperties):
        self.use_volume = properties.use_volume
        self.use_weights = properties.use_weights
        self.weight_scale_km = properties.weight_scale_km
        self.min_num_neighbors = properties.min_num_neighbors
        self.fit_intercept = properties.fit_intercept
        self.default_lapse_rate = properties.default_lapse_rate
        return self

    def compute_temperatures(self, target: SurfaceDataset, source: SurfaceDataset, samples: SampleBatch) -> Dict[str, SurfaceDataset]:
        output = self._compute_lapse_rates_at_sample_locations(source, samples)
        output = self._interpolate_to_target_locations(output, target)
        return {'surface_o8000': output, 'surface_o1280': source}

    def _compute_lapse_rates_at_sample_locations(self, source: SurfaceDataset, samples: SampleBatch) -> Tuple[LocationBatch, np.ndarray, np.ndarray, np.ndarray]:
        logging.info('Computing lapse rate at sample locations')
        num_links = samples.source_reference.num_links
        site_id_at_link = samples.source_reference.links['location'].values
        t2m_at_site = samples.data[0].t2m.values
        t2m_site_at_link = t2m_at_site[site_id_at_link]
        z_at_site = samples.data[0].z.values
        z_site_at_link = z_at_site[site_id_at_link]
        split_indices = np.unique(np.cumsum(num_links))
        dt_around_site = np.split(samples.data[1].t2m.values - t2m_site_at_link, split_indices)
        dz_around_site = np.split(samples.data[1].z.values - z_site_at_link, split_indices)
        lapse_rates = np.full_like(t2m_at_site, - 0.0065)
        mask = num_links > 0
        count = int(np.sum(mask))
        lapse_rates[mask] = np.fromiter(
            (self._estimate_lapse_rate(dt, dz) for dt, dz in zip(dt_around_site, dz_around_site)),
            count=count, dtype=float
        )
        source.add_scalar_field(lapse_rates, 'lapse_rate')
        source.add_scalar_field(t2m_at_site, 't2m_o1280')
        source.add_scalar_field(z_at_site, 'z_o1280')
        return samples.locations, t2m_at_site, z_at_site, lapse_rates

    def _interpolate_to_target_locations(self, output, target: SurfaceDataset) -> SurfaceDataset:
        logging.info('Interpolating data to target locations')
        locations, t2m_at_site, z_at_site, lapse_rates = output
        source_coords = locations.coords.as_geocentric().values
        tree = KDTree(source_coords)
        target_coords = target.mesh.locations.coords.as_geocentric().values
        nearest = tree.query(target_coords, k=1, return_distance=False).ravel()
        lapse_rate_at_target = lapse_rates[nearest]
        t2m_lowres_at_target = t2m_at_site[nearest]
        z_lowres_at_target = z_at_site[nearest]
        t2m_highres_at_target = t2m_lowres_at_target + lapse_rate_at_target * (target.z - z_lowres_at_target)
        target.add_scalar_field(lapse_rate_at_target, 'lapse_rate')
        target.add_scalar_field(t2m_lowres_at_target, 't2m_o1280')
        target.add_scalar_field(t2m_highres_at_target, 't2m_o8000')
        target.add_scalar_field(z_lowres_at_target, 'z_o1280')
        return target

    def _estimate_lapse_rate(self, dt, dz):
        if len(dt) < self.min_num_neighbors:
            return self.default_lapse_rate
        lm = LinearRegression(fit_intercept=self.fit_intercept)
        lm.fit(dz[:, None], dt)
        return lm.coef_[0]

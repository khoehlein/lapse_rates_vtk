import logging
from enum import Enum
from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.linear_model import LinearRegression

from src.interaction.downscaling.domain_selection import DomainData
from src.interaction.downscaling.downscaling import DownscalerModel, InterpolationType, NearestNeighborInterpolation, \
    BarycentricInterpolation, SurfaceFieldType, OutputDataset, GridConfiguration
from src.interaction.downscaling.neighborhood_lookup import NeighborhoodModel
from src.model.visualization.interface import PropertyModel

import xarray as xr


class DownscalingProcess(PropertyModel):

    def __init__(self, source_domain: DomainData, target_domain: DomainData):
        super().__init__(None)
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.cache = {}
        self.output = None

    def reset_cache(self):
        self.cache.clear()

    def run(self):
        raise NotImplementedError()


class _LowresDownscalingProcess(DownscalingProcess):

    class Properties(DownscalingProcess.Properties):
        interpolation_type = InterpolationType

    def __init__(self, source_domain: DomainData, target_domain: DomainData):
        super().__init__(source_domain, target_domain)
        self.interpolator = None
        self._interpolators = {
            InterpolationType.NEAREST_NEIGHBOR: NearestNeighborInterpolation,
            InterpolationType.BARYCENTRIC: BarycentricInterpolation,
        }

    def set_properties(self, properties) -> '_LowresDownscalingProcess':
        super().set_properties(properties)
        self.update_interpolator()
        return self

    def update_interpolator(self):
        self.interpolator = self._get_interpolator()

    def _get_interpolator(self, properties = None):
        if self.properties is None:
            return None
        cls = self._interpolators[properties.interpolation_type]
        return cls(self.source_domain.data)


class DefaultDownscalingProcess(_LowresDownscalingProcess):

    class Properties(_LowresDownscalingProcess.Properties):
        lapse_rate: float

    def _get_interpolated_source_data(self):
        if 'interpolated_source_data' not in self.cache:
            target_sites = self.target_domain.sites
            self.cache['interpolated_source_data'] = self.interpolator.interpolate(target_sites, variables=['t2m', 'z'])
        return self.cache['interpolated_source_data']

    def update_interpolator(self):
        self.interpolator = self._get_interpolator()
        if 'interpolated_source_data' in self.cache:
            del self.cache['interpolated_source_data']

    def run(self):
        if self.properties is None:
            raise RuntimeError('[ERROR] Error applying default downscaler: Properties not set.')

        data_lr_to_hr = self._get_interpolated_source_data()
        z_hr = self.target_domain.data['z']
        dz = z_hr.values - data_lr_to_hr['z'].values
        dt2m =  (self.properties.lapse_rate / 1000.) * dz
        t2m_hr = data_lr_to_hr['t2m'].values + dt2m

        data_lr = self.source_domain.data
        data_hr = self.target_domain.data
        data_hr = data_hr.assign({
            SurfaceFieldType.T2M.value: ('values', t2m_hr),
            SurfaceFieldType.T2M_INTERPOLATION: data_lr_to_hr['t2m'].rename(SurfaceFieldType.T2M_INTERPOLATION),
            SurfaceFieldType.T2M_DIFFERENCE: ('values', dt2m),
            SurfaceFieldType.LAPSE_RATE.value: ('values', np.full_like(t2m_hr, self.properties.lapse_rate)),
            SurfaceFieldType.Z_INTERPOLATION.value: ('values', data_lr_to_hr['z'].rename(SurfaceFieldType.Z_INTERPOLATION.value)),
            SurfaceFieldType.Z_DIFFERENCE.value: ('values', dz),
        })

        output = OutputDataset()
        output_hr = output.get_group(GridConfiguration.O8000)
        for field_type in self.outputs_highres():
            output_hr.add_surface_field(field_type, data_hr[field_type.value])
        output_lr = output.get_group(GridConfiguration.O1280)
        for field_type in self.outputs_lowres():
            output_lr.add_surface_field(field_type, data_lr[field_type.value])

        self.cache['output'] = output

        return output

    def outputs_highres(self):
        return [
            SurfaceFieldType.T2M,
            SurfaceFieldType.T2M_INTERPOLATION,
            SurfaceFieldType.T2M_DIFFERENCE,
            SurfaceFieldType.LAPSE_RATE,
            SurfaceFieldType.Z,
            SurfaceFieldType.Z_INTERPOLATION,
            SurfaceFieldType.Z_DIFFERENCE,
            SurfaceFieldType.LSM
        ]

    def outputs_lowres(self):
        return [
            SurfaceFieldType.T2M,
            SurfaceFieldType.LAPSE_RATE,
            SurfaceFieldType.Z,
            SurfaceFieldType.LSM
        ]


class DownscalingMethod(PropertyModel):

    def process(self, *args, **kwargs):
        raise NotImplementedError()


class _NeighborhoodDownscalingProcess(_LowresDownscalingProcess):

    class Properties(_LowresDownscalingProcess.Properties):
        neighborhood_model: NeighborhoodModel.Properties
        downscaling_method: DownscalingMethod.Properties

    def __init__(
            self,
            source_domain: DomainData, target_domain: DomainData,
            downscaling_method: DownscalingMethod
    ):
        super().__init__(source_domain, target_domain)
        self.neighborhood_model = NeighborhoodModel(source_domain.data_store)
        self._neighborhood_data = None
        self.downscaling_method = downscaling_method

    def synchronize_properties(self):
        self.properties.neighborhood_model = self.neighborhood_model.properties
        self.properties.downscaling_method = self.downscaling_method.properties


    def set_properties(self, properties: '_NeighborhoodDownscalingProcess.Properties') -> '_NeighborhoodDownscalingProcess':
        super().set_properties(properties)
        self.neighborhood_model.set_properties(properties.neighborhood_model)
        if 'neighborhood_data' in self.cache:
            del self.cache['neighborhood_data']
        self.downscaling_method.set_properties(properties.downscaling_method)
        return self


class AdaptiveLapseRateDownscaling(DownscalingMethod):

    class Properties(DownscalingMethod.Properties):
        max_lapse_rate: float
        min_lapse_rate: float
        min_num_neighbors: int
        fit_intercept: bool
        default_lapse_rate: float
        weight_scale: float
        use_weight_scale: bool
        use_volume_data: bool

    def __init__(self):
        super().__init__(None)

    def process(self, site_data: xr.Dataset, neighborhood_data: xr.Dataset, target_data):
        if self.properties is None:
            raise RuntimeError('[ERROR] Error in applying lapse rate downscaler: properties not set')
        neighborhood_data, neighborhood_graph = neighborhood_data
        logging.info('Computing lapse rate at sample locations')
        num_links = neighborhood_graph.num_links
        site_id_at_link = neighborhood_graph.links['location'].values
        distance_at_link = neighborhood_graph.links['distance'].values
        t2m_at_site = site_data.t2m.values
        t2m_site_at_link = t2m_at_site[site_id_at_link]
        z_at_site = site_data.z.values
        z_site_at_link = z_at_site[site_id_at_link]
        split_indices = np.unique(np.cumsum(num_links))
        dt_around_site = np.split(neighborhood_data.t2m.values - t2m_site_at_link, split_indices)
        dz_around_site = np.split(neighborhood_data.z.values - z_site_at_link, split_indices)
        distance_around_site = np.split(distance_at_link, split_indices)
        lapse_rates = np.full_like(t2m_at_site, - 0.0065)
        mask = num_links > 0
        count = int(np.sum(mask))
        lapse_rates[mask] = np.fromiter(
            (
                self._estimate_lapse_rate(dt, dz, d)
                for dt, dz, d in zip(dt_around_site, dz_around_site, distance_around_site)
            ),
            count=count, dtype=float
        )
        # TODO: Implement output preparation
        return

    def _estimate_lapse_rate(self, dt, dz, d):
        if len(dt) < self.properties.min_num_neighbors:
            return self.properties.default_lapse_rate
        lm = LinearRegression(fit_intercept=self.properties.fit_intercept)
        if self.properties.use_weights:
            weights = np.exp(-(d / (self.properties.weight_scale_km * 1000.)) ** 2.)
        else:
            weights = None
        lm.fit(dz[:, None], dt, sample_weight=weights)
        return lm.coef_[0]

class AdaptiveLapseRateDownscalingProcess(_NeighborhoodDownscalingProcess):

    def __init__(
            self,
            source_domain: DomainData, target_domain: DomainData,
    ):
        method = AdaptiveLapseRateDownscaling()
        super().__init__(
            source_domain, target_domain,
            method
        )

    def _get_neighborhood_data(self):
        if 'neighborhood_data' not in self.cache:
            self.cache['neighborhood_data'] = self.neighborhood_model.query_neighbor_data(self.source_domain.sites)
        return self.cache['neighborhood_data']

    def run(self):
        site_data = self.source_domain.data
        neighborhood_data = self._get_neighborhood_data()
        target_data = self.target_domain.sites
        self.output = self.downscaling_method.process(site_data, neighborhood_data, target_data)


class DownscalingPipelineController(QObject):

    def from_settings(
            self,
            neighborhood_settings: NeighborhoodModel.Properties,
            downscaling_settings: DownscalerModel.Properties,
    ):


    def __init__(self, view: DownscalingPipelineView, model: DownscalingModel, parent=None):
        super(DownscalingPipelineController, self).__init__(parent)
        self.view = view
        self.model = model


class NeighborhoodData(object):

    def __init__(self, domain_model: DomainData):
        self.domain_model = domain_model
        self.lookup = NeighborhoodLookup(self.data_store)
        self._neighborhood_graph = None
        self.data = None

    @property
    def data_store(self):
        return self.domain_model.data_store

    def set_neighborhood_properties(self, properties: NeighborhoodLookup.Properties):
        self.lookup.set_properties(properties)
        self.update()
        return self

    def update(self):
        domain_sites = self.domain_model.sites
        self._neighborhood_graph = self.lookup.query_neighbor_graph(domain_sites)
        self.data = self.data_store.query_site_data(self._neighborhood_graph.links['neighbors'])

    def query_neighbor_graph(self, locations: LocationBatch) -> NeighborhoodGraph:
        if self.lookup is None:
            raise RuntimeError('[ERROR]  Error while querying neighborhood graph: lookup not found.')
        return self.lookup.query_neighbor_graph(locations)

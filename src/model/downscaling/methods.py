from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from src.model.data.data_source import MultiFieldSource
from src.model.data.data_store import DomainData, GlobalData
from src.model.data.output import OutputDataset
from src.model.downscaling.interpolation import InterpolationType, InterpolationModel
from src.model.downscaling.neighborhood import NeighborhoodModel, NeighborhoodType, DEFAULT_NEIGHBORHOOD_RADIAL
from src.model.interface import PropertyModel, SurfaceFieldType, GridConfiguration, FilterNodeModel


class DownscalerType(Enum):
    FIXED_LAPSE_RATE = 'fixed_lapse_rate'
    ADAPTIVE_LAPSE_RATE = 'adaptive_lapse_rate'
    NETWORK = 'network'


class DownscalingMethodModel(FilterNodeModel):

    class Properties(FilterNodeModel.Properties):
        pass

    @classmethod
    def from_settings(cls, settings: 'DownscalingMethodModel.Properties', data_store: GlobalData):
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self.source_data: MultiFieldSource = None
        self.target_data: MultiFieldSource = None
        self.outputs = MultiFieldSource([SurfaceFieldType.T2M.value, SurfaceFieldType.T2M_DIFFERENCE])
        self.register_input('source_data', MultiFieldSource)
        self.register_input('target_data', MultiFieldSource)
        self.register_output('outputs', MultiFieldSource)

    def set_data(self, source_data: MultiFieldSource, target_data: MultiFieldSource):
        self.source_data = source_data
        self.target_data = target_data
        self.outputs.set_mesh(self.target_data.mesh_source)
        self.set_outputs_valid(False)
        return self

    def set_properties(self, properties: 'DownscalingMethodModel.Properties') -> 'DownscalingMethodModel':
        super().set_properties(properties)
        if self.properties_changed():
            self.outputs.set_valid(False)
        return self

    def supports_update(self, properties):
        return isinstance(properties, self.Properties)


class _InterpolatedDownscaler(DownscalingMethodModel):

    @dataclass
    class Properties(DownscalingMethodModel.Properties):
        interpolation: InterpolationModel.Properties

    def __init__(self, interpolation: InterpolationModel):
        super().__init__()
        self.interpolation: InterpolationModel = interpolation

    def synchronize_properties(self):
        self.properties.interpolation = self.interpolation.properties
        return self

    def set_data(self, source_data: MultiFieldSource, target_data: MultiFieldSource):
        self.interpolation.set_source(source_data)
        self.interpolation.set_target_mesh(target_data.mesh_source)
        super().set_data(source_data, target_data)
        return self

    def set_properties(self, properties: '_InterpolatedDownscaler.Properties') -> '_InterpolatedDownscaler':
        self.interpolation.set_properties(properties)
        super().set_properties(properties)
        return self


class FixedLapseRateDownscaler(_InterpolatedDownscaler):

    @dataclass
    class Properties(_InterpolatedDownscaler.Properties):
        lapse_rate: float

    def __init__(self):
        interpolation = InterpolationModel([SurfaceFieldType.T2M.value, SurfaceFieldType.Z.value])
        super().__init__(interpolation)

    @classmethod
    def from_settings(cls, settings: 'FixedLapseRateDownscaler.Properties', data_store: GlobalData):
        model = cls()
        model.set_properties(settings)
        return model

    def update_outputs(self):
        self.interpolation.update()
        interp_data = self.interpolation.outputs.data()
        target_data = self.target_data.data()

        t2m_model_interp = interp_data.t2m.values
        lapse_rate = np.full_like(t2m_model_interp, self.properties.lapse_rate / 1000.)

        dz = target_data.z.values - interp_data.z.values
        dt2m = lapse_rate * dz
        t2m_hr = t2m_model_interp + dt2m

        output_dataset = xr.Dataset(
            data_vars={
                SurfaceFieldType.T2M.value: ('values', t2m_hr),
                SurfaceFieldType.T2M_DIFFERENCE.value: ('values', dt2m),
                SurfaceFieldType.Z_DIFFERENCE.value: ('values', dz),
            },
        )
        self.outputs.set_data(output_dataset)

        return self


DEFAULTS_FIXED_LAPSE_RATE = FixedLapseRateDownscaler.Properties(
    interpolation=InterpolationModel.Properties(
        type=InterpolationType.NEAREST_NEIGHBOR
    ),
    lapse_rate=-6.5,
)


class LapseRateEstimator(FilterNodeModel):

    @dataclass
    class Properties(FilterNodeModel.Properties):
        use_volume: bool
        use_weights: bool
        weight_scale_km: float
        min_num_neighbors: int
        fit_intercept: bool
        default_lapse_rate: float
        neighborhood: NeighborhoodModel.Properties

    def __init__(self, neighborhood_model: NeighborhoodModel):
        super().__init__()
        self.neighborhood = neighborhood_model
        self.source_data: MultiFieldSource = None
        self.target_data: MultiFieldSource = None
        self.output = MultiFieldSource(scalar_names=[SurfaceFieldType.LAPSE_RATE.value])
        self.register_input('source_data', MultiFieldSource)
        self.register_input('target_data', MultiFieldSource)
        self.register_output('output', MultiFieldSource)

    def set_data(self, source_data: MultiFieldSource, target_data: MultiFieldSource):
        self.source_data = source_data
        self.target_data = target_data
        self.neighborhood.set_domain(self.source_data)
        self.set_outputs_valid(False)
        return self

    def set_properties(self, properties: 'LapseRateEstimator.Properties') -> 'LapseRateEstimator':
        super().set_properties(properties)
        if self.properties_changed():
            self.neighborhood.set_properties(self.properties.neighborhood)
            self.set_outputs_valid(False)
        return self

    def synchronize_properties(self):
        self.properties.neighborhood = self.neighborhood.properties
        return self

    def update_outputs(self):
        self.neighborhood.update()

        site_data = self.source_data.data
        neighbor_data = self.neighborhood.data
        neighbor_graph = self.neighborhood.graph

        num_links = neighbor_graph.num_links
        site_id_at_link = neighbor_graph.links['location'].values
        distance_at_link = neighbor_graph.links['distance'].values
        t2m_at_site = site_data.t2m.values
        t2m_site_at_link = t2m_at_site[site_id_at_link]
        z_at_site = site_data.z.values
        z_site_at_link = z_at_site[site_id_at_link]
        split_indices = np.unique(np.cumsum(num_links))
        dt_around_site = np.split(neighbor_data.t2m.values - t2m_site_at_link, split_indices)
        dz_around_site = np.split(neighbor_data.z.values - z_site_at_link, split_indices)
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

        coords = self.source_data.sites.coords.as_lat_lon()
        self.output = xr.DataArray(
            data=lapse_rates,
            dims=['values'],
            name=SurfaceFieldType.LAPSE_RATE.value,
            coords={
                'latitude': ('values', coords.y),
                'longitude': ('values', coords.x),
            },
        )

        return self.output

    def _estimate_lapse_rate(self, dt, dz, d):
        props = self.properties
        if len(dt) < props.min_num_neighbors:
            return props.default_lapse_rate
        lm = LinearRegression(fit_intercept=props.fit_intercept)
        if props.use_weights:
            weights = np.exp(-(d / (props.weight_scale_km * 1000.)) ** 2.)
        else:
            weights = None
        lm.fit(dz[:, None], dt, sample_weight=weights)
        return lm.coef_[0]


DEFAULTS_ADAPTIVE_ESTIMATOR = LapseRateEstimator.Properties(
    False, False, 30., 10, False, -6.5,
    DEFAULT_NEIGHBORHOOD_RADIAL
)


class AdaptiveLapseRateDownscaler(_InterpolatedDownscaler):

    FIELDS_O1280 = [
        SurfaceFieldType.LAPSE_RATE,
        SurfaceFieldType.T2M,
        SurfaceFieldType.Z,
        SurfaceFieldType.LSM
    ]
    FIELDS_O8000 = [
        SurfaceFieldType.LAPSE_RATE_INTERPOLATION,
        SurfaceFieldType.T2M_INTERPOLATION, SurfaceFieldType.T2M_DIFFERENCE,
        SurfaceFieldType.Z, SurfaceFieldType.Z_INTERPOLATION, SurfaceFieldType.Z_DIFFERENCE,
        SurfaceFieldType.LSM
    ]

    @dataclass
    class Properties(_InterpolatedDownscaler.Properties):
        estimator: LapseRateEstimator.Properties

    @classmethod
    def from_settings(cls, settings: 'AdaptiveLapseRateDownscaler.Properties', data_store: GlobalData):
        neighborhood = NeighborhoodModel(data_store)
        estimator = LapseRateEstimator(neighborhood)
        model = cls(estimator)
        model.set_properties(settings)
        return model

    def __init__(self, estimator: LapseRateEstimator):
        super().__init__()
        self.estimator = estimator

    def set_data(self, source_data: DomainData, target_data: DomainData):
        super().set_data(source_data, target_data)
        self.estimator.set_data(source_data, target_data)

    def set_properties(self, properties: 'AdaptiveLapseRateDownscaler.Properties') -> 'AdaptiveLapseRateDownscaler':
        if self.properties == properties:
            return self
        super().set_properties(properties)
        self.estimator.set_properties(properties.estimator)
        return self

    def update(self):
        if self.output is None:
            super().update()
            self.estimator.update()
            self._interpolate_estimator_outputs()
        return self

    def _interpolate_estimator_outputs(self):
        source_data = self.source_data.data
        target_data = self.target_data.data
        sites = self.target_data.sites
        interpolated_source = self.interpolator.interpolate(sites, source_data, variables=['t2m', 'z'])
        interpolated_lapse_rate = self.interpolator.interpolate(sites, self.estimator.output)

        dz = target_data.z.values - interpolated_source.z.values
        lapse_rate = interpolated_lapse_rate.values
        dt2m = lapse_rate * dz
        t2m_hr = interpolated_source.t2m.values + dt2m

        data_hr = target_data.assign({
            SurfaceFieldType.T2M.value: ('values', t2m_hr),
            SurfaceFieldType.T2M_INTERPOLATION: interpolated_source['t2m'].rename(SurfaceFieldType.T2M_INTERPOLATION),
            SurfaceFieldType.T2M_DIFFERENCE: ('values', dt2m),
            SurfaceFieldType.LAPSE_RATE.value: ('values', np.full_like(t2m_hr, lapse_rate)),
            SurfaceFieldType.Z_INTERPOLATION.value: (
            'values', interpolated_source['z'].rename(SurfaceFieldType.Z_INTERPOLATION.value)),
            SurfaceFieldType.Z_DIFFERENCE.value: ('values', dz),
        })

        output = OutputDataset()

        output_hr = output.get_group(GridConfiguration.O8000)
        for field_type in self.FIELDS_O8000:
            output_hr.add_surface_field(field_type, data_hr[field_type.value])

        output_lr = output.get_group(GridConfiguration.O1280)
        for field_type in self.FIELDS_O1280:
            output_lr.add_surface_field(field_type, source_data[field_type.value])

        self.output = output

    def synchronize_properties(self):
        self.properties.estimator = self.estimator.properties


DEFAULTS_ADAPTIVE_LAPSE_RATE = AdaptiveLapseRateDownscaler.Properties(
    InterpolationType.NEAREST_NEIGHBOR,
    DEFAULTS_ADAPTIVE_ESTIMATOR
)


class NetworkDownscaler(DownscalingMethodModel):
    pass

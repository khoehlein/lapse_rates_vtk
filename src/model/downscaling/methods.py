from dataclasses import dataclass
from enum import Enum

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from src.model.data.data_store import DomainData, GlobalData
from src.model.data.output import OutputDataset
from src.model.downscaling.interpolation import InterpolationType, NearestNeighborInterpolation, \
    BarycentricInterpolation
from src.model.downscaling.neighborhood import NeighborhoodModel, NeighborhoodType, DEFAULT_NEIGHBORHOOD_RADIAL
from src.model.interface import PropertyModel, SurfaceFieldType, GridConfiguration


class DownscalerType(Enum):
    FIXED_LAPSE_RATE = 'fixed_lapse_rate'
    ADAPTIVE_LAPSE_RATE = 'adaptive_lapse_rate'
    NETWORK = 'network'


class DownscalingMethodModel(PropertyModel):

    class Properties(PropertyModel.Properties):
        pass

    @classmethod
    def from_settings(cls, settings: 'DownscalingMethodModel.Properties', data_store: GlobalData):
        raise NotImplementedError()

    def __init__(self):
        super().__init__(None)
        self.source_data: DomainData = None
        self.target_data: DomainData = None
        self.output: OutputDataset = None

    def set_data(self, source_data: DomainData, target_data: DomainData):
        self.source_data = source_data
        self.target_data = target_data
        self.output = None
        return self

    def update(self):
        raise NotImplementedError()

    def supports_update(self, properties):
        return isinstance(properties, self.Properties)


class _InterpolatedDownscaler(DownscalingMethodModel):

    @dataclass
    class Properties:
        interpolation: InterpolationType

    def __init__(self):
        super().__init__()
        self.interpolator = None
        self._interpolators = {
            InterpolationType.NEAREST_NEIGHBOR: NearestNeighborInterpolation,
            InterpolationType.BARYCENTRIC: BarycentricInterpolation,
        }

    def set_data(self, source_data: DomainData, target_data: DomainData):
        super().set_data(source_data, target_data)
        self.interpolator = None

    def set_properties(self, properties: '_InterpolatedDownscaler.Properties') -> '_InterpolatedDownscaler':
        if self.properties == properties:
            return self
        super().set_properties(properties)
        self.interpolator = None
        return self

    def update(self):
        if self.properties is None or self.source_data is None:
            raise RuntimeError('[ERROR] Error updating interpolated downscaler')
        self._update_interpolator()
        return self

    def _update_interpolator(self):
        interpolation_type = self.properties.interpolation
        if (self.interpolator is None) or (self.interpolator.type != interpolation_type):
            cls = self._interpolators.get(interpolation_type)
            self.interpolator = cls(self.source_data)
        else:
            self.interpolator.set_source(self.source_data)
        return self


class FixedLapseRateDownscaler(_InterpolatedDownscaler):

    FIELDS_O1280 = [
        SurfaceFieldType.T2M,
        SurfaceFieldType.Z,
        SurfaceFieldType.LSM
    ]
    FIELDS_O8000 = [
        SurfaceFieldType.T2M_INTERPOLATION, SurfaceFieldType.T2M_DIFFERENCE,
        SurfaceFieldType.Z, SurfaceFieldType.Z_INTERPOLATION, SurfaceFieldType.Z_DIFFERENCE,
        SurfaceFieldType.LSM
    ]

    @dataclass
    class Properties(_InterpolatedDownscaler.Properties):
        lapse_rate: float

    @classmethod
    def from_settings(cls, settings: 'FixedLapseRateDownscaler.Properties', data_store: GlobalData):
        model = cls()
        model.set_properties(settings)
        return model

    def update(self):
        if self.output is None:
            super().update()
            self._interpolate_temperatures()
        return self

    def _interpolate_temperatures(self):
        source_data = self.source_data.data
        target_data = self.target_data.data
        sites = self.target_data.sites
        interpolated_source = self.interpolator.interpolate(sites, source_data, variables=['t2m', 'z'])
        lapse_rate = np.full_like(interpolated_source.t2m.values, self.properties.lapse_rate / 1000.)

        dz = target_data.z.values - interpolated_source.z.values
        dt2m = lapse_rate * dz
        t2m_hr = interpolated_source.t2m.values + dt2m

        data_hr = target_data.assign({
            SurfaceFieldType.T2M.value: ('values', t2m_hr),
            SurfaceFieldType.T2M_INTERPOLATION: interpolated_source['t2m'].rename(
                SurfaceFieldType.T2M_INTERPOLATION),
            SurfaceFieldType.T2M_DIFFERENCE: ('values', dt2m),
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


DEFAULTS_FIXED_LAPSE_RATE = FixedLapseRateDownscaler.Properties(
    InterpolationType.NEAREST_NEIGHBOR,
    -6.5,
)


class LapseRateEstimator(PropertyModel):

    @dataclass
    class Properties:
        use_volume: bool
        use_weights: bool
        weight_scale_km: float
        min_num_neighbors: int
        fit_intercept: bool
        default_lapse_rate: float
        neighborhood: NeighborhoodModel.Properties

    def __init__(self, neighborhood_model: NeighborhoodModel):
        super().__init__(None)
        self.neighborhood = neighborhood_model
        self.source_data: DomainData = None
        self.target_data: DomainData = None
        self.output = None

    def set_data(self, source_data: DomainData, target_data: DomainData):
        self.source_data = source_data
        self.target_data = target_data
        self.neighborhood.set_domain(self.source_data)
        self.output = None
        return self

    def set_properties(self, properties: 'LapseRateEstimator.Properties') -> 'LapseRateEstimator':
        if properties == self.properties:
            return self
        super().set_properties(properties)
        self.neighborhood.set_properties(self.properties.neighborhood)
        self.output = None
        return self

    def update(self):
        if self.output is None:
            if self.neighborhood.data is None:
                self.neighborhood.update()
            self._compute_lapse_rates()
        return self

    def synchronize_properties(self):
        self.properties.neighborhood = self.neighborhood.properties
        return self

    def _compute_lapse_rates(self):
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

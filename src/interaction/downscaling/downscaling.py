from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Any

import numpy as np
import xarray as xr
from PyQt5.QtWidgets import QWidget
from sklearn.neighbors import NearestNeighbors

from src.interaction.downscaling.domain_selection import DomainData
from src.interaction.downscaling.geometry import LocationBatch, Coordinates
from src.interaction.downscaling.neighborhood_lookup import NeighborhoodModel
from src.model.visualization.interface import PropertyModel, PropertySettingsView, PropertyController


class GridConfiguration(Enum):
    O1280 = 'o1280'
    O8000 = 'o8000'


class SurfaceFieldType(Enum):
    T2M = 't2m'
    T2M_INTERPOLATION = 't2m_interpolation'
    T2M_DIFFERENCE = 't2m_difference'
    LAPSE_RATE = 'lapse_rate'
    Z = 'z'
    Z_INTERPOLATION = 'z_interpolation'
    Z_DIFFERENCE = 'z_difference'
    LSM = 'lsm'


class VolumeFieldType(Enum):
    Z_QUANTILES = 'z_quantiles'
    T2M_VOLUME = 't2m_volume'


class OutputDataset(object):

    def __init__(self, parent=None):
        self.parent = parent
        self.groups = {}
        self.surface_fields = {}
        self.volume_fields = {}

    def get_group(self, key: Any):
        if key in self.groups:
            return self.groups[key]
        group = OutputDataset(parent=self)
        self.groups[key] = group
        return group

    def add_surface_field(self, field_type: SurfaceFieldType, data: xr.DataArray):
        if field_type in self.surface_fields:
            raise RuntimeError(f'[ERROR] Error assigning surface field: field {field_type.name} already occupied.')
        self.surface_fields[field_type] = data
        return self

    def add_volume_field(self, field_type: VolumeFieldType, data: xr.Dataset):
        if field_type in self.volume_fields:
            raise RuntimeError(f'[ERROR] Error assigning surface field: field {field_type.name} already occupied.')
        assert 'z' in data.data_vars, \
            f'[ERROR] Error in assigning volume field {field_type.name}: parameter z is not defined.'
        self.volume_fields[field_type] = data
        return self


class DownscalerType(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    NETWORK = 'network'


class InterpolationType(Enum):
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    BARYCENTRIC = 'barycentric'


class InterpolationMethod(object):

    def __init__(self, source: xr.Dataset):
        self.source = source

    def interpolate(self, target: LocationBatch, variables: Optional[List[str]]=None):
        raise NotImplementedError()


class NearestNeighborInterpolation(InterpolationMethod):

    def __init__(self, source: xr.Dataset):
        super(NearestNeighborInterpolation, self).__init__(source)
        self._build_search_structure()

    @staticmethod
    def _to_search_structure_coords(coords: Coordinates):
        return coords.as_xyz().values

    def _build_search_structure(self):
        self.search_structure = NearestNeighbors(n_neighbors=1)
        xyz = self._to_search_structure_coords(Coordinates.from_xarray(self.source))
        self.search_structure.fit(xyz)

    def interpolate(self, target: LocationBatch, variables=None):
        xyz = self._to_search_structure_coords(target.coords)
        indices = self.search_structure.kneighbors(xyz, return_distance=False)
        source = self.source if variables is None else self.source[variables]
        return source.isel(values=indices)


class BarycentricInterpolation(object):
    pass


class DownscalingMethod(PropertyModel):

    def __init__(self, properties: 'DownscalingMethod.Properties' = None):
        super().__init__(properties)

    def process(self, site_data: xr.Dataset, neighbor_data: xr.Dataset, target_domain: DomainData) -> OutputDataset:
        raise NotImplementedError()


class _LowResolutionDownscaler(DownscalingMethod):

    class Properties(DownscalingMethod.Properties):
        interpolation_type: InterpolationType

    def __init__(
            self, properties: '_LowResolutionDownscaler.Properties' = None):
        super().__init__(properties)
        self._interpolators = {
            InterpolationType.NEAREST_NEIGHBOR: NearestNeighborInterpolation,
            InterpolationType.BARYCENTRIC: BarycentricInterpolation,
        }

    @property
    def interpolation_type(self):
        return self.properties.interpolation_type

    def get_interpolator(self, source: xr.Dataset):
        if self.properties is None:
            raise RuntimeError('[ERROR] Error in building interpolator: properties unavailable.')
        cls = self._interpolators.get(self.interpolation_type)
        return cls(source)


class DefaultDownscaler(_LowResolutionDownscaler):

    @dataclass
    class Properties(_LowResolutionDownscaler.Properties):
        lapse_rate: float

    def __init__(self, properties: 'DefaultDownscaler.Properties' = None):
        super().__init__(properties)

    @property
    def lapse_rate(self):
        return self.properties.lapse_rate

    def process(self, site_data: xr.Dataset, neighbor_data: xr.Dataset, target_domain: DomainData):
        if self.properties is None:
            raise RuntimeError('[ERROR] Error applying default downscaler: Properties not set.')

        interpolator = self.get_interpolator(site_data)

        target_sites = target_domain.sites
        data_lr_to_hr = interpolator.interpolate(target_sites, variables=['t2m', 'z'])
        z_hr = target_domain.data['z']
        dz = z_hr.values - data_lr_to_hr['z'].values
        dt2m =  (self.lapse_rate / 1000.) * dz
        t2m_hr = data_lr_to_hr['t2m'].values + dt2m

        data_hr = target_domain.data
        data_hr = data_hr.assign({
            SurfaceFieldType.T2M.value: ('values', t2m_hr),
            SurfaceFieldType.T2M_INTERPOLATION: data_lr_to_hr['t2m'].rename(SurfaceFieldType.T2M_INTERPOLATION),
            SurfaceFieldType.T2M_DIFFERENCE: ('values', dt2m),
            SurfaceFieldType.LAPSE_RATE.value: ('values', np.full_like(t2m_hr, self.lapse_rate)),
            SurfaceFieldType.Z_INTERPOLATION.value: ('values', data_lr_to_hr['z'].rename(SurfaceFieldType.Z_INTERPOLATION.value)),
            SurfaceFieldType.Z_DIFFERENCE.value: ('values', dz),
        })

        output = OutputDataset()
        output_hr = output.get_group(GridConfiguration.O8000)
        for field_type in self.outputs_highres():
            output_hr.add_surface_field(field_type, data_hr[field_type.value])
        output_lr = output.get_group(GridConfiguration.O1280)
        for field_type in self.outputs_lowres():
            output_lr.add_surface_field(field_type, site_data[field_type.value])

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


class _NeighborhoodDownscaler(_LowResolutionDownscaler):

    @dataclass
    class Properties(_LowResolutionDownscaler):
        neighborhood_properties: NeighborhoodModel.Properties

class DownscalerModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        downscaler_type: DownscalerType
        downscaler_properties: DownscalingMethod.Properties

    def __init__(self):
        super().__init__(None)
        self.downscaler = None

    @property
    def downscaler_type(self) -> DownscalerType:
        return self.properties.downscaler_type

    def set_properties(self, properties: 'DownscalerModel.Properties'):
        reset_required = self.downscaler_reset_required(properties)
        super().set_properties(properties)
        if self.downscaler is None or reset_required:
            self._reset_downscaler()
        else:
            self.downscaler.set_properties(properties.downscaler_properties)
        return self

    def downscaler_reset_required(self, properties: 'DownscalerModel.Properties'):
        if self.properties is None:
            return True
        if properties.downscaler_type != self.downscaler_type:
            return True
        return False

    def _reset_downscaler(self):
        if self.properties is None:
            raise RuntimeError('[ERROR] Error building downscaler: properties not set.')
        cls = {
            DownscalerType.CONSTANT: DefaultDownscaler
        }.get(self.properties.downscaler_type, None)
        assert cls is not None, f'[ERROR] Error building downscaler: unknown downscaler type: {self.properties.downscaler_type}.'
        self.downscaler = cls(self.properties.downscaler_properties)

    def downscale(self, site_data: xr.Dataset, neighbor_data: xr.Dataset, target_domain: DomainData):
        return self.downscaler.process(site_data, neighbor_data, target_domain)


class DownscalerSettingsView(QWidget):
    downscaler_settings_changed = None

    def get_settings(self):
        raise NotImplementedError()

    def update_settings(self, settings):
        raise NotImplementedError()


class DownscalerController(PropertyController):

    @classmethod
    def from_settings(cls, settings: DownscalerModel.Properties, parent=None):
        view = DownscalerSettingsView(parent=parent)
        view.update_settings(settings)
        model = DownscalerModel()
        return cls(view, model, parent, apply_defaults=False)

    def default_settings(self):
        return DownscalerModel.Properties(
            downscaler_type=DownscalerType.CONSTANT,
            downscaler_properties=DefaultDownscaler.Properties(lapse_rate=-6.5)
        )

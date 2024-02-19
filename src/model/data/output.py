from typing import Any

import xarray as xr

from src.model.interface import SurfaceFieldType, VolumeFieldType


class OutputDataset(object):

    def __init__(self, parent=None, reference=None):
        self.parent = parent
        self.reference = reference
        self.groups = {}
        self.surface_fields = {}
        self.volume_fields = {}

    def get_group(self, key: Any):
        if key not in self.groups:
            raise RuntimeError()
        return self.groups[key]

    def create_group(self, key: Any, reference=None):
        if key in self.groups:
            raise RuntimeError()
        group = OutputDataset(parent=self, reference=reference)
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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union

import xarray as xr

from src.model.data.data_source import MeshDataSource, MultiFieldSource, MergedFieldSource
from src.model.geometry import LocationBatch, Coordinates
from src.model.interface import PropertyModel, FilterNodeModel


class InterpolationType(Enum):
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    BARYCENTRIC = 'barycentric'


class InterpolationModel(FilterNodeModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        type: InterpolationType

    def __init__(self, output_scalars: List[str]):
        super().__init__()
        self.source_data = None
        self.target_mesh = None
        self.output = MultiFieldSource(scalar_names=output_scalars)
        self.register_input('source_data', (MultiFieldSource, MergedFieldSource), instance=None)
        self.register_input('target_mesh', MeshDataSource, instance=None)
        self.register_output('outputs', MultiFieldSource)
        self.method = None

    def set_properties(self, properties) -> 'PropertyModel':
        super().set_properties(properties)
        if self.properties_changed():
            self.method = None
            self.set_outputs_valid(False)
        return self

    def set_source(self, source_data: Union[MultiFieldSource, MergedFieldSource]):
        self.source_data = source_data
        if self.method is not None:
            self.method.set_mesh(source_data.mesh_source)
        self.set_outputs_valid(False)
        return self

    def set_target_mesh(self, target_mesh: MeshDataSource):
        self.target_mesh = target_mesh
        self.output.set_mesh(target_mesh)
        self.set_outputs_valid(False)
        return self

    def update_outputs(self):
        if self.method is None:
            self._update_interpolation_method()
        else:
            assert self.method.type == self.properties.type
            assert self.method.mesh_source is self.source_data.mesh_source
        mesh_data = self.target_mesh.data()
        data = self.method.interpolate(
            mesh_data.locations, self.source_data.data,
            variables=self.output.scalar_names()
        )
        self.output.set_data(data)
        return self

    def _update_interpolation_method(self):
        if self.properties.type == InterpolationType.NEAREST_NEIGHBOR:
            self.method = NearestNeighborInterpolation()
            self.method.set_mesh(self.source_data.mesh_source)
        else:
            raise NotImplementedError()
        return self


class InterpolationMethod(object):

    def __init__(self, type_: InterpolationType):
        self.type = type_
        self.mesh_source = None

    def set_mesh(self, mesh_source: MeshDataSource):
        self.mesh_source = mesh_source
        return self

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables: Optional[List[str]] = None):
        raise NotImplementedError()


class NearestNeighborInterpolation(InterpolationMethod):

    def __init__(self):
        super(NearestNeighborInterpolation, self).__init__(InterpolationType.NEAREST_NEIGHBOR)
        self._build_search_structure()

    def set_mesh(self, mesh_source: MeshDataSource):
        super().set_mesh(mesh_source)
        self._build_search_structure()
        return self

    @staticmethod
    def _to_search_structure_coords(coords: Coordinates):
        return coords.as_xyz().values

    def _build_search_structure(self):
        if self.mesh_source.is_valid():
            mesh_data = self.mesh_source.data()
            self.search_structure = mesh_data.get_grid_lookup()
        else:
            self.search_structure = None

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables=None):
        xyz = self._to_search_structure_coords(target.coords)
        indices = self.search_structure.kneighbors(xyz, return_distance=False)
        data = data if variables is None else data[variables]
        return data.isel(values=indices)


class BarycentricInterpolation(object):
    pass

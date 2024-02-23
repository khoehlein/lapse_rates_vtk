from typing import List, Union, Iterable

import numpy as np
import xarray as xr

from src.model.downscaling.neighborhood_graphs import NeighborhoodGraph
from src.model.interface import DataNodeModel


class DataItem(DataNodeModel):
    """
    Base class for objects that actually carry data
    """

    def __init__(self, name=None, callback=None):
        super().__init__(name)
        self._data = None
        self._is_valid = False
        self._callback = callback

    def data(self):
        if self.is_valid():
            return self._data
        return None

    def set_data(self, data) -> 'DataItem':
        self._data = data
        return self

    def set_valid(self, valid: bool) -> 'DataItem':
        self._is_valid = valid
        return self

    def update(self) -> 'DataItem':
        if self.is_valid():
            return self
        if self._callback is not None:
            self._callback()
        self.set_valid(True)
        return self


class ScalarDataSource(DataItem):
    """
    object that serves as a source of scalar data
    """

    def __init__(self, name: str = None, callback=None):
        super().__init__(name, callback)

    def data(self) -> np.ndarray:
        return super().data()

    def set_data(self, data: np.ndarray) -> 'ScalarDataSource':
        super().set_data(data)
        return self


class DatasetSource(DataItem):
    """
    object that serves as a source of scalar data
    """

    def __init__(self, name: str = None, callback=None):
        super().__init__(name, callback)

    def data(self) -> xr.Dataset:
        return super().data()

    def set_data(self, data: xr.Dataset) -> 'DatasetSource':
        super().set_data(data)
        return self


class GraphDataSource(DataItem):
    """
    object that serves as a source of scalar data
    """

    def __init__(self, name: str = None, callback=None):
        super().__init__(name, callback)

    def data(self) -> NeighborhoodGraph:
        return super().data()

    def set_data(self, data: NeighborhoodGraph) -> 'GraphDataSource':
        super().set_data(data)
        return self


class MeshData(object):
    """
    data class for representing 3d mesh data
    """

    def x(self) -> np.ndarray:
        raise NotImplementedError()

    def y(self) -> np.ndarray:
        raise NotImplementedError()

    def z(self) -> np.ndarray:
        raise NotImplementedError()

    def as_polydata(self):
        raise NotImplementedError()


class MeshDataSource(DataItem):
    """
    object that serves as a source of 3d mesh data
    """
    COMPONENT_NAMES = ['x', 'y', 'z']

    def __init__(self, name: str = None):
        super().__init__(name)
        self._build_component_sources()

    def _build_component_sources(self):
        self._component_fields = {}
        for component_name in self.COMPONENT_NAMES:
            scalar_name = f'{self.name()}_{component_name}'
            scalar_source = ScalarDataSource(name=scalar_name, callback=self.update)
            field_source = ScalarFieldSource(scalar_source, self, name=scalar_name)
            self._component_fields[component_name] = field_source

    def get_component_field(self, name: str) -> Union['ScalarFieldSource', None]:
        return self._component_fields.get(name)

    def set_data(self, data: MeshData) -> 'MeshDataSource':
        super().set_data(data)
        self._update_component_data()
        return self

    def _update_component_data(self):
        for component_name, field_source in self._component_fields.items():
            scalar_data = getattr(self._data, component_name) if self._data is not None else None
            field_source.scalar.set_data(scalar_data)

    def set_valid(self, valid: bool) -> 'MeshDataSource':
        super().set_valid(valid)
        for field_source in self._component_fields.values():
            field_source.scalar.set_valid(valid)
        return self

    def update(self):
        super().update()
        self._update_component_data()
        return self


class ScalarFieldData(object):

    def __init__(self, name: str, scalars: np.ndarray, mesh: MeshData):
        self.name = str(name)
        self._scalars = scalars
        self._mesh = mesh

    def as_polydata(self):
        polydata = self._mesh.as_polydata()
        polydata.scalars[self.name] = self._scalars
        return polydata


class ScalarFieldSource(DataNodeModel):

    def __init__(self, scalar: ScalarDataSource, mesh_source: MeshDataSource, name: str = None):
        super().__init__(name)
        self.mesh_source = mesh_source
        self.scalar = scalar

    def set_mesh(self, mesh_source: MeshDataSource):
        self.mesh_source = mesh_source
        return self

    def data(self):
        if not self.is_valid():
            return None
        name = self.name or self.uid
        return ScalarFieldData(name, self.scalar.data(), self.mesh_source.data())

    def is_valid(self):
        return (self.mesh_source is not None and self.mesh_source.is_valid()) and self.scalar.is_valid()

    def update(self):
        if self.mesh_source is not None:
            self.mesh_source.update()
        self.scalar.update()
        return self


class MultiFieldSource(DataItem):

    def __init__(self, scalar_names: List[str] = None, mesh_source: MeshDataSource = None, name: str = None):
        super().__init__(name)
        self.mesh_source = mesh_source
        self.fields = {}
        if scalar_names is not None:
            self.add_scalar_fields(*scalar_names)

    def reset_scalar_fields(self):
        self.fields.clear()
        return self

    def add_scalar_fields(self, *scalar_names: List[str]):
        for scalar_name in scalar_names:
            scalar_source_name = f'{self.name()}_{scalar_name}'
            scalar = ScalarDataSource(name=scalar_source_name, callback=self.update)
            field_source = ScalarFieldSource(scalar, self.mesh_source, name=scalar_source_name)
            self.fields[scalar_name] = field_source
        return self

    def set_mesh(self, mesh_source: MeshDataSource):
        self.mesh_source = mesh_source
        for field in self.fields.values():
            field.set_mesh(mesh_source)
     
    def is_valid(self):
        return super().is_valid() and self.mesh_source.is_valid()
    
    def set_valid(self, valid: bool) -> 'MultiFieldSource':
        super().set_valid(valid)
        for field_source in self.fields.values():
            field_source.scalar.set_valid(valid)
        return self

    def set_data(self, data: xr.Dataset) -> 'MultiFieldSource':
        self._data = data
        self._update_field_data()
        return self

    def _update_field_data(self):
        for scalar_name, field_source in self.fields.items():
            scalar_data = self._data[scalar_name].values if self._data is not None else None
            field_source.scalar.set_data(scalar_data)

    def scalar_names(self):
        return list(self.fields.keys())

    def get_scalar_field(self, scalar_name: str) -> ScalarFieldSource:
        return self.fields[scalar_name]

    def update(self):
        if self.is_valid():
            return self
        super().update()
        self.mesh_source.update()
        self._update_field_data()
        self.set_valid(True)
        return self

    def clear(self):
        return self.set_data(None)


class MergedFieldSource(DataNodeModel):

    def __init__(self, sources: Iterable[MultiFieldSource], name=None):
        super().__init__(name)
        self._sources = list(sources)

    def data(self):
        if not self.is_valid():
            return None
        return xr.merge([source.data() for source in self._sources])

    def is_valid(self):
        return all([source.is_valid() for source in self._sources])

    def clear(self):
        for source in self._sources:
            source.clear()
        return self
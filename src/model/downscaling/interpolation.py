from enum import Enum
from typing import Optional, List

import xarray as xr

from src.model.geometry import LocationBatch, Coordinates
from src.model.data.data_store import DomainData


class InterpolationType(Enum):
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    BARYCENTRIC = 'barycentric'


class InterpolationMethod(object):

    def __init__(self, source: DomainData):
        self.source = source

    def set_source(self, source: DomainData):
        self.source = source
        return self

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables: Optional[List[str]] = None):
        raise NotImplementedError()


class NearestNeighborInterpolation(InterpolationMethod):

    def __init__(self, source: DomainData):
        super(NearestNeighborInterpolation, self).__init__(source)
        self._build_search_structure()

    def set_source(self, source: DomainData):
        super().set_source(source)
        self._build_search_structure()
        return self

    @staticmethod
    def _to_search_structure_coords(coords: Coordinates):
        return coords.as_xyz().values

    def _build_search_structure(self):
        self.search_structure = self.source.get_grid_lookup()

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables=None):
        xyz = self._to_search_structure_coords(target.coords)
        indices = self.search_structure.kneighbors(xyz, return_distance=False)
        data = data if variables is None else data[variables]
        return data.isel(values=indices)


class BarycentricInterpolation(object):
    pass

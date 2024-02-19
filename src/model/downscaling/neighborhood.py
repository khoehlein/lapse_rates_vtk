from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple

import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.model.data.data_store import GlobalData, DomainData
from src.model.geometry import Coordinates, LocationBatch
from src.model.interface import PropertyModel
from src.model.downscaling.neighborhood_graphs import NeighborhoodGraph, UniformNeighborhoodGraph, \
    RadialNeighborhoodGraph


class NeighborhoodType(Enum):
    RADIAL = 'radial'
    NEAREST_NEIGHBORS = 'nearest_neighbors'


class LookupType(Enum):
    AUTO = 'auto'
    KD_TREE = 'kd_tree'
    BALL_TREE = 'ball_tree'
    BRUTE = 'brute'


class NeighborhoodModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        neighborhood_type: NeighborhoodType
        neighborhood_size: Union[int, float]
        tree_type: LookupType
        num_jobs: int # 1 for single-process, -1 for all processors
        lsm_threshold: float


    def __init__(self, data_store: GlobalData):
        super().__init__(None)
        self.data_store = data_store
        self.domain: DomainData = None

        self.search_structure = None
        self.data: xr.Dataset = None
        self.graph: NeighborhoodGraph = None

        self._actions = {
            NeighborhoodType.NEAREST_NEIGHBORS: self._query_k_nearest_neighbors,
            NeighborhoodType.RADIAL: self._query_radial_neighbors,
        }

    def update(self):
        if self.properties is None or self.domain is None:
            message = []
            if self.properties is None:
                message.append('properties not set')
            if self.domain is None:
                message.append('domain not set')
            message = ' and '.join(message)
            raise RuntimeError(f'[ERROR] Error in updating neighborhood data: {message}')
        if self.search_structure is None:
            self._build_search_structure()
        self.graph = self.query_neighbor_graph(self.domain.sites)
        self.data = self.data_store.query_link_data(self.graph.links)
        return self

    def tree_update_required(self, old_properties: 'NeighborhoodModel.Properties') -> bool:
        new_properties = self.properties
        if new_properties is None:
            return True
        if new_properties.lsm_threshold != old_properties.lsm_threshold:
            return True
        if new_properties.tree_type != old_properties.tree_type:
            return True
        if new_properties.num_jobs != old_properties.num_jobs:
            return True
        return False

    def set_properties(self, properties: 'NeighborhoodModel.Properties') -> 'NeighborhoodModel':
        old_properties = self.properties
        if old_properties == properties:
            return self
        super().set_properties(properties)
        if self.tree_update_required(old_properties):
            self.search_structure = None
        self.data = None
        self.graph = None
        return self

    def set_domain(self, domain: DomainData):
        self.domain = domain
        self.search_structure = None
        self.data = None
        self.graph = None
        return self

    def _build_search_structure(self):
        properties = self.properties
        self.search_structure = NearestNeighbors(
            algorithm=self.properties.tree_type.value,
            n_jobs=self.properties.num_jobs,
            leaf_size=100,
        )
        data = self.data_store.get_lsm()
        if data is None:
            raise RuntimeError('[ERROR] Error while building neighborhood lookup from properties: LSM unavailable.')
        mask = np.argwhere(data.values >= properties.lsm_threshold)
        data = data.isel(values=mask)
        coords = Coordinates.from_xarray(data).as_geocentric().values
        self.search_structure.fit(coords)

    def query_neighbor_graph(self, sites: LocationBatch) -> NeighborhoodGraph:
        action = self._actions.get(self.properties.neighborhood_type, None)
        if action is None:
            raise RuntimeError()
        return action(sites)

    def query_neighbor_data(self, sites: LocationBatch) -> Tuple[xr.Dataset, NeighborhoodGraph]:
        graph = self.query_neighbor_graph(sites)
        data = self.data_store.query_link_data(graph.links)
        return data, graph

    def _query_k_nearest_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(
            locations, self.search_structure, self.properties.neighborhood_size)

    def _query_radial_neighbors(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return RadialNeighborhoodGraph.from_tree_query(
            locations, self.search_structure, self.properties.neighborhood_size)


DEFAULT_NEIGHBORHOOD_RADIAL = NeighborhoodModel.Properties(
    neighborhood_type=NeighborhoodType.RADIAL,
    neighborhood_size=30.,
    tree_type=LookupType.AUTO,
    num_jobs=-1,
    lsm_threshold=0.5
)


DEFAULT_NEIGHBORHOOD_NEAREST_NEIGHBORS = NeighborhoodModel.Properties(
    neighborhood_type=NeighborhoodType.NEAREST_NEIGHBORS,
    neighborhood_size=32,
    tree_type=LookupType.AUTO,
    num_jobs=-1,
    lsm_threshold=0.5
)
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple

import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.interaction.downscaling.data_store import DataStore
from src.interaction.downscaling.geometry import Coordinates, LocationBatch
from src.model.neighborhood_lookup.neighborhood_graphs import NeighborhoodGraph, UniformNeighborhoodGraph, \
    RadialNeighborhoodGraph
from src.model.visualization.interface import PropertyModel, PropertyController, PropertySettingsView


class NeighborhoodType(Enum):
    RADIAL = 'radial'
    NEAREST_NEIGHBORS = 'nearest_neighbors'


class TreeType(Enum):
    AUTO = 'auto'
    KD_TREE = 'kd_tree'
    BALL_TREE = 'ball_tree'
    BRUTE = 'brute'


class NeighborhoodModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        neighborhood_type: NeighborhoodType
        neighborhood_size: Union[int, float]
        tree_type: TreeType
        num_jobs: int # 1 for single-process, -1 for all processors
        lsm_threshold: float

    def __init__(self, data_store: DataStore):
        super().__init__(None)
        self.data_store = data_store
        self.search_structure: NearestNeighbors = None
        self._actions = {
            NeighborhoodType.NEAREST_NEIGHBORS: self._query_k_nearest_neighbors,
            NeighborhoodType.RADIAL: self._query_radial_neighbors,
        }

    @property
    def neighborhood_type(self) -> NeighborhoodType:
        return self.properties.neighborhood_type

    @property
    def neighborhood_size(self) -> Union[int, float]:
        return self.properties.neighborhood_size

    @property
    def tree_type(self) -> TreeType:
        return self.properties.tree_type

    @property
    def num_jobs(self) -> int:
        return self.properties.num_jobs

    @property
    def lsm_threshold(self) -> float:
        return self.properties.lsm_threshold

    def _reset_search_structure(self):
        properties = self.properties
        self.search_structure = NearestNeighbors(algorithm=self.tree_type.value, leaf_size=100,
                                            n_jobs=self.num_jobs)
        data = self.data_store.get_lsm()
        if data is None:
            raise RuntimeError('[ERROR] Error while building neighborhood lookup from properties: LSM unavailable.')
        mask = np.argwhere(data.values >= properties.lsm_threshold)
        data = data.isel(values=mask)
        coords = Coordinates.from_xarray(data).as_geocentric().values
        self.search_structure.fit(coords)

    def set_properties(self, properties: 'NeighborhoodModel.Properties'):
        reset = self.tree_update_required(properties)
        super().set_properties(properties)
        if reset:
            self._reset_search_structure()
        self.data = self.data_store.query_link_data()
        return self

    def tree_update_required(self, properties: 'NeighborhoodModel.Properties') -> bool:
        if self.properties is None:
            return True
        if properties.lsm_threshold != self.lsm_threshold:
            return True
        if properties.tree_type != self.tree_type:
            return True
        if properties.num_jobs != self.num_jobs:
            return True
        return False

    def query_neighbor_graph(self, sites: LocationBatch) -> NeighborhoodGraph:
        action = self._actions.get(self.neighborhood_type, None)
        if action is None:
            raise RuntimeError()
        return action(sites)

    def query_neighbor_data(self, sites: LocationBatch) -> Tuple[xr.Dataset, NeighborhoodGraph]:
        graph = self.query_neighbor_graph(sites)
        data = self.data_store.query_link_data(graph.links)
        return data, graph

    def _query_k_nearest_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)

    def _query_radial_neighbors(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return RadialNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)


class NeighborhoodLookupSettingsView(PropertySettingsView):

    def get_settings(self) -> NeighborhoodModel.Properties:
        raise NotImplementedError()

    def update_settings(self, settings: NeighborhoodModel.Properties):
        raise NotImplementedError()


class NeighborhoodLookupController(PropertyController):

    @classmethod
    def from_settings(cls, settings: NeighborhoodModel.Properties, data_store: DataStore, parent=None):
        view = NeighborhoodLookupSettingsView(parent)
        view.update_settings(settings)
        model = NeighborhoodModel(data_store)
        return cls(view, model, parent, apply_defaults=False)

    def default_settings(self):
        return NeighborhoodModel.Properties(
            neighborhood_type=NeighborhoodType.RADIAL,
            neighborhood_size=60.,
            tree_type=TreeType.AUTO,
            num_jobs=-1,
            lsm_threshold=0.5
        )

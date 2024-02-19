from itertools import chain
from typing import List
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import LocationBatch


class NeighborhoodGraph(object):

    def __init__(
            self,
            locations: LocationBatch,
            neighbors: List[np.ndarray],
            distances: List[np.ndarray],
            source_reference: np.ndarray = None
    ):
        self.locations = locations
        self.neighbors = neighbors
        self.distances = distances
        self.source_reference = source_reference
        self.num_links: int = None
        self.nearest_neighbor: np.ndarray = None
        self.total_links: int = None
        self.links: pd.DataFrame = None
        self._compute_index_stats()
        self._compute_distance_stats()

    def _compute_links(self) -> pd.DataFrame:
        raise NotImplementedError()

    def _compute_index_stats(self):
        raise NotImplementedError()

    def _compute_distance_stats(self):
        raise NotImplementedError()

    def get_subset(self, location_ids: np.ndarray):
        locations = self.locations.get_subset(location_ids)
        neighbors = self.neighbors[location_ids]
        distances = self.distances[location_ids]
        source_reference = location_ids if self.source_reference is None else self.source_reference[location_ids]
        return self.__class__(locations, neighbors, distances, source_reference)

    def get_uniform_neighborhoods(self) -> List['NeighborhoodGraph']:
        raise NotImplementedError()


class UniformNeighborhoodGraph(NeighborhoodGraph):

    def __init__(
            self,
            locations: LocationBatch,
            neighbors: np.ndarray,
            distances: np.ndarray,
            source_reference: np.ndarray = None,
    ):
        self.uniform_size = self.neighbors.shape[-1]
        super().__init__(locations, neighbors, distances, source_reference)

    def _compute_index_stats(self):
        self.num_links = np.full((len(self.locations),), self.uniform_size)
        self.nearest_neighbor = self.neighbors[:, 0]
        self.total_links = self.neighbors.size
        self.links = self._compute_links()

    def _compute_links(self) -> pd.DataFrame:
        flat_locids = np.repeat(np.arange(len(self.locations)), self.uniform_size)
        return pd.DataFrame({'location': flat_locids, 'neighbor': self.neighbors.ravel(), 'distance': self.distances.ravel()})

    def _compute_distance_stats(self):
        self.min_distance = self.distances[0]
        self.max_distance = self.distances[-1]

    def get_uniform_neighborhoods(self) -> List['UniformNeighborhoodGraph']:
        return [self]

    @classmethod
    def from_tree_query(cls, locations: LocationBatch, tree: NearestNeighbors, k: int):
        xyz = locations.coords.as_geocentric().values
        distances, neighbors = tree.kneighbors(xyz, n_neighbors=k, return_distance=True)
        data = cls(locations, neighbors, distances)
        return data


class RadialNeighborhoodGraph(NeighborhoodGraph):

    def _compute_links(self):
        flat_locids = np.zeros((self.total_links,), dtype=int)
        counter = 0
        for i, num_links in enumerate(self.num_links):
            flat_locids[counter:(counter + num_links)] = i
            counter += num_links
        return pd.DataFrame({
            'location': flat_locids,
            'neighbor': np.fromiter(chain.from_iterable(self.neighbors), count=self.total_links, dtype=int),
            'distance': np.fromiter(chain.from_iterable(self.distances), count=self.total_links, dtype=float)
        })

    def _compute_index_stats(self):
        self.num_links = np.fromiter((len(nids) for nids in self.neighbors), count=len(self.neighbors), dtype=int)
        self.nearest_neighbor = np.fromiter(((nids[0] if len(nids) else -1) for nids in self.neighbors), count=len(self.neighbors), dtype=int)
        self.total_links = np.sum(self.num_links)
        self.links = self._compute_links()

    def _compute_distance_stats(self):
        self.min_distance = np.fromiter(((d[0] if len(d) else -1.) for d in self.distances), count=len(self.distances), dtype=float)
        self.max_distance = np.fromiter(((d[-1] if len(d) else -1.) for d in self.distances), count=len(self.distances), dtype=float)

    @classmethod
    def from_tree_query(cls, locations: LocationBatch, tree: NearestNeighbors, radius_km: float):
        xyz = locations.coords.as_geocentric().values
        distances, neighbors = tree.radius_neighbors(
            xyz, radius=1000. * radius_km,
            return_distance=True, sort_results=True
        )
        data = cls(locations, neighbors, distances)
        return data

    def get_uniform_neighborhoods(self):
        groups = pd.DataFrame({
            'location': np.arange(self.num_links),
            'num_links': self.num_links
        }).set_index('location').groupby(by='num_links').groups

        def build_uniform_neighborhood(key):
            location_ids = groups.get(key).values
            locations = self.locations.get_subset(location_ids)
            neighbors = np.asarray(self.neighbors[location_ids])
            distances = np.asarray(self.distances[location_ids])
            return UniformNeighborhoodGraph(locations, neighbors, distances)

        neighborhoods = [build_uniform_neighborhood(key) for key in groups.keys()]
        return neighborhoods

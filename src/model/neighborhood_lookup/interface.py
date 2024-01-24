from typing import Dict, Any

import numpy as np
import xarray as xr
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget
from sklearn.neighbors import KDTree

from src.model.geometry import LocationBatch, Coordinates
from src.model.neighborhood_lookup.neighborhood_graphs import RadialNeighborhoodGraph, UniformNeighborhoodGraph


class AboveThresholdFilter(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.threshold = None
        self._compute_mask()

    def set_threshold(self, threshold: float):
        if threshold != self.threshold:
            self.threshold = float(threshold)
            self._compute_mask()
        return self

    def set_data(self, data: np.ndarray):
        self.data = data
        self._compute_mask()
        return self

    def _compute_mask(self):
        if self.is_valid():
            self.mask = self.data >= self.threshold
        else:
            self.mask = None
        return self

    def is_valid(self):
        return not (self.data is None or self.threshold is None)


class NeighborhoodLookupModel(QWidget):

    def __init__(self, property_class,parent=None):
        super().__init__(parent)
        self._property_class = property_class
        self.locations = None
        self.filter = AboveThresholdFilter(parent=self)
        self.update_tree_lookup()

    def set_source_data(self, data: xr.DataArray):
        self.locations = Coordinates.from_xarray(data).as_geocentric().values
        self.filter.set_data(data.values)
        self.update_tree_lookup()
        return self

    def validate_neighborhood_properties(self, properties):
        assert isinstance(properties, self._property_class)

    def update_tree_lookup(self):
        if self.locations is not None and self.filter.is_valid():
            self.tree_lookup = KDTree(self.locations[self.filter.mask], leaf_size=100)
        else:
            self.tree_lookup = None
        return self

    def query_radial_neighbors(self, locations: LocationBatch, radius_km: float) -> RadialNeighborhoodGraph:
        graph = RadialNeighborhoodGraph.from_tree_query(locations, self.tree_lookup, radius_km)
        return graph

    def query_nearest_neighbors(self, locations: LocationBatch, k: int) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(locations, self.tree_lookup, k)

    def query_neighborhood(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        raise NotImplementedError()

    def set_neighborhood_properties(self, properties):
        raise NotImplementedError

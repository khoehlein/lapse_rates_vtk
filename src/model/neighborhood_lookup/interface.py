from typing import Dict, Any

import numpy as np
import xarray as xr
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget
from sklearn.neighbors import KDTree

from src.model.geometry import LocationBatch, Coordinates
from src.model.neighborhood_lookup.neighborhood_graphs import RadialNeighborhoodGraph, UniformNeighborhoodGraph


class AboveThresholdFilter(QWidget):

    mask_changed = pyqtSignal()

    def __init__(self, data: np.ndarray, threshold: float, parent=None):
        super().__init__(parent)
        self.data = data
        self.threshold = threshold
        self._compute_mask()

    @pyqtSlot(float)
    def set_threshold(self, threshold: float):
        if threshold != self.threshold:
            self.threshold = float(threshold)
            self.mask_changed.emit(threshold)

    def _compute_mask(self):
        self.mask = self.data >= self.threshold


class NeighborhoodLookup(QWidget):

    tree_lookup_changed = pyqtSignal()

    def __init__(self, data: xr.DataArray, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        if config is None:
            config = {}
        self.locations = Coordinates.from_xarray(data).as_geocentric().values
        self.filter = AboveThresholdFilter(data.values, config.get('lsm_threshold', 0.), parent=self)
        self._tree_kws = config.get('tree_kws', {'leaf_size': 100})
        self._refresh_tree_lookup()
        self.filter.mask_changed.connect(self._refresh_tree_lookup)

    def _refresh_tree_lookup(self):
        self.tree_lookup = KDTree(self.locations[self.filter.mask], **self._tree_kws)
        self.tree_lookup_changed.emit()

    def query_neighborhood_graph(self, locations: LocationBatch, radius_km: float) -> RadialNeighborhoodGraph:
        graph = RadialNeighborhoodGraph.from_tree_query(locations, self.tree_lookup, radius_km)
        return graph

    def query_nearest_neighbors(self, locations: LocationBatch, k: int) -> UniformNeighborhoodGraph:
        raise NotImplementedError()
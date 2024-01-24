from dataclasses import dataclass
from typing import Dict, Any

import xarray as xr
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from src.model.geometry import LocationBatch
from src.model.neighborhood_lookup.interface import NeighborhoodLookupModel
from src.model.neighborhood_lookup.neighborhood_graphs import UniformNeighborhoodGraph


@dataclass(init=True, repr=True)
class KNNNeighborhoodProperties(object):
    neighborhood_size: float
    lsm_threshold: float


class KNNNeighborhoodLookup(NeighborhoodLookupModel):

    def __init__(self,parent=None):
        super().__init__(KNNNeighborhoodProperties, parent)
        self.neighborhood_size = None

    def set_neighborhood_properties(self, properties: KNNNeighborhoodProperties):
        self.validate_neighborhood_properties(properties)
        self.neighborhood_size = properties.neighborhood_size
        self.filter.set_threshold(properties.lsm_threshold)
        self.update_tree_lookup()
        return self

    def query_neighborhood(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return self.query_nearest_neighbors(locations, self.neighborhood_size)
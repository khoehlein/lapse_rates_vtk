from dataclasses import dataclass
from typing import Dict, Any
import xarray as xr

from PyQt5.QtCore import pyqtSignal, pyqtSlot

from src.model.geometry import LocationBatch
from src.model.neighborhood_lookup.interface import NeighborhoodLookupModel
from src.model.neighborhood_lookup.neighborhood_graphs import RadialNeighborhoodGraph


@dataclass(init=True, repr=True)
class RadialNeighborhoodProperties(object):
    lookup_radius: float
    lsm_threshold: float


class RadialNeighborhoodLookup(NeighborhoodLookupModel):

    def __init__(self, parent=None):
        super().__init__(RadialNeighborhoodProperties, parent)
        self.lookup_radius = None

    def set_neighborhood_properties(self, properties: RadialNeighborhoodProperties):
        self.validate_neighborhood_properties(properties)
        self.lookup_radius = properties.lookup_radius
        self.filter.set_threshold(properties.lsm_threshold)
        self.update_tree_lookup()
        return self

    def query_neighborhood(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return self.query_radial_neighbors(locations, self.lookup_radius)
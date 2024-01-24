from dataclasses import dataclass
from typing import Dict, Any
import xarray as xr

from PyQt5.QtCore import pyqtSignal, pyqtSlot

from src.model.geometry import LocationBatch
from src.model.neighborhood_lookup.interface import NeighborhoodLookup
from src.model.neighborhood_lookup.neighborhood_graphs import RadialNeighborhoodGraph


@dataclass(init=True, repr=True)
class RadialNeighborhoodProperties(object):
    lookup_radius: float
    lsm_threshold: float


class RadialNeighborhoodLookup(NeighborhoodLookup):

    lookup_radius_changed = pyqtSignal()

    def __init__(self, land_sea_mask: xr.DataArray, config: Dict[str, Any] = None, parent=None):
        if config is None:
            config = {}
        super().__init__(land_sea_mask, config, parent)
        self.lookup_radius = config.get('lookup_radius', 30.)

    @pyqtSlot(float)
    def set_lookup_radius(self, radius_km: float):
        if radius_km != self.lookup_radius:
            self.lookup_radius = radius_km
            self.lookup_radius_changed.emit()

    def query_neighborhood(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return self.query_neighborhood_graph(locations, self.lookup_radius)
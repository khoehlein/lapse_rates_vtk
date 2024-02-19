from dataclasses import dataclass

from src.model._legacy.geometry import LocationBatch
from src.model._legacy.interface import NeighborhoodLookupModel
from src.model.downscaling.neighborhood_graphs import RadialNeighborhoodGraph


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
        old_filter_threshold = self.filter.threshold
        new_filter_threshold = properties.lsm_threshold
        if old_filter_threshold is None or new_filter_threshold != old_filter_threshold:
            self.filter.set_threshold(new_filter_threshold)
            self.update_tree_lookup()
        return self

    def query_neighborhood(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return self.query_radial_neighbors(locations, self.lookup_radius)
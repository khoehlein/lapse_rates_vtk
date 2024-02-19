from dataclasses import dataclass

from src.model._legacy.geometry import LocationBatch
from src.model._legacy.interface import NeighborhoodLookupModel
from src.model.downscaling.neighborhood_graphs import UniformNeighborhoodGraph


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
        old_filter_threshold = self.filter.threshold
        new_filter_threshold = properties.lsm_threshold
        if old_filter_threshold is None or new_filter_threshold != old_filter_threshold:
            self.filter.set_threshold(properties.lsm_threshold)
            self.update_tree_lookup()
        return self

    def query_neighborhood(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return self.query_nearest_neighbors(locations, self.neighborhood_size)
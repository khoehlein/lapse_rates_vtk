from dataclasses import dataclass


@dataclass(init=True, repr=True)
class KNNNeighborhoodProperties(object):
    neighborhood_size: float
    lsm_threshold: float


class KNNNeighborhoodLookup(NeighborhoodLookup):

    neighborhood_size_changed = pyqtSignal()

    def __init__(self, land_sea_mask: xr.DataArray, config: Dict[str, Any] = None, parent=None):
        if config is None:
            config = {}
        super().__init__(land_sea_mask, config, parent)
        self.neighborhood_size = config.get('neighborhood_size', 32)

    @pyqtSlot(float)
    def set_neighborhood_size(self, k: int):
        if k != self.neighborhood_size:
            self.neighborhood_size = k
            self.neighborhood_size_changed.emit()

    def query_neighborhood(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return self.query_nearest_neighbors(locations, self.neighborhood_size)
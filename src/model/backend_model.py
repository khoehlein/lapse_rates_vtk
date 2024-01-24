from PyQt5.QtWidgets import QWidget

from src.model.downscaling import DownscalerInterface
from src.model.data_store.world_data import WorldData, NeighborhoodLookup


class BackendModel(QWidget):

    def __init__(
            self,
            data_store: WorldData = None,
            neighborhood_lookup: NeighborhoodLookup = None,
            downscaler: DownscalerInterface = None,
            parent=None
    ):
        super().__init__(parent)

from PyQt5.QtWidgets import QWidget
from src.model.world_data import WorldData


class BackendModel(QWidget):

    def __init__(
            self,
            data_store: WorldData,
            downscaler,
            parent=None
    ):
        super().__init__(parent)

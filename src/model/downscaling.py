from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.model.geometry import SurfaceGeometry
from src.model.world_data import WorldData


class DownscalerInterface(QWidget):

    def compute_temperatures(self, target: SurfaceGeometry, source: WorldData) -> np.ndarray:
        raise NotImplementedError()

    def set(self, **kwargs) -> 'DownscalerInterface':
        raise NotImplementedError()

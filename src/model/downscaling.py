from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QWidget

from src.model.geometry import SurfaceGeometry
from src.model.data_store.world_data import WorldData, SampleBatch
from src.model.neighborhood_lookup.interface import NeighborhoodLookupModel


class DownscalerModel(QObject):

    def __init__(self, properties_class, parent=None):
        super().__init__(parent)
        self._properties_class = properties_class

    def set_downscaler_properties(self, properties):
        raise NotImplementedError()

    def validate_downscaler_properties(self, properties):
        assert isinstance(properties, self._properties_class)

    def compute_temperatures(self, target: SurfaceGeometry, samples: SampleBatch) -> np.ndarray:
        raise NotImplementedError()


@dataclass(init=True, repr=True)
class LapseRateDownscalerProperties():
    use_volume: bool
    use_weights: bool
    weight_scale_km: float = None


class LapseRateDownscaler(DownscalerModel):

    def __init__(self, parent=None):
        super().__init__(LapseRateDownscalerProperties, parent)
        self.use_volume = None
        self.use_weights = None
        self.weight_scale_km = None


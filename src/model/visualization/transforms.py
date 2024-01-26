import numpy as np
from PyQt5.QtCore import QObject


class SceneSpaceTransform(QObject):

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AffineLinear(SceneSpaceTransform):

    def __init__(self, offset: float, scale: float, parent=None):
        super().__init__(parent)
        self.offset = float(offset)
        self.scale = float(scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale

    @classmethod
    def identity(cls):
        return cls(0., 1.)

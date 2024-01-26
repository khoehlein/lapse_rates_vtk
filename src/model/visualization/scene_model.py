from typing import List

from PyQt5.QtCore import QObject
from src.model.visualization.lighting import LightingModel
from src.model.visualization.visualizations import SurfaceVisualization


class SceneModel(QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.surface_o1280: SurfaceVisualization = None
        self.surface_o8000: SurfaceVisualization = None
        self.lighting: LightingModel = None

    @property
    def visuals(self):
        return [self.surface_o1280, self.surface_o8000]

    def reset_vertical_scale(self, new_scale: float) -> 'SceneModel':
        for v in self.visuals:
            if v is not None:
                v.reset_vertical_scale(new_scale)
        return self

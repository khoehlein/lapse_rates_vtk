from typing import List, Dict

from PyQt5.QtCore import QObject
from src.model.visualization.lighting import LightingModel
from src.model.visualization.visualizations import VisualizationModel


class SceneModel(QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.visuals: Dict[str, VisualizationModel] = {}
        self.lighting: LightingModel = None

    def reset_vertical_scale(self, new_scale: float) -> 'SceneModel':
        for v in self.visuals.values():
            if v is not None:
                v.reset_vertical_scale(new_scale)
        return self

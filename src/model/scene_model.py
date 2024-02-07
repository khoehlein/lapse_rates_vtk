from typing import Dict

from PyQt5.QtCore import QObject
import pyvista as pv

from src.model.visualization.interface import VisualizationModel


class SceneModel(QObject):

    def __init__(self, host: pv.Plotter, parent=None):
        super().__init__(parent)
        self.host = host
        self.visuals: Dict[str, VisualizationModel] = {}
        self._vertical_scale = 1.

    def add_visualization(self, visualization: VisualizationModel) -> VisualizationModel:
        return self._add_visualization(visualization)

    def add_or_replace_visualization(self, visualization: VisualizationModel):
        key = visualization.key
        if key in self.visuals:
            self.remove_visualization(key)
        self._add_visualization(visualization)
        return visualization

    def replace_visualization(self, visualization: VisualizationModel):
        key = visualization.key
        assert key in self.visuals
        self.remove_visualization(key)
        self._add_visualization(visualization)
        return visualization

    def _add_visualization(self, visualization: VisualizationModel) -> 'SceneModel':
        self.visuals[visualization.key] = visualization
        self.host.suppress_render = True
        visualization.set_vertical_scale(self._vertical_scale)
        visualization.set_host(self.host)
        self.host.update_bounds_axes()
        self.host.suppress_render = False
        self.host.render()
        return visualization

    def remove_visualization(self, key: str):
        if key in self.visuals:
            visualization = self.visuals[key]
            visualization.clear_host()
            del self.visuals[key]

    def reset(self):
        self.visuals.clear()
        self.host.clear_actors()
        self.host.scalar_bars.clear()

    def set_vertical_scale(self, scale):
        self._vertical_scale = scale
        for visualization in self.visuals.values():
            visualization.set_vertical_scale(self._vertical_scale)
        self.host.render()
        return self

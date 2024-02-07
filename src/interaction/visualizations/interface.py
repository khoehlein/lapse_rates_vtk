import uuid
from typing import Dict, Type, Any

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget

from src.model.geometry import SurfaceDataset
from src.model.visualization.interface import VisualizationModel


class VisualizationSettingsView(QWidget):

    source_data_changed = pyqtSignal(str) # trigger signal for scene controller
    visibility_changed = pyqtSignal(bool)

    def __init__(self, key: str = None, parent=None):
        super().__init__(parent)
        if key is None:
            key = str(uuid.uuid4())
        self.key = key

    def get_visibility(self) -> bool:
        raise NotImplementedError()

    def get_properties_summary(self) -> Dict[str, Any]:
        raise NotImplementedError()


class VisualizationController(QObject):

    def __init__(
            self,
            settings_view: VisualizationSettingsView,
            visualization_class: Type[VisualizationModel],
            parent=None):
        super().__init__(parent)
        self.settings_view: VisualizationSettingsView = settings_view
        self.settings_view.visibility_changed.connect(self._on_visibility_changed)
        self._visualization_class = visualization_class
        self.visualization: VisualizationModel = None

    @property
    def key(self):
        return self.settings_view.key

    def _on_visibility_changed(self, visible: bool):
        self.visualization.set_visibility(visible)

    def visualize(self, dataset: Dict[str, SurfaceDataset]):
        properties = self.settings_view.get_properties_summary()
        visualization = self._visualization_class.from_dataset(dataset, properties, key=self.key)
        visualization.set_visibility(self.settings_view.get_visibility())
        self.visualization = visualization
        return visualization

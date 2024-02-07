from typing import Tuple

from PyQt5.QtCore import QObject

from src.interaction.visualizations.interface import VisualizationSettingsView, VisualizationController
from src.interaction.visualizations.surface_scalar_field.controller import SurfaceScalarFieldController
from src.interaction.visualizations.surface_scalar_field.view import SurfaceScalarFieldSettingsView
from src.model.visualization.interface import VisualizationType


class VisualizationFactory(QObject):

    def __init__(
            self,
            scene_settings_view,
            scene_controller,
            parent=None
    ):
        super(VisualizationFactory, self).__init__(parent)
        self.scene_settings_view = scene_settings_view
        self.scene_controller = scene_controller

    def setup_visualization_interface(
            self,
            visualization_type: VisualizationType, label: str
    ) -> Tuple[VisualizationSettingsView, VisualizationController]:
        if visualization_type == VisualizationType.SURFACE_SCALAR_FIELD:
            settings_view = SurfaceScalarFieldSettingsView(parent=self.scene_settings_view)
            controller = SurfaceScalarFieldController(settings_view, parent=self.scene_controller)
        else:
            raise NotImplementedError()
        self.scene_controller.register_visualization_interface(settings_view, controller, label)
        return settings_view, controller
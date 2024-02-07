from typing import Tuple

from PyQt5.QtCore import QObject

from src.interaction.visualizations.interface import VisualizationSettingsView, VisualizationController
from src.interaction.visualizations.projection_lines.controller import ProjectionLinesController
from src.interaction.visualizations.projection_lines.view import ProjectionLinesSettingsView
from src.interaction.visualizations.surface_isocontours import SurfaceIsocontoursController, SurfaceIsocontoursSettingsView
from src.interaction.visualizations.surface_scalar_field import SurfaceScalarFieldController, SurfaceScalarFieldSettingsView
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

    def build_visualization_interface(
            self,
            visualization_type: VisualizationType
    ) -> Tuple[VisualizationSettingsView, VisualizationController]:
        if visualization_type == VisualizationType.SURFACE_SCALAR_FIELD:
            settings_view = SurfaceScalarFieldSettingsView(parent=self.scene_settings_view)
            controller = SurfaceScalarFieldController(settings_view, parent=self.scene_controller)
        elif visualization_type == VisualizationType.SURFACE_ISOCONTOURS:
            settings_view = SurfaceIsocontoursSettingsView(parent=self.scene_settings_view)
            controller = SurfaceIsocontoursController(settings_view, parent=self.scene_controller)
        elif visualization_type == VisualizationType.PROJECTION_LINES:
            settings_view = ProjectionLinesSettingsView(parent=self.scene_settings_view)
            controller = ProjectionLinesController(settings_view, parent=self.scene_controller)
        else:
            raise NotImplementedError()
        return settings_view, controller
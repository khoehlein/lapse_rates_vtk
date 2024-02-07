import logging
from src.interaction.visualizations.interface import VisualizationController
from src.interaction.visualizations.projection_lines.view import ProjectionLinesSettingsView
from src.model.visualization.projection_lines import ProjectionLines


class ProjectionLinesController(VisualizationController):

    def __init__(self, settings_view: ProjectionLinesSettingsView, parent=None):
        super().__init__(settings_view, ProjectionLines, parent)
        self.settings_view.geometry_changed.connect(self._on_geometry_changed)
        self.settings_view.color_changed.connect(self._on_color_changed)

    def _on_geometry_changed(self):
        logging.info('Handling vis properties change')
        geo_properties = self.settings_view.get_geometry_properties()
        self.visualization.update_geometry(geo_properties)

    def _on_color_changed(self):
        logging.info('Handling color change')
        color_properties = self.settings_view.get_color_properties()
        self.visualization.update_color(color_properties)

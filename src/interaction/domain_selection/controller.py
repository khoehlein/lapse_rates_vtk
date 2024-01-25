from PyQt5.QtCore import QObject

from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.model.backend_model import DownscalingPipeline


class DownscalingController(QObject):

    def __init__(self, settings_view: SettingsViewTabbed, render_view: PyvistaView, pipeline_model: DownscalingPipeline, parent=None):
        super().__init__(parent)
        self.settings_view = settings_view
        self.render_view = render_view
        self.pipeline_model = pipeline_model
        self.settings_view.domain_settings.domain_changed.connect(self._handle_domain_change)
        self.settings_view.neighborhood_settings.neighborhood_changed.connect(self._handle_neighborhood_change)
        self._synchronize_domain_bounds()
        self._synchronize_neighborhood_properties()
        self._synchronize_downscaler_properties()
        self.pipeline_model.update()

    def _handle_domain_change(self):
        print('Found this')
        self.render_view.plotter.clear()
        self._synchronize_domain_bounds()
        self.pipeline_model.update()

    def _handle_neighborhood_change(self):
        self._synchronize_neighborhood_properties()
        self.pipeline_model.update()

    def _handle_downscaler_change(self):
        self._synchronize_downscaler_properties()
        self.pipeline_model.update()

    def _synchronize_domain_bounds(self):
        domain_bounds = self.settings_view.domain_settings.get_domain_boundaries()
        self.pipeline_model.set_domain_bounds(domain_bounds)

    def _synchronize_neighborhood_properties(self):
        properties = self.settings_view.neighborhood_settings.get_neighborhood_properties()
        self.pipeline_model.set_neighborhood_properties(properties)

    def _synchronize_downscaler_properties(self):
        properties = self.settings_view.downscaler_settings.get_downscaler_properties()
        self.pipeline_model.set_downscaler_properties(properties)

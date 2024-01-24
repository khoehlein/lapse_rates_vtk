from PyQt5.QtCore import QObject

from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.model.backend_model import BackendModel


class DomainController(QObject):

    def __init__(self, settings_view: SettingsViewTabbed, render_view: PyvistaView, backend: BackendModel, parent=None):
        super().__init__(parent)
        self.settings_view = settings_view
        self.render_view = render_view
        self.backend = backend
        self.settings_view.domain_settings.domain_changed.connect(self._handle_domain_changed)

    def _get_current_domain_bounds(self):
        return self.settings_view.domain_settings.get_domain_boundaries()

    def get_current_neighborhood_settings(self):
        return self.settings_view.neighborhood_settings.get_settings()

    def _handle_domain_changed(self):
        print('Found this')
        self.render_view.plotter.clear()
        domain_bounds = self.settings_view.domain_settings.get_domain_boundaries()
        self.backend.reset_domain_data(domain_bounds)
        self.backend.update_neighborhood_graph()
        self.backend.update_downscaler_output()

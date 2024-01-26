import logging

from PyQt5.QtCore import QObject

from src.interaction.domain_selection.controller import DownscalingController
from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.interaction.visualizations.view import SurfaceVisSettingsView
from src.model.backend_model import DownscalingPipeline
from src.model.geometry import SurfaceDataset
from src.model.visualization.scene_model import SceneModel
from src.model.visualization.visualizations import WireframeSurface, SurfaceVisualization


class SceneController(QObject):

    def __init__(
            self,
            settings_view: SettingsViewTabbed, render_view: PyvistaView,
            pipeline_controller: DownscalingController,
            pipeline_model: DownscalingPipeline, scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.settings_view = settings_view.visualization_settings
        self.pipeline_controller = pipeline_controller
        self.pipeline_model = pipeline_model
        self.render_view = render_view
        self.scene_model = scene_model
        self.pipeline_controller.domain_changed.connect(self._handle_domain_change)
        self.pipeline_controller.data_changed.connect(self._handle_data_change)
        self.settings_view.surface_o8000_settings.vis_properties_changed.connect(self._handle_vis_properties_change)
        self.settings_view.surface_o8000_settings.vis_method_changed.connect(self._handle_domain_change)
        self._handle_domain_change()

    @property
    def plotter(self):
        return self.render_view.plotter

    def _handle_domain_change(self):
        self.plotter.clear()
        pipeline_output = self.pipeline_model.get_output()
        self.scene_model.surface_o8000 = self._build_surface_visualization(
            'surface_o8000', pipeline_output, self.settings_view.surface_o1280_settings
        )

    def _build_surface_visualization(self, name: str, surface_data: SurfaceDataset, vis_settings: SurfaceVisSettingsView) -> SurfaceVisualization:
        logging.info('Building surface visualization for {}'.format(name))
        vis_properties = vis_settings.get_vis_properties()
        if isinstance(vis_properties, WireframeSurface.Properties):
            visualization = WireframeSurface(surface_data, parent=self.scene_model)
            visualization.set_vertical_scale(4000.)
            visualization.set_properties(vis_properties)
        else:
            raise NotImplementedError()
        render_data = visualization.get_render_data()
        render_data.draw(self.plotter, name=name)
        return visualization

    def _handle_data_change(self):
        return self._handle_domain_change()

    def _handle_vis_properties_change(self):
        logging.info('Handling vis properties change')
        vis_properties = self.settings_view.get_vis_properties()
        visualization = self.scene_model.surface_o8000
        render_data = visualization.get_render_data()
        render_data.update_properties(vis_properties)
        self.plotter.render()

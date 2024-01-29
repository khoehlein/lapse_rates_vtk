import logging

from PyQt5.QtCore import QObject, pyqtSignal

from src.interaction.domain_selection.controller import DownscalingController
from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.visualizations.view import SurfaceVisSettingsView, SceneSettingsView
from src.model.backend_model import DownscalingPipeline
from src.model.geometry import SurfaceDataset
from src.model.visualization.scene_model import SceneModel
from src.model.visualization.visualizations import WireframeSurface, SurfaceVisualization, TranslucentSurface


class VisualizationController(QObject):

    visualization_changed = pyqtSignal(str)

    def __init__(self, key: str, settings_view: SurfaceVisSettingsView, scene_model: SceneModel, parent=None):
        super().__init__(parent)
        self.key = str(key)
        self.settings_view = settings_view
        self.scene_model = scene_model
        self.settings_view.vis_properties_changed.connect(self._handle_vis_properties_change)
        self.settings_view.vis_method_changed.connect(self._handle_vis_method_change)
        self.settings_view.visibility_changed.connect(self._handle_visibility_change)

    def _handle_vis_properties_change(self):
        logging.info('Handling vis properties change')
        vis_properties = self.settings_view.get_vis_properties()
        visualization = self.scene_model.visuals[self.key]
        visualization.set_properties(vis_properties)

    def _handle_visibility_change(self):
        visible = self.settings_view.checkbox_visibility.isChecked()
        visualization = self.scene_model.visuals[self.key]
        visualization.set_visible(visible)

    def _handle_vis_method_change(self):
        surface_data = self.scene_model.visuals[self.key].dataset
        self.build_visualization(surface_data)
        self.visualization_changed.emit(self.key)

    def build_visualization(self, surface_data: SurfaceDataset) -> SurfaceVisualization:
        logging.info('Building visualization for {}'.format(self.key))
        vis_properties = self.settings_view.get_vis_properties()
        if isinstance(vis_properties, WireframeSurface.Properties):
            visualization = WireframeSurface(
                surface_data,
                plotter_key=self.key,
                parent=self.scene_model
            )
        elif isinstance(vis_properties, TranslucentSurface.Properties):
            visualization = TranslucentSurface(
                surface_data,
                plotter_key=self.key,
                parent=self.scene_model
            )
        else:
            raise NotImplementedError()
        visualization.set_properties(vis_properties)
        visualization.set_vertical_scale(4000.)
        visualization.set_visible(self.settings_view.get_visibility())
        self.scene_model.visuals.update({self.key: visualization})
        return visualization

    def update_visualization_data(self, visualization: SurfaceVisualization):
        raise NotImplementedError()


class SceneController(QObject):

    def __init__(
            self,
            settings_view: SceneSettingsView, render_view: PyvistaView,
            pipeline_controller: DownscalingController,
            pipeline_model: DownscalingPipeline, scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.settings_view = settings_view
        self.pipeline_controller = pipeline_controller
        self.pipeline_model = pipeline_model
        self.render_view = render_view
        self.vis_controllers = {
            key: VisualizationController(key, self.settings_view.vis_settings[key], scene_model)
            for key in self.settings_view.keys()
        }
        for key in self.vis_controllers:
            self.vis_controllers[key].visualization_changed.connect(self._on_visualization_changed)
        self._scene_model = scene_model
        self.pipeline_controller.domain_changed.connect(self._handle_domain_change)
        self.pipeline_controller.data_changed.connect(self._handle_data_change)
        self.reset_scene()

    @property
    def plotter(self):
        return self.render_view.plotter

    def reset_scene(self):
        self.plotter.clear()
        domain_data = self.pipeline_model.get_output()
        for controller in self.vis_controllers.values():
            visualization = controller.build_visualization(domain_data)
            visualization.draw(self.plotter)
        self.plotter.render()
        return self

    def _on_visualization_changed(self, key: str):
        self._scene_model.visuals[key].draw(self.plotter)

    def _handle_domain_change(self):
        return self.reset_scene()

    def update_visualization_data(self):
        domain_data = self.pipeline_model.get_output()
        for controller in self.vis_controllers.values():
            controller.update_visualization_data(domain_data)
        return self

    def _handle_data_change(self):
        self.update_visualization_data()


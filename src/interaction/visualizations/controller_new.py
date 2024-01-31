import logging

from PyQt5.QtCore import QObject, pyqtSignal
import pyvista as pv
from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.visualizations.view_new import VisualizationSettingsView
from src.model.backend_model import DownscalingPipeline
from src.model.geometry import SurfaceDataset
from src.model.visualization.scene_model_new import SceneModel, WireframeSurface, TranslucentSurface, PointsSurface, \
    SurfaceGeometry


class VisualizationController(QObject):

    visualization_changed = pyqtSignal(str)

    def __init__(self, key: str, settings_view: VisualizationSettingsView, scene_model: SceneModel, parent=None):
        super().__init__(parent)
        self.key = str(key)
        self.settings_view = settings_view
        self.scene_model = scene_model
        self.settings_view.vis_properties_changed.connect(self._handle_vis_properties_change)
        self.settings_view.representation_changed.connect(self._handle_vis_method_change)
        # self.settings_view.vis_method_changed.connect(self._handle_vis_method_change)
        # self.settings_view.visibility_changed.connect(self._handle_visibility_change)

    def _handle_vis_properties_change(self):
        logging.info('Handling vis properties change')
        vis_properties = self.settings_view.get_vis_properties()
        visualization = self.scene_model.visuals[self.key]
        visualization.set_properties(vis_properties)
        self.visualization_changed.emit(self.key)

    def _handle_visibility_change(self):
        visible = self.settings_view.checkbox_visibility.isChecked()
        visualization = self.scene_model.visuals[self.key]
        visualization.set_visibility(visible)

    def _handle_vis_method_change(self):
        visualisation = self.scene_model.visuals[self.key]
        host = visualisation.clear_host()
        surface_data = visualisation._dataset
        self.build_visualization(surface_data, host)
        self.visualization_changed.emit(self.key)

    def build_visualization(self, surface_data: SurfaceDataset, plotter: pv.Plotter) -> SurfaceGeometry:
        logging.info('Building visualization for {}'.format(self.key))
        vis_properties = self.settings_view.get_vis_properties()
        if isinstance(vis_properties, WireframeSurface.Properties):
            visualization = WireframeSurface(
                surface_data,
                visual_key=self.key,
                parent=self.scene_model
            )
        elif isinstance(vis_properties, TranslucentSurface.Properties):
            visualization = TranslucentSurface(
                surface_data,
                visual_key=self.key,
                parent=self.scene_model
            )
        elif isinstance(vis_properties, PointsSurface.Properties):
            visualization = PointsSurface(
                surface_data,
                visual_key=self.key,
                parent=self.scene_model
            )
        else:
            raise NotImplementedError()
        visualization.set_properties(vis_properties)
        visualization.set_vertical_scale(4000.)
        visualization.set_visibility(self.settings_view.get_visibility())
        visualization.set_host(plotter)
        self.scene_model.visuals.update({self.key: visualization})
        return visualization

    def update_visualization_data(self, surface_data: SurfaceDataset):
        self.build_visualization(surface_data)
        self.visualization_changed.emit(self.key)


class SceneController(QObject):

    def __init__(
            self,
            settings_view: VisualizationSettingsView, render_view: PyvistaView,
            pipeline_model: DownscalingPipeline, scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.settings_view = settings_view
        self.pipeline_model = pipeline_model
        self.render_view = render_view
        self.vis_controller = VisualizationController(
            'surface_o8000',
            self.settings_view,
            scene_model
        )
        # self.vis_controller.visualization_changed.connect(self._on_visualization_changed)
        self._scene_model = scene_model
        self.reset_scene()

    @property
    def plotter(self):
        return self.render_view.plotter

    def reset_scene(self):
        self.plotter.clear()
        domain_data = self.pipeline_model.get_output()
        visualization = self.vis_controller.build_visualization(domain_data['surface_o8000'], self.plotter)
        self.plotter.render()
        return self

    def _on_visualization_changed(self, key: str):
        self._scene_model.visuals[key].draw(self.plotter)
        self.plotter.render()

    def _handle_domain_change(self):
        return self.reset_scene()

    def update_visualization_data(self):
        domain_data = self.pipeline_model.get_output()
        for controller in self.vis_controllers.values():
            controller.update_visualization_data(domain_data)
        return self

    def _handle_data_change(self):
        self.update_visualization_data()


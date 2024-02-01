import logging
import uuid
from typing import Dict

from PyQt5.QtCore import QObject, pyqtSignal
import pyvista as pv
from src.interaction.visualizations.view import VisualizationSettingsView, DataConfiguration
from src.model.backend_model import DownscalingPipeline
from src.model.geometry import SurfaceDataset
from src.model.visualization.scene_model import SceneModel, WireframeSurface, TranslucentSurface, PointsSurface, \
    SurfaceGeometry, VisualizationModel


class VisualizationController(QObject):

    def __init__(self, settings_view: VisualizationSettingsView, parent=None):
        super().__init__(parent)
        self.settings_view = settings_view
        self.visualization: VisualizationModel = None
        self.settings_view.vis_properties_changed.connect(self._on_vis_properties_changed)
        self.settings_view.visibility_changed.connect(self._on_visibility_changed)

    @property
    def key(self):
        return self.settings_view.key

    def _on_vis_properties_changed(self):
        logging.info('Handling vis properties change')
        vis_properties = self.settings_view.get_vis_properties()
        self.visualization.set_properties(vis_properties)

    def _on_visibility_changed(self, visible: bool):
        self.visualization.set_visibility(visible)

    def visualize(self, domain_data: Dict[str, SurfaceDataset]) -> VisualizationModel:
        logging.info('Building visualization for {}'.format(self.key))
        vis_properties = self.settings_view.get_vis_properties()
        source_properties = self.settings_view.get_source_properties()
        surface_data = self._select_source_data(domain_data, source_properties)
        if isinstance(vis_properties, WireframeSurface.Properties):
            visualization = WireframeSurface(*surface_data, visual_key=self.key)
        elif isinstance(vis_properties, TranslucentSurface.Properties):
            visualization = TranslucentSurface(*surface_data, visual_key=self.key)
        elif isinstance(vis_properties, PointsSurface.Properties):
            visualization = PointsSurface(*surface_data, visual_key=self.key)
        else:
            raise NotImplementedError()
        visualization.set_properties(vis_properties)
        visualization.set_vertical_scale(4000.)
        visualization.set_visibility(self.settings_view.get_visibility())
        self.visualization = visualization
        return visualization

    def _select_source_data(self, domain_data: Dict[str, SurfaceDataset], source_properties: DataConfiguration):
        selection = {
            DataConfiguration.SURFACE_O1280: ['surface_o1280'],
            DataConfiguration.SURFACE_O8000: ['surface_o8000'],
        }[source_properties]
        return [domain_data[key] for key in selection]


class SceneController(QObject):

    def __init__(
            self,
            pipeline_model: DownscalingPipeline,
            scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.pipeline_model = pipeline_model
        self.scene_model = scene_model
        self._visualization_controls = {}
        self.vis_controllers: Dict[str, VisualizationController] = {}

    def register_settings_view(self, settings_view: VisualizationSettingsView) -> VisualizationController:
        controller = VisualizationController(settings_view, parent=self)
        self._visualization_controls[controller.key] = controller
        settings_view.visualization_changed.connect(self._on_visualization_changed)
        return controller

    def reset_scene(self):
        self.scene_model.reset()
        domain_data = self.pipeline_model.get_output()
        for controller in self._visualization_controls.values():
            visualization = controller.visualize(domain_data)
            self.scene_model.add_visualization(visualization)
        # self.scene_model.host.render()
        return self

    def _on_visualization_changed(self, key: str):
        controller = self._visualization_controls[key]
        dataset = self.pipeline_model.get_output()
        visualization = controller.visualize(dataset)
        self.scene_model.replace_visualization(visualization)

    def _on_domain_changed(self):
        return self.reset_scene()

    def _update_visualization_data(self):
        for key in self._visualization_controls:
            self._on_visualization_changed(key)

    def _on_data_changed(self):
        self._update_visualization_data()

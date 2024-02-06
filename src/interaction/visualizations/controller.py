import logging
from enum import Enum
from typing import Dict, Tuple

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QComboBox, QLineEdit, QFormLayout

from src.interaction.visualizations.view import VisualizationSettingsView, DataConfiguration, SceneSettingsView
from src.model.backend_model import DownscalingPipeline
from src.model.geometry import SurfaceDataset
from src.model.visualization.scene_model import SceneModel, VisualizationModel, ColorModel, MeshGeometryModel


class VisualizationController(QObject):

    def __init__(self, settings_view: VisualizationSettingsView, parent=None):
        super().__init__(parent)
        self.settings_view: VisualizationSettingsView = settings_view
        self.visualization: VisualizationModel = None
        self.settings_view.geometry_changed.connect(self._on_geometry_changed)
        self.settings_view.color_changed.connect(self._on_color_changed)
        self.settings_view.visibility_changed.connect(self._on_visibility_changed)

    @property
    def key(self):
        return self.settings_view.key

    def _on_geometry_changed(self):
        logging.info('Handling vis properties change')
        geo_properties = self.settings_view.get_vis_properties()
        self.visualization.update_geometry(geo_properties)

    def _on_color_changed(self):
        logging.info('Handling color change')
        color_properties = self.settings_view.get_color_properties()
        self.visualization.update_color(color_properties)

    def _on_visibility_changed(self, visible: bool):
        self.visualization.set_visibility(visible)

    def visualize(self, domain_data: Dict[str, SurfaceDataset]) -> VisualizationModel:
        logging.info('Building visualization for {}'.format(self.key))
        geo_properties = self.settings_view.get_vis_properties()
        color_properties = self.settings_view.get_color_properties()
        color_model = ColorModel.from_properties(color_properties)
        source_properties = self.settings_view.get_source_properties()
        surface_data = self._select_source_data(domain_data, source_properties)
        geometry_model = MeshGeometryModel(*surface_data)
        geometry_model.set_properties(geo_properties)
        visualization = VisualizationModel(geometry_model, color_model, visual_key=self.key)
        visibility = self.settings_view.get_visibility()
        visualization.set_visibility(visibility)
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
            scene_settings_view: SceneSettingsView,
            pipeline_model: DownscalingPipeline,
            scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.pipeline_model = pipeline_model
        self.scene_model = scene_model
        self.scene_settings_view = scene_settings_view
        self.scene_settings_view.new_interface_requested.connect(self._on_new_interface_requested)
        self._visualization_controls: Dict[str, VisualizationController] = {}

    def _on_new_interface_requested(self):
        dialog = VisualizationRequestDialog(self.scene_settings_view)
        if dialog.exec():
            label = dialog.get_label()
            visualization_type = dialog.get_visualization_type()
            factory = VisualizationFactory(self.scene_settings_view, self)
            factory.setup_visualization_interface(visualization_type, label)
        else:
            print('Nothing happened')

    def register_visualization_interface(self, settings_view: VisualizationSettingsView, controller: VisualizationController):
        self.scene_settings_view.register_settings_view(settings_view)
        settings_view.source_data_changed.connect(self._on_source_data_changed)
        self._visualization_controls[controller.key] = controller
        domain_data = self.pipeline_model.get_output()
        visualization = controller.visualize(domain_data)
        self.scene_model.add_visualization(visualization)
        return controller

    def reset_scene(self):
        self.scene_model.reset()
        domain_data = self.pipeline_model.get_output()
        for controller in self._visualization_controls.values():
            visualization = controller.visualize(domain_data)
            self.scene_model.add_visualization(visualization)
        return self

    def set_vertical_scale(self, scale):
        self.scene_model.set_vertical_scale(scale)

    def _on_visualization_changed(self, key: str):
        controller = self._visualization_controls[key]
        dataset = self.pipeline_model.get_output()
        visualization = controller.visualize(dataset)
        self.scene_model.replace_visualization(visualization)

    def _on_source_data_changed(self, key: str):
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


class VisualizationType(Enum):
    SURFACE_SCALAR_FIELD = 'surface_scalar_field'


class VisualizationRequestDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Select new visualization')

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.combo_vis_type = QComboBox(self)
        self.combo_vis_type.addItem('Surface scalar field', VisualizationType.SURFACE_SCALAR_FIELD)
        self.line_edit_label = QLineEdit()

        layout = QVBoxLayout()
        form = QFormLayout()
        form.addRow('Visualization type:', self.combo_vis_type)
        form.addRow('Label:', self.line_edit_label)
        layout.addLayout(form)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

    def get_label(self):
        text = self.line_edit_label.text()
        if not text:
            text = 'unlabeled'
        return text

    def get_visualization_type(self):
        return self.combo_vis_type.currentData()


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
            settings_view = VisualizationSettingsView(parent=self.scene_settings_view)
            controller = VisualizationController(settings_view, parent=self.scene_controller)
        else:
            raise NotImplementedError()
        self.scene_controller.register_visualization_interface(settings_view, controller)
        return settings_view, controller

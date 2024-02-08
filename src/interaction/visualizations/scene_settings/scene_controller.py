from typing import Dict

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QComboBox, QLineEdit, QVBoxLayout, QFormLayout

from src.interaction.domain_selection.controller import DownscalingController
from src.interaction.visualizations.factory import VisualizationFactory
from src.interaction.visualizations.scene_settings.scene_settings_view import SceneSettingsView
from src.interaction.visualizations.surface_scalar_field.controller import SurfaceScalarFieldController
from src.interaction.visualizations.surface_scalar_field.view import SurfaceScalarFieldSettingsView
from src.model.backend_model import DownscalingPipeline
from src.model.scene_model import SceneModel
from src.model.visualization.interface import VisualizationType


class SceneController(QObject):

    def __init__(
            self,
            scene_settings_view: SceneSettingsView,
            pipeline_controller: DownscalingController,
            pipeline_model: DownscalingPipeline,
            scene_model: SceneModel,
            parent=None,
    ):
        super().__init__(parent)
        self.pipeline_controller = pipeline_controller
        self.pipeline_controller.data_changed.connect(self._on_data_changed)
        self.pipeline_model = pipeline_model
        self.scene_model = scene_model
        self.scene_settings_view = scene_settings_view
        self.scene_settings_view.new_interface_requested.connect(self._on_new_interface_requested)
        self.scene_settings_view.reset_requested.connect(self._on_reset_requested)
        self._visualization_controls: Dict[str, SurfaceScalarFieldController] = {}

    def _on_reset_requested(self):
        for key in self._visualization_controls:
            self.scene_model.remove_visualization(key)
            self.scene_settings_view.remove_settings_view(key)
        self._visualization_controls.clear()

    def _on_new_interface_requested(self):
        dialog = VisualizationRequestDialog(self.scene_settings_view)
        if dialog.exec():
            visualization_type = dialog.get_visualization_type()
            factory = VisualizationFactory(self.scene_settings_view, self)
            settings_view, controller = factory.build_visualization_interface(visualization_type)
            controller, label = self.register_visualization_interface(settings_view, controller, dialog.get_label())
            visualization = self.build_visualization(controller, label)
            self.scene_model.add_visualization(visualization)
        else:
            print('Nothing happened')

    def build_visualization(self, controller, label):
        domain_data = self.pipeline_model.get_output()
        visualization = controller.visualize(domain_data)
        visualization.gui_label = label
        return visualization

    def _get_unique_label(self, label: str):
        labels = self.scene_model.list_labels()
        counter = 1
        while label in labels:
            label = f'{label}-{counter}'
            counter += 1
        return label

    def register_visualization_interface(
            self,
            settings_view: SurfaceScalarFieldSettingsView,
            controller: SurfaceScalarFieldController,
            raw_label: str
    ):
        label = self._get_unique_label(raw_label)
        self.scene_settings_view.register_settings_view(settings_view, label)
        settings_view.source_data_changed.connect(self._on_source_data_changed)
        self._visualization_controls[controller.key] = controller
        return controller, label

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
        self.combo_vis_type.addItem('Surface isocontours', VisualizationType.SURFACE_ISOCONTOURS)
        self.combo_vis_type.addItem('Projection lines', VisualizationType.PROJECTION_LINES)
        self.combo_vis_type.currentIndexChanged.connect(self._on_selection_changed)
        self.line_edit_label = QLineEdit()
        self._update_label_hint()

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
            text = self._get_default_label()
        return text

    def _get_default_label(self):
        suffix = {
            VisualizationType.SURFACE_SCALAR_FIELD: 'surface',
            VisualizationType.SURFACE_ISOCONTOURS: 'contours',
            VisualizationType.PROJECTION_LINES: 'hint-lines'
        }[self.combo_vis_type.currentData()]
        return f'unnamed-{suffix}'

    def _on_selection_changed(self):
        self._update_label_hint()

    def _update_label_hint(self):
        if not self.line_edit_label.text():
            self.line_edit_label.setPlaceholderText(self._get_default_label())

    def get_visualization_type(self):
        return self.combo_vis_type.currentData()

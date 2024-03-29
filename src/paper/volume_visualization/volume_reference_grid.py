from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QFormLayout, QCheckBox, QVBoxLayout

from src.paper.volume_visualization.plotter_slot import SurfaceReferenceProperties, PlotterSlot
from src.paper.volume_visualization.scaling import ScalingParameters
from src.paper.volume_visualization.volume import MeshSettingsView
from src.paper.volume_visualization.volume_data import VolumeData
from src.paper.volume_visualization.volume_data_representation import MeshDataRepresentation
from src.widgets import SelectColorButton


class ReferenceGridVisualization(MeshDataRepresentation):

    def __init__(self, slot: PlotterSlot, volume_data: VolumeData, properties: SurfaceReferenceProperties = None, scaling: ScalingParameters = None, parent=None):
        if properties is None:
            properties = SurfaceReferenceProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.mesh = None

    def set_properties(self, properties: SurfaceReferenceProperties, render=True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_reference_mesh(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self


class ReferenceLevelVisualization(MeshDataRepresentation):

    def __init__(self, slot: PlotterSlot, volume_data: VolumeData, properties: SurfaceReferenceProperties = None, scaling: ScalingParameters = None, parent=None):
        if properties is None:
            properties = SurfaceReferenceProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.mesh = None

    def set_properties(self, properties: SurfaceReferenceProperties, render=True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_level_mesh(self.scaling, use_scalar_key=False)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self


class ReferenceGridSettingsView(MeshSettingsView):

    visibility_changed = pyqtSignal()

    def build_handles(self):
        self.button_surface_color = SelectColorButton(parent=self)
        self.checkbox_visibility = QCheckBox(self)
        self.checkbox_visibility.setText('show')
        return super().build_handles()

    def _connect_signals(self):
        self.button_surface_color.color_changed.connect(self.settings_changed)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed)
        return super()._connect_signals()

    def _build_form_layout(self):
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.checkbox_visibility)
        layout = QFormLayout()
        layout.addRow("Style:", self.combo_surface_style)
        layout.addRow("Culling:", self.combo_culling)
        layout.addRow("Color:", self.button_surface_color)
        layout.addRow("Opacity:", self.spinner_opacity)
        layout.addRow("Line width:", self.spinner_line_width)
        layout.addRow("Lines as tubes:", self.checkbox_lines_as_tubes)
        layout.addRow("Point size:", self.spinner_point_size)
        layout.addRow("Points as spheres:", self.checkbox_points_as_spheres)
        layout.addRow("Metallic:", self.spinner_metallic)
        layout.addRow("Roughness:", self.spinner_roughness)
        layout.addRow("Ambient:", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power:", self.spinner_specular_power)
        layout.addRow("Edge color:", self.button_edge_color)
        layout.addRow("Edge opacity:", self.spinner_edge_opacity)
        layout.addRow("Show edges:", self.checkbox_show_edges)
        layout.addRow("Lighting:", self.checkbox_lighting)
        outer_layout.addLayout(layout)
        return outer_layout

    def get_settings(self):
        return SurfaceReferenceProperties(
            self.combo_surface_style.currentData(),
            self.spinner_line_width.value(),
            self.checkbox_lines_as_tubes.isChecked(),
            self.spinner_metallic.value(),
            self.spinner_roughness.value(),
            self.spinner_point_size.value(),
            self.checkbox_points_as_spheres.isChecked(),
            self.spinner_opacity.value(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            self.checkbox_show_edges.isChecked(),
            self.spinner_edge_opacity.value(), self.button_edge_color.current_color.getRgb(),
            self.checkbox_lighting.isChecked(),self.combo_culling.currentData(),
            self.button_surface_color.current_color.getRgb(),
        )

    def apply_settings(self, settings: SurfaceReferenceProperties):
        self.button_surface_color.set_current_color(QColor(*settings.color))
        return super().apply_settings(settings)

    def apply_visibility(self, visible: bool):
        self.checkbox_visibility.setChecked(visible)
        return self

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()


class ReferenceGridController(QObject):

    def __init__(self, view: ReferenceGridSettingsView, model: ReferenceGridVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self._synchronize_settings()
        self.view.settings_changed.connect(self.on_settings_changed)
        self.view.visibility_changed.connect(self.on_visibility_changed)
        self.model.visibility_changed.connect(self.view.apply_visibility)

    def _synchronize_settings(self):
        self.view.apply_settings(self.model.properties)
        self.view.apply_visibility(self.model.is_visible())

    def on_visibility_changed(self):
        self.model.set_visible(self.view.get_visibility())

    def on_settings_changed(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)
        return self

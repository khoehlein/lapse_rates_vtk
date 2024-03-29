from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QCheckBox, QFormLayout, QVBoxLayout

from src.interaction.visualizations.geometry_settings_view import SurfaceSettingsView
from src.paper.volume_visualization.plotter_slot import StationSiteReferenceProperties, \
    StationOnTerrainReferenceProperties
from src.paper.volume_visualization.station_data_representation import StationDataRepresentation
from src.widgets import SelectColorButton


class StationSiteReferenceVisualization(StationDataRepresentation):

    def _update_mesh_scaling(self):
        self.mesh.points[:, -1] = self.station_data.compute_station_elevation(self.scaling).ravel()

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_sites(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)


class StationOnTerrainReferenceVisualization(StationDataRepresentation):

    def _update_mesh_scaling(self):
        z_site = self.station_data.compute_station_elevation(self.scaling).ravel()
        z_surf = self.station_data.compute_terrain_elevation(self.scaling).ravel()
        n = len(z_site)
        self.mesh.points[:n, -1] = z_site
        self.mesh.points[n:, -1] = z_surf

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_reference(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)


class StationSiteReferenceSettingsView(QWidget):

    settings_changed = pyqtSignal()
    visibility_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.build_handles()
        self._connect_signals()
        self._set_layout()

    def build_handles(self):
        self.button_point_color = SelectColorButton(parent=self)
        self.spinner_point_size = QDoubleSpinBox(self)
        self.spinner_point_size.setRange(0.25, 100)
        self.spinner_point_size.setSingleStep(0.25)
        self.checkbox_points_as_spheres = QCheckBox(self)
        self.spinner_metallic = QDoubleSpinBox(self)
        self.spinner_metallic.setRange(0., 1.)
        self.spinner_metallic.setSingleStep(0.05)
        self.spinner_roughness = QDoubleSpinBox(self)
        self.spinner_roughness.setRange(0., 1.)
        self.spinner_roughness.setSingleStep(0.05)
        self.spinner_ambient = QDoubleSpinBox(self)
        self.spinner_ambient.setRange(0., 1.)
        self.spinner_ambient.setSingleStep(0.05)
        self.spinner_diffuse = QDoubleSpinBox(self)
        self.spinner_diffuse.setRange(0., 1.)
        self.spinner_diffuse.setSingleStep(0.05)
        self.spinner_specular = QDoubleSpinBox(self)
        self.spinner_specular.setRange(0., 1.)
        self.spinner_specular.setSingleStep(0.05)
        self.spinner_specular_power = QDoubleSpinBox(self)
        self.spinner_specular_power.setRange(0., 128.)
        self.spinner_specular_power.setSingleStep(0.5)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setRange(0., 1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.checkbox_lighting = QCheckBox(self)
        self.checkbox_visibility = QCheckBox(self)
        self.checkbox_visibility.setText('show')

    def _connect_signals(self):
        self.button_point_color.color_changed.connect(self.settings_changed)
        self.spinner_point_size.valueChanged.connect(self.settings_changed)
        self.checkbox_points_as_spheres.stateChanged.connect(self.settings_changed)
        self.spinner_metallic.valueChanged.connect(self.settings_changed)
        self.spinner_roughness.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.spinner_opacity.valueChanged.connect(self.settings_changed)
        self.checkbox_lighting.stateChanged.connect(self.settings_changed)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        self.setLayout(layout)

    def _build_form_layout(self):
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.checkbox_visibility)
        layout = QFormLayout()
        layout.addRow("Color:", self.button_point_color)
        layout.addRow("Opacity:", self.spinner_opacity)
        layout.addRow("Point size:", self.spinner_point_size)
        layout.addRow("Points as spheres:", self.checkbox_points_as_spheres)
        layout.addRow("Metallic:", self.spinner_metallic)
        layout.addRow("Roughness:", self.spinner_roughness)
        layout.addRow("Ambient:", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power:", self.spinner_specular_power)
        layout.addRow("Lighting:", self.checkbox_lighting)
        outer_layout.addLayout(layout)
        return outer_layout

    def apply_settings(self, settings: StationSiteReferenceProperties):
        self.button_point_color.set_current_color(QColor(*settings.color))
        self.spinner_point_size.setValue(settings.point_size)
        self.checkbox_points_as_spheres.setChecked(settings.render_points_as_spheres)
        self.spinner_metallic.setValue(settings.metallic)
        self.spinner_roughness.setValue(settings.roughness)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.spinner_opacity.setValue(settings.opacity)
        self.checkbox_lighting.setChecked(settings.lighting)
        return self

    def get_settings(self):
        return StationSiteReferenceProperties(
            self.button_point_color.current_color.getRgb(),
            self.spinner_metallic.value(),
            self.spinner_roughness.value(),
            self.spinner_point_size.value(),
            self.checkbox_points_as_spheres.isChecked(),
            self.spinner_opacity.value(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            self.checkbox_lighting.isChecked()
        )

    def apply_visibility(self, visible: bool):
        self.checkbox_visibility.setChecked(visible)
        return self

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()


class StationOnTerrainReferenceSettingsView(QWidget):

    settings_changed = pyqtSignal()
    visibility_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.build_handles()
        self._connect_signals()
        self._set_layout()

    def build_handles(self):
        self.spinner_line_width = QDoubleSpinBox(self)
        self.spinner_line_width.setRange(0.25, 20)
        self.spinner_line_width.setSingleStep(0.25)
        self.checkbox_lines_as_tubes = QCheckBox('')
        self.spinner_metallic = QDoubleSpinBox(self)
        self.spinner_metallic.setRange(0., 1.)
        self.spinner_metallic.setSingleStep(0.05)
        self.spinner_roughness = QDoubleSpinBox(self)
        self.spinner_roughness.setRange(0., 1.)
        self.spinner_roughness.setSingleStep(0.05)
        self.spinner_ambient = QDoubleSpinBox(self)
        self.spinner_ambient.setRange(0., 1.)
        self.spinner_ambient.setSingleStep(0.05)
        self.spinner_diffuse = QDoubleSpinBox(self)
        self.spinner_diffuse.setRange(0., 1.)
        self.spinner_diffuse.setSingleStep(0.05)
        self.spinner_specular = QDoubleSpinBox(self)
        self.spinner_specular.setRange(0., 1.)
        self.spinner_specular.setSingleStep(0.05)
        self.spinner_specular_power = QDoubleSpinBox(self)
        self.spinner_specular_power.setRange(0., 128.)
        self.spinner_specular_power.setSingleStep(0.5)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setRange(0., 1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.button_edge_color = SelectColorButton(parent=self)
        self.checkbox_lighting = QCheckBox(self)
        self.checkbox_visibility = QCheckBox(self)
        self.checkbox_visibility.setText('show')

    def _connect_signals(self):
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed)
        self.spinner_line_width.valueChanged.connect(self.settings_changed)
        self.checkbox_lines_as_tubes.stateChanged.connect(self.settings_changed)
        self.spinner_metallic.valueChanged.connect(self.settings_changed)
        self.spinner_roughness.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.spinner_opacity.valueChanged.connect(self.settings_changed)
        self.button_edge_color.color_changed.connect(self.settings_changed)
        self.checkbox_lighting.stateChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        self.setLayout(layout)

    def _build_form_layout(self):
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.checkbox_visibility)
        layout = QFormLayout()
        layout.addRow("Color:", self.button_edge_color)
        layout.addRow("Opacity:", self.spinner_opacity)
        layout.addRow("Line width:", self.spinner_line_width)
        layout.addRow("Lines as tubes:", self.checkbox_lines_as_tubes)
        layout.addRow("Metallic:", self.spinner_metallic)
        layout.addRow("Roughness:", self.spinner_roughness)
        layout.addRow("Ambient:", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power:", self.spinner_specular_power)
        layout.addRow("Lighting:", self.checkbox_lighting)
        outer_layout.addLayout(layout)
        return outer_layout

    def apply_settings(self, settings: StationOnTerrainReferenceProperties):
        self.spinner_line_width.setValue(settings.line_width)
        self.checkbox_lines_as_tubes.setChecked(settings.render_lines_as_tubes)
        self.spinner_metallic.setValue(settings.metallic)
        self.spinner_roughness.setValue(settings.roughness)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.spinner_opacity.setValue(settings.opacity)
        self.button_edge_color.set_current_color(QColor(*settings.color))
        self.checkbox_lighting.setChecked(settings.lighting)
        return self

    def get_settings(self):
        return StationOnTerrainReferenceProperties(
            self.button_edge_color.current_color.getRgb(),
            self.spinner_line_width.value(),
            self.checkbox_lines_as_tubes.isChecked(),
            self.spinner_metallic.value(),
            self.spinner_roughness.value(),
            self.spinner_opacity.value(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            self.checkbox_lighting.isChecked(),
        )

    def apply_visibility(self, visible: bool):
        self.checkbox_visibility.setChecked(visible)
        return self

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()
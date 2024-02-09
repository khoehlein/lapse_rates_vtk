import pyvista as pv

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QFormLayout, QLabel, \
    QCheckBox, QStackedLayout
from pyvista.plotting.opts import InterpolationType

from src.interaction.background_color.view import SelectColorButton
from src.model.visualization.mesh_geometry import ShadingType, LightingProperties, WireframeProperties, \
    TranslucentSurfaceProperties, PointsSurfaceProperties


class LightingSettingsView(QWidget):

    lighting_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_shading = QComboBox(self)
        for shading_method in ShadingType:
            self.combo_shading.addItem(shading_method.value)
        self.combo_shading.currentIndexChanged.connect(self.lighting_changed.emit)
        self.spinner_metallic = self._get_unit_spinner()
        self.spinner_roughness = self._get_unit_spinner()
        self.spinner_ambient = self._get_unit_spinner()
        self.spinner_diffuse = self._get_unit_spinner()
        self.spinner_specular = self._get_unit_spinner()
        self.spinner_specular_power = QDoubleSpinBox(self)
        self.spinner_specular_power.setMinimum(0.)
        self.spinner_specular_power.setMaximum(128.)
        self.spinner_specular_power.setSingleStep(0.5)
        self.spinner_specular_power.valueChanged.connect(self.lighting_changed.emit)
        self.button_reset = QPushButton(self)
        self.button_reset.setText("Reset")
        self.button_reset.clicked.connect(self._on_reset_clicked)
        self._set_defaults()
        self._set_layout()

    def _on_reset_clicked(self):
        self._set_defaults()
        self.lighting_changed.emit()

    def _set_layout(self):
        outer = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow(QLabel('Shading:'), self.combo_shading)
        layout.addRow(QLabel('Metallic:'), self.spinner_metallic)
        layout.addRow(QLabel('Roughness:'), self.spinner_roughness)
        layout.addRow(QLabel('Ambient:'), self.spinner_ambient)
        layout.addRow(QLabel('Diffuse:'), self.spinner_diffuse)
        layout.addRow(QLabel('Specular:'), self.spinner_specular)
        layout.addRow(QLabel('Specular power:'), self.spinner_specular_power)
        outer.addLayout(layout)
        outer.addWidget(self.button_reset)
        self.setLayout(outer)

    def _get_unit_spinner(self):
        widget = QDoubleSpinBox(self)
        widget.setMinimum(0.)
        widget.setMaximum(1.)
        widget.setSingleStep(0.05)
        widget.valueChanged.connect(self.lighting_changed.emit)
        return widget

    def _set_defaults(self):
        lighting_config = pv.global_theme.lighting_params
        interpolation_type = {
            InterpolationType.FLAT: ShadingType.FLAT,
            InterpolationType.PBR: ShadingType.PBR,
            InterpolationType.PHONG: ShadingType.PHONG,
            InterpolationType.GOURAUD: ShadingType.GOURAUD,
        }[lighting_config.interpolation]
        self.combo_shading.setCurrentText(interpolation_type.value)
        self.spinner_metallic.setValue(lighting_config.metallic)
        self.spinner_roughness.setValue(lighting_config.roughness)
        self.spinner_ambient.setValue(lighting_config.ambient)
        self.spinner_diffuse.setValue(lighting_config.diffuse)
        self.spinner_specular.setValue(lighting_config.specular)
        self.spinner_specular_power.setValue(lighting_config.specular_power)

    def get_settings(self):
        return LightingProperties(**{
            'shading': ShadingType(self.combo_shading.currentText()),
            'metallic': self.spinner_metallic.value(),
            'roughness': self.spinner_roughness.value(),
            'ambient': self.spinner_ambient.value(),
            'diffuse': self.spinner_diffuse.value(),
            'specular': self.spinner_specular.value(),
            'specular_power': self.spinner_specular_power.value(),
        })


class WireframeSettingsView(QWidget):

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_line_width = QDoubleSpinBox(self)
        self.spinner_line_width.setMinimum(0.)
        self.spinner_line_width.setMaximum(32.)
        self.spinner_line_width.setSingleStep(0.25)
        self.spinner_line_width.valueChanged.connect(self.representation_changed.emit)
        self.checkbox_lines_as_tubes = QCheckBox(self)
        self.checkbox_lines_as_tubes.stateChanged.connect(self.representation_changed.emit)
        self._set_defaults()
        self._set_layout()

    def _set_defaults(self):
        color_theme = pv.global_theme
        self.spinner_line_width.setValue(color_theme.line_width)
        self.checkbox_lines_as_tubes.setChecked(color_theme.render_lines_as_tubes)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Line width:'), self.spinner_line_width)
        layout.addRow(QLabel('Lines as tubes:'), self.checkbox_lines_as_tubes)
        self.setLayout(layout)

    def get_settings(self):
        return WireframeProperties(
            line_width=self.spinner_line_width.value(),
            render_lines_as_tubes=self.checkbox_lines_as_tubes.isChecked(),
        )


class SurfaceSettingsView(QWidget):

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_edge_color = SelectColorButton(parent=self)
        self.button_edge_color.color_changed.connect(self.representation_changed)
        self.spinner_edge_opacity = QDoubleSpinBox(self)
        self.spinner_edge_opacity.setMinimum(0.)
        self.spinner_edge_opacity.setMaximum(1.)
        self.spinner_edge_opacity.setSingleStep(0.05)
        self.spinner_edge_opacity.valueChanged.connect(self.representation_changed.emit)

        self.checkbox_show_edges = QCheckBox(self)
        self.checkbox_show_edges.stateChanged.connect(self._on_edge_state_changed)
        self.checkbox_show_edges.stateChanged.connect(self.representation_changed.emit)
        self._set_defaults()
        self._set_layout()

    def _on_edge_state_changed(self, show_edges: bool):
        self.spinner_edge_opacity.setEnabled(show_edges)
        self.button_edge_color.setEnabled(show_edges)

    def _set_defaults(self):
        color_theme = pv.global_theme
        default_color = QColor(*color_theme.edge_color.int_rgb)
        self.button_edge_color.set_current_color(default_color)
        self.spinner_edge_opacity.setValue(color_theme.edge_opacity)
        self.checkbox_show_edges.setChecked(color_theme.show_edges)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Show edges:'), self.checkbox_show_edges)
        layout.addRow(QLabel('Edge color:'), self.button_edge_color)
        layout.addRow(QLabel('Edge opacity:'), self.spinner_edge_opacity)
        self.setLayout(layout)

    def get_settings(self):
        return TranslucentSurfaceProperties(
            show_edges=self.checkbox_show_edges.isChecked(),
            edge_color=self.button_edge_color.current_color,
            edge_opacity=self.spinner_edge_opacity.value(),
        )


class PointsSettingsView(QWidget):

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_point_size = QDoubleSpinBox(self)
        self.spinner_point_size.setMinimum(0.)
        self.spinner_point_size.setMaximum(32.)
        self.spinner_point_size.setSingleStep(0.25)
        self.spinner_point_size.valueChanged.connect(self.representation_changed.emit)

        self.checkbox_points_as_spheres = QCheckBox(self)
        self.checkbox_points_as_spheres.stateChanged.connect(self.representation_changed.emit)
        self._set_defaults()
        self._set_layout()

    def _set_defaults(self):
        color_theme = pv.global_theme
        self.spinner_point_size.setValue(color_theme.point_size)
        self.checkbox_points_as_spheres.setChecked(color_theme.render_points_as_spheres)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Point size:'), self.spinner_point_size)
        layout.addRow(QLabel('Points as spheres:'), self.checkbox_points_as_spheres)
        self.setLayout(layout)

    def get_settings(self):
        return PointsSurfaceProperties(
            point_size=self.spinner_point_size.value(),
            render_points_as_spheres=self.checkbox_points_as_spheres.isChecked(),
        )


class RepresentationSettingsView(QWidget):

    representation_changed = pyqtSignal()

    def __init__(
            self,
            enable_wireframe=True,
            enable_surface=True,
            enable_points=True,
            parent=None
    ):
        super().__init__(parent)
        self.combo_geometry_style = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self.combo_geometry_style.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_geometry_style.currentIndexChanged.connect(self.representation_changed.emit)

        if enable_surface:
            self.surface_settings = SurfaceSettingsView(self)
            self.surface_settings.representation_changed.connect(self.representation_changed.emit)
            self.combo_geometry_style.addItem('Surface')
            self.interface_stack.addWidget(self.surface_settings)

        if enable_wireframe:
            self.wireframe_settings = WireframeSettingsView(self)
            self.wireframe_settings.representation_changed.connect(self.representation_changed.emit)
            self.combo_geometry_style.addItem('Wireframe')
            self.interface_stack.addWidget(self.wireframe_settings)


        if enable_points:
            self.points_settings = PointsSettingsView(self)
            self.points_settings.representation_changed.connect(self.representation_changed.emit)
            self.combo_geometry_style.addItem('Points')
            self.interface_stack.addWidget(self.points_settings)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_geometry_style)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        self.vbox_layout = layout
        # self.setLayout(layout)

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

import uuid
from enum import Enum

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QTabWidget, QStackedWidget, \
    QCheckBox, QVBoxLayout, QStackedLayout, QPushButton
import pyvista as pv
from pyvista.plotting.opts import InterpolationType

from src.interaction.background_color.view import SelectColorButton
from src.model.visualization.scene_model import ShadingMethod, WireframeSurface, TranslucentSurface, PointsSurface


class LightingSettingsView(QWidget):

    lighting_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_shading = QComboBox(self)
        for shading_method in ShadingMethod:
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
        self.checkbox_lighting = QCheckBox()
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
        layout.addRow(QLabel('Use lighting:'), self.checkbox_lighting)
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
            InterpolationType.FLAT: ShadingMethod.FLAT,
            InterpolationType.PBR: ShadingMethod.PBR,
            InterpolationType.PHONG: ShadingMethod.PHONG,
            InterpolationType.GOURAUD: ShadingMethod.GOURAUD,
        }[lighting_config.interpolation]
        self.combo_shading.setCurrentText(interpolation_type.value)
        self.spinner_metallic.setValue(lighting_config.metallic)
        self.spinner_roughness.setValue(lighting_config.roughness)
        self.spinner_ambient.setValue(lighting_config.ambient)
        self.spinner_diffuse.setValue(lighting_config.diffuse)
        self.spinner_specular.setValue(lighting_config.specular)
        self.spinner_specular_power.setValue(lighting_config.specular_power)
        self.checkbox_lighting.setChecked(pv.global_theme.lighting)

    def get_settings(self):
        return {
            'shading': ShadingMethod(self.combo_shading.currentText()),
            'metallic': self.spinner_metallic.value(),
            'roughness': self.spinner_roughness.value(),
            'ambient': self.spinner_ambient.value(),
            'diffuse': self.spinner_diffuse.value(),
            'specular': self.spinner_specular.value(),
            'specular_power': self.spinner_specular_power.value(),
            'lighting': self.checkbox_lighting.isChecked()
        }


class WireframeSettingsView(QWidget):
    REPRESENTATION_CLASS = WireframeSurface.Properties

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_frame_color = SelectColorButton(parent=self)
        self.button_frame_color.color_changed.connect(self.representation_changed.emit)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.valueChanged.connect(self.representation_changed.emit)
        self.spinner_opacity.setSingleStep(0.05)
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
        default_color = QColor(*color_theme.color.int_rgb)
        self.button_frame_color.set_current_color(default_color)
        self.spinner_opacity.setValue(color_theme.opacity)
        self.spinner_line_width.setValue(color_theme.line_width)
        self.checkbox_lines_as_tubes.setChecked(color_theme.render_lines_as_tubes)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Color:'), self.button_frame_color)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        layout.addRow(QLabel('Line width:'), self.spinner_line_width)
        layout.addRow(QLabel('Lines as tubes:'), self.checkbox_lines_as_tubes)
        self.setLayout(layout)

    def get_settings(self):
        return {
            'color': self.button_frame_color.current_color,
            'opacity': self.spinner_opacity.value(),
            'line_width': self.spinner_line_width.value(),
            'render_lines_as_tubes': self.checkbox_lines_as_tubes.isChecked(),
        }


class SurfaceSettingsView(QWidget):
    REPRESENTATION_CLASS = TranslucentSurface.Properties

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_color = SelectColorButton(parent=self)
        self.button_color.color_changed.connect(self.representation_changed)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.representation_changed.emit)

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
        default_color = QColor(*color_theme.color.int_rgb)
        self.button_color.set_current_color(default_color)
        default_color = QColor(*color_theme.edge_color.int_rgb)
        self.button_edge_color.set_current_color(default_color)
        self.spinner_opacity.setValue(color_theme.opacity)
        self.spinner_edge_opacity.setValue(color_theme.edge_opacity)
        self.checkbox_show_edges.setChecked(color_theme.show_edges)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Color:'), self.button_color)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        layout.addRow(QLabel('Show edges:'), self.checkbox_show_edges)
        layout.addRow(QLabel('Edge color:'), self.button_edge_color)
        layout.addRow(QLabel('Edge opacity:'), self.spinner_edge_opacity)
        self.setLayout(layout)

    def get_settings(self):
        return {
            'color': self.button_color.current_color,
            'opacity': self.spinner_opacity.value(),
            'show_edges': self.checkbox_show_edges.isChecked(),
            'edge_color': self.button_edge_color.current_color,
            'edge_opacity': self.spinner_edge_opacity.value(),
        }


class PointsSettingsView(QWidget):
    REPRESENTATION_CLASS = PointsSurface.Properties

    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_color = SelectColorButton(parent=self)
        self.button_color.color_changed.connect(self.representation_changed)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.representation_changed.emit)

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
        default_color = QColor(*color_theme.color.int_rgb)
        self.button_color.set_current_color(default_color)
        self.spinner_opacity.setValue(color_theme.opacity)
        self.spinner_point_size.setValue(color_theme.point_size)
        self.checkbox_points_as_spheres.setChecked(color_theme.render_points_as_spheres)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Color:'), self.button_color)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        layout.addRow(QLabel('Point size:'), self.spinner_point_size)
        layout.addRow(QLabel('Points as spheres:'), self.checkbox_points_as_spheres)
        self.setLayout(layout)

    def get_settings(self):
        return {
            'color': self.button_color.current_color,
            'opacity': self.spinner_opacity.value(),
            'point_size': self.spinner_point_size.value(),
            'render_points_as_spheres': self.checkbox_points_as_spheres.isChecked(),
        }


class GeometrySettingsView(QWidget):

    properties_changed = pyqtSignal()
    representation_changed = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_geometry_style = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self.combo_geometry_style.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_geometry_style.currentIndexChanged.connect(self.representation_changed.emit)

        self.wireframe_settings = WireframeSettingsView(self)
        self.wireframe_settings.representation_changed.connect(self.properties_changed.emit)
        self.surface_settings = SurfaceSettingsView(self)
        self.surface_settings.representation_changed.connect(self.properties_changed.emit)
        self.points_settings = PointsSettingsView(self)
        self.points_settings.representation_changed.connect(self.properties_changed.emit)

        self.combo_geometry_style.addItems(['Wireframe', 'Surface', 'Points'])
        self.interface_stack.addWidget(self.wireframe_settings)
        self.interface_stack.addWidget(self.surface_settings)
        self.interface_stack.addWidget(self.points_settings)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Style:'))
        layout.addWidget(self.combo_geometry_style)
        layout.addLayout(self.interface_stack)
        self.setLayout(layout)

    def get_settings(self):
        current_widget = self.interface_stack.currentWidget()
        return current_widget.REPRESENTATION_CLASS, current_widget.get_settings()


class RepresentationSettingsView(QTabWidget):

    vis_properties_changed = pyqtSignal()
    visualization_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lighting_settings = LightingSettingsView(self)
        self.geometry_settings = GeometrySettingsView(self)
        self.addTab(self._to_tab_widget(self.geometry_settings), 'Representation')
        self.addTab(self._to_tab_widget(self.lighting_settings), 'Lighting')
        self.lighting_settings.lighting_changed.connect(self.vis_properties_changed.emit)
        self.geometry_settings.properties_changed.connect(self.vis_properties_changed.emit)
        self.geometry_settings.representation_changed.connect(self.visualization_changed.emit)

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper

    def get_vis_properties(self):
        properties_class, geometry_settings = self.geometry_settings.get_settings()
        lighting_settings = self.lighting_settings.get_settings()
        return properties_class(**geometry_settings, **lighting_settings)


class DataConfiguration(Enum):
    SURFACE_O1280 = 'Surface (O1280)'
    SURFACE_O8000 = 'Surface (O8000)'


class VisualizationType(Enum):
    GEOMETRY = 'Geometry'
    LAPSE_RATE_O1280 = 'Lapse rate (O1280)'
    LAPSE_RATE_O8000 = 'Lapse rate (O8000)'
    T2M_O1280 = 'T2M (O1280)'
    T2M_O8000 = 'T2M (O8000)'
    T2M_DIFFERENCE = 'T2M (difference)'
    Z_O1280 = 'Z (O1280)'
    Z_O8000 = 'Z (O8000)'
    Z_DIFFERENCE = 'Z (difference)'


_available_visualizations = {
    DataConfiguration.SURFACE_O1280: [
        VisualizationType.GEOMETRY,
        # VisualizationType.LAPSE_RATE_O1280,
        # VisualizationType.T2M_O1280,
        # VisualizationType.Z_O1280
    ],
    DataConfiguration.SURFACE_O8000: [
        VisualizationType.GEOMETRY,
        # VisualizationType.LAPSE_RATE_O1280,
        # VisualizationType.LAPSE_RATE_O8000,
        # VisualizationType.T2M_O1280,
        # VisualizationType.T2M_O8000,
        # VisualizationType.Z_O1280,
        # VisualizationType.Z_O8000,
        # VisualizationType.Z_DIFFERENCE,
    ]
}


class VisualizationSettingsView(QWidget):

    vis_properties_changed = pyqtSignal()
    visualization_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(bool)

    def __init__(self, key: str = None, parent=None):
        super().__init__(parent)
        if key is None:
            key = str(uuid.uuid4())
        self.key = key
        self.combo_source_data = QComboBox(self)
        self.combo_source_data.addItems([config.value for config in DataConfiguration])
        self.combo_visualization_type = QComboBox(self)
        self._populate_visualization_combo(DataConfiguration(self.combo_source_data.currentText()))
        self.combo_source_data.currentTextChanged.connect(self._on_source_data_changed)
        self.representation_settings = RepresentationSettingsView(self)
        self.representation_settings.visualization_changed.connect(self._on_visualization_changed)
        self.representation_settings.vis_properties_changed.connect(self.vis_properties_changed.emit)
        self.checkbox_visibility = QCheckBox('Visible')
        self.checkbox_visibility.setChecked(True)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self._set_layout()

    def _on_visualization_changed(self):
        self.visualization_changed.emit(self.key)

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_source_data)
        layout.addWidget(self.combo_visualization_type)
        layout.addWidget(self.representation_settings)
        layout.addWidget(self.checkbox_visibility)
        layout.addStretch()
        self.setLayout(layout)

    def get_vis_properties(self):
        return self.representation_settings.get_vis_properties()

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()

    def get_source_properties(self) -> DataConfiguration:
        return DataConfiguration(self.combo_source_data.currentText())

    def _on_source_data_changed(self, source_type: str) -> None:
        source_type = DataConfiguration(source_type)
        self._populate_visualization_combo(source_type)
        self.visualization_changed.emit(self.key)

    def _populate_visualization_combo(self, source_type: DataConfiguration) -> None:
        self.combo_visualization_type.clear()
        self.combo_visualization_type.addItems([
            vis_type.value
            for vis_type in _available_visualizations[source_type]
        ])

import uuid
from enum import Enum

from PyQt5.QtCore import pyqtSignal, Qt, QAbstractTableModel
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QTabWidget, QStackedWidget, \
    QCheckBox, QVBoxLayout, QStackedLayout, QPushButton, QHBoxLayout, QTableView
import pyvista as pv
from pyvista.plotting.opts import InterpolationType

from src.interaction.background_color.view import SelectColorButton
from src.model.visualization.scene_model import ShadingMethod, ScalarColormapModel, UniformColorModel, \
    WireframeProperties, TranslucentSurfaceProperties, PointsSurfaceProperties, MeshGeometryModel, LightingProperties, \
    VisualizationType


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

    def get_settings(self):
        return LightingProperties(**{
            'shading': ShadingMethod(self.combo_shading.currentText()),
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_geometry_style = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self.combo_geometry_style.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_geometry_style.currentIndexChanged.connect(self.representation_changed.emit)

        self.wireframe_settings = WireframeSettingsView(self)
        self.wireframe_settings.representation_changed.connect(self.representation_changed.emit)
        self.surface_settings = SurfaceSettingsView(self)
        self.surface_settings.representation_changed.connect(self.representation_changed.emit)
        self.points_settings = PointsSettingsView(self)
        self.points_settings.representation_changed.connect(self.representation_changed.emit)

        self.combo_geometry_style.addItems(['Wireframe', 'Surface', 'Points'])
        self.interface_stack.addWidget(self.wireframe_settings)
        self.interface_stack.addWidget(self.surface_settings)
        self.interface_stack.addWidget(self.points_settings)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_geometry_style)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        self.setLayout(layout)

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()


class DataConfiguration(Enum):
    SURFACE_O1280 = 'Surface (O1280)'
    SURFACE_O8000 = 'Surface (O8000)'


_available_visualizations = {
    DataConfiguration.SURFACE_O1280: [
        VisualizationType.GEOMETRY,
        VisualizationType.LAPSE_RATE,
        VisualizationType.T2M_O1280,
        VisualizationType.Z_O1280
    ],
    DataConfiguration.SURFACE_O8000: [
        VisualizationType.GEOMETRY,
        VisualizationType.LAPSE_RATE,
        VisualizationType.T2M_O1280,
        VisualizationType.T2M_O8000,
        VisualizationType.Z_O1280,
        VisualizationType.Z_O8000,
        VisualizationType.Z_DIFFERENCE,
    ]
}

_vis_default_lapse_rate = ScalarColormapModel.Properties(
    None, 'RdBu', 1., (-14, 14), None, None
)

_vis_default_temperature = ScalarColormapModel.Properties(
    None, 'RdBu', 1.,  (260, 320), None, None
)

_vis_default_temperature_difference = ScalarColormapModel.Properties(
    None, 'RdBu', 1.,  (-40, 40), None, None
)

_vis_default_elevation = ScalarColormapModel.Properties(
    None, 'greys', 1., (-500, 9000), None, None
)

_vis_default_elevation_difference = ScalarColormapModel.Properties(
    None, 'RdBu', 1., (-1500, 1500), None, None
)

_vis_defaults = {
    VisualizationType.GEOMETRY: UniformColorModel.Properties('k', 1.),
    VisualizationType.LAPSE_RATE: _vis_default_lapse_rate,
    VisualizationType.T2M_O1280: _vis_default_temperature,
    VisualizationType.T2M_O8000: _vis_default_temperature,
    VisualizationType.T2M_DIFFERENCE: _vis_default_temperature_difference,
    VisualizationType.Z_O1280: _vis_default_elevation,
    VisualizationType.Z_O8000: _vis_default_elevation,
    VisualizationType.Z_DIFFERENCE: _vis_default_elevation_difference,
}


class UniformColorSettingsView(QWidget):

    color_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(UniformColorSettingsView, self).__init__(parent)
        self.button_color = SelectColorButton(parent=self)
        self.button_color.color_changed.connect(self.color_changed.emit)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setValue(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.color_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow('Color:', self.button_color)
        layout.addRow('Opacity', self.spinner_opacity)
        self.setLayout(layout)

    def set_defaults(self, defaults = None):
        color_theme = pv.global_theme
        default_color = QColor(*color_theme.color.int_rgb)
        self.button_color.set_current_color(default_color)
        self.spinner_opacity.setValue(color_theme.opacity)

    def get_settings(self, scalar_name: str = None):
        return UniformColorModel.Properties(
            color=self.button_color.current_color,
            opacity=self.spinner_opacity.value()
        )


class ColormapSettingsView(QWidget):

    color_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(ColormapSettingsView, self).__init__(parent)
        self.combo_cmap_name = QComboBox(self)
        self.combo_cmap_name.addItems([
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
            'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
            'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ])
        self.combo_cmap_name.currentTextChanged.connect(self.color_changed.emit)
        self.spinner_scalar_min = QDoubleSpinBox(self)
        self.spinner_scalar_max = QDoubleSpinBox(self)
        self.spinner_scalar_min.valueChanged.connect(self.spinner_scalar_max.setMinimum)
        self.spinner_scalar_max.valueChanged.connect(self.spinner_scalar_min.setMaximum)
        self.spinner_scalar_min.valueChanged.connect(self.color_changed.emit)
        self.spinner_scalar_min.setPrefix('min: ')
        self.spinner_scalar_max.valueChanged.connect(self.color_changed.emit)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_scalar_max.setPrefix('max: ')
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.color_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Colormap:'), self.combo_cmap_name)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.spinner_scalar_min)
        hlayout.addWidget(self.spinner_scalar_max)
        layout.addRow(QLabel('Scalar range:'), hlayout)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        self.setLayout(layout)

    def set_defaults(self, settings: ScalarColormapModel.Properties):
        self.combo_cmap_name.setCurrentText(settings.colormap_name)
        self.spinner_scalar_min.setMinimum(-9999)
        self.spinner_scalar_max.setMaximum(9999)
        self.spinner_scalar_min.setMaximum(settings.scalar_range[1])
        self.spinner_scalar_max.setMinimum(settings.scalar_range[0])
        self.spinner_scalar_min.setValue(settings.scalar_range[0])
        self.spinner_scalar_max.setValue(settings.scalar_range[1])
        self.spinner_opacity.setValue(settings.opacity)

    def get_settings(self, scalar_name: str):
        return ScalarColormapModel.Properties(
            scalar_name, self.combo_cmap_name.currentText(),
            self.spinner_opacity.value(), (self.spinner_scalar_min.value(), self.spinner_scalar_max.value())
        )


class VisualizationSettingsView(QWidget):

    source_data_changed = pyqtSignal(str)
    geometry_changed = pyqtSignal()
    visibility_changed = pyqtSignal(bool)
    color_changed = pyqtSignal()

    def __init__(self, key: str = None, parent=None):
        super().__init__(parent)
        if key is None:
            key = str(uuid.uuid4())
        self.key = key

        self.combo_source_data = QComboBox(self)
        self.combo_source_data.addItems([config.value for config in DataConfiguration])

        self.combo_visualization_type = QComboBox(self)
        self.combo_visualization_type.addItems([config.value for config in VisualizationType])
        self._toggle_visualization_types()
        self.combo_source_data.currentTextChanged.connect(self._on_source_data_changed)
        self.combo_visualization_type.currentTextChanged.connect(self.color_changed.emit)

        self.tabs = QTabWidget(self)
        self._build_color_tab()
        self._build_representation_tab()
        self._build_lighting_tab()

        self.checkbox_visibility = QCheckBox('Visible')
        self.checkbox_visibility.setChecked(True)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self._set_layout()

    def _toggle_visualization_types(self):
        model = self.combo_visualization_type.model()
        selected_source = DataConfiguration(self.combo_source_data.currentText())
        for i in range(model.rowCount()):
            item = model.item(i)
            item_text = self.combo_visualization_type.itemText(i)
            item.setEnabled(VisualizationType(item_text) in _available_visualizations[selected_source])

    def _on_source_data_changed(self, source_type: str) -> None:
        self._toggle_visualization_types()
        self.source_data_changed.emit(self.key)

    def _build_color_tab(self):
        self.interface_stack = QStackedLayout()
        for color_type in VisualizationType:
            widget = UniformColorSettingsView(self) if color_type == VisualizationType.GEOMETRY else ColormapSettingsView(self)
            self.interface_stack.addWidget(widget)
            widget.set_defaults(_vis_defaults[color_type])
            widget.color_changed.connect(self.color_changed.emit)
        self.combo_visualization_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        color_stack_widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(self.combo_visualization_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        color_stack_widget.setLayout(layout)
        self.tabs.addTab(color_stack_widget, 'Color')

    def _build_representation_tab(self):
        self.representation_settings = RepresentationSettingsView(self)
        self.representation_settings.representation_changed.connect(self.geometry_changed.emit)
        self.tabs.addTab(self.representation_settings, 'Representation')

    def _build_lighting_tab(self):
        self.lighting_settings = LightingSettingsView(self)
        self.lighting_settings.lighting_changed.connect(self.geometry_changed.emit)
        self.tabs.addTab(self._to_tab_widget(self.lighting_settings), 'Lighting')

    def _to_tab_widget(self, x):
        widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(x)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_source_data)
        layout.addWidget(self.tabs)
        layout.addWidget(self.checkbox_visibility)
        layout.addStretch()
        self.setLayout(layout)

    def get_vis_properties(self):
        rep_settings = self.representation_settings.get_settings()
        lighting_settings = self.lighting_settings.get_settings()
        prop = MeshGeometryModel.Properties(mesh=rep_settings, lighting=lighting_settings)
        return prop

    def get_color_properties(self):
        scalar_name = VisualizationType(self.combo_visualization_type.currentText()).name.lower()
        return self.interface_stack.currentWidget().get_settings(scalar_name)

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()

    def get_source_properties(self) -> DataConfiguration:
        return DataConfiguration(self.combo_source_data.currentText())

    def _populate_visualization_combo(self, source_type: DataConfiguration) -> None:
        self.combo_visualization_type.clear()
        self.combo_visualization_type.addItems([
            vis_type.value
            for vis_type in _available_visualizations[source_type]
        ])

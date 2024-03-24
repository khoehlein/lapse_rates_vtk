import math
from enum import Enum
from typing import Dict
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout, QStackedLayout, \
    QVBoxLayout, QSpinBox, QSlider
from src.paper.volume_visualization.color_lookup import InteractiveColorLookup
from src.paper.volume_visualization.plotter_slot import ActorProperties, ContourParameters, PlotterSlot, \
    VolumeProperties, IsocontourProperties, SurfaceProperties, InterpolationType, SurfaceStyle, CullingMethod
from src.paper.volume_visualization.scaling import ScalingParameters
from src.paper.volume_visualization.volume_data import VolumeData
from src.paper.volume_visualization.volume_data_representation import VolumeDataRepresentation, MeshDataRepresentation
from src.widgets import SelectColorButton


class VolumeRepresentationMode(Enum):
    MODEL_LEVELS = 'model_levels'
    DVR = 'dvr'
    ISO_CONTOURS = 'iso_levels'


class DVRRepresentation(MeshDataRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: VolumeProperties = None, scaling: ScalingParameters = None, parent=None
    ):
        if properties is None:
            properties = VolumeProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup

    def set_properties(self, properties: VolumeProperties, render: bool = True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_volume_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_volume_mesh(self.mesh, lookup_table, self.properties, render=False)
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class ModelLevelRepresentation(MeshDataRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: SurfaceProperties = None, scaling: ScalingParameters = None, parent=None
    ):
        if properties is None:
            properties = SurfaceProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup

    def set_properties(self, properties: SurfaceProperties, render: bool = True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_level_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_surface_mesh(self.mesh, lookup_table, self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class IsocontourRepresentation(VolumeDataRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: IsocontourProperties = None, scaling: ScalingParameters = None, parent=None
    ):
        if properties is None:
            properties = IsocontourProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup
        self.mesh = None

    def set_properties(self, properties: IsocontourProperties, render: bool = True):
        if properties.contours != self.properties.contours:
            self.properties = properties
            if self.is_visible():
                self.blockSignals(True)
                self.clear(render=False)
                self.show(render=render)
                self.blockSignals(False)
        else:
            super().set_properties(properties, render=render)
        return self

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_contour_mesh(self.properties.contours, self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_surface_mesh(self.mesh, lookup_table, self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

    def clear(self, render: bool = True):
        self.slot.clear(render=render)
        self.mesh = None
        self.visibility_changed.emit(False)
        return self

    def is_visible(self):
        return self.mesh is not None

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.is_visible():
            self.blockSignals(True)
            self.clear(render=False)
            self.scaling = scaling
            self.show(render=render)
            self.blockSignals(False)
        return self

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class ScalarVolumeVisualization(VolumeDataRepresentation):

    properties_changed = pyqtSignal(str)
    visibility_changed = pyqtSignal(bool)

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: ActorProperties, scaling: ScalingParameters = None, visible=True, parent: QObject = None
    ):
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup
        self.representation = None

        self.color_lookup.lookup_table_changed.connect(self.on_color_lookup_changed)

        if visible:
            self.show()

    def on_color_lookup_changed(self):
        if self.representation is not None:
            self.representation.update_scalar_colors()

    def clear(self, render: bool = True):
        if self.representation is not None:
            self.representation.clear(render=render)
            self.representation = None
        self.visibility_changed.emit(False)
        return self

    def show(self, render: bool = True):
        if self.representation is not None:
            return self
        properties_type = type(self.properties)
        if properties_type == VolumeProperties:
            self.representation = DVRRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == SurfaceProperties:
            self.representation = ModelLevelRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == IsocontourProperties:
            self.representation = IsocontourRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        else:
            raise NotImplementedError()
        self.representation.show(render=render)
        self.visibility_changed.emit(True)
        return self

    def change_representation(self, properties: ActorProperties, render: bool = True):
        self.properties = properties
        if self.representation is not None:
            self.blockSignals(True)
            self.clear(render=False)
            self.show()
            self.blockSignals(False)
        return self

    def set_properties(self, properties: ActorProperties, render: bool = True):
        self.properties = properties
        if self.representation is not None:
            self.representation.set_properties(properties, render=render)
        return self

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.representation is not None:
            self.representation.set_scaling(scaling, render=render)
        return self

    def is_visible(self):
        return self.representation is not None

    @property
    def representation_mode(self):
        return {
            VolumeProperties: VolumeRepresentationMode.DVR,
            SurfaceProperties: VolumeRepresentationMode.MODEL_LEVELS,
            IsocontourProperties: VolumeRepresentationMode.ISO_CONTOURS,
        }.get(type(self.properties))


class DVRSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_interpolation_type = QComboBox(self)
        self.combo_interpolation_type.addItem('linear', InterpolationType.LINEAR)
        self.combo_interpolation_type.addItem('nearest', InterpolationType.NEAREST)
        self.spinner_opacity_unit_length = QSlider(Qt.Orientation.Horizontal, self)
        self.spinner_opacity_unit_length.setRange(-20, 20)
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
        self.checkbox_shade = QCheckBox(self)
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.combo_interpolation_type.currentIndexChanged.connect(self.settings_changed)
        self.spinner_opacity_unit_length.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.checkbox_shade.stateChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow("Interpolation", self.combo_interpolation_type)
        layout.addRow("Opacity unit length:", self.spinner_opacity_unit_length)
        layout.addRow("Ambient", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power", self.spinner_specular_power)
        layout.addRow("Shade", self.checkbox_shade)
        self.setLayout(layout)

    def apply_settings(self, settings: VolumeProperties):
        self.combo_interpolation_type.setCurrentText(settings.interpolation_type.value)
        slider_setting = int(round(math.log2(settings.opacity_unit_distance) * 4))
        self.spinner_opacity_unit_length.setValue(slider_setting)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.checkbox_shade.setChecked(settings.shade)
        return self

    def get_settings(self):
        opacity = 2. ** (self.spinner_opacity_unit_length.value() / 4.)
        return VolumeProperties(
            self.combo_interpolation_type.currentData(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            opacity,
            self.checkbox_shade.isChecked()
        )


class SurfaceSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(SurfaceSettingsView, self).__init__(parent)
        self.build_handles()
        self._connect_signals()
        self._set_layout()

    def build_handles(self):
        self.combo_surface_style = QComboBox(self)
        self.combo_surface_style.addItem('wireframe', SurfaceStyle.WIREFRAME)
        self.combo_surface_style.addItem('surface', SurfaceStyle.SURFACE)
        self.combo_surface_style.addItem('points', SurfaceStyle.POINTS)
        self.combo_culling = QComboBox(self)
        self.combo_culling.addItem('front', CullingMethod.FRONT)
        self.combo_culling.addItem('back', CullingMethod.BACK)
        self.combo_culling.addItem('none', CullingMethod.NONE)
        self.spinner_line_width = QDoubleSpinBox(self)
        self.spinner_line_width.setRange(0.25, 20)
        self.spinner_line_width.setSingleStep(0.25)
        self.checkbox_lines_as_tubes = QCheckBox('')
        self.spinner_point_size = QDoubleSpinBox(self)
        self.spinner_point_size.setRange(0.25, 20)
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
        self.spinner_edge_opacity = QDoubleSpinBox(self)
        self.spinner_edge_opacity.setRange(0., 1.)
        self.spinner_edge_opacity.setSingleStep(0.05)
        self.checkbox_show_edges = QCheckBox(self)
        self.button_edge_color = SelectColorButton()
        self.checkbox_lighting = QCheckBox(self)

    def _connect_signals(self):
        self.combo_surface_style.currentTextChanged.connect(self.settings_changed)
        self.combo_culling.currentTextChanged.connect(self.settings_changed)
        self.spinner_line_width.valueChanged.connect(self.settings_changed)
        self.checkbox_lines_as_tubes.stateChanged.connect(self.settings_changed)
        self.spinner_point_size.valueChanged.connect(self.settings_changed)
        self.checkbox_points_as_spheres.stateChanged.connect(self.settings_changed)
        self.spinner_metallic.valueChanged.connect(self.settings_changed)
        self.spinner_roughness.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.spinner_opacity.valueChanged.connect(self.settings_changed)
        self.spinner_edge_opacity.valueChanged.connect(self.settings_changed)
        self.checkbox_show_edges.stateChanged.connect(self.settings_changed)
        self.button_edge_color.color_changed.connect(self.settings_changed)
        self.checkbox_lighting.stateChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        self.setLayout(layout)

    def _build_form_layout(self):
        layout = QFormLayout()
        layout.addRow("Style:", self.combo_surface_style)
        layout.addRow("Culling:", self.combo_culling)
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
        return layout

    def apply_settings(self, settings: SurfaceProperties):
        self.combo_surface_style.setCurrentText(settings.style.value)
        self.combo_culling.setCurrentText(settings.culling.value)
        self.spinner_line_width.setValue(settings.line_width)
        self.checkbox_lines_as_tubes.setChecked(settings.render_lines_as_tubes)
        self.spinner_point_size.setValue(settings.point_size)
        self.checkbox_points_as_spheres.setChecked(settings.render_points_as_spheres)
        self.spinner_metallic.setValue(settings.metallic)
        self.spinner_roughness.setValue(settings.roughness)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.spinner_opacity.setValue(settings.opacity)
        self.spinner_edge_opacity.setValue(settings.edge_opacity)
        self.checkbox_show_edges.setChecked(settings.show_edges)
        self.button_edge_color.set_current_color(QColor(*settings.edge_color))
        self.checkbox_lighting.setChecked(settings.lighting)
        return self

    def get_settings(self):
        return SurfaceProperties(
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
            self.checkbox_lighting.isChecked(),self.combo_culling.currentData()
        )


class IsocontourSettingsView(SurfaceSettingsView):

    def build_handles(self):
        super().build_handles()
        self.combo_contour_key = QComboBox(self)
        self.combo_contour_key.addItems(['z_model_levels', 't', 'grad_t'])
        self.spinner_num_contours = QSpinBox(self)
        self.spinner_num_contours.setRange(4, 128)

    def _connect_signals(self):
        super()._connect_signals()
        self.combo_contour_key.currentTextChanged.connect(self.settings_changed)
        self.spinner_num_contours.valueChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        layout.addRow('Contour parameter:', self.combo_contour_key)
        layout.addRow('Num. of contours:', self.spinner_num_contours)
        self.setLayout(layout)

    def get_settings(self):
        return IsocontourProperties(
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
            ContourParameters(
                self.combo_contour_key.currentText(),
                self.spinner_num_contours.value()
            )
        )

    def apply_settings(self, settings: IsocontourProperties):
        super().apply_settings(settings)
        contours = settings.contours
        self.combo_contour_key.setCurrentText(contours.contour_key)
        self.spinner_num_contours.setValue(contours.num_levels)
        return self


class ScalarVolumeSettingsView(QWidget):

    settings_changed = pyqtSignal()
    representation_changed = pyqtSignal()
    visibility_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_representation_type = QComboBox(self)
        self.combo_representation_type.addItem("DVR", VolumeRepresentationMode.DVR)
        self.combo_representation_type.addItem("model levels", VolumeRepresentationMode.MODEL_LEVELS)
        self.combo_representation_type.addItem("isocontours", VolumeRepresentationMode.ISO_CONTOURS)
        self.representation_views = {
            VolumeRepresentationMode.DVR: DVRSettingsView(self),
            VolumeRepresentationMode.MODEL_LEVELS: SurfaceSettingsView(self),
            VolumeRepresentationMode.ISO_CONTOURS: IsocontourSettingsView(self)
        }
        self.interface_stack = QStackedLayout()
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentationMode.DVR])
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentationMode.MODEL_LEVELS])
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentationMode.ISO_CONTOURS])
        self.checkbox_visible = QCheckBox(self)
        self.checkbox_visible.setChecked(True)
        self.checkbox_visible.setText('show')
        self._connect_signals()
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.checkbox_visible)
        layout.addWidget(self.combo_representation_type)
        layout.addLayout(self.interface_stack)
        # layout.addWidget(self.representation_views[VolumeRepresentation.DVR])
        # layout.addWidget(self.representation_views[VolumeRepresentation.MODEL_LEVELS])
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self):
        self.combo_representation_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_representation_type.currentIndexChanged.connect(self.representation_changed)
        for key in VolumeRepresentationMode:
            if key in self.representation_views:
                self.representation_views[key].settings_changed.connect(self.settings_changed)
        self.checkbox_visible.stateChanged.connect(self.visibility_changed)

    def get_representation_mode(self):
        return self.combo_representation_type.currentData()

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

    def apply_settings(self, settings: Dict[VolumeRepresentationMode, ActorProperties], use_defaults=False):
        defaults = {
            VolumeRepresentationMode.DVR: VolumeProperties(),
            VolumeRepresentationMode.MODEL_LEVELS: SurfaceProperties(),
            VolumeRepresentationMode.ISO_CONTOURS: IsocontourProperties()
        }
        for key in VolumeRepresentationMode:
            if key in settings:
                props = settings[key]
            elif use_defaults:
                props = defaults.get(key)
            else:
                props = None
            if props is not None and key in self.representation_views:
                self.representation_views[key].apply_settings(props)
        return self

    def apply_visibility(self, visible):
        self.checkbox_visible.setChecked(visible)
        return self

    def get_visibility(self):
        return self.checkbox_visible.isChecked()


class ScalarVolumeController(QObject):

    def __init__(self, view: ScalarVolumeSettingsView, model: ScalarVolumeVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self._synchronize_settings()
        self.view.settings_changed.connect(self.on_settings_changed)
        self.view.representation_changed.connect(self.on_representation_changed)
        self.view.visibility_changed.connect(self.on_visibility_changed)
        self.model.visibility_changed.connect(self.view.apply_visibility)

    def _synchronize_settings(self):
        self.view.apply_settings({self.model.representation_mode: self.model.properties}, use_defaults=True)
        self.view.apply_visibility(self.model.is_visible())

    def on_visibility_changed(self):
        self.model.set_visible(self.view.get_visibility())

    def on_settings_changed(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)
        return self

    def on_representation_changed(self):
        settings = self.view.get_settings()
        self.model.change_representation(settings)
        return self
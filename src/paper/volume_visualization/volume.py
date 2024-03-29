import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union

import pandas as pd
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout, QStackedLayout, \
    QVBoxLayout, QSpinBox, QSlider
from src.paper.volume_visualization.color_lookup import InteractiveColorLookup
from src.paper.volume_visualization.multi_method_visualization import MultiMethodScalarVisualization, \
    MultiMethodSettingsView
from src.paper.volume_visualization.plotter_slot import ActorProperties, ContourParameters, PlotterSlot, \
    VolumeProperties, IsocontourProperties, MyMeshProperties, InterpolationType, SurfaceStyle, CullingMethod, \
    LevelProperties
from src.paper.volume_visualization.scaling import ScalingParameters
from src.paper.volume_visualization.volume_data import VolumeData
from src.paper.volume_visualization.volume_data_representation import VolumeDataRepresentation, MeshDataRepresentation
from src.widgets import SelectColorButton

import xarray as xr


class VolumeRepresentationMode(Enum):
    MODEL_LEVELS = 'model_levels'
    MODEL_MESH = 'model_mesh'
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
        self.slot.show_scalar_volume(self.mesh, lookup_table, self.properties, render=False)
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
            properties: LevelProperties = None, scaling: ScalingParameters = None, parent=None
    ):
        if properties is None:
            properties = LevelProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup

    def set_properties(self, properties: LevelProperties, render: bool = True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_level_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_scalar_mesh(self.mesh, lookup_table, self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class ModelMeshRepresentation(MeshDataRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: MyMeshProperties = None, scaling: ScalingParameters = None, parent=None
    ):
        if properties is None:
            properties = MyMeshProperties()
        super().__init__(slot, volume_data, properties, scaling, parent)
        self.color_lookup = color_lookup

    def set_properties(self, properties: MyMeshProperties, render: bool = True):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.volume_data.get_volume_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_scalar_mesh(self.mesh, lookup_table, self.properties, render=render)
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
        self.slot.show_scalar_mesh(self.mesh, lookup_table, self.properties, render=render)
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


class VolumeScalarVisualization(MultiMethodScalarVisualization):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData, color_lookup: InteractiveColorLookup,
            properties: ActorProperties, scaling: ScalingParameters = None, parent: QObject = None
    ):
        super().__init__(slot, color_lookup, properties, scaling, parent)
        self.volume_data = volume_data

    def _build_representation(self):
        properties_type = type(self.properties)
        if properties_type == VolumeProperties:
            self.representation = DVRRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == MyMeshProperties:
            self.representation = ModelMeshRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == LevelProperties:
            self.representation = ModelLevelRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == IsocontourProperties:
            self.representation = IsocontourRepresentation(
                self.slot, self.volume_data, self.color_lookup, self.properties, self.scaling
            )
        else:
            raise NotImplementedError()

    @property
    def representation_mode(self):
        return {
            VolumeProperties: VolumeRepresentationMode.DVR,
            LevelProperties: VolumeRepresentationMode.MODEL_LEVELS,
            MyMeshProperties: VolumeRepresentationMode.MODEL_MESH,
            IsocontourProperties: VolumeRepresentationMode.ISO_CONTOURS,
        }.get(type(self.properties))

    def update_data(self, new_data: Union[xr.Dataset, pd.DataFrame], render: bool = True):
        self.blockSignals(True)
        self.volume_data.update_field_data(new_data)
        if self.is_visible():
            self.representation.update(render=render)
        self.blockSignals(False)
        return self


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


class MeshSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(MeshSettingsView, self).__init__(parent)
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

    def apply_settings(self, settings: MyMeshProperties):
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
        return MyMeshProperties(
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


class LevelSettingsView(MeshSettingsView):

    def get_settings(self):
        return LevelProperties(
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

    def apply_settings(self, settings: LevelProperties):
        return super().apply_settings(settings)


class IsocontourSettingsView(MeshSettingsView):

    def build_handles(self):
        super().build_handles()
        self.combo_contour_key = QComboBox(self)
        self.set_contour_keys(['z_model_levels', 't', 'grad_t', 'latitude_3d', 'longitude_3d', 'model_level_3d'])
        self.spinner_num_contours = QSpinBox(self)
        self.spinner_num_contours.setRange(2, 128)

    def set_contour_keys(self, keys):
        self.combo_contour_key.clear()
        self.combo_contour_key.addItems(keys)
        return self

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


class VolumeScalarSettingsView(MultiMethodSettingsView):

    def __init__(self, use_dvr=True, use_model_levels=True, use_contours=True, use_mesh=True,  parent=None):
        defaults = {
            VolumeRepresentationMode.DVR: VolumeProperties(),
            VolumeRepresentationMode.MODEL_LEVELS: LevelProperties(),
            VolumeRepresentationMode.MODEL_MESH: MyMeshProperties(),
            VolumeRepresentationMode.ISO_CONTOURS: IsocontourProperties()
        }
        view_mapping = {}
        labels = {}
        if use_dvr:
            view_mapping[VolumeRepresentationMode.DVR] = DVRSettingsView
            labels[VolumeRepresentationMode.DVR] = 'DVR'
        if use_model_levels:
            view_mapping[VolumeRepresentationMode.MODEL_LEVELS] = LevelSettingsView
            labels[VolumeRepresentationMode.MODEL_LEVELS] = 'model levels'
        if use_mesh:
            view_mapping[VolumeRepresentationMode.MODEL_MESH] = MeshSettingsView
            labels[VolumeRepresentationMode.MODEL_MESH] = 'model mesh'
        if use_contours:
            view_mapping[VolumeRepresentationMode.ISO_CONTOURS] = IsocontourSettingsView
            labels[VolumeRepresentationMode.ISO_CONTOURS] = 'isocontours'
        super().__init__(view_mapping, defaults, labels, parent)

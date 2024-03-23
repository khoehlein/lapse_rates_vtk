import math
import uuid
from collections import namedtuple
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Union

import xarray as xr
import pyvista as pv
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout, QStackedWidget, QStackedLayout, \
    QVBoxLayout

from src.model.geometry import TriangleMesh, LocationBatch, Coordinates, WedgeMesh
from src.paper.volume_visualization.color_lookup import InteractiveColorLookup


class VolumeRepresentation(Enum):
    MODEL_LEVELS = 'model_levels'
    DVR = 'dvr'
    ISO_LEVELS = 'iso_levels'


class InterpolationType(Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'


class CullingMethod(Enum):
    BACK = 'back'
    FRONT = 'front'
    NONE = 'none'


class SurfaceStyle(Enum):
    WIREFRAME = 'wireframe'
    POINTS = 'points'
    SURFACE = 'surface'


@dataclass
class ActorProperties(object):
    pass


@dataclass
class SurfaceProperties(ActorProperties):
    color: Any = pv.global_theme.color.int_rgb
    style: SurfaceStyle = SurfaceStyle.SURFACE
    line_width: float = pv.global_theme.line_width
    render_lines_as_tubes: bool = pv.global_theme.render_lines_as_tubes
    metallic: float = pv.global_theme.lighting_params.metallic
    roughness: float = pv.global_theme.lighting_params.roughness
    point_size: float = pv.global_theme.point_size
    render_points_as_spheres: bool = pv.global_theme.render_points_as_spheres
    opacity: float = pv.global_theme.opacity
    ambient: float = pv.global_theme.lighting_params.ambient
    diffuse: float = pv.global_theme.lighting_params.diffuse
    specular: float = pv.global_theme.lighting_params.specular
    specular_power: float = pv.global_theme.lighting_params.specular_power
    show_edges: bool = pv.global_theme.show_edges
    edge_opacity: float = pv.global_theme.edge_opacity
    edge_color: Any = pv.global_theme.edge_color.int_rgb
    lighting: bool = pv.global_theme.lighting
    culling: CullingMethod = CullingMethod.NONE


@dataclass
class VolumeProperties(ActorProperties):
    interpolation_type: InterpolationType = InterpolationType.LINEAR
    ambient: float = pv.global_theme.lighting_params.ambient
    diffuse: float = pv.global_theme.lighting_params.diffuse
    specular: float = pv.global_theme.lighting_params.specular
    specular_power: float = pv.global_theme.lighting_params.specular_power
    opacity_unit_distance: float = 1.
    shade: bool = True


ScaleParameters = namedtuple('ScaleParameters', ['scale', 'offset_scale'])


class VolumeFieldData(object):

    def __init__(self, key: str, field_data: xr.Dataset, terrain_data: xr.Dataset):
        self.key = key
        self.field_data = field_data
        self.terrain_data = terrain_data

    def get_volume_mesh(self, scale_params: ScaleParameters) -> pv.PolyData:
        surface_mesh = TriangleMesh(
            LocationBatch(Coordinates.from_xarray(self.terrain_data)),
            self.terrain_data['triangles'].values
        )
        z = self._relative_elevation.copy()
        if self.offset_scale != 1.:
            z *= self.offset_scale
        if self.offset != 0.:
            z += self.offset
        z += self._baseline_elevation
        z /= self.vertical_scale
        mesh = WedgeMesh(surface_mesh, z)
        mesh = mesh.to_wedge_grid()
        mesh[self.key] = self.model_data[self.key].values.ravel()


# class DVRVisualization(object):
#
#     def __init__(
#             self,
#             key: str, model_data: xr.Dataset, terrain_data: xr.Dataset,
#             canvas: pv.Plotter, actor_label: str, scalar_bar_label: str,
#             color_lookup: InteractiveColorLookup,
#     ):

class VolumeVisualization(QObject):
    properties_changed = pyqtSignal(str)

    def __init__(
            self,
            key: str, color_label: str,
            vertical_scale: float,
            model_data: xr.Dataset, terrain_data: xr.Dataset,
            color_lookup: InteractiveColorLookup,
            canvas: pv.Plotter,
            offset_scale: float = None,
            offset: float = None,
            parent: QObject = None
    ):
        super().__init__(parent)
        self.uid = str(uuid.uuid4())
        self.key = key
        self.scalar_bar_label = color_label
        self.model_data = model_data
        self.terrain_data = terrain_data
        self.color_lookup = color_lookup
        self.color_lookup.lookup_table_changed.connect(self.on_colormap_changed)
        self.canvas = canvas
        self.vertical_scale = float(vertical_scale)
        self.offset_scale = float(offset_scale) if offset_scale is not None else 1.
        self.offset = float(offset) if offset is not None else 0.

        self._baseline_elevation = self.terrain_data['z_surf'].values
        self._relative_elevation = self.model_data['z_model_levels'].values - self._baseline_elevation

        self.representation_mode = VolumeRepresentation.DVR
        self.props = VolumeProperties()
        self.actors = None
        self._draw_representation = {
            VolumeRepresentation.MODEL_LEVELS: self._draw_model_levels,
            VolumeRepresentation.ISO_LEVELS: self._draw_iso_levels,
            VolumeRepresentation.DVR: self._draw_dvr,
        }
        self.visible = True

    # def clear(self):
    #     if self.actors is None

    def set_properties(self, properties: ActorProperties):
        self._verify_properties(properties)
        self.props = properties
        if self.actors is not None:
            self._update_actor_properties()

    def set_visible(self, visible: bool):
        self.visible = visible
        if self.actors is not None:
            if self.representation_mode == VolumeRepresentation.DVR:
                self.actors['mesh'].mapper.scalar_visibility = visible
            else:
                self.actors['mesh'].visibility = visible
        return self

    def _update_actor_properties(self):
        actor_props = self.actors['mesh'].prop
        for field in fields(self.props):
            prop_name = field.name
            value = getattr(self.props, prop_name)
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            setattr(actor_props, prop_name, value)

    def _verify_properties(self, properties: ActorProperties):
        expected_class = {
            VolumeRepresentation.MODEL_LEVELS: SurfaceProperties,
            VolumeRepresentation.ISO_LEVELS: SurfaceProperties,
            VolumeRepresentation.DVR: VolumeProperties,
        }.get(self.representation_mode)
        assert isinstance(properties, expected_class)

    def on_colormap_changed(self):
        if self.actors is not None:
            if self.representation_mode == VolumeRepresentation.DVR:
                self._update_color_in_volume_actor()
            else:
                self._update_color_in_surface_actor()
            self._draw_scalar_bar()

    def _update_color_in_volume_actor(self):
        lut = self.color_lookup.lookup_table
        scalar_range = lut.scalar_range
        mapper = self.actors['mesh'].mapper
        mapper.scalar_range = scalar_range
        mapper.lookup_table = lut
        self.actors['mesh'].prop.apply_lookup_table(lut)

    def _update_color_in_surface_actor(self):
        mapper = self.actors['mesh'].mapper
        mapper.lookup_table = self.color_lookup.lookup_table

    def _remove_scalar_bar(self):
        # self.canvas.remove_actor(self.actors['scalar_bar'])
        self.canvas.remove_scalar_bar(title=self.scalar_bar_label)
        del self.actors['scalar_bar']

    def _draw_scalar_bar(self, render=True):
        actor = self.canvas.add_scalar_bar(
            title=self.scalar_bar_label,
            mapper=self.actors['mesh'].mapper,
            interactive=True, render=render
        )
        self.actors['scalar_bar'] = actor

    def set_representation_mode(self, mode: VolumeRepresentation):
        self.representation_mode = mode
        self.draw()
        return self

    def draw(self):
        actor = self._draw_representation[self.representation_mode](render=False)
        if self.actors is not None:
            self.actors['mesh'] = actor
        else:
            self.actors = {'mesh': actor}
        self._draw_scalar_bar()
        return self

    def _build_volume_mesh(self):
        surface_mesh = TriangleMesh(
            LocationBatch(Coordinates.from_xarray(self.terrain_data)),
            self.terrain_data['triangles'].values
        )
        z = self._relative_elevation.copy()
        if self.offset_scale != 1.:
            z *= self.offset_scale
        if self.offset != 0.:
            z += self.offset
        z += self._baseline_elevation
        z /= self.vertical_scale
        mesh = WedgeMesh(surface_mesh, z)
        mesh = mesh.to_wedge_grid()
        mesh[self.key] = self.model_data[self.key].values.ravel()
        return mesh

    def _draw_dvr(self, render=True) -> pv.volume.Volume:
        mesh = self._build_volume_mesh()
        drawing_kws = self._get_drawing_kws()
        lut = self.color_lookup.lookup_table
        scalar_range = lut.scalar_range
        actor = self.canvas.add_volume(
            mesh, name=self.uid, clim=scalar_range, cmap=lut,
            blending='composite', mapper='smart', show_scalar_bar=False, render=render,
            **drawing_kws
        )
        return actor

    def _get_drawing_kws(self):
        if self.props is None:
            return {}
        kws = {}
        for field in fields(self.props):
            field_name = field.name
            if field_name in ['interpolation_type']:
                continue
            value = getattr(self.props, field_name)
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            kws[field_name] = value
        return kws

    def _draw_iso_levels(self, render=True):
        raise NotImplementedError()

    def _draw_model_levels(self, render=True):
        raise NotImplementedError()


class DVRSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_interpolation_type = QComboBox(self)
        self.combo_interpolation_type.addItem('linear', InterpolationType.LINEAR)
        self.combo_interpolation_type.addItem('nearest', InterpolationType.NEAREST)
        self.spinner_opacity_unit_length = QDoubleSpinBox(self)
        self.spinner_opacity_unit_length.setRange(-4, 4)
        self.spinner_opacity_unit_length.setSingleStep(0.05)
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
        self.spinner_opacity_unit_length.setValue(math.log(settings.opacity_unit_distance))
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.checkbox_shade.setChecked(settings.shade)
        return self

    def get_settings(self):
        return VolumeProperties(
            self.combo_interpolation_type.currentData(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            math.exp(self.spinner_opacity_unit_length.value()),
            self.checkbox_shade.isChecked()
        )


class VolumeVisualSettingsView(QWidget):

    settings_changed = pyqtSignal()
    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_representation_type = QComboBox(self)
        self.combo_representation_type.addItem("DVR", VolumeRepresentation.DVR)
        self.representation_views = {
            VolumeRepresentation.DVR: DVRSettingsView(self)
        }
        self.interface_stack = QStackedLayout()
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentation.DVR])
        self._connect_signals()
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_representation_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self):
        self.combo_representation_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_representation_type.currentIndexChanged.connect(self.representation_changed)
        self.representation_views[VolumeRepresentation.DVR].settings_changed.connect(self.settings_changed)

    def get_representation_mode(self):
        return self.combo_representation_type.currentData()

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

    def apply_settings(self, settings: Dict[VolumeRepresentation, ActorProperties], use_defaults=False):
        for key in VolumeRepresentation:
            if key in settings:
                props = settings[key]
            elif use_defaults:
                props = {VolumeRepresentation.DVR: VolumeProperties()}.get(key, SurfaceProperties())
            else:
                props = None
            if props is not None and key in self.representation_views:
                self.representation_views[key].apply_settings(props)
        return self


class VolumeVisualController(QObject):

    def __init__(self, view: VolumeVisualSettingsView, model: VolumeVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings({self.model.representation_mode: self.model.props}, use_defaults=True)
        self.view.settings_changed.connect(self.on_settings_changed)
        self.view.representation_changed.connect(self.on_representation_changed)

    def on_settings_changed(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)
        return self

    def on_representation_changed(self):
        raise NotImplementedError()

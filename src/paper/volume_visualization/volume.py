import math
import uuid
from collections import namedtuple
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
import pyvista as pv
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout, QStackedWidget, QStackedLayout, \
    QVBoxLayout, QSpinBox, QPushButton, QSlider
from pyvista.plotting import Plotter, Volume

from src.model.geometry import TriangleMesh, LocationBatch, Coordinates, WedgeMesh
from src.model.scene_model import SceneModel
from src.paper.volume_visualization.color_lookup import InteractiveColorLookup
from src.widgets import SelectColorButton


class VolumeRepresentation(Enum):
    MODEL_LEVELS = 'model_levels'
    DVR = 'dvr'
    ISO_CONTOURS = 'iso_levels'


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


ScalingParameters = namedtuple('ScaleParameters', ['scale', 'offset_scale'])
ContourParameters = namedtuple('ContourParameters', ['contour_key', 'num_levels'])


@dataclass
class ActorProperties(object):
    pass


@dataclass
class VolumeProperties(ActorProperties):
    interpolation_type: InterpolationType = InterpolationType.LINEAR
    ambient: float = pv.global_theme.lighting_params.ambient
    diffuse: float = pv.global_theme.lighting_params.diffuse
    specular: float = pv.global_theme.lighting_params.specular
    specular_power: float = pv.global_theme.lighting_params.specular_power
    opacity_unit_distance: float = 1.
    shade: bool = True


@dataclass
class SurfaceProperties(ActorProperties):
    # color: Any = pv.global_theme.color.int_rgb
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
class IsocontourProperties(SurfaceProperties):
    contours: ContourParameters = ContourParameters('z_model_levels', 10)


class VolumeFieldData(object):

    def __init__(self, key: str, field_data: xr.Dataset, terrain_data: xr.Dataset):
        self.key = key
        self.field_data = field_data
        self.terrain_data = terrain_data
        self._baseline_elevation = self.terrain_data['z_surf'].values
        self._relative_elevation = self.field_data['z_model_levels'].values - self._baseline_elevation

    def get_volume_mesh(self, scale_params: ScalingParameters) -> pv.UnstructuredGrid:
        surface_mesh = TriangleMesh(
            LocationBatch(Coordinates.from_xarray(self.terrain_data)),
            self.terrain_data['triangles'].values
        )
        z = self._compute_elevation_coordinate(scale_params)
        mesh = WedgeMesh(surface_mesh, z)
        mesh = mesh.to_wedge_grid()
        mesh[self.key] = self.field_data[self.key].values.ravel()
        return mesh

    def get_level_mesh(self, scale_params: ScalingParameters) -> pv.PolyData:
        coords = Coordinates.from_xarray(self.terrain_data)
        num_nodes = len(coords)
        triangles_base = self.terrain_data['triangles'].values
        num_triangles = len(triangles_base)
        field_data = self.field_data[self.key].values
        num_levels = len(field_data)
        faces = np.zeros((num_triangles * num_levels, 4), dtype=int)
        faces[:, 0] = 3
        for i in range(1, num_levels):
            faces[(i * num_triangles): ((i + 1) * num_triangles), 1:] = triangles_base + i * num_nodes
        coords = np.stack([coords.x, coords.y], axis=-1)
        coords = np.tile(coords, (num_levels, 1))
        z = self._compute_elevation_coordinate(scale_params)
        coords = np.concatenate([coords, np.reshape(z, (-1, 1))], axis=-1)
        mesh = pv.PolyData(coords, faces)
        mesh[self.key] = self.field_data[self.key].values.ravel()
        return mesh

    def _compute_elevation_coordinate(self, scale_params):
        z = self._relative_elevation.copy()
        if scale_params.offset_scale != 1.:
            z *= scale_params.offset_scale
        z += self._baseline_elevation
        z /= scale_params.scale
        return z

    def get_iso_mesh(self, contour_params: ContourParameters, scale_params: ScalingParameters) -> pv.PolyData:
        mesh = self.get_volume_mesh(scale_params)
        contour_key = contour_params.contour_key
        if contour_key is None:
            contour_key = self.key
        if contour_key != self.key:
            mesh[self.key] = self.field_data[self.key].values.ravel()
        mesh[contour_key] = self.field_data[contour_key].values.ravel()
        iso_mesh = mesh.contour(
            scalars=contour_key, isosurfaces=contour_params.num_levels, method='contour', compute_scalars=True,
        )
        iso_mesh.set_active_scalars(self.key)
        return iso_mesh


class PlotterSlot(object):

    def __init__(self, plotter: pv.Plotter, scalar_bar_id: str):
        assert scalar_bar_id not in plotter.scalar_bars.keys()
        self.plotter = plotter
        self.id = str(uuid.uuid4())
        self.scalar_bar_id = scalar_bar_id
        self.actor = None
        self.scalar_bar_actor = None

    def clear(self, render=True):
        if self.actor is not None:
            render_ = render if self.scalar_bar_actor is None else False
            self.plotter.remove_actor(self.actor, render=render_)
        if self.scalar_bar_actor is not None:
            try:
                self.plotter.remove_scalar_bar(title=self.scalar_bar_id, render=render)
            except KeyError:
                pass
        self.scalar_bar_actor = None
        return self

    @staticmethod
    def _actor_props_to_plotter_kws(actor_props: ActorProperties):
        if actor_props is None:
            return {}
        kws = {}
        for field in fields(actor_props):
            field_name = field.name
            if field_name in ['interpolation_type', 'contours']:
                continue
            value = getattr(actor_props, field_name)
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            kws[field_name] = value
        return kws

    def draw_scalar_bar(self, mapper, render: bool = True, interactive=True):
        self.scalar_bar_actor = self.plotter.add_scalar_bar(
            title=self.scalar_bar_id, mapper=mapper, render=render, interactive=interactive,
        )
        return self

    def show_volume_mesh(self, mesh: pv.UnstructuredGrid, lookup_table: pv.LookupTable, properties: VolumeProperties, render: bool = True):
        plotter_kws = self._actor_props_to_plotter_kws(properties)
        clim = lookup_table.scalar_range
        actor = self.plotter.add_volume(
            mesh, cmap=lookup_table, clim=clim, render=False, **plotter_kws,
            show_scalar_bar=False, name=self.id
        )
        self.actor = actor
        self.draw_scalar_bar(actor.mapper, render=render)
        return actor

    def show_surface_mesh(self, mesh: pv.PolyData, lookup_table: pv.LookupTable, properties: SurfaceProperties, render: bool = True):
        plotter_kws = self._actor_props_to_plotter_kws(properties)
        clim = lookup_table.scalar_range
        actor = self.plotter.add_mesh(
            mesh, cmap=lookup_table, clim=clim, render=False, **plotter_kws,
            show_scalar_bar=False, name=self.id
        )
        self.actor = actor
        self.draw_scalar_bar(actor.mapper, render=render)
        return actor

    def update_actor(self, properties: ActorProperties):
        if self.actor is not None:
            actor_props = self.actor.prop
            for field in fields(properties):
                prop_name = field.name
                if prop_name in ['contours']:
                    continue
                value = getattr(properties, prop_name)
                if value is None:
                    continue
                if isinstance(value, Enum):
                    value = value.value
                setattr(actor_props, prop_name, value)
        return True

    def update_scalar_colors(self, lookup_table: pv.LookupTable):
        if self.actor is not None:
            scalar_range = lookup_table.scalar_range
            mapper = self.actor.mapper
            mapper.scalar_range = scalar_range
            mapper.lookup_table = lookup_table
            if isinstance(self.actor, Volume):
                self.actor.prop.apply_lookup_table(lookup_table)
            self.draw_scalar_bar(mapper)
        return self


class VisualRepresentation3d(object):

    def __init__(
            self,
            slot: PlotterSlot, field_data: VolumeFieldData, scaling: ScalingParameters,
            properties: ActorProperties, color_lookup: InteractiveColorLookup
    ):
        self.slot = slot
        self.field_data = field_data
        self.scaling = scaling
        self.properties = properties
        self.color_lookup = color_lookup

    def clear(self, render: bool = True):
        self.slot.clear(render=render)
        return self

    def show(self, render: bool = True):
        raise NotImplementedError()

    def set_properties(self, properties: ActorProperties):
        self.properties = properties
        self.slot.update_actor(properties)
        return self

    def set_scaling(self, scaling: ScalingParameters, render=True):
        self.clear(render=False)
        self.scaling = scaling
        self.show(render=render)
        return self

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class DVRRepresentation(VisualRepresentation3d):

    def __init__(self, slot: PlotterSlot, field_data: VolumeFieldData, color_lookup, properties: VolumeProperties = None, scaling: ScalingParameters = None):
        if properties is None:
            properties = VolumeProperties()
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        super().__init__(slot, field_data, scaling, properties, color_lookup)

    def set_properties(self, properties: VolumeProperties):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        volume_mesh = self.field_data.get_volume_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_volume_mesh(volume_mesh, lookup_table, self.properties, render=render)
        return self


class IsocontourRepresentation(VisualRepresentation3d):

    def __init__(self, slot: PlotterSlot, field_data: VolumeFieldData, color_lookup, properties: VolumeProperties = None, scaling: ScalingParameters = None):
        if properties is None:
            properties = VolumeProperties()
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        super().__init__(slot, field_data, scaling, properties, color_lookup)

    def set_properties(self, properties: IsocontourProperties):
        if properties.contours != self.properties.contours:
            self.clear(render=False)
            self.properties = properties
            self.show()
        else:
            super().set_properties(properties)
        return self

    def show(self, render: bool = True):
        iso_mesh = self.field_data.get_iso_mesh(self.properties.contours, self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_surface_mesh(iso_mesh, lookup_table, self.properties, render=render)
        return self


class ModelLevelRepresentation(VisualRepresentation3d):

    def __init__(self, slot: PlotterSlot, field_data: VolumeFieldData, color_lookup, properties: SurfaceProperties = None, scaling: ScalingParameters = None):
        if properties is None:
            properties = VolumeProperties()
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        super().__init__(slot, field_data, scaling, properties, color_lookup)

    def set_properties(self, properties: VolumeProperties):
        return super().set_properties(properties)

    def show(self, render: bool = True):
        level_mesh = self.field_data.get_level_mesh(self.scaling)
        lookup_table = self.color_lookup.lookup_table
        self.slot.show_surface_mesh(level_mesh, lookup_table, self.properties, render=render)
        return self


class ScalarVolumeVisualization(QObject):
    properties_changed = pyqtSignal(str)

    def __init__(
            self,
            field_data: VolumeFieldData,
            color_lookup: InteractiveColorLookup,
            properties: ActorProperties,
            plotter_slot: PlotterSlot,
            scaling: ScalingParameters = None,
            visible=True,
            parent: QObject = None
    ):
        super().__init__(parent)
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        self.data = field_data
        self.color_lookup = color_lookup
        self.plotter_slot = plotter_slot
        self.properties = properties
        self.scaling = scaling
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
        return self

    def show(self, render: bool = True):
        if self.representation is not None:
            return self
        properties_type = type(self.properties)
        if properties_type == VolumeProperties:
            self.representation = DVRRepresentation(
                self.plotter_slot, self.data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == SurfaceProperties:
            self.representation = ModelLevelRepresentation(
                self.plotter_slot, self.data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == IsocontourProperties:
            self.representation = IsocontourRepresentation(
                self.plotter_slot, self.data, self.color_lookup, self.properties, self.scaling
            )
        else:
            raise NotImplementedError()
        self.representation.show(render=render)
        return self

    def change_representation(self, properties: ActorProperties):
        self.properties = properties
        if self.representation is not None:
            self.clear(render=False)
            self.show()
        return self

    def set_properties(self, properties: ActorProperties):
        self.properties = properties
        if self.representation is not None:
            self.representation.set_properties(properties)
        return self

    def set_scaling(self, scaling: ScalingParameters):
        self.scaling = scaling
        if self.representation is not None:
            self.representation.set_scaling(scaling)
        return self

    def set_visible(self, visible: bool):
        if visible and not self.is_visible():
            self.show()
        if not visible and self.is_visible():
            self.clear()
        return self

    def is_visible(self):
        return self.representation is not None

    @property
    def representation_mode(self):
        return {
            VolumeProperties: VolumeRepresentation.DVR,
            SurfaceProperties: VolumeRepresentation.MODEL_LEVELS,
            IsocontourProperties: VolumeRepresentation.ISO_CONTOURS,
        }.get(type(self.properties))


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
        self.spinner_num_contours.setRange(4, 32)

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


class VolumeVisualSettingsView(QWidget):

    settings_changed = pyqtSignal()
    representation_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_representation_type = QComboBox(self)
        self.combo_representation_type.addItem("DVR", VolumeRepresentation.DVR)
        self.combo_representation_type.addItem("model levels", VolumeRepresentation.MODEL_LEVELS)
        self.combo_representation_type.addItem("isocontours", VolumeRepresentation.ISO_CONTOURS)
        self.representation_views = {
            VolumeRepresentation.DVR: DVRSettingsView(self),
            VolumeRepresentation.MODEL_LEVELS: SurfaceSettingsView(self),
            VolumeRepresentation.ISO_CONTOURS: IsocontourSettingsView(self)
        }
        self.interface_stack = QStackedLayout()
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentation.DVR])
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentation.MODEL_LEVELS])
        self.interface_stack.addWidget(self.representation_views[VolumeRepresentation.ISO_CONTOURS])
        self._connect_signals()
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_representation_type)
        layout.addLayout(self.interface_stack)
        # layout.addWidget(self.representation_views[VolumeRepresentation.DVR])
        # layout.addWidget(self.representation_views[VolumeRepresentation.MODEL_LEVELS])
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self):
        self.combo_representation_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_representation_type.currentIndexChanged.connect(self.representation_changed)
        for key in VolumeRepresentation:
            if key in self.representation_views:
                self.representation_views[key].settings_changed.connect(self.settings_changed)

    def get_representation_mode(self):
        return self.combo_representation_type.currentData()

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

    def apply_settings(self, settings: Dict[VolumeRepresentation, ActorProperties], use_defaults=False):
        defaults = {
            VolumeRepresentation.DVR: VolumeProperties(),
            VolumeRepresentation.MODEL_LEVELS: SurfaceProperties(),
            VolumeRepresentation.ISO_CONTOURS: IsocontourProperties()
        }
        for key in VolumeRepresentation:
            if key in settings:
                props = settings[key]
            elif use_defaults:
                props = defaults.get(key)
            else:
                props = None
            if props is not None and key in self.representation_views:
                self.representation_views[key].apply_settings(props)
        return self


class VolumeVisualController(QObject):

    def __init__(self, view: VolumeVisualSettingsView, model: ScalarVolumeVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings({self.model.representation_mode: self.model.properties}, use_defaults=True)
        self.view.settings_changed.connect(self.on_settings_changed)
        self.view.representation_changed.connect(self.on_representation_changed)

    def on_settings_changed(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)
        return self

    def on_representation_changed(self):
        settings = self.view.get_settings()
        self.model.change_representation(settings)
        return self


class SceneScalingModel(QObject):

    def __init__(self, scaling: ScalingParameters = None, parent=None):
        super().__init__(parent)
        if scaling is None:
            scaling = ScalingParameters(2. ** 12, 1.)
        self.scaling = scaling
        self.visuals = []
        self.plotters = []

    def add_visual(self, visual: ScalarVolumeVisualization):
        plotter = visual.plotter_slot.plotter
        self.visuals.append(visual)
        if plotter not in self.plotters:
            self.plotters.append(plotter)
        plotter.suppress_rendering = True
        visual.set_scaling(self.scaling)
        plotter.update_bounds_axes()
        plotter.suppress_rendering=False
        plotter.render()
        return self

    def suppress_rendering(self, suppress: bool):
        for plotter in self.plotters:
            plotter.suppress_rendering = suppress
        return self

    def render(self):
        for plotter in self.plotters:
            plotter.render()
        return self

    def set_scaling(self, scaling: ScalingParameters):
        self.scaling = scaling
        self.suppress_rendering(True)
        for visual in self.visuals:
            visual.set_scaling(self.scaling)
        self.suppress_rendering(False)
        self.render()


class SceneScalingSettingsView(QWidget):

    scaling_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider_log_vertical_scale = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_vertical_scale.setRange(0, 64)
        self.scale_step = 0.25
        self.slider_log_offset_exaggeration = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_offset_exaggeration.setRange(0, 32)
        self.button_apply = QPushButton(self)
        self.button_apply.setText("Apply")
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_apply.clicked.connect(self.scaling_changed)
        # self.slider_log_vertical_scale.valueChanged.connect(self.scaling_changed)
        # self.slider_log_offset_exaggeration.valueChanged.connect(self.scaling_changed)

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow("Vertical scale:", self.slider_log_vertical_scale)
        layout.addRow("Offset exaggeration:", self.slider_log_offset_exaggeration)
        outer_layout.addLayout(layout)
        outer_layout.addWidget(self.button_apply)
        self.setLayout(outer_layout)

    def get_settings(self):
        return ScalingParameters(
            2. ** (self.scale_step * self.slider_log_vertical_scale.value()),
            2. ** (self.scale_step * self.slider_log_offset_exaggeration.value()),
        )

    def apply_settings(self, scaling: ScalingParameters):
        self.slider_log_vertical_scale.setValue(int(math.log2(scaling.scale) / self.scale_step))
        self.slider_log_offset_exaggeration.setValue(int(math.log2(scaling.offset_scale) / self.scale_step))
        return self


class SceneScalingController(QObject):

    def __init__(self, view: SceneScalingSettingsView, model: SceneScalingModel, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(self.model.scaling)
        self.view.scaling_changed.connect(self.on_scaling_changed)

    def on_scaling_changed(self):
        scaling = self.view.get_settings()
        self.model.set_scaling(scaling)
        return self

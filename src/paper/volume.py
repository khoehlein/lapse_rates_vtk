import uuid
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import xarray as xr
import pyvista as pv
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget

from src.model.geometry import TriangleMesh, LocationBatch, Coordinates, WedgeMesh
from src.paper.color_lookup import InteractiveColorLookup


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
class SurfaceProperties():
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
    interpolation_type: InterpolationType
    ambient: float
    diffuse: float
    specular: float
    specular_power: float
    opacity_unit_distance: float


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
        self.canvas = canvas
        self.vertical_scale = float(vertical_scale)
        self.offset_scale = float(offset_scale) if offset_scale is not None else 1.
        self.offset = float(offset) if offset is not None else 0.

        self._baseline_elevation = self.terrain_data['z_surf'].values
        self._relative_elevation = self.model_data['z_model_levels'].values - self._baseline_elevation

        self.representation_mode = VolumeRepresentation.DVR
        self.props = None
        self.actors = None
        self._draw_representation = {
            VolumeRepresentation.MODEL_LEVELS: self._draw_model_levels,
            VolumeRepresentation.ISO_LEVELS: self._draw_iso_levels,
            VolumeRepresentation.DVR: self._draw_dvr,
        }
        self.visible = True
        self.draw()

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
        actor_props = self.actors['mesh'].props
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
            self._remove_scalar_bar()
            self._draw_scalar_bar()

    def _update_color_in_volume_actor(self):
        self.actors['mesh'].apply_lookup_table(self.color_lookup.lookup_table)

    def _update_color_in_surface_actor(self):
        mapper = self.actors['mesh'].mapper
        mapper.lookup_table = self.color_lookup.lookup_table

    def _remove_scalar_bar(self):
        self.canvas.remove_actor(self.actors['scalar_bar'])
        del self.actors['scalar_bar']

    def _draw_scalar_bar(self):
        actor = self.canvas.add_scalar_bar(
            title=self.scalar_bar_label,
            mapper=self.actors['mesh'].mapper,
            interactive=True
        )
        self.actors['scalar_bar'] = actor

    def set_representation_mode(self, mode: VolumeRepresentation):
        self.representation_mode = mode
        self.draw()
        return self

    def draw(self):
        actor = self._draw_representation[self.representation_mode]()
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

    def _draw_dvr(self) -> pv.volume.Volume:
        mesh = self._build_volume_mesh()
        drawing_kws = self._get_drawing_kws()
        actor = self.canvas.add_volume(
            mesh, cmap=self.color_lookup.lookup_table, name=self.uid,
            blending='composite', mapper='smart', show_scalar_bar=False,
            **drawing_kws
        )
        return actor

    def _get_drawing_kws(self):
        if self.props is None:
            return {}
        kws = {}
        for field in fields(self.props):
            field_name = field.name
            value = getattr(self.props, field_name)
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            kws[field_name] = value
        return kws

    def _draw_iso_levels(self):
        raise NotImplementedError()

    def _draw_model_levels(self):
        raise NotImplementedError()

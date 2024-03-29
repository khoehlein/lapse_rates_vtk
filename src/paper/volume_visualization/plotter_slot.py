import uuid
from collections import namedtuple
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Union
import pyvista as pv

from pyvista.plotting import Volume


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
class MeshProperties(ActorProperties):
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
class SurfaceReferenceProperties(MeshProperties):
    color: Any = pv.global_theme.color.int_rgb


@dataclass
class IsocontourProperties(MeshProperties):
    contours: ContourParameters = ContourParameters('z_model_levels', 10)


@dataclass
class StationSiteProperties(MeshProperties):

    def __init__(
            self,
            metallic: float = pv.global_theme.lighting_params.metallic,
            roughness: float = pv.global_theme.lighting_params.roughness,
            point_size: float = pv.global_theme.point_size,
            render_points_as_spheres: bool = pv.global_theme.render_points_as_spheres,
            opacity: float = pv.global_theme.opacity,
            ambient: float = pv.global_theme.lighting_params.ambient,
            diffuse: float = pv.global_theme.lighting_params.diffuse,
            specular: float = pv.global_theme.lighting_params.specular,
            specular_power: float = pv.global_theme.lighting_params.specular_power,
            lighting: bool = pv.global_theme.lighting,
    ):
        super().__init__(
            style=SurfaceStyle.POINTS,
            metallic=metallic,
            roughness=roughness,
            point_size=point_size,
            render_points_as_spheres=render_points_as_spheres,
            opacity=opacity,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            lighting=lighting,
        )


@dataclass(init=False)
class StationSiteReferenceProperties(StationSiteProperties):
    color: Any = pv.global_theme.color.int_rgb

    def __init__(
            self,
            color: Any = pv.global_theme.color.int_rgb,
            metallic: float = pv.global_theme.lighting_params.metallic,
            roughness: float = pv.global_theme.lighting_params.roughness,
            point_size: float = pv.global_theme.point_size,
            render_points_as_spheres: bool = pv.global_theme.render_points_as_spheres,
            opacity: float = pv.global_theme.opacity,
            ambient: float = pv.global_theme.lighting_params.ambient,
            diffuse: float = pv.global_theme.lighting_params.diffuse,
            specular: float = pv.global_theme.lighting_params.specular,
            specular_power: float = pv.global_theme.lighting_params.specular_power,
            lighting: bool = pv.global_theme.lighting,
    ):
        super().__init__(metallic, roughness, point_size, render_points_as_spheres, opacity, ambient, diffuse, specular, specular_power, lighting)
        self.color = color


@dataclass(init=False)
class StationOnTerrainProperties(MeshProperties):
    preference: str = 'cell'

    def __init__(
            self,
            line_width: float = pv.global_theme.line_width,
            render_lines_as_tubes: bool = pv.global_theme.render_lines_as_tubes,
            metallic: float = pv.global_theme.lighting_params.metallic,
            roughness: float = pv.global_theme.lighting_params.roughness,
            opacity: float = pv.global_theme.opacity,
            ambient: float = pv.global_theme.lighting_params.ambient,
            diffuse: float = pv.global_theme.lighting_params.diffuse,
            specular: float = pv.global_theme.lighting_params.specular,
            specular_power: float = pv.global_theme.lighting_params.specular_power,
            lighting: bool = pv.global_theme.lighting,
    ):
        super().__init__(
            style=SurfaceStyle.WIREFRAME,
            line_width=line_width,
            render_lines_as_tubes=render_lines_as_tubes,
            metallic=metallic,
            roughness=roughness,
            opacity=opacity,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            show_edges=True,
            lighting=lighting,
        )


@dataclass(init=False)
class StationOnTerrainReferenceProperties(StationOnTerrainProperties):
    color: Any = pv.global_theme.color.int_rgb

    def __init__(
            self,
            color: Any = pv.global_theme.color.int_rgb,
            line_width: float = pv.global_theme.line_width,
            render_lines_as_tubes: bool = pv.global_theme.render_lines_as_tubes,
            metallic: float = pv.global_theme.lighting_params.metallic,
            roughness: float = pv.global_theme.lighting_params.roughness,
            opacity: float = pv.global_theme.opacity,
            ambient: float = pv.global_theme.lighting_params.ambient,
            diffuse: float = pv.global_theme.lighting_params.diffuse,
            specular: float = pv.global_theme.lighting_params.specular,
            specular_power: float = pv.global_theme.lighting_params.specular_power,
            lighting: bool = pv.global_theme.lighting,
    ):
        super().__init__(line_width, render_lines_as_tubes, metallic, roughness, opacity, ambient, diffuse, specular, specular_power, lighting)
        self.color = color


class PlotterSlot(object):

    def __init__(self, plotter: pv.Plotter, scalar_bar_id: str = None):
        # assert scalar_bar_id not in plotter.scalar_bars.keys() or scalar_bar_id is None
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
                self.plotter.remove_actor(self.scalar_bar_actor)
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

    def draw_scalar_bar(self, mapper, render: bool = True, interactive=False):
        if self.scalar_bar_id in self.plotter.scalar_bars.keys():
            self.plotter.remove_scalar_bar(self.scalar_bar_id)
        self.scalar_bar_actor = self.plotter.add_scalar_bar(
            title=self.scalar_bar_id, mapper=mapper, render=render, interactive=interactive, height=0.08,
            position_x=0.02, title_font_size=28, label_font_size=28
        )
        return self

    def show_scalar_volume(self, mesh: pv.UnstructuredGrid, lookup_table: pv.LookupTable, properties: VolumeProperties, render: bool = True):
        plotter_kws = self._actor_props_to_plotter_kws(properties)
        clim = lookup_table.scalar_range
        actor = self.plotter.add_volume(
            mesh, cmap=lookup_table, clim=clim, render=False, **plotter_kws,
            show_scalar_bar=False, name=self.id
        )
        self.actor = actor
        self.draw_scalar_bar(actor.mapper, render=render)
        return actor

    def show_scalar_mesh(
            self, mesh: Union[pv.PolyData, pv.UnstructuredGrid], lookup_table: pv.LookupTable,
            properties: Union[
                MeshProperties,
                StationSiteProperties,
                StationOnTerrainProperties
            ],
            render: bool = True
    ):
        plotter_kws = self._actor_props_to_plotter_kws(properties)
        clim = lookup_table.scalar_range
        actor = self.plotter.add_mesh(
            mesh, cmap=lookup_table, clim=clim, render=False, **plotter_kws,
            show_scalar_bar=False, name=self.id
        )
        self.actor = actor
        self.draw_scalar_bar(actor.mapper, render=render)
        return actor

    def show_reference_mesh(
            self, mesh: pv.PolyData,
            properties: Union[
                SurfaceReferenceProperties,
                StationSiteReferenceProperties,
                StationOnTerrainReferenceProperties
            ],
            render: bool = True
    ):
        plotter_kws = self._actor_props_to_plotter_kws(properties)
        mesh.set_active_scalars(None)
        actor = self.plotter.add_mesh(
            mesh, render=render, name=self.id, **plotter_kws
        )
        self.actor = actor
        return actor

    def update_actor(self, properties: ActorProperties, render: bool = True):
        if self.actor is not None:
            actor_props = self.actor.prop
            for field in fields(properties):
                prop_name = field.name
                if prop_name in ['contours', 'preference']:
                    continue
                value = getattr(properties, prop_name)
                if value is None:
                    continue
                if isinstance(value, Enum):
                    value = value.value
                setattr(actor_props, prop_name, value)
            if render:
                self.plotter.render()
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

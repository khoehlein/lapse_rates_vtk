from dataclasses import dataclass
from enum import Enum
from typing import Dict
import pyvista as pv
from PyQt5.QtCore import QObject

from src.model.visualization.interface import PropertyModel, standard_adapter


class ShadingType(Enum):
    PBR = 'pbr'
    PHONG = 'Phong'
    GOURAUD = 'Gouraud'
    FLAT = 'Flat'


@dataclass
class LightingProperties:
    shading: ShadingType = None  # Pyvista property: interpolation
    metallic: float = None
    roughness: float = None
    ambient: float = None
    diffuse: float = None
    specular: float = None
    specular_power: float = None


class MeshStyle(Enum):
    WIREFRAME = 'wireframe'
    SURFACE = 'surface'
    POINTS = 'points'


@dataclass
class MeshProperties(object):
    pass


@dataclass
class WireframeProperties(MeshProperties):
    line_width: float = None
    render_lines_as_tubes: bool = None


@dataclass
class TranslucentSurfaceProperties(MeshProperties):
    show_edges: bool = None
    edge_color: str = None
    edge_opacity: float = None


@dataclass
class PointsSurfaceProperties(MeshProperties):
    point_size: float = None
    render_points_as_spheres: bool = None


_mesh_style_mapping = {
    WireframeProperties: MeshStyle.WIREFRAME,
    PointsSurfaceProperties: MeshStyle.POINTS,
    TranslucentSurfaceProperties: MeshStyle.SURFACE
}


class MeshGeometryModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        lighting: LightingProperties
        mesh: MeshProperties

    def __init__(
            self,
            dataset: pv.PolyData,
            properties: 'MeshGeometryModel.Properties' = None,
            parent: QObject = None,
            scalar_preference='point'
    ):
        super().__init__(parent)
        if properties is not None:
            self.set_properties(properties)
        self._mesh = dataset
        self._vertical_coordinates = self._mesh.points[:, -1].copy()
        self._vertical_scale = 1.
        self._scalar_preference = scalar_preference

    @property
    def mesh_style(self):
        return _mesh_style_mapping[type(self._properties.mesh)]

    def set_properties(self, properties: 'MeshGeometryModel.Properties') -> 'MeshGeometryModel':
        return super().set_properties(properties)

    def set_vertical_scale(self, scale: float):
        self._vertical_scale = float(scale)
        self._mesh.points[:, -1] = self._vertical_coordinates / self._vertical_scale
        return self

    def write_to_host(self, host: pv.Plotter, **kwargs) -> Dict[str, pv.Actor]:
        lighting_kws = standard_adapter.read(self._properties.lighting)
        mesh_properties = self._properties.mesh
        mesh_kws = standard_adapter.read(mesh_properties)
        actor = host.add_mesh(
            self._mesh,
            style=self.mesh_style.name.lower(),
            preference=self._scalar_preference,
            **mesh_kws, **lighting_kws, **kwargs
        )
        return {'mesh': actor}

    def update_actors(self, actors: Dict[str, pv.Actor]) -> Dict[str, pv.Actor]:
        actor_props = actors['mesh'].prop
        style = self.mesh_style.name.lower()
        actor_props.style = style
        lighting_kws = standard_adapter.read(self._properties.lighting)
        for key, value in lighting_kws.items():
            setattr(actor_props, key, value)
        mesh_kws = standard_adapter.read(self._properties.mesh)
        for key, value in mesh_kws.items():
            setattr(actor_props, key, value)
        return actors

    def supports_update(self, properties: 'MeshGeometryModel.Properties'):
        return True

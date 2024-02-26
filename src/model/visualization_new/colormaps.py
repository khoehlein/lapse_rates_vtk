import dataclasses
from dataclasses import dataclass
from enum import Enum

import pyvista as pv

from src.model.interface import PropertyModel
from src.model.visualization.mesh_geometry import MeshStyle
from src.model.visualization_new.utils import KeywordAdapter


DEFAULT_ADAPTER = KeywordAdapter(
    {
        'shading': 'interpolation',
        'isolevels': 'isosurfaces',
        'contour_scalar': 'scalars',
        'contour_method': 'method'
    },
    {
        'style': lambda x: x.name.lower(),
        'shading': lambda x: x.value,
        'color': lambda x: x.name(),
        'edge_color': lambda x: x.name(),
        'contour_method': lambda x: x.name.lower(),
        'contour_scalar': lambda x: x.name.lower()
    }
)


class ShadingType(Enum):
    PBR = 'pbr'
    PHONG = 'Phong'
    GOURAUD = 'Gouraud'
    FLAT = 'Flat'


class _ActorPropertyModel(PropertyModel):

    def update_actor(self, actor: pv.Actor, adapter: KeywordAdapter = None):
        actor_props = actor.props
        for key, value in self.get_actor_properties(adapter):
            setattr(actor_props, key, value)
        return self

    def get_actor_properties(self, adapter: KeywordAdapter = None):
        if self.properties is None:
            return {}
        if adapter is None:
            adapter = DEFAULT_ADAPTER
        return adapter.read(self.properties)


class LightingModel(_ActorPropertyModel):

    @dataclass
    class Properties(_ActorPropertyModel.Properties):
        shading: ShadingType
        metallic: float
        roughness: float
        ambient: float
        diffuse: float
        specular: float
        specular_power: float


class MeshStyleModel(_ActorPropertyModel):

    def __init__(self, style: MeshStyle):
        super().__init__()
        self.style = style

    @dataclass
    class Properties(_ActorPropertyModel.Properties):
        pass


class MeshSurfaceModel(MeshStyleModel):

    @dataclass
    class Properties(MeshStyleModel.Properties):
        show_edges: bool = None
        edge_color: str = None
        edge_opacity: float = None


class MeshWireframeModel(MeshStyleModel):
    @dataclass
    class Properties(MeshStyleModel.Properties):
        line_width: float = None
        render_lines_as_tubes: bool = None


class MeshPointsModel(MeshStyleModel):

    @dataclass
    class Properties(MeshStyleModel.Properties):
        point_size: float = None
        render_points_as_spheres: bool = None


class ColorModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        pass


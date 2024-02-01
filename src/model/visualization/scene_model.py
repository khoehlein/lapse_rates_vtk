import copy
import dataclasses
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from PyQt5.QtCore import QObject
import pyvista as pv

from src.model.geometry import SurfaceDataset
from src.model.visualization.transforms import AffineLinear


class VisualizationUpdateModel(object):
    pass


class VisualizationModel(QObject):

    def __init__(self, visual_key: str = None, parent=None):
        super().__init__(parent)
        if visual_key is None:
            visual_key = str(uuid.uuid4())
        self.key = str(visual_key)
        self._host = None
        self._host_actors = {}
        self._properties = None
        self._is_visible = True

    def set_host(self, host: pv.Plotter) -> 'VisualizationModel':
        self.clear_host()
        if host is not None:
            self._host = host
            self._write_to_host()
        return self

    def clear_host(self):
        host_reference = self._host
        if self._host is not None:
            for actor in self._host_actors.values():
                self._host.remove_actor(actor)
        self._host = None
        self._host_actors = {}
        return host_reference

    def _write_to_host(self):
        raise NotImplementedError()

    def set_properties(self, properties) -> 'VisualizationModel':
        self._properties = properties
        if self._host is not None:
            self._update_actor_properties()
        return self

    def set_visibility(self, visible: bool) -> 'VisualizationModel':
        self._is_visible = visible
        for actor in self._host_actors.values():
            actor.visibility = self._is_visible
        return self

    def _update_actor_properties(self):
        raise NotImplementedError()

    def update(self, update: VisualizationUpdateModel):
        raise NotImplementedError()


class ShadingMethod(Enum):
    PBR = 'pbr'
    PHONG = 'Phong'
    GOURAUD = 'Gouraud'
    FLAT = 'Flat'


class KeywordAdapter(object):

    def __init__(self, aliases: Dict[str, str], transforms: Dict[str, Any]):
        self._aliases = aliases
        self._transforms = transforms

    def _field_value(self, name: str, source):
        value = getattr(source, name)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def _field_name(self, name: str):
        return self._aliases.get(name, name)

    def read(self, properties):
        return {
            self._field_name(field.name): self._field_value(field.name, properties)
            for field in dataclasses.fields(properties)
        }


_keyword_adapter = KeywordAdapter(
    {
        'shading': 'interpolation'
    },
    {
        'shading': lambda x: x.value,
        'color': lambda x: x.name(),
        'edge_color': lambda x: x.name(),
    }
)


class GeometryStyle(Enum):
    WIREFRAME = 'wireframe'
    SURFACE = 'surface'
    POINTS = 'points'


class MeshVisualizationModel(VisualizationModel):

    class ActorKey(Enum):
        MESH = 'mesh'

    @dataclass(init=True)
    class Properties:
        shading: ShadingMethod = None # Pyvista property: interpolation
        metallic: float = None
        roughness: float = None
        ambient: float = None
        diffuse: float = None
        specular: float = None
        specular_power: float = None
        lighting: bool = None

    def __init__(self, dataset: SurfaceDataset, style: GeometryStyle, visual_key: str = None, parent=None):
        super().__init__(visual_key, parent)
        self._dataset = dataset
        self._mesh = dataset.get_polydata()
        self._vertical_transform = AffineLinear.identity()
        self._geometry_style = style

    def set_vertical_scale(self, scale: float):
        self._vertical_transform.scale = float(scale)
        self._mesh.points[:, -1] = self._vertical_transform.apply(self._dataset.z)
        return self

    def _write_to_host(self):
        property_kws = _keyword_adapter.read(self._properties)
        color_kws = self._get_color_kws()
        plotter_kws = self._get_plotter_kws()
        actor = self._host.add_mesh(self._mesh, **property_kws, **plotter_kws, **color_kws)
        self._host_actors[self.ActorKey.MESH] = actor

    def _get_plotter_kws(self) -> Dict[str, Any]:
        return {'name': self.key, 'reset_camera': True, 'style': self._geometry_style.value}

    def _update_actor_properties(self):
        actor = self._host_actors[self.ActorKey.MESH]
        actor_props = actor.prop
        new_actor_props = _keyword_adapter.read(self._properties)
        for key, value in new_actor_props.items():
            setattr(actor_props, key, value)

    def set_color(self, properties) -> 'MeshVisualizationModel':
        raise NotImplementedError()

    def _get_color_kws(self) -> Dict[str, Any]:
        raise NotImplementedError()


@dataclass
class UniformColorMixin():
    color: str = None
    opacity: float = None


class GeometryVisualizationModel(MeshVisualizationModel):

    @dataclass
    class ColorProperties(UniformColorMixin):
        pass

    def __init__(self, dataset: SurfaceDataset, style: GeometryStyle, visual_key: str = None, parent=None):
        super().__init__(dataset, style, visual_key, parent)
        self._color_properties = None

    def set_color(self, properties: 'GeometryVisualizationModel.ColorProperties') -> 'GeometryVisualizationModel':
        self._color_properties = properties
        if self._host is not None:
            self._update_actor_color()
        return self

    def _update_actor_color(self):
        actor = self._host_actors[self.ActorKey.MESH]
        actor_props = actor.prop
        new_actor_props = _keyword_adapter.read(self._color_properties)
        for key, value in new_actor_props.items():
            setattr(actor_props, key, value)

    def _get_color_kws(self) -> Dict[str, Any]:
        if self._color_properties is None:
            return {}
        kws = _keyword_adapter.read(self._color_properties)
        return kws


@dataclass
class WireframeMixin(object):
    line_width: float = None
    render_lines_as_tubes: bool = None


@dataclass
class TranslucentSurfaceMixin(object):
    show_edges: bool = None
    edge_color: str = None
    edge_opacity: float = None


@dataclass
class PointsSurfaceMixin(object):
    point_size: float = None
    render_points_as_spheres: bool = None


class WireframeGeometry(GeometryVisualizationModel):

    @dataclass
    class Properties(MeshVisualizationModel.Properties, UniformColorMixin, WireframeMixin):
        pass

    def __init__(self, dataset: SurfaceDataset, visual_key: str = None, parent=None):
        super().__init__(dataset, GeometryStyle.WIREFRAME, visual_key, parent)

    def set_properties(self, properties: 'WireframeGeometry.Properties') -> 'VisualizationModel':
        return super().set_properties(properties)


class SurfaceGeometry(GeometryVisualizationModel):

    @dataclass
    class Properties(MeshVisualizationModel.Properties, UniformColorMixin, TranslucentSurfaceMixin):
        pass

    def __init__(self, dataset: SurfaceDataset, visual_key: str = None, parent=None):
        super().__init__(dataset, GeometryStyle.SURFACE, visual_key, parent)

    def set_properties(self, properties: 'SurfaceGeometry.Properties') -> 'VisualizationModel':
        return super().set_properties(properties)


class PointsGeometry(GeometryVisualizationModel):

    @dataclass
    class Properties(MeshVisualizationModel.Properties, UniformColorMixin, PointsSurfaceMixin):
        pass

    def __init__(self, dataset: SurfaceDataset, visual_key: str = None, parent=None):
        super().__init__(dataset, GeometryStyle.POINTS, visual_key, parent)

    def set_properties(self, properties: 'PointsGeometry.Properties') -> 'VisualizationModel':
        return super().set_properties(properties)


class ScalarFieldModel(MeshVisualizationModel):

    def __init__(self, dataset: SurfaceDataset, style: GeometryStyle, visual_key: str = None, parent=None):
        super().__init__(dataset, style, visual_key, parent)
        self._colormap = None
        self._lookup_reference = None



class SceneModel(QObject):

    def __init__(self, host: pv.Plotter, parent=None):
        super().__init__(parent)
        self.host = host
        self.visuals: Dict[str, VisualizationModel] = {}

    def add_visualization(self, visualization: VisualizationModel) -> VisualizationModel:
        return self._add_visualization(visualization)

    def add_or_replace_visualization(self, visualization: VisualizationModel):
        key = visualization.key
        if key in self.visuals:
            self.remove_visualization(key)
        self._add_visualization(visualization)
        return visualization

    def replace_visualization(self, visualization: VisualizationModel):
        key = visualization.key
        assert key in self.visuals
        self.remove_visualization(key)
        self._add_visualization(visualization)
        return visualization

    def _add_visualization(self, visualization: VisualizationModel) -> 'SceneModel':
        self.visuals[visualization.key] = visualization
        visualization.set_host(self.host)
        return visualization

    def remove_visualization(self, key: str):
        if key in self.visuals:
            visualization = self.visuals[key]
            visualization.clear_host()
            del self.visuals[key]

    def reset(self):
        self.visuals.clear()
        self.host.clear_actors()
        self.host.scalar_bars.clear()


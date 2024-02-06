import copy
import dataclasses
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple, Union

from PyQt5.QtCore import QObject
import pyvista as pv

from src.model.geometry import SurfaceDataset


class ShadingMethod(Enum):
    PBR = 'pbr'
    PHONG = 'Phong'
    GOURAUD = 'Gouraud'
    FLAT = 'Flat'


@dataclass
class LightingProperties:
    shading: ShadingMethod = None  # Pyvista property: interpolation
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
        'style': lambda x: x.name.lower(),
        'shading': lambda x: x.value,
        'color': lambda x: x.name(),
        'edge_color': lambda x: x.name(),
    }
)


class VisualizationType(Enum):
    GEOMETRY = 'Geometry'
    LAPSE_RATE = 'Lapse rate'
    T2M_O1280 = 'T2M (O1280)'
    T2M_O8000 = 'T2M (O8000)'
    T2M_DIFFERENCE = 'T2M (difference)'
    Z_O1280 = 'Z (O1280)'
    Z_O8000 = 'Z (O8000)'
    Z_DIFFERENCE = 'Z (difference)'


class PropertyModel(QObject):

    class Properties(object):
        pass

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._properties = None

    def set_properties(self, properties) -> 'PropertyModel':
        self._properties = properties
        return self

    def supports_update(self, properties):
        raise NotImplementedError()

    def get_kws(self):
        raise NotImplementedError()


class ColorModel(PropertyModel):

    class Properties(object):
        pass

    @property
    def scalar_bar_title(self) -> Union[str, None]:
        raise NotImplementedError()

    def update_actors(self, actors):
        raise NotImplementedError()

    @staticmethod
    def from_properties(properties):
        if isinstance(properties, UniformColorModel.Properties):
            model = UniformColorModel()
        elif isinstance(properties, ScalarColormapModel.Properties):
            model = ScalarColormapModel()
        else:
            raise NotImplementedError()
        model.set_properties(properties)
        return model

    def get_kws(self):
        raise NotImplementedError()


class UniformColorModel(ColorModel):

    @dataclass
    class Properties(ColorModel.Properties):
        color: str
        opacity: float

    @property
    def scalar_bar_title(self) -> Union[str, None]:
        return None

    def update_actors(self, actors: Dict[str, pv.Actor]) -> pv.Actor:
        actor = actors['mesh']
        actor_props = actor.prop
        new_actor_props = _keyword_adapter.read(self._properties)
        for key, value in new_actor_props.items():
            setattr(actor_props, key, value)
        return actor

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, UniformColorModel.Properties)

    def get_kws(self):
        kws = _keyword_adapter.read(self._properties)
        kws['scalars'] = None
        return kws


class ScalarColormapModel(ColorModel):

    @dataclass
    class Properties(ColorModel.Properties):
        scalar_name: str
        colormap_name: str
        opacity: float
        scalar_range: Tuple[float, float] = None
        below_range_color: str = None
        above_range_color: str = None

    @property
    def scalar_bar_title(self) -> Union[str, None]:
        if self._properties is None:
            return None
        scalar_type = getattr(VisualizationType, self._properties.scalar_name.upper())
        return scalar_type.value

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, ScalarColormapModel.Properties)

    def update_actors(self, actors: Dict[str, pv.Actor]) -> pv.Actor:
        actor = actors['mesh']
        actor_props = actor.prop
        actor_props.opacity = self._properties.opacity
        actor.mapper.array_name = self._properties.scalar_name

        scalar_range = self._properties.scalar_range
        if scalar_range is None:
            mesh = actor.mapper.mesh
            scalar_range = mesh.get_data_range(self._properties.scalar_name)
        actor.mapper.scalar_range = scalar_range

        actor.mapper.lookup_table.cmap = self._properties.colormap_name
        below_range_color = self._properties.below_range_color
        if below_range_color is not None:
            actor.mapper.lookup_table.below_range_color = below_range_color
        above_range_color = self._properties.above_range_color
        if above_range_color is not None:
            actor.mapper.lookup_table.above_range_color = above_range_color
        return actor

    def get_kws(self):
        props = self._properties
        lut = pv.LookupTable(cmap=props.colormap_name)
        lut.scalar_range = props.scalar_range
        return {'scalars': props.scalar_name, 'cmap': lut, 'opacity': props.opacity, 'show_scalar_bar': False}


class MeshGeometryModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        lighting: LightingProperties
        mesh: MeshProperties

    def __init__(
            self,
            dataset: SurfaceDataset,
            properties: 'MeshGeometryModel.Properties' = None,
            parent: QObject = None
    ):
        super().__init__(parent)
        if properties is not None:
            self.set_properties(properties)
        self._dataset = dataset
        self._mesh = dataset.get_polydata()
        self._mesh.points[:, -1] /= 4000.
        self._scale = None

    @property
    def mesh_style(self):
        return _mesh_style_mapping[type(self._properties.mesh)]

    def set_properties(self, properties: 'MeshGeometryModel.Properties') -> 'MeshGeometryModel':
        return super().set_properties(properties)

    def set_vertical_scale(self, scale: float):
        self._scale = float(scale)
        return self

    def write_to_host(self, host: pv.Plotter, **kwargs) -> Dict[str, pv.Actor]:
        lighting_kws = _keyword_adapter.read(self._properties.lighting)
        mesh_properties = self._properties.mesh
        mesh_kws = _keyword_adapter.read(mesh_properties)
        actor = host.add_mesh(self._mesh, style=self.mesh_style.name.lower(), **mesh_kws, **lighting_kws, **kwargs)
        # actor.scale = (1., 1., 1 / self._scale)
        return {'mesh': actor}

    def update_actors(self, actors: Dict[str, pv.Actor]) -> Dict[str, pv.Actor]:
        actor_props = actors['mesh'].prop
        style = self.mesh_style.name.lower()
        actor_props.style = style
        lighting_kws = _keyword_adapter.read(self._properties.lighting)
        for key, value in lighting_kws.items():
            setattr(actor_props, key, value)
        mesh_kws = _keyword_adapter.read(self._properties.mesh)
        for key, value in mesh_kws.items():
            setattr(actor_props, key, value)
        # actors['mesh'].scale = (1, 1, 1 / self._scale)
        return actors

    def supports_update(self, properties: 'MeshGeometryModel.Properties'):
        return True


class VisualizationModel(QObject):

    def __init__(self, geometry: MeshGeometryModel, color: ColorModel, visual_key: str = None, parent=None):
        super().__init__(parent)
        if visual_key is None:
            visual_key = str(uuid.uuid4())
        self.key = str(visual_key)
        self.geometry = geometry
        self.color = color
        self._host = None
        self._host_actors = {}
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

    def update_geometry(self, properties: MeshGeometryModel.Properties) -> None:
        if self.geometry.supports_update(properties):
            self.geometry.set_properties(properties)
            if self._host_actors:
                self.geometry.update_actors(self._host_actors)
            return self
        raise NotImplementedError()

    def update_color(self, properties: ColorModel.Properties) -> 'VisualizationModel':
        scalar_bar_old = self.color.scalar_bar_title
        if self.color.supports_update(properties):
            self.color.set_properties(properties)
            scalar_bar_new = self.color.scalar_bar_title
            if self._host_actors:
                self.color.update_actors(self._host_actors)
            if scalar_bar_old is not None:
                if scalar_bar_new is None or scalar_bar_old != scalar_bar_new:
                    self._host.remove_actor(self._host_actors['scalar_bar'])
                    self._host.remove_scalar_bar(scalar_bar_old)
            if scalar_bar_new is not None:
                if not (scalar_bar_old is not None and scalar_bar_new == scalar_bar_old):
                    actor = self._host.add_scalar_bar(mapper=self._host_actors['mesh'].mapper, title=scalar_bar_new)
                    self._host_actors['scalar_bar'] = actor
        else:
            self.color = ColorModel.from_properties(properties)
            self._write_to_host()
        return self

    def _write_to_host(self):
        color_kws = self.color.get_kws()
        actors = self.geometry.write_to_host(self._host, **color_kws, name=self.key)
        self._host_actors.update(actors)
        scalar_title = self.color.scalar_bar_title
        if scalar_title is not None:
            actor = self._host.add_scalar_bar(mapper=self._host_actors['mesh'].mapper, title=scalar_title)
            self._host_actors['scalar_bar'] = actor

    def set_visibility(self, visible: bool) -> 'VisualizationModel':
        self._is_visible = visible
        for actor in self._host_actors.values():
            actor.visibility = self._is_visible
        return self

    def set_vertical_scale(self, scale: float) -> 'VisualizationModel':
        self.geometry.set_vertical_scale(scale)
        if self._host_actors:
            self.geometry.update_actors(self._host_actors)
        return self


class SceneModel(QObject):

    def __init__(self, host: pv.Plotter, parent=None):
        super().__init__(parent)
        self.host = host
        self.visuals: Dict[str, VisualizationModel] = {}
        self._scale = 1.

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
        self.host.suppress_render = True
        visualization.set_vertical_scale(self._scale)
        visualization.set_host(self.host)
        self.host.update_bounds_axes()
        self.host.suppress_render = False
        self.host.render()
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

    def set_vertical_scale(self, scale):
        self._scale = scale
        for visualization in self.visuals.values():
            visualization.set_vertical_scale(self._scale)
        self.host.render()
        return self


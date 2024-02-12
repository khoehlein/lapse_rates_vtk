import dataclasses
import uuid
from enum import Enum
from typing import Dict, Any

import pyvista as pv

from PyQt5.QtCore import QObject

from src.model.geometry import SurfaceDataset


class VisualizationType(Enum):
    SURFACE_SCALAR_FIELD = 'surface_scalar_field'
    SURFACE_ISOCONTOURS = 'surface_isocontours'
    PROJECTION_LINES = 'projection_lines'


class DataConfiguration(Enum):
    SURFACE_O1280 = 'Surface (O1280)'
    SURFACE_O8000 = 'Surface (O8000)'


class ScalarType(Enum):
    GEOMETRY = 'Geometry'
    LAPSE_RATE = 'Lapse rate'
    T2M_O1280 = 'T2M (O1280)'
    T2M_O8000 = 'T2M (O8000)'
    T2M_DIFFERENCE = 'T2M (difference)'
    Z_O1280 = 'Z (O1280)'
    Z_O8000 = 'Z (O8000)'
    Z_DIFFERENCE = 'Z (difference)'


available_scalars = {
    DataConfiguration.SURFACE_O1280: [
        ScalarType.GEOMETRY,
        ScalarType.LAPSE_RATE,
        ScalarType.T2M_O1280,
        ScalarType.Z_O1280
    ],
    DataConfiguration.SURFACE_O8000: [
        ScalarType.GEOMETRY,
        ScalarType.LAPSE_RATE,
        ScalarType.T2M_O1280,
        ScalarType.T2M_O8000,
        ScalarType.Z_O1280,
        ScalarType.Z_O8000,
        ScalarType.Z_DIFFERENCE,
    ]
}


class PropertyModelUpdateError(Exception):
    pass


class PropertyModel(QObject):

    class Properties(object):
        pass

    def __init__(self, properties: 'PropertyModel.Properties' = None, parent: QObject = None):
        super().__init__(parent)
        self.set_properties(properties)

    def set_properties(self, properties) -> 'PropertyModel':
        if not self.new_properties_valid(properties):
            raise PropertyModelUpdateError()
        self._properties = properties
        return self

    def new_properties_valid(self, properties: 'PropertyModel.Properties') -> bool:
        return True

    def supports_update(self, properties):
        raise NotImplementedError()

    def get_kws(self):
        raise NotImplementedError()


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


standard_adapter = KeywordAdapter(
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


class VisualizationModel(QObject):

    def __init__(self, visual_type: VisualizationType, visual_key: str = None, parent=None):
        super().__init__(parent)
        self.visual_type: VisualizationType = visual_type
        if visual_key is None:
            visual_key = str(uuid.uuid4())
        self.key = str(visual_key)
        self.host = None
        self.host_actors = {}
        self._is_visible = True
        self.gui_label = None

    def set_host(self, host: pv.Plotter) -> 'VisualizationModel':
        self.clear_host()
        if host is not None:
            self.host = host
            self._write_to_host()
        return self

    def reset(self):
        return self.set_host(self.host)

    def clear_host(self):
        host_reference = self.host
        if self.host is not None:
            for actor in self.host_actors.values():
                self.host.remove_actor(actor)
        self.host = None
        self.host_actors = {}
        return host_reference

    def _write_to_host(self):
        raise NotImplementedError()

    def set_visibility(self, visible: bool) -> 'VisualizationModel':
        self._is_visible = visible
        for actor in self.host_actors.values():
            actor.visibility = self._is_visible
        return self

    def set_vertical_scale(self, scale: float) -> 'VisualizationModel':
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key=None) -> 'VisualizationModel':
        raise NotImplementedError()

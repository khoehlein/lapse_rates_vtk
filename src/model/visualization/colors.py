from dataclasses import dataclass
from typing import Union, Dict, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtGui import QColor

from src.model.visualization.interface import standard_adapter, PropertyModel, ScalarType


class ColorModel(PropertyModel):

    @dataclass
    class Properties(object):
        pass

    @property
    def scalar_bar_title(self) -> Union[str, None]:
        raise NotImplementedError()

    def update_actors(self, actors, host, scalar_bar_old):
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

    def update_actors(self, actors: Dict[str, pv.Actor], host, scalar_bar_old) -> pv.Actor:
        actor = actors['mesh']
        actor_props = actor.prop
        new_actor_props = standard_adapter.read(self._properties)
        for key, value in new_actor_props.items():
            setattr(actor_props, key, value)
        return actor

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, UniformColorModel.Properties)

    def get_kws(self):
        kws = standard_adapter.read(self._properties)
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
        scalar_type = getattr(ScalarType, self._properties.scalar_name.upper())
        return scalar_type.value

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, ScalarColormapModel.Properties)

    def update_actors(self, actors: Dict[str, pv.Actor], host, scalar_bar_old) -> pv.Actor:
        self._update_mesh_actors(actors)
        self._update_scalar_bar(actors, host, scalar_bar_old)
        return actors

    def _update_mesh_actors(self, actors: Dict[str, pv.Actor]):
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
        return actors

    def _update_scalar_bar(self, actors: Dict[str, pv.Actor], host, scalar_bar_old: str):
        scalar_bar_new = self.scalar_bar_title
        if 'scalar_bar' in actors and (scalar_bar_new is None or scalar_bar_old != scalar_bar_new):
            host.remove_actor(actors['scalar_bar'])
            host.remove_scalar_bar(scalar_bar_old)
            del actors['scalar_bar']
        if scalar_bar_new is not None and scalar_bar_new != scalar_bar_old:
            actor = host.add_scalar_bar(mapper=actors['mesh'].mapper, title=scalar_bar_new)
            actors['scalar_bar'] = actor
        return actors

    def get_kws(self):
        props = self._properties
        lut = pv.LookupTable(cmap=props.colormap_name)
        lut.scalar_range = props.scalar_range
        return {'scalars': props.scalar_name, 'cmap': lut, 'opacity': props.opacity, 'show_scalar_bar': False}


def numpy_to_qcolor(numpy_color: np.ndarray) -> QColor:
    return QColor(*np.floor(256 * numpy_color).astype(int).clip(max=255))
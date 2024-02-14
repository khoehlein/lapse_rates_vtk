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

    def get_scalar_bar_title(self, gui_label: str = None) -> Union[str, None]:
        raise NotImplementedError()

    def update_actors(self, actors, host, scalar_bar_old, gui_label: str = None):
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

    def get_scalar_bar_title(self, gui_label: str = None) -> Union[str, None]:
        return None

    def update_actors(self, actors: Dict[str, pv.Actor], host, scalar_bar_old, gui_label=None) -> pv.Actor:
        actor = actors['mesh']
        actor.mapper.array_name = None
        actor.mapper.dataset.active_scalars_name = None
        actor_props = actor.prop
        new_actor_props = standard_adapter.read(self.properties)
        for key, value in new_actor_props.items():
            setattr(actor_props, key, value)
        if 'scalar_bar' in actors:
            host.remove_actor(actors['scalar_bar'])
            # host.remove_scalar_bar(scalar_bar_old) # apparently not required here
            del actors['scalar_bar']
        return actor

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, UniformColorModel.Properties)

    def get_kws(self):
        kws = standard_adapter.read(self.properties)
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

    def get_scalar_bar_title(self, gui_label: str = None) -> Union[str, None]:
        if self.properties is None:
            return None
        scalar_type = getattr(ScalarType, self.properties.scalar_name.upper())
        title = scalar_type.value
        if gui_label is None:
            return title
        return f'{title} ({gui_label})'

    def supports_update(self, properties: ColorModel.Properties):
        return isinstance(properties, ScalarColormapModel.Properties)

    def update_actors(self, actors: Dict[str, pv.Actor], host, scalar_bar_old, gui_label: str = None) -> pv.Actor:
        self._update_mesh_actors(actors)
        self._update_scalar_bar(actors, host, scalar_bar_old, gui_label)
        return actors

    def _update_mesh_actors(self, actors: Dict[str, pv.Actor]):
        actor = actors['mesh']
        actor_props = actor.prop
        actor_props.opacity = self.properties.opacity
        actor.mapper.array_name = self.properties.scalar_name
        actor.mapper.dataset.active_scalars_name = self.properties.scalar_name

        scalar_range = self.properties.scalar_range
        if scalar_range is None:
            mesh = actor.mapper.mesh
            scalar_range = mesh.get_data_range(self.properties.scalar_name)
        actor.mapper.scalar_range = scalar_range

        actor.mapper.lookup_table.cmap = self.properties.colormap_name
        below_range_color = self.properties.below_range_color
        if below_range_color is not None:
            actor.mapper.lookup_table.below_range_color = below_range_color
        above_range_color = self.properties.above_range_color
        if above_range_color is not None:
            actor.mapper.lookup_table.above_range_color = above_range_color
        return actors

    def _update_scalar_bar(self, actors: Dict[str, pv.Actor], host, scalar_bar_old: str, gui_label):
        scalar_bar_new = self.get_scalar_bar_title(gui_label)
        if ('scalar_bar' in actors) and (scalar_bar_new != scalar_bar_old):
            host.remove_actor(actors['scalar_bar'])
            host.remove_scalar_bar(scalar_bar_old)
            del actors['scalar_bar']
        if 'scalar_bar' not in actors:
            actor = host.add_scalar_bar(
                mapper=actors['mesh'].mapper, title=scalar_bar_new,
                interactive=True, title_font_size=12, label_font_size=12, vertical=False
            )
            actors['scalar_bar'] = actor
        return actors

    def get_kws(self):
        props = self.properties
        lut = pv.LookupTable(cmap=props.colormap_name)
        lut.scalar_range = props.scalar_range
        kws = {'scalars': props.scalar_name, 'cmap': lut, 'opacity': props.opacity}
        if props.scalar_name is not None:
            kws['show_scalar_bar'] = False
        return kws


def numpy_to_qcolor(numpy_color: np.ndarray) -> QColor:
    return QColor(*np.floor(256 * numpy_color).astype(int).clip(max=255))
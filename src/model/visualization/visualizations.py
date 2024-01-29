import copy
import uuid
from dataclasses import dataclass

from PyQt5.QtCore import QObject

import pyvista as pv
from PyQt5.QtGui import QColor
from pyvista.plotting import Plotter

from src.model.geometry import SurfaceDataset
from src.model.visualization.transforms import SceneSpaceTransform, AffineLinear


class VisualizationModel(QObject):

    def __init__(
            self,
            vertical_transform: SceneSpaceTransform = None,
            plotter_key: str = None,
            parent=None
    ):
        super().__init__(parent)
        if plotter_key is None:
            plotter_key = str(uuid.uuid4())
        self.plotter_key = plotter_key
        if vertical_transform is None:
            vertical_transform = AffineLinear.identity()
        self._vertical_transform = vertical_transform
        self.properties = None
        self.is_visible = True

    def set_properties(self, properties) -> 'VisualizationModel':
        self.properties = properties
        return self

    def set_visible(self, visible: bool) -> 'VisualizationModel':
        self.is_visible = visible
        return self

    def set_vertical_scale(self, scale: float) -> 'VisualizationModel':
        self._vertical_transform.scale = scale
        return self

    def set_vertical_offset(self, offset: float) -> 'VisualizationModel':
        self._vertical_transform.offset = offset
        return self

    def draw(self, plotter: Plotter) -> 'VisualizationModel':
        raise NotImplementedError()


class SurfaceVisualization(VisualizationModel):

    def __init__(
            self,
            surface_data: SurfaceDataset,
            vertical_transform: SceneSpaceTransform = None,
            plotter_key: str = None,
            parent=None
    ):
        super().__init__(vertical_transform, plotter_key, parent)
        self.dataset = surface_data
        self.polydata_reference = self.dataset.get_polydata()
        self._world_coordinates = copy.deepcopy(self.polydata_reference.points)
        self._actors = []
        self.update()

    def update(self) -> 'WireframeSurface.PyvistaReference':
        self._update_geometry_coordinates()
        self._update_actor_props()
        self._update_actor_visibility()
        return self

    def set_vertical_offset(self, offset: float) -> 'WireframeSurface':
        super().set_vertical_offset(offset)
        self._update_geometry_coordinates()
        return self

    def set_vertical_scale(self, scale: float) -> 'WireframeSurface':
        super().set_vertical_scale(scale)
        self._update_geometry_coordinates()
        return self

    def _update_geometry_coordinates(self):
        self.polydata_reference.points[:, -1] = self._vertical_transform.apply(self._world_coordinates[:, -1])

    def set_properties(self, properties: 'WireframeSurface.Properties') -> 'WireframeSurface':
        super().set_properties(properties)
        self._update_actor_props()
        return self

    def _update_actor_props(self):
        raise NotImplementedError()

    def set_visible(self, visible: bool):
        super().set_visible(visible)
        self._update_actor_visibility()
        return self

    def _update_actor_visibility(self):
        for actor in self._actors:
            actor.visibility = self.is_visible


class WireframeSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(object):
        line_width: float = None
        opacity: float = None
        color: QColor = None

    def _update_actor_props(self):
        for actor in self._actors:
            actor_props = actor.prop
            actor_props.line_width = self.properties.line_width
            actor_props.opacity = self.properties.opacity
            actor_props.color = self.properties.color.name()

    def draw(self, plotter: pv.Plotter) -> 'WireframeSurface.PyvistaReference':
        actor = plotter.add_mesh(self.polydata_reference, name=self.plotter_key, style='wireframe')
        self._actors.append(actor)
        self._update_actor_props()
        self._update_actor_visibility()
        return self


class TranslucentSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(object):
        opacity: float = None
        color: QColor = None
        show_edges: bool = None

    def __init__(
            self,
            surface_data: SurfaceDataset,
            vertical_transform: SceneSpaceTransform = None,
            plotter_key: str = None,
            parent=None
    ):
        self.colormap = None
        super().__init__(surface_data, vertical_transform, plotter_key, parent)

    def draw(self, plotter: pv.Plotter) -> 'WireframeSurface.PyvistaReference':
        actor = plotter.add_mesh(self.polydata_reference, name=self.plotter_key, style='surface')
        self._actors.append(actor)
        self._update_actor_props()
        self._update_actor_visibility()
        return self

    def _update_actor_props(self):
        for actor in self._actors:
            actor_props = actor.prop
            actor_props.opacity = self.properties.opacity
            actor_props.color = self.properties.color.name()
            actor_props.show_edges = self.properties.show_edges
            actor_props.edge_color = 'k'




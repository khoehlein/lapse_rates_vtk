from copy import deepcopy
from dataclasses import dataclass

from PyQt5.QtCore import QObject

import pyvista as pv
from PyQt5.QtGui import QColor

from src.model.geometry import SurfaceDataset
from src.model.visualization.transforms import SceneSpaceTransform, AffineLinear


class RenderDataReference(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)

    def update(self) -> 'RenderDataReference':
        raise NotImplementedError()

    def update_properties(self, properties=None):
        raise NotImplementedError

    def draw(self, plotter: pv.Plotter, name: str = None) -> None:
        raise NotImplementedError()


class VisualizationModel(QObject):

    def __init__(self, vertical_transform: SceneSpaceTransform = None, parent=None):
        super().__init__(parent)
        if vertical_transform is None:
            vertical_transform = AffineLinear.identity()
        self.properties = None
        self._vertical_transform = vertical_transform
        self._render_data: RenderDataReference = None

    def set_properties(self, properties) -> 'VisualizationModel':
        self.properties = properties
        if self._render_data is not None:
            self._render_data.update_properties(properties)
        return self

    def set_vertical_scale(self, scale: float) -> 'VisualizationModel':
        self._vertical_transform.scale = scale
        if self._render_data is not None:
            self._render_data.update()
        return self

    def set_vertical_offset(self, offset: float) -> 'VisualizationModel':
        self._vertical_transform.offset = offset
        if self._render_data is not None:
            self._render_data.update()
        return self

    def get_render_data(self) -> RenderDataReference:
        if self._render_data is None:
            self._render_data = self._build_render_data()
        return self._render_data

    def _build_render_data(self) -> RenderDataReference:
        raise NotImplementedError()


class SurfaceVisualization(VisualizationModel):

    def __init__(self, surface_data: SurfaceDataset, parent=None):
        super().__init__(parent=parent)
        self.dataset = surface_data


class WireframeSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(object):
        line_width: float = None
        color: QColor = None

    class PyvistaReference(RenderDataReference):

        def __init__(self, vis: 'WireframeSurface', parent=None):
            super().__init__(parent)
            self.polydata_reference = vis.dataset.get_polydata()
            self.transform = vis._vertical_transform
            self.properties = vis.properties
            self._world_coordinates = deepcopy(self.polydata_reference.points)
            self._actors = []
            self.update()

        def update(self) -> 'WireframeSurface.PyvistaReference':
            self.polydata_reference.points[:, -1] = self.transform.apply(self._world_coordinates[:, -1])
            self.update_properties()
            return self

        def update_properties(self, properties: 'WireframeSurface.Properties' = None) -> 'WireframeSurface.PyvistaReference':
            if properties is not None:
                self.properties = properties
            for actor in self._actors:
                actor_props = actor.prop
                actor_props.line_width = self.properties.line_width
                actor_props.color = self.properties.color.name()
            return self

        def draw(self, plotter: pv.Plotter, name: str = None) -> 'WireframeSurface.PyvistaReference':
            actor = plotter.add_mesh(self.polydata_reference, name=name, style='wireframe')
            self._actors.append(actor)
            self.update_properties()
            return self

    def __init__(self, surface_data: SurfaceDataset, parent=None):
        super().__init__(surface_data, parent)
        self.properties = None

    def _build_render_data(self) -> RenderDataReference:
        return self.PyvistaReference(self)

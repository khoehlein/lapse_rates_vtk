import copy
import uuid
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject

import pyvista as pv
from src.model.geometry import SurfaceDataset
from src.model.visualization.colors import ColormapModel
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

    def set_visibility(self, visible: bool) -> 'VisualizationModel':
        self.is_visible = visible
        return self

    def set_vertical_scale(self, scale: float) -> 'VisualizationModel':
        self._vertical_transform.scale = scale
        return self

    def set_vertical_offset(self, offset: float) -> 'VisualizationModel':
        self._vertical_transform.offset = offset
        return self

    def set_properties(self, properties) -> 'VisualizationModel':
        self.properties = properties
        return self


class SurfaceVisualization(VisualizationModel):

    @dataclass
    class Properties(object):
        colormap: ColormapModel = None

    class PyvistaReference(object):

        def __init__(self, dataset: pv.DataSet, actor: pv.Actor):
            self.dataset = dataset
            self.actor = actor

        def set_vertical_coordinate(self, z: np.ndarray) -> 'SurfaceVisualization.PyvistaReference':
            self.dataset.points[:, -1] = z
            return self

        def set_visibility(self, visible: bool) -> 'SurfaceVisualization.PyvistaReference':
            self.actor.visibility = visible
            return self

        def set_properties(self, properties: 'SurfaceVisualization.Properties') -> 'SurfaceVisualization.PyvistaReference':
            raise NotImplementedError()

        def update_colormap(self, colormap: ColormapModel) -> 'SurfaceVisualization.PyvistaReference':
            if colormap.is_uniform:
                self.dataset.set_active_scalars(None)
                color = colormap.color.name()
                self.actor.prop.color = color
            else:
                scalar_name = colormap.scalar_name
                self.dataset.set_active_scalars(scalar_name)
                scalar_data = self.dataset.point_data[scalar_name]
                mapper = self.actor.mapper
                lookup_table = pv.LookupTable(
                    cmap=colormap.name,
                    below_range_color=colormap.color_below_range.name(),
                    above_range_color=colormap.color_above_range.name(),
                    scalar_range=(colormap.vmin, colormap.vmax),
                )
                mapper.set_scalars(scalar_data, scalar_name, cmap=lookup_table)
                # mapper.array_name = scalar_name
                mapper.update()
            return self

        @classmethod
        def from_plotter_call(
                cls,
                visualization: 'SurfaceVisualization',
                plotter: pv.Plotter
        ) -> 'SurfaceVisualization.PyvistaReference':
            raise NotImplementedError()

    def __init__(
            self,
            surface_data: SurfaceDataset,
            vertical_transform: SceneSpaceTransform = None,
            plotter_key: str = None,
            parent=None
    ):
        super().__init__(vertical_transform, plotter_key, parent)
        self.dataset = surface_data
        self.properties: 'SurfaceVisualization.Properties' = None
        self.colormap: ColormapModel = None
        self.plotter_handle: 'SurfaceVisualization.PyvistaReference' = None

    def set_vertical_offset(self, offset: float) -> 'WireframeSurface':
        super().set_vertical_offset(offset)
        if self.plotter_handle is not None:
            z_new = self._vertical_transform.apply(self.dataset.z)
            self.plotter_handle.set_vertical_coordinate(z_new)
        return self

    def set_vertical_scale(self, scale: float) -> 'WireframeSurface':
        super().set_vertical_scale(scale)
        if self.plotter_handle is not None:
            z_new = self._vertical_transform.apply(self.dataset.z)
            self.plotter_handle.set_vertical_coordinate(z_new)
        return self

    def set_visibility(self, visible: bool):
        super().set_visibility(visible)
        if self.plotter_handle is not None:
            self.plotter_handle.set_visibility(self.is_visible)
        return self

    def set_properties(self, properties: 'SurfaceVisualization.Properties') -> 'SurfaceVisualization':
        self.properties = properties
        if self.plotter_handle is not None:
            self.plotter_handle.set_properties(properties)
            self.plotter_handle.update_colormap(properties.colormap)
        return self

    def draw(self, plotter: pv.Plotter) -> 'SurfaceVisualization':
        plotter.suppress_rendering = True
        # plotter.add_mesh(
        #     self.get_polydata(),
        # )
        self.plotter_handle = self.PyvistaReference.from_plotter_call(self, plotter)
        z_new = self._vertical_transform.apply(self.dataset.z)
        # self.plotter_handle.set_vertical_coordinate(z_new)
        self.plotter_handle.set_visibility(self.is_visible)
        self.plotter_handle.set_properties(self.properties)
        self.plotter_handle.update_colormap(self.properties.colormap)
        plotter.update_bounds_axes()
        plotter.suppress_rendering = False
        return self

    def get_polydata(self) -> pv.PolyData:
        polydata = self.dataset.get_polydata()
        polydata.points[:, -1] = self._vertical_transform.apply(self.dataset.z)
        return polydata


class WireframeSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(SurfaceVisualization.Properties):
        line_width: float = None
        opacity: float = None

    class PyvistaReference(SurfaceVisualization.PyvistaReference):

        @classmethod
        def from_plotter_call(
                cls,
                visualization: 'WireframeSurface',
                plotter: pv.Plotter
        ) -> 'WireframeSurface.PyvistaReference':
            dataset = visualization.get_polydata()
            actor = plotter.add_mesh(dataset, style='wireframe', name=visualization.plotter_key, color='k')
            return cls(dataset, actor)

        def set_properties(self, properties: 'WireframeSurface.Properties') -> 'WireframeSurface.PyvistaReference':
            actor_props = self.actor.prop
            actor_props.line_width = properties.line_width
            actor_props.opacity = properties.opacity


class TranslucentSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(SurfaceVisualization.Properties):
        opacity: float = None
        show_edges: bool = None

    class PyvistaReference(SurfaceVisualization.PyvistaReference):

        @classmethod
        def from_plotter_call(
                cls,
                visualization: 'TranslucentSurface',
                plotter: pv.Plotter
        ) -> 'TranslucentSurface.PyvistaReference':
            dataset = visualization.get_polydata()
            actor = plotter.add_mesh(dataset, style='surface', name=visualization.plotter_key, color='k')
            return cls(dataset, actor)

        def set_properties(self, properties: 'TranslucentSurface.Properties') -> 'WireframeSurface.PyvistaReference':
            actor_props = self.actor.prop
            actor_props.opacity = properties.opacity
            actor_props.show_edges = properties.show_edges
            actor_props.edge_color = 'k'


class BallSurface(SurfaceVisualization):

    @dataclass(init=True, repr=False, eq=True)
    class Properties(SurfaceVisualization.Properties):
        opacity: float = None
        point_size: float = None




import pyvista as pv
from PyQt5.QtCore import pyqtSignal

from src.paper.volume_visualization.plotter_slot import ActorProperties, PlotterSlot
from src.paper.volume_visualization.scaling import ScalingParameters, VolumeVisual
from src.paper.volume_visualization.volume_data import VolumeData


class VolumeDataRepresentation(VolumeVisual):

    def __init__(
            self,
            slot: PlotterSlot, volume_data: VolumeData,
            properties: ActorProperties,
            scaling: ScalingParameters = None,
            parent=None
    ):
        super(VolumeDataRepresentation, self).__init__(parent)
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        self.slot = slot
        self.volume_data = volume_data
        self.scaling = scaling
        self.properties = properties

    def is_visible(self):
        raise NotImplementedError()

    def clear(self, render: bool = True):
        raise NotImplementedError()

    def show(self, render: bool = True):
        raise NotImplementedError()

    def set_properties(self, properties: ActorProperties, render: bool = True):
        self.properties = properties
        self.slot.update_actor(properties, render=render)
        return self

    def set_scaling(self, scaling: ScalingParameters, render=True):
        raise NotImplementedError()

    def get_plotter(self) -> pv.Plotter:
        return self.slot.plotter


class MeshDataRepresentation(VolumeDataRepresentation):

    def __init__(self, slot: PlotterSlot, volume_data: VolumeData, properties: ActorProperties, scaling: ScalingParameters = None, parent=None):
        super(MeshDataRepresentation, self).__init__(slot, volume_data, properties, scaling, parent=parent)
        self.mesh = None

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.is_visible():
            self.mesh.points[:, -1] = self.volume_data.compute_elevation_coordinate(self.scaling).ravel()
            if render:
                self.slot.plotter.render()
        return self

    def clear(self, render: bool = True):
        self.slot.clear(render=render)
        self.mesh = None
        self.visibility_changed.emit(False)
        return self

    def is_visible(self, render: bool = True):
        return self.mesh is not None

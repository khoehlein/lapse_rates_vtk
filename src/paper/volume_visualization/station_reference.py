import pyvista as pv
from PyQt5.QtCore import QObject

from src.paper.volume_visualization.scaling import ScalingParameters, VolumeVisual
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.plotter_slot import PlotterSlot, ReferenceGridProperties


class StationDataVisualization(VolumeVisual):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData,
            properties: ReferenceGridProperties = None, scaling: ScalingParameters = None,
            parent: QObject = None
    ):
        super().__init__(parent)
        self.slot = slot
        self.station_data = station_data
        if properties is None:
            properties = ReferenceGridProperties()
        self.properties = properties
        if scaling is None:
            scaling = ScalingParameters(1., 1.)
        self.scaling = scaling
        self.mesh = None

    def is_visible(self):
        return self.mesh is not None

    def clear(self, render: bool = True):
        self.slot.clear(render=render)
        self.mesh = None
        self.visibility_changed.emit(False)
        return self

    def get_plotter(self) -> pv.Plotter:
        return self.slot.plotter


class StationSiteVisualization(StationDataVisualization):

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.is_visible():
            self.mesh.points[:, -1] = self.volume_data.compute_station_elevation(self.scaling).ravel()
            if render:
                self.slot.plotter.render()
        return self

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.station_data.get_station_sites(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self


class StationReferenceVisualization(StationDataVisualization):

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.is_visible():
            z_site = self.station_data.compute_station_elevation(self.scaling).ravel()
            z_surf = self.station_data.compute_terrain_elevation(self.scaling).ravel()
            n = len(z_site)
            self.mesh.points[:n, -1] = z_site
            self.mesh.points[n:, -1] = z_surf
            if render:
                self.slot.plotter.render()
        return self

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self.mesh = self.station_data.get_station_reference(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

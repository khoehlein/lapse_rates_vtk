import pandas as pd
import pyvista as pv
from PyQt5.QtCore import QObject

from src.paper.volume_visualization.scaling import ScalingParameters, VolumeVisual
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.plotter_slot import PlotterSlot, MeshProperties


class StationDataVisualization(VolumeVisual):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData,
            properties: MeshProperties,
            scaling: ScalingParameters = None,
            parent: QObject = None
    ):
        super().__init__(parent)
        if scaling is None:
            scaling = ScalingParameters(1., 1., False, False, False)
        self.slot = slot
        self.station_data = station_data
        self.scaling = scaling
        self.properties = properties

    def get_plotter(self) -> pv.Plotter:
        return self.slot.plotter

    def set_properties(self, properties: MeshProperties, render: bool = True):
        self.properties = properties
        self.slot.update_actor(properties, render=render)
        return self

    def update_data(self, new_data: pd.DataFrame, render: bool = True):
        self.blockSignals(True)
        was_visible = self.is_visible()
        if was_visible:
            self.clear(render=False)
        self.station_data.update_station_data(new_data)
        if was_visible:
            self.show(render=render)
        self.blockSignals(False)
        return self

    def update(self, render: bool = True):
        self.blockSignals(True)
        if self.is_visible():
            self.clear(render=False)
            self.show(render=render)
        self.blockSignals(False)
        return self


class StationDataRepresentation(StationDataVisualization):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData,
            properties: MeshProperties,
            scaling: ScalingParameters = None,
            parent: QObject = None
    ):
        super().__init__(slot, station_data, properties, scaling, parent)
        self.mesh = None

    def is_visible(self):
        return self.mesh is not None

    def clear(self, render: bool = True):
        self.slot.clear(render=render)
        self.mesh = None
        self.visibility_changed.emit(False)
        return self

    def show(self, render: bool = True):
        if self.is_visible():
            return self
        self._build_and_show_mesh()
        self.slot.update_actor(self.properties, render=render)
        self.visibility_changed.emit(True)
        return self

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.is_visible():
            self._update_mesh_scaling()
            if render:
                self.slot.plotter.render()
        return self

    def _update_mesh_scaling(self):
        raise NotImplementedError()

    def _build_and_show_mesh(self):
        raise NotImplementedError()

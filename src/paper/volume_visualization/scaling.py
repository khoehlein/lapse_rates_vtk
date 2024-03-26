import math
from collections import namedtuple
from enum import Enum

import pyvista as pv
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QPushButton, QSlider, QHBoxLayout


class SceneGeometry(Enum):
    SURFACE = 'surface'
    VOLUME = 'volume'
    CONTOUR = 'contour'
    STATION = 'station'


class SurfaceSlot(Enum):
    O1280 = 'o1280'
    O8000 = 'o8000'


class VolumeSlot(Enum):
    MODEL_LEVELS = 'model_levels'
    DVR = 'dvr'


class ContourSlot(Enum):
    T = 't'
    GRAD_T = 'grad_t'
    Z_MODEL_LEVELS = 'z_model_levels'
    LATITUDE_3D = 'latitude_3d'
    LONGITUDE_3D = 'longitude_3d'


class StationSlot(Enum):
    STATION_SITE = 'station_site'
    STATION_ON_TERRAIN = 'station_on_terrain'


ScalingParameters = namedtuple('ScaleParameters', ['scale', 'offset_scale'])
VisualizationSlot = namedtuple('VisualizationSlot', ['geometry', 'slot'])
VisualizationSlotRequest = namedtuple('VisualizationSlotRequest', ['uid', 'slot'])


class VolumeVisual(QObject):

    visibility_changed = pyqtSignal(bool)
    visualization_slot_requested = pyqtSignal(str, VisualizationSlotRequest)

    def is_visible(self) -> bool:
        raise NotImplementedError()

    def set_visible(self, visible: bool):
        if visible and not self.is_visible():
            self.show()
        if not visible and self.is_visible():
            self.clear()
        return self

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        raise NotImplementedError()

    def get_plotter(self) -> pv.Plotter:
        raise NotImplementedError()

    def show(self, render: bool = True):
        raise NotImplementedError()

    def clear(self, render: bool = True):
        raise NotImplementedError()

    def update(self, render: bool = True):
        if self.is_visible():
            self.clear(render=False)
            self.show(render=render)
        return self


class SceneScalingModel(QObject):

    def __init__(self, scaling: ScalingParameters = None, parent=None):
        super().__init__(parent)
        if scaling is None:
            scaling = ScalingParameters(2. ** 12, 1.)
        self.scaling = scaling
        self.visuals = []
        self.plotters = []

    def add_visual(self, visual: VolumeVisual):
        plotter = visual.get_plotter()
        self.visuals.append(visual)
        if plotter not in self.plotters:
            self.plotters.append(plotter)
        plotter.suppress_rendering = True
        visual.set_scaling(self.scaling, render=False)
        plotter.update_bounds_axes()
        plotter.suppress_rendering=False
        plotter.render()
        return self

    def suppress_rendering(self, suppress: bool):
        for plotter in self.plotters:
            plotter.suppress_rendering = suppress
        return self

    def render(self):
        for plotter in self.plotters:
            plotter.render()
        return self

    def set_scaling(self, scaling: ScalingParameters):
        self.scaling = scaling
        self.suppress_rendering(True)
        for visual in self.visuals:
            visual.set_scaling(self.scaling)
        self.suppress_rendering(False)
        self.render()

    def clear_all(self):
        for visual in self.visuals:
            if visual.is_visible():
                visual.clear()
        return self


class SceneScalingSettingsView(QWidget):

    scaling_changed = pyqtSignal()
    clear_request = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider_log_vertical_scale = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_vertical_scale.setRange(-64, 0)
        self.scale_step = 0.25
        self.slider_log_offset_exaggeration = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_offset_exaggeration.setRange(-32, 32)
        self.slider_log_offset_exaggeration.setValue(0)
        self.button_clear = QPushButton(self)
        self.button_clear.setText("Clear scene")
        self.button_reset = QPushButton(self)
        self.button_reset.setText("Reset exaggeration")
        self.button_apply = QPushButton(self)
        self.button_apply.setText("Apply")
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_apply.clicked.connect(self.scaling_changed)
        self.button_reset.clicked.connect(self.on_reset)
        self.button_clear.clicked.connect(self.clear_request)

    def on_reset(self):
        self.slider_log_offset_exaggeration.setValue(0)
        self.scaling_changed.emit()

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow("Vertical scale:", self.slider_log_vertical_scale)
        layout.addRow("Offset exaggeration:", self.slider_log_offset_exaggeration)
        outer_layout.addLayout(layout)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.button_reset)
        hlayout.addWidget(self.button_apply)
        outer_layout.addLayout(hlayout)
        outer_layout.addWidget(self.button_clear)
        self.setLayout(outer_layout)

    def get_settings(self):
        return ScalingParameters(
            2. ** (- self.scale_step * self.slider_log_vertical_scale.value()),
            2. ** (self.scale_step * self.slider_log_offset_exaggeration.value()),
        )

    def apply_settings(self, scaling: ScalingParameters):
        self.slider_log_vertical_scale.setValue(- int(math.log2(scaling.scale) / self.scale_step))
        self.slider_log_offset_exaggeration.setValue(int(math.log2(scaling.offset_scale) / self.scale_step))
        return self


class SceneScalingController(QObject):

    def __init__(self, view: SceneScalingSettingsView, model: SceneScalingModel, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(self.model.scaling)
        self.view.scaling_changed.connect(self.on_scaling_changed)
        self.view.clear_request.connect(self.handle_clear_request)

    def handle_clear_request(self):
        self.model.clear_all()

    def on_scaling_changed(self):
        scaling = self.view.get_settings()
        self.model.set_scaling(scaling)
        return self

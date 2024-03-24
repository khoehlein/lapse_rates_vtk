import math
import uuid
from collections import namedtuple
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
import pyvista as pv
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QComboBox, QFormLayout, QStackedWidget, QStackedLayout, \
    QVBoxLayout, QSpinBox, QPushButton, QSlider

ScalingParameters = namedtuple('ScaleParameters', ['scale', 'offset_scale'])


class VolumeVisual(QObject):

    visibility_changed = pyqtSignal(bool)

    def is_visible(self) -> bool:
        raise NotImplementedError()

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        raise NotImplementedError()

    def get_plotter(self) -> pv.Plotter:
        raise NotImplementedError()

    def show(self, render: bool = True):
        raise NotImplementedError()

    def clear(self, render: bool = True):
        raise NotImplementedError()


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


class SceneScalingSettingsView(QWidget):

    scaling_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider_log_vertical_scale = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_vertical_scale.setRange(0, 64)
        self.scale_step = 0.25
        self.slider_log_offset_exaggeration = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_log_offset_exaggeration.setRange(0, 32)
        self.button_apply = QPushButton(self)
        self.button_apply.setText("Apply")
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_apply.clicked.connect(self.scaling_changed)

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow("Vertical scale:", self.slider_log_vertical_scale)
        layout.addRow("Offset exaggeration:", self.slider_log_offset_exaggeration)
        outer_layout.addLayout(layout)
        outer_layout.addWidget(self.button_apply)
        self.setLayout(outer_layout)

    def get_settings(self):
        return ScalingParameters(
            2. ** (self.scale_step * self.slider_log_vertical_scale.value()),
            2. ** (self.scale_step * self.slider_log_offset_exaggeration.value()),
        )

    def apply_settings(self, scaling: ScalingParameters):
        self.slider_log_vertical_scale.setValue(int(math.log2(scaling.scale) / self.scale_step))
        self.slider_log_offset_exaggeration.setValue(int(math.log2(scaling.offset_scale) / self.scale_step))
        return self


class SceneScalingController(QObject):

    def __init__(self, view: SceneScalingSettingsView, model: SceneScalingModel, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(self.model.scaling)
        self.view.scaling_changed.connect(self.on_scaling_changed)

    def on_scaling_changed(self):
        scaling = self.view.get_settings()
        self.model.set_scaling(scaling)
        return self

import dataclasses
from typing import Union, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QFormLayout, QHBoxLayout
from src.widgets import RangeSpinner


@dataclasses.dataclass
class RampClipProperties(object):
    cutoff_lower: float
    cutoff_upper: float
    value_lower: float
    value_upper: float


class InteractiveRampClip(QObject):

    cutoff_changed = pyqtSignal()

    def __init__(self, properties: RampClipProperties, parent=None):
        super().__init__(parent)
        self.properties = properties

    def set_properties(self, properties: RampClipProperties):
        self.properties = properties
        self.cutoff_changed.emit()
        return self

    def clip(self, values: np.ndarray, other: np.ndarray, return_thresholds=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def compute_threshold(self, other: np.ndarray):
        output = np.zeros_like(other)
        output[other < self.properties.value_lower] = self.properties.cutoff_lower
        output[other >= self.properties.value_upper] = self.properties.cutoff_upper
        mask = np.logical_and(
            other >= self.properties.value_lower,
            other < self.properties.value_upper
        )
        if np.any(mask):
            scale = (self.properties.cutoff_upper - self.properties.cutoff_lower) / (self.properties.value_upper - self.properties.value_lower)
            output[mask] = scale * (other[mask] - self.properties.value_lower) + self.properties.cutoff_lower
        return output


class RampMaxClip(InteractiveRampClip):

    def clip(self, values: np.ndarray, other: np.ndarray, return_thresholds=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        thresholds = self.compute_threshold(other)
        clipped = np.minimum(values, thresholds)
        if return_thresholds:
            return clipped, thresholds
        return clipped


class RampMinClip(InteractiveRampClip):

    def clip(self, values: np.ndarray, other: np.ndarray, return_thresholds=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        thresholds = self.compute_threshold(other)
        clipped = np.maximum(values, thresholds)
        if return_thresholds:
            return clipped, thresholds
        return clipped


class RampClipSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_lower_cutoff = QDoubleSpinBox(self)
        self.spinner_lower_cutoff.setRange(-20, 100)
        self.spinner_lower_cutoff.setSingleStep(0.05)
        self.spinner_lower_cutoff.setPrefix('below: ')
        self.spinner_upper_cutoff = QDoubleSpinBox(self)
        self.spinner_upper_cutoff.setRange(-20, 100)
        self.spinner_upper_cutoff.setSingleStep(0.05)
        self.spinner_upper_cutoff.setPrefix('above: ')
        self.range_slope = RangeSpinner(self, 0.85, 0.95, 0., 1.)
        self._connect_signals()

    def _connect_signals(self):
        self.spinner_lower_cutoff.valueChanged.connect(self.settings_changed)
        self.spinner_upper_cutoff.valueChanged.connect(self.settings_changed)
        self.range_slope.range_changed.connect(self.settings_changed)

    def get_layout(self):
        layout = QFormLayout()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.spinner_lower_cutoff)
        hlayout.addWidget(self.spinner_upper_cutoff)
        layout.addRow('Cutoffs:', hlayout)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.range_slope.min_spinner)
        hlayout.addWidget(self.range_slope.max_spinner)
        layout.addRow('Ramp range:', hlayout)
        return layout

    def get_settings(self) -> RampClipProperties:
        return RampClipProperties(
            self.spinner_lower_cutoff.value(),
            self.spinner_upper_cutoff.value(),
            *self.range_slope.limits()
        )

    def apply_settings(self, settings: RampClipProperties):
        self.range_slope.set_limits(settings.value_lower, settings.value_upper)
        self.spinner_lower_cutoff.setValue(settings.cutoff_lower)
        self.spinner_upper_cutoff.setValue(settings.cutoff_upper)
        return self


class RampClipController(QObject):

    def __init__(self, view: RampClipSettingsView, model: InteractiveRampClip, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(self.model.properties)
        self.view.settings_changed.connect(self.on_settings_changed)

    def on_settings_changed(self):
        self.model.set_properties(self.view.get_settings())

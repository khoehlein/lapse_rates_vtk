import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib as mpl
import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QSpinBox

from src.widgets import RangeSpinner, SelectColorButton


class InteractiveColorLookup(QObject):
    lookup_table: pv.LookupTable
    samples: np.ndarray
    colors: np.ndarray

    colormap_changed = pyqtSignal()

    @dataclass
    class Properties(object):
        pass

    def __init__(self, properties: 'InteractiveColorLookup.Properties', parent=None):
        super().__init__(parent)
        self.props = properties
        self.update_lookup_table()

    def update_lookup_table(self):
        raise NotImplementedError()

    def get_opacity_function(self):
        raise NotImplementedError()

    def set_properties(self, properties: 'InteractiveColorLookup.Properties'):
        self.props = properties
        self.update_lookup_table()
        return self


class AsymmetricDivergentColorLookup(InteractiveColorLookup):

    @dataclass
    class Properties(InteractiveColorLookup.Properties):
        cmap_name: str
        vmin: float
        vmax: float
        vcenter: float
        num_samples: int
        log_n_lower: float
        log_n_upper: float
        opacity_lower: float
        opacity_upper: float
        color_below: Any
        color_above: Any

    def __init__(self, properties: 'AsymmetricDivergentColorLookup.Properties'):
        super().__init__(properties)

    def update_lookup_table(self):
        vmin = self.props.vmin
        vmax = self.props.vmax
        vcenter = self.props.vcenter
        num_samples = self.props.num_samples
        self.samples = np.linspace(vmin, vmax, num_samples)
        cmap = mpl.colormaps[self.props.cmap_name]
        lower_samples = self.samples < vcenter
        colors_lower = [
            cmap(0.5 * (x - vmin) / (vcenter - vmin))
            for x in self.samples[lower_samples]
        ]
        upper_samples = self.samples >= vcenter
        colors_upper = [
            cmap(0.5 + 0.5 * (x - vcenter) / (vmax - vcenter))
            for x in self.samples[upper_samples]
        ]
        self.colors = np.asarray(colors_lower + colors_upper)
        self.colors[lower_samples, -1] = 1. - (self.samples[lower_samples] - vmin) / (vcenter - vmin)
        self.colors[upper_samples, -1] = (self.samples[upper_samples] - vcenter) / (vmax - vcenter)
        self.colors[lower_samples, -1] = np.power(self.colors[lower_samples, -1], math.exp(self.props.log_n_lower)) * self.props.opacity_lower
        self.colors[upper_samples, -1] = np.power(self.colors[upper_samples, -1], math.exp(self.props.log_n_upper)) * self.props.opacity_upper
        listed_cmap = mpl.colors.ListedColormap(self.colors.tolist())
        lut = pv.LookupTable(
            cmap=listed_cmap, scalar_range=(vmin, vmax),
            below_range_color=self.props.color_below,
            above_range_color=self.props.color_above,
        )
        self.lookup_table = lut

    def set_properties(self, properties: 'AsymmetricDivergentColorLookup.Properties'):
        self.props = properties
        self.update_lookup_table()
        return self

    def get_opacity_function(self):
        return self.samples, self.colors[:, -1]


CMAP_NAMES = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
]


class AsymmetricLookupSettingsView(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self.combo_cmap_name = QComboBox(self)
        self.combo_cmap_name.addItems(CMAP_NAMES)
        self.combo_cmap_name.currentIndexChanged.connect(self.settings_changed.emit)
        self.value_range = RangeSpinner(self, -13., 50, -15., 100.)
        self.spinner_vcenter = QDoubleSpinBox(self)
        self.spinner_vcenter.setValue(-6.5)
        self.value_range.range_changed.connect(self.on_range_changed)
        self.spinner_vcenter.valueChanged.connect(self.settings_changed)
        self.value_range.range_changed.connect(self.settings_changed.emit)
        self.spinner_num_samples = QSpinBox(self)
        self.spinner_num_samples.setMinimum(3)
        self.spinner_num_samples.setMaximum(1024)
        self.spinner_num_samples.valueChanged.connect(self.settings_changed)
        self.spinner_log_n_lower = QDoubleSpinBox(self)
        self.spinner_log_n_lower.setMinimum(-4)
        self.spinner_log_n_lower.setMaximum(4)
        self.spinner_log_n_lower.valueChanged.connect(self.settings_changed)
        self.spinner_log_n_upper = QDoubleSpinBox(self)
        self.spinner_log_n_upper.setMinimum(-4)
        self.spinner_log_n_upper.setMaximum(4)
        self.spinner_log_n_upper.valueChanged.connect(self.settings_changed)
        self.spinner_opacity_lower = QDoubleSpinBox(self)
        self.spinner_opacity_lower.setMinimum(0.)
        self.spinner_opacity_lower.setMaximum(1.)
        self.spinner_opacity_lower.valueChanged.connect(self.settings_changed)
        self.spinner_opacity_upper = QDoubleSpinBox(self)
        self.spinner_opacity_upper.setMinimum(0.)
        self.spinner_opacity_upper.setMaximum(1.)
        self.spinner_opacity_upper.valueChanged.connect(self.settings_changed)
        self.button_color_below = SelectColorButton(parent=self)
        self.button_color_below.color_changed.connect(self.settings_changed)
        self.button_color_above = SelectColorButton(parent=self)
        self.button_color_above.color_changed.connect(self.settings_changed)
        self._set_layout()

    def _set_layout(self):
        raise NotImplementedError()

    def on_range_changed(self):
        new_range = self.value_range.limits()
        self.spinner_vcenter.setMinimum(new_range[0])
        self.spinner_vcenter.setMaximum(new_range[1])

    def apply_settings(self, settings: AsymmetricDivergentColorLookup.Properties):
        self.combo_cmap_name.setCurrentText(settings.cmap_name)
        self.value_range.set_limits(settings.vmin, settings.vmax)
        self.spinner_log_n_lower.setValue(settings.log_n_lower)
        self.spinner_log_n_upper.setValue(settings.log_n_upper)
        self.spinner_opacity_lower.setValue(settings.opacity_lower)
        self.spinner_opacity_upper.setValue(settings.opacity_upper)
        self.button_color_above.set_current_color(QColor(settings.color_above))
        self.button_color_above.set_current_color(QColor(settings.color_below))
        return self

    def get_settings(self):
        limits = self.value_range.limits()
        return AsymmetricDivergentColorLookup.Properties(
            cmap_name=self.combo_cmap_name.currentText(),
            vmin=limits[0], vmax=limits[1],
            vcenter=self.spinner_vcenter.value(),
            num_samples=self.spinner_num_samples.value(),
            log_n_lower=self.spinner_log_n_lower.value(),
            log_n_upper=self.spinner_log_n_upper.value(),
            opacity_lower=self.spinner_opacity_lower.value(),
            opacity_upper=self.spinner_opacity_upper.value(),
            color_above=self.button_color_above.current_color,
            color_below=self.button_color_below.current_color,
        )


class AsymetricColorLookupControls(QObject):

    def __init__(self, view: AsymmetricLookupSettingsView, model: AsymmetricDivergentColorLookup, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(model.props)
        self.view.settings_changed.connect(self.synchronize_settings)

    def synchronize_settings(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)

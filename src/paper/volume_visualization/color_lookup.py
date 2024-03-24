import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib as mpl
import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QFormLayout, QHBoxLayout, \
    QPushButton, QLabel

from src.widgets import RangeSpinner, SelectColorButton


class InteractiveColorLookup(QObject):
    lookup_table: pv.LookupTable
    samples: np.ndarray
    colors: np.ndarray

    lookup_table_changed = pyqtSignal()

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


@dataclass
class CustomOpacityProperties(object):
    log_n_lower: float = 0.
    log_n_upper: float = 0.
    opacity_lower: float = 1.
    opacity_upper: float = 1.
    opacity_center: float = 0.


class AsymmetricDivergentColorLookup(InteractiveColorLookup):

    @dataclass
    class Properties(InteractiveColorLookup.Properties):
        cmap_name: str
        vmin: float
        vmax: float
        vcenter: float
        num_samples: int
        color_below: Any
        color_above: Any
        opacity: CustomOpacityProperties

    def __init__(self, properties: 'AsymmetricDivergentColorLookup.Properties'):
        super().__init__(properties)

    def update_lookup_table(self):
        self._update_samples()
        self._update_colors()
        self._update_opacity()
        listed_cmap = mpl.colors.ListedColormap(self.colors.tolist())
        lut = pv.LookupTable(
            cmap=listed_cmap, scalar_range=(self.props.vmin, self.props.vmax),
            below_range_color=self.props.color_below,
            above_range_color=self.props.color_above,
        )
        self.lookup_table = lut

    def _update_samples(self):
        vmin = self.props.vmin
        vmax = self.props.vmax
        num_samples = self.props.num_samples
        self.samples = np.linspace(vmin, vmax, num_samples)

    def _update_colors(self):
        vmin = self.props.vmin
        vmax = self.props.vmax
        vcenter = self.props.vcenter
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

    def _update_opacity(self):
        oprops = self.props.opacity
        ocenter = oprops.opacity_center
        vcenter = self.props.vcenter
        self._update_opacity_section(
            self.samples < vcenter, self.props.vmin, vcenter, oprops.opacity_lower, ocenter, oprops.log_n_lower
        )
        self._update_opacity_section(
            self.samples >= vcenter, vcenter, self.props.vmax, ocenter, oprops.opacity_upper, oprops.log_n_upper
        )

    def _update_opacity_section(self, mask, vlow, vup, olow, oup, log_exponent):
        if np.any(mask):
            opacity = olow + ((oup - olow) / (vup - vlow)) * (self.samples[mask] - vlow)
            omin = min(olow, oup)
            omax = max(olow, oup)
            difference = (omax - omin)
            if difference > 0.:
                opacity_relative = (opacity - omin) / difference
                opacity_relative = np.power(opacity_relative, 2. ** log_exponent)
                opacity = omin + difference * opacity_relative
            self.colors[mask, -1] = opacity

    def set_properties(self, properties: 'AsymmetricDivergentColorLookup.Properties'):
        self.props = properties
        self.update_lookup_table()
        self.lookup_table_changed.emit()
        return self

    def get_opacity_function(self):
        return self.samples, self.colors[:, -1]


CMAP_NAMES = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
]


class CustomOpacitySettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_log_n_lower = QDoubleSpinBox(self)
        self.spinner_log_n_lower.setPrefix('lower: ')
        self.spinner_log_n_lower.setMinimum(-4)
        self.spinner_log_n_lower.setMaximum(4)
        self.spinner_log_n_lower.setSingleStep(0.05)
        self.spinner_log_n_upper = QDoubleSpinBox(self)
        self.spinner_log_n_upper.setPrefix('upper: ')
        self.spinner_log_n_upper.setMinimum(-4)
        self.spinner_log_n_upper.setMaximum(4)
        self.spinner_log_n_upper.setSingleStep(0.05)
        self.spinner_opacity_lower = QDoubleSpinBox(self)
        self.spinner_opacity_lower.setPrefix('lower: ')
        self.spinner_opacity_lower.setMinimum(0.)
        self.spinner_opacity_lower.setMaximum(1.)
        self.spinner_opacity_lower.setSingleStep(0.05)
        self.spinner_opacity_upper = QDoubleSpinBox(self)
        self.spinner_opacity_upper.setPrefix('upper: ')
        self.spinner_opacity_upper.setMinimum(0.)
        self.spinner_opacity_upper.setMaximum(1.)
        self.spinner_opacity_upper.setSingleStep(0.05)
        self.spinner_opacity_center = QDoubleSpinBox(self)
        self.spinner_opacity_center.setMinimum(0.)
        self.spinner_opacity_center.setMaximum(1.)
        self.spinner_opacity_center.setSingleStep(0.05)
        self.button_uniform = QPushButton(self)
        self.button_uniform.setText('uniform')
        self.button_vshape = QPushButton(self)
        self.button_vshape.setText('v-shape')
        self.button_opaque = QPushButton(self)
        self.button_opaque.setText('opaque')
        self._connect_signals()

    def get_layout(self, layout: QFormLayout = None) -> QFormLayout:
        if layout is None:
            layout = QFormLayout()
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.spinner_opacity_lower)
        opacity_layout.addWidget(self.spinner_opacity_upper)
        layout.addRow("Max. opacity:", opacity_layout)
        layout.addRow("Center opacity:", self.spinner_opacity_center)
        exponent_layout = QHBoxLayout()
        exponent_layout.addWidget(self.spinner_log_n_lower)
        exponent_layout.addWidget(self.spinner_log_n_upper)
        layout.addRow("Exponents:", exponent_layout)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(self.button_uniform)
        preset_layout.addWidget(self.button_vshape)
        preset_layout.addWidget(self.button_opaque)
        layout.addRow('Preset: ', preset_layout)
        return layout

    def on_button_uniform(self):
        self.spinner_log_n_lower.setValue(0.)
        self.spinner_log_n_upper.setValue(0.)
        ocenter = self.spinner_opacity_center.value()
        self.spinner_opacity_lower.setValue(ocenter)
        self.spinner_opacity_upper.setValue(ocenter)

    def on_button_vshape(self):
        self.spinner_log_n_lower.setValue(0.)
        self.spinner_log_n_upper.setValue(0.)
        self.spinner_opacity_center.setValue(0.)
        self.spinner_opacity_lower.setValue(1.)
        self.spinner_opacity_upper.setValue(1.)

    def on_button_opaque(self):
        self.spinner_opacity_center.setValue(1.)
        self.on_button_uniform()

    def _connect_signals(self):
        # self.spinner_log_n_lower.valueChanged.connect(self.settings_changed.emit)
        # self.spinner_log_n_upper.valueChanged.connect(self.settings_changed.emit)
        # self.spinner_opacity_lower.valueChanged.connect(self.settings_changed.emit)
        # self.spinner_opacity_upper.valueChanged.connect(self.settings_changed.emit)
        self.button_uniform.clicked.connect(self.on_button_uniform)
        self.button_vshape.clicked.connect(self.on_button_vshape)
        self.button_opaque.clicked.connect(self.on_button_opaque)

    def get_settings(self):
        return CustomOpacityProperties(
            log_n_lower=self.spinner_log_n_lower.value(),
            log_n_upper=self.spinner_log_n_upper.value(),
            opacity_lower=self.spinner_opacity_lower.value(),
            opacity_upper=self.spinner_opacity_upper.value(),
            opacity_center=self.spinner_opacity_center.value()
        )

    def opacity_below(self):
        return self.spinner_opacity_lower.value()

    def opacity_above(self):
        return self.spinner_opacity_upper.value()

    def apply_settings(self, settings: CustomOpacityProperties):
        self.spinner_log_n_lower.setValue(settings.log_n_lower)
        self.spinner_log_n_upper.setValue(settings.log_n_upper)
        self.spinner_opacity_lower.setValue(settings.opacity_lower)
        self.spinner_opacity_upper.setValue(settings.opacity_upper)
        self.spinner_opacity_center.setValue(settings.opacity_center)
        return self


class ADCLSettingsView(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self.combo_cmap_name = QComboBox(self)
        self.combo_cmap_name.addItems(CMAP_NAMES)
        self.scalar_range = RangeSpinner(self, -13., 50, -15., 1000.)
        self.spinner_vcenter = QDoubleSpinBox(self)
        self.spinner_vcenter.setValue(-6.5)
        self.spinner_vcenter.setSingleStep(0.05)
        self.spinner_num_samples = QSpinBox(self)
        self.spinner_num_samples.setMinimum(3)
        self.spinner_num_samples.setMaximum(1024)
        self.button_color_below = SelectColorButton(parent=self)
        self.button_color_below.setText(' Below')
        self.button_color_above = SelectColorButton(parent=self)
        self.button_color_above.setText(' Above')
        self.custom_opacity_view = CustomOpacitySettingsView(self)
        self.button_apply = QPushButton(self)
        self.button_apply.setText('Apply')
        self._connect_signals()
        self._set_layout()

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow("Colormap:", self.combo_cmap_name)
        range_layout = QHBoxLayout()
        range_layout.addWidget(self.scalar_range.min_spinner)
        range_layout.addWidget(self.scalar_range.max_spinner)
        layout.addRow("Scalar range:", range_layout)
        layout.addRow("Center:", self.spinner_vcenter)
        layout.addRow("Samples:", self.spinner_num_samples)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.button_color_below)
        color_layout.addWidget(self.button_color_above)
        layout.addRow("Outlier colors:", color_layout)
        self.custom_opacity_view.get_layout(layout=layout)
        outer_layout.addLayout(layout)
        outer_layout.addWidget(self.button_apply)
        self.setLayout(outer_layout)

    def _connect_signals(self):
        # self.combo_cmap_name.currentIndexChanged.connect(self.settings_changed.emit)
        self.scalar_range.range_changed.connect(self.on_range_changed)
        self.button_apply.clicked.connect(self.settings_changed)
        # self.spinner_vcenter.valueChanged.connect(self.settings_changed.emit)
        # self.scalar_range.range_changed.connect(self.settings_changed.emit)
        # self.spinner_num_samples.valueChanged.connect(self.settings_changed.emit)
        # self.button_color_below.color_changed.connect(self.settings_changed.emit)
        # self.button_color_above.color_changed.connect(self.settings_changed.emit)

    def on_range_changed(self):
        new_range = self.scalar_range.limits()
        self.spinner_vcenter.setMinimum(new_range[0])
        self.spinner_vcenter.setMaximum(new_range[1])

    def apply_settings(self, settings: AsymmetricDivergentColorLookup.Properties):
        self.combo_cmap_name.setCurrentText(settings.cmap_name)
        self.scalar_range.set_limits(settings.vmin, settings.vmax)
        self.spinner_vcenter.setValue(settings.vcenter)
        self.spinner_num_samples.setValue(settings.num_samples)
        self.button_color_below.set_current_color(QColor(settings.color_below))
        self.button_color_above.set_current_color(QColor(settings.color_above))
        self.custom_opacity_view.apply_settings(settings.opacity)
        return self

    def get_settings(self):
        limits = self.scalar_range.limits()
        color_below = list(self.button_color_below.current_color.getRgb())
        color_below[-1] = int(self.custom_opacity_view.opacity_below() * 255)
        color_above = list(self.button_color_above.current_color.getRgb())
        color_above[-1] = int(self.custom_opacity_view.opacity_above() * 255)
        return AsymmetricDivergentColorLookup.Properties(
            cmap_name=self.combo_cmap_name.currentText(),
            vmin=limits[0], vmax=limits[1],
            vcenter=self.spinner_vcenter.value(),
            num_samples=self.spinner_num_samples.value(),
            color_above=color_above,
            color_below=color_below,
            opacity=self.custom_opacity_view.get_settings()
        )


class ADCLController(QObject):

    def __init__(self, view: ADCLSettingsView, model: AsymmetricDivergentColorLookup, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.apply_settings(model.props)
        self.view.settings_changed.connect(self.synchronize_settings)

    def synchronize_settings(self):
        settings = self.view.get_settings()

        self.model.set_properties(settings)
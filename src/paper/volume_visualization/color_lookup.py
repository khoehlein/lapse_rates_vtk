import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import matplotlib as mpl
import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QFormLayout, QHBoxLayout, \
    QPushButton, QLabel

from src.widgets import RangeSpinner, SelectColorButton


ECMWF_COLORS = [
    '#ffffff',
    '#e5e5e5',
    '#cccccc',
    '#b2b2b2',
    '#ad99ad',
    '#7a667a',
    '#473347',
    '#330066',
    '#59007f',
    '#7f00ff',
    '#007fff',
    '#00ccff',
    '#00ffff',
    '#26e599',
    '#66bf26',
    '#bfe526',
    '#ffff7f',
    '#ffff00',
    '#ffd900',
    '#ffb000',
    '#ff7200',
    '#ff0000',
    '#cc0000',
    '#7f002c',
    '#cc3d6e',
    '#ff00ff',
    '#ff7fff',
    '#ffbfff',
    '#e5cce5',
    '#e5e5e5',
    '#ffffff'
]


class OutlierColor(Enum):
    ABOVE = '#fa39fa'
    BELOW = '#39fafa'


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
        raise NotImplementedError()

    def get_controller(self, settings_view):
        raise NotImplementedError()


class ECMWFColors(InteractiveColorLookup):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmap = mpl.colors.ListedColormap(
            ECMWF_COLORS[1:-1]
        ).with_extremes(over=ECMWF_COLORS[-1], under=ECMWF_COLORS[0])
        bounds = 40 - 2 * np.arange(len(ECMWF_COLORS) - 1)
        self.samples = np.linspace(bounds.min(), bounds.max(), 256)
        self.clim = (bounds.min(), bounds.max())
        colors = [self.cmap(x) for x in np.linspace(0, 1, 256)]
        self.lookup_table = pv.LookupTable(
            cmap=mpl.colors.ListedColormap(colors), scalar_range=self.clim,
            below_range_color=ECMWF_COLORS[0],
            above_range_color=ECMWF_COLORS[-1],
        )
        # self.lookup_table = pv.LookupTable(
        #     cmap='hsv', scalar_range=self.clim,
        #     # below_range_color=ECMWF_COLORS[0],
        #     # above_range_color=ECMWF_COLORS[-1],
        # )

    def update_lookup_table(self):
        pass

    def get_opacity_function(self):
        return self.samples, np.ones_like(self.samples)

    def set_properties(self, properties: 'InteractiveColorLookup.Properties'):
        pass

    def get_controller(self, settings_view):
        return None


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
        cmap_center: float
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
        cmap_center = self.props.cmap_center
        colors_lower = [
            cmap(cmap_center * (x - vmin) / (vcenter - vmin))
            for x in self.samples[lower_samples]
        ]
        upper_samples = self.samples >= vcenter
        colors_upper = [
            cmap(cmap_center + (1. - cmap_center) * (x - vcenter) / (vmax - vcenter))
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
            diff = (vup - vlow)
            opacity = olow + ((oup - olow) / diff) * (self.samples[mask] - vlow)
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

    def get_controller(self, settings_view):
        return ADCLController(settings_view, self)


CMAP_NAMES = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hsv',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'ocean', 'gist_earth', 'terrain', 'gist_heat', 'gist_gray', 'gist_rainbow'
]
CMAP_NAMES = CMAP_NAMES + [cmap_name + '_r' for cmap_name in CMAP_NAMES]


class CustomOpacitySettingsView(QWidget):

    settings_changed = pyqtSignal()
    preset_selected = pyqtSignal()

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
        self.button_vshape.setText('V-shape')
        self.button_ashape = QPushButton(self)
        self.button_ashape.setText('A-shape')
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
        preset_layout.addWidget(self.button_ashape)
        preset_layout.addWidget(self.button_opaque)
        layout.addRow('Preset: ', preset_layout)
        return layout

    def on_button_uniform(self):
        self.spinner_log_n_lower.setValue(0.)
        self.spinner_log_n_upper.setValue(0.)
        ocenter = self.spinner_opacity_center.value()
        self.spinner_opacity_lower.setValue(ocenter)
        self.spinner_opacity_upper.setValue(ocenter)
        self.preset_selected.emit()

    def on_button_vshape(self):
        self.spinner_log_n_lower.setValue(0.)
        self.spinner_log_n_upper.setValue(0.)
        self.spinner_opacity_center.setValue(0.)
        self.spinner_opacity_lower.setValue(1.)
        self.spinner_opacity_upper.setValue(1.)
        self.preset_selected.emit()

    def on_button_ashape(self):
        self.spinner_log_n_lower.setValue(0.)
        self.spinner_log_n_upper.setValue(0.)
        self.spinner_opacity_center.setValue(1.)
        self.spinner_opacity_lower.setValue(0.)
        self.spinner_opacity_upper.setValue(0.)
        self.preset_selected.emit()

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
        self.button_ashape.clicked.connect(self.on_button_ashape)
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
        self.spinner_cmap_center = QDoubleSpinBox(self)
        self.spinner_cmap_center.setRange(0., 1.)
        self.spinner_cmap_center.setValue(0.5)
        self.spinner_cmap_center.setSingleStep(0.05)
        self.scalar_range = RangeSpinner(self, -13., 50, -10000., 10000.)
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
        layout.addRow("Colormap center:", self.spinner_cmap_center)
        range_layout = QHBoxLayout()
        range_layout.addWidget(self.scalar_range.min_spinner)
        range_layout.addWidget(self.scalar_range.max_spinner)
        layout.addRow("Scalar range:", range_layout)
        layout.addRow("Scalar center:", self.spinner_vcenter)
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
        self.custom_opacity_view.preset_selected.connect(self.settings_changed)
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
        self.spinner_cmap_center.setValue(settings.cmap_center)
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
            cmap_center=self.spinner_cmap_center.value(),
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


def make_lsm_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'gist_earth', 0.5, 0.25, 0.75, 0.5, 7, 'black', 'white',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_elevation_offset_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'BrBG_r', 0.5, -1500, 1500, 0., 29, 'green', 'orange',
            CustomOpacityProperties()
        )
    )


def make_elevation_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'gist_earth', 0.3, -100., 3000., 0., 58, 'black', 'white',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_temperature_lookup():
    return ECMWFColors()


def make_diverging_temp_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'coolwarm', 0.5, -20, 40, 0., 29, 'blue', 'red',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_lapse_rate_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'PuOr_r', 0.5, -12, 50, -6.5, 256, '#55007f', '#713900',
            CustomOpacityProperties()
        )
    )

def make_model_level_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'magma_r', 0.5, 118, 137, 127, 8, 'white', 'black',
            CustomOpacityProperties()
        )
    )


def make_temperature_difference_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'coolwarm', 0.5, -15, 15, 0., 29, 'blue', 'red',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_quantile_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'gist_gray_r', 0.5, 0., 1., 0.5, 7, 'white', 'black',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_quantile_difference_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'coolwarm', 0.5, -0.5, 0.5, 0., 7, 'blue', 'red',
            CustomOpacityProperties()
        )
    )


def make_count_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'gist_gray', 0., 0., 1000., 0., 7, 'black', 'white',
            CustomOpacityProperties(opacity_center=1.)
        )
    )


def make_score_lookup():
    return AsymmetricDivergentColorLookup(
        AsymmetricDivergentColorLookup.Properties(
            'gist_gray', 0., 0., 1., 0., 7, 'black', 'white',
            CustomOpacityProperties(opacity_center=1.)
        )
    )
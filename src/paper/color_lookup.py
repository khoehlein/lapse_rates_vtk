import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib as mpl
import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal


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

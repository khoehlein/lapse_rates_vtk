import pyvista as pv

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QFormLayout, QComboBox, QLabel, QHBoxLayout

from src.interaction.background_color.view import SelectColorButton
from src.model.visualization.colors import ScalarColormapModel, UniformColorModel
from src.model.visualization.interface import ScalarType

_vis_default_lapse_rate = ScalarColormapModel.Properties(
    None, 'RdBu', 1., (-14, 14), None, None
)

_vis_default_temperature = ScalarColormapModel.Properties(
    None, 'RdBu', 1.,  (260, 320), None, None
)

_vis_default_temperature_difference = ScalarColormapModel.Properties(
    None, 'RdBu', 1.,  (-40, 40), None, None
)

_vis_default_elevation = ScalarColormapModel.Properties(
    None, 'greys', 1., (-500, 9000), None, None
)

_vis_default_elevation_difference = ScalarColormapModel.Properties(
    None, 'RdBu', 1., (-1500, 1500), None, None
)

cmap_defaults = {
    ScalarType.GEOMETRY: UniformColorModel.Properties('k', 1.),
    ScalarType.LAPSE_RATE: _vis_default_lapse_rate,
    ScalarType.T2M_O1280: _vis_default_temperature,
    ScalarType.T2M_O8000: _vis_default_temperature,
    ScalarType.T2M_DIFFERENCE: _vis_default_temperature_difference,
    ScalarType.Z_O1280: _vis_default_elevation,
    ScalarType.Z_O8000: _vis_default_elevation,
    ScalarType.Z_DIFFERENCE: _vis_default_elevation_difference,
}


class UniformColorSettingsView(QWidget):

    color_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(UniformColorSettingsView, self).__init__(parent)
        self.button_color = SelectColorButton(parent=self)
        self.button_color.color_changed.connect(self.color_changed.emit)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setValue(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.color_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow('Color:', self.button_color)
        layout.addRow('Opacity', self.spinner_opacity)
        self.setLayout(layout)

    def set_defaults(self, defaults = None):
        color_theme = pv.global_theme
        default_color = QColor(*color_theme.color.int_rgb)
        self.button_color.set_current_color(default_color)
        self.spinner_opacity.setValue(color_theme.opacity)

    def get_settings(self, scalar_name: str = None):
        return UniformColorModel.Properties(
            color=self.button_color.current_color,
            opacity=self.spinner_opacity.value()
        )


class ColormapSettingsView(QWidget):

    color_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(ColormapSettingsView, self).__init__(parent)
        self.combo_cmap_name = QComboBox(self)
        self.combo_cmap_name.addItems([
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
            'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
            'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ])
        self.combo_cmap_name.currentTextChanged.connect(self.color_changed.emit)
        self.spinner_scalar_min = QDoubleSpinBox(self)
        self.spinner_scalar_max = QDoubleSpinBox(self)
        self.spinner_scalar_min.valueChanged.connect(self.spinner_scalar_max.setMinimum)
        self.spinner_scalar_max.valueChanged.connect(self.spinner_scalar_min.setMaximum)
        self.spinner_scalar_min.valueChanged.connect(self.color_changed.emit)
        self.spinner_scalar_min.setPrefix('min: ')
        self.spinner_scalar_max.valueChanged.connect(self.color_changed.emit)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_scalar_max.setPrefix('max: ')
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.spinner_opacity.valueChanged.connect(self.color_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Colormap:'), self.combo_cmap_name)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.spinner_scalar_min)
        hlayout.addWidget(self.spinner_scalar_max)
        layout.addRow(QLabel('Scalar range:'), hlayout)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        self.setLayout(layout)

    def set_defaults(self, settings: ScalarColormapModel.Properties):
        self.combo_cmap_name.setCurrentText(settings.colormap_name)
        self.spinner_scalar_min.setMinimum(-9999)
        self.spinner_scalar_max.setMaximum(9999)
        self.spinner_scalar_min.setMaximum(settings.scalar_range[1])
        self.spinner_scalar_max.setMinimum(settings.scalar_range[0])
        self.spinner_scalar_min.setValue(settings.scalar_range[0])
        self.spinner_scalar_max.setValue(settings.scalar_range[1])
        self.spinner_opacity.setValue(settings.opacity)

    def get_settings(self, scalar_name: str):
        return ScalarColormapModel.Properties(
            scalar_name, self.combo_cmap_name.currentText(),
            self.spinner_opacity.value(), (self.spinner_scalar_min.value(), self.spinner_scalar_max.value())
        )

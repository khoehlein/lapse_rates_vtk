from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QToolButton, QColorDialog, QVBoxLayout, QPushButton, QComboBox, QStackedLayout, \
    QLabel, QSpinBox, QFormLayout, QDoubleSpinBox, QGridLayout, QCheckBox
from matplotlib import pyplot as plt

from src.model.visualization.colors import UniformColormap, SequentialColormap, DivergingColormap, numpy_to_qcolor


class SelectColorButton(QPushButton):

    color_changed = pyqtSignal(QColor)
    size_changed = pyqtSignal(QSize)

    def __init__(self, color: QColor = None, parent=None):
        super().__init__(parent)
        if color is None:
            color = QColor(0, 0, 0)
        self.current_color = color
        # self.display_button = QToolButton(self)
        self._update_button_icon()
        self.clicked.connect(self._select_color)

    def set_current_color(self, color: QColor):
        self.current_color = color
        self._update_button_icon()
        return self

    def resizeEvent(self, a0=None):
        out = super().resizeEvent(a0)
        self._update_button_icon()
        return out

    def _update_button_icon(self):
        pixmap = QPixmap(12, 12)
        pixmap.fill(self.current_color)
        self.setIcon(QIcon(pixmap))

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
            self._update_button_icon()
            self.color_changed.emit(self.current_color)


class UniformColorSelector(QWidget):

    colormap_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_select_color = SelectColorButton(QColor(0, 0, 0))
        self.button_select_color.setText('Select color')
        layout = QVBoxLayout()
        layout.addWidget(self.button_select_color)
        layout.addStretch()
        self.setLayout(layout)

    def get_colormap(self):
        return UniformColormap(self.button_select_color.current_color)


class ScalarColormapSelector(QWidget):

    colormap_changed = pyqtSignal()

    DIVERGING_CMAPS = [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    ]
    SEQUENTIAL_CMAPS = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_vmin = QDoubleSpinBox()
        self.spinner_vmin.setPrefix('min: ')
        self.spinner_vmax = QDoubleSpinBox()
        self.spinner_vmax.setPrefix('max: ')
        self.spinner_center = QDoubleSpinBox()
        self.spinner_center.setPrefix('center: ')
        self.spinner_vmin.setMinimum(-99999)
        self.spinner_vmax.setMaximum(99999)
        self.spinner_vmax.valueChanged.connect(self.spinner_vmin.setMaximum)
        self.spinner_vmin.valueChanged.connect(self.spinner_vmax.setMinimum)
        self.spinner_vmin.valueChanged.connect(self.spinner_center.setMinimum)
        self.spinner_vmax.valueChanged.connect(self.spinner_center.setMaximum)
        self.spinner_vmin.setValue(0.)
        self.spinner_vmax.setValue(1.)
        self.spinner_center.setValue(0.5)
        self.button_color_below = SelectColorButton(QColor(0, 0, 0))
        self.button_color_below.setText('Color below')
        self.button_color_above = SelectColorButton(QColor(255, 255, 255))
        self.button_color_above.setText('Color above')
        self.combo_cmap_name = QComboBox()
        self.combo_cmap_name.addItems(self.DIVERGING_CMAPS + self.SEQUENTIAL_CMAPS)
        self.combo_cmap_name.currentIndexChanged.connect(self._toggle_center_visibility)
        self.combo_cmap_name.currentIndexChanged.connect(self._update_outlier_colors)
        self.checkbox_block_outliers = QCheckBox('Block outlier color')
        self._set_layout()

    def _toggle_center_visibility(self, index: int):
        self.spinner_center.setDisabled(index >= len(self.DIVERGING_CMAPS))

    def _update_outlier_colors(self):
        if not self.checkbox_block_outliers.isChecked():
            cmap_name = self.combo_cmap_name.currentText()
            cmap = plt.get_cmap(cmap_name)
            color_min = numpy_to_qcolor(cmap.get_under())
            color_max = numpy_to_qcolor(cmap.get_over())
            self.button_color_below.set_current_color(color_min)
            self.button_color_above.set_current_color(color_max)

    def get_colormap(self):
        current_idx = self.combo_cmap_name.currentIndex()
        if current_idx >= len(self.DIVERGING_CMAPS):
            return SequentialColormap(
                self.combo_cmap_name.currentText(),
                self.spinner_vmin.value(),
                self.spinner_vmax.value(),
                color_below_range=self.button_color_below.current_color,
                color_above_range=self.button_color_above.current_color,
            )
        return DivergingColormap(
            self.combo_cmap_name.currentText(),
            self.spinner_vmin.value(),
            self.spinner_vmax.value(),
            self.spinner_center.value(),
            color_below_range=self.button_color_below.current_color,
            color_above_range=self.button_color_above.current_color,
        )

    def _set_layout(self):
        layout = QGridLayout()
        layout.addWidget(self.combo_cmap_name, 0, 0, 1, 4)
        layout.addWidget(self.spinner_vmin, 1, 0, 1, 2)
        layout.addWidget(self.spinner_vmax, 1, 2, 1, 2)
        layout.addWidget(self.button_color_below, 2, 0, 1, 2)
        layout.addWidget(self.button_color_above, 2, 2, 1, 2)
        layout.addWidget(self.spinner_center, 3, 1, 1, 2)
        layout.addWidget(self.checkbox_block_outliers, 4, 1, 1, 4)
        self.setLayout(layout)

    def load_defaults(self, cmap):
        self.combo_cmap_name.setCurrentText(cmap.name)
        self.spinner_vmin.setMaximum(float(cmap.vmax))
        self.spinner_vmin.setValue(float(cmap.vmin))
        self.spinner_vmax.setValue(float(cmap.vmax))
        self.button_color_below.current_color = cmap.color_below_range
        self.button_color_above.current_color = cmap.color_above_range
        if isinstance(cmap, DivergingColormap):
            self.spinner_center.setValue(float(cmap.center))
        else:
            self.spinner_center.setValue((float(cmap.vmax) + float(cmap.vmin)) / 2.)


class ColorSelectionMenu(QWidget):

    colormap_changed = pyqtSignal()
    color_scalar_changed = pyqtSignal()

    SCALARS = ['lapse_rate', 't2m_o1280', 't2m_o8000', 't2m_difference']
    DEFAULTS = {
        'lapse_rate': DivergingColormap('RdBu', -13.5, 13.5, -6.5),
        't2m_o1280': SequentialColormap('YlOrBr', 265., 315.),
        't2m_o8000': SequentialColormap('YlOrBr', 265., 315.),
        't2m_difference': DivergingColormap('RdBu', -20., 20., 0.)
    }
    LABELS = {
        'lapse_rate': 'Lapse rate',
        't2m_o1280': 'T2m (O1280)',
        't2m_o8000': 'T2m (O8000)',
        't2m_difference': 'T2m (difference)',
    }

    def __init__(self, add_o8000=True, parent=None):
        super().__init__(parent)
        self.combo_color_type = QComboBox()
        self.interface_stack = QStackedLayout()
        self.menu_uniform_color = UniformColorSelector(self)
        self.combo_color_type.addItem('Uniform')
        self.interface_stack.addWidget(self.menu_uniform_color)
        self.menu_uniform_color.colormap_changed.connect(self.colormap_changed)
        self.colormaps = {}
        scalar_names = ['lapse_rate', 't2m_o1280']
        if add_o8000:
            scalar_names += ['t2m_o8000', 't2m_difference']
        for scalar_name in scalar_names:
            selector = ScalarColormapSelector(self)
            self.combo_color_type.addItem(self.LABELS[scalar_name])
            self.interface_stack.addWidget(selector)
            self.colormaps[scalar_name] = selector
            selector.load_defaults(self.DEFAULTS[scalar_name])
            selector.colormap_changed.connect(self.colormap_changed)
        self.combo_color_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_color_type.currentIndexChanged.connect(self.colormap_changed.emit)
        self._set_layout()

    def get_current_scalar(self):
        return self.combo_color_type.currentText()

    def get_colormap(self):
        colormap = self.interface_stack.currentWidget().get_colormap()
        selected = self.combo_color_type.currentIndex()
        if selected > 0:
            colormap.scalar_name = self.SCALARS[selected - 1]
        return colormap

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_color_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        self.setLayout(layout)


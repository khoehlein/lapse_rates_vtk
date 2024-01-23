import math
from typing import Any

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QLineEdit, QSlider, QLabel, QWidget, QVBoxLayout, QDoubleSpinBox, QGridLayout, QHBoxLayout


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class DoubleSliderWrapper(QWidget):

    value_changed = pyqtSignal(float)

    def __init__(self, vmin: float = 0., vmax: float = 1., steps: int = 128, parent=None):
        super().__init__(parent)
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(steps))
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._steps = int(steps)
        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def value(self):
        return self._int_value_to_float(self.slider.value())

    def _int_value_to_float(self, value: int):
        return (value / self._steps * (self._vmax - self._vmin)) + self._vmin

    def _float_value_to_int(self, value: float):
        return int(round((value - self._vmin) / (self._vmax - self._vmin) * self._steps))

    def set_value(self, value: float):
        self.slider.setValue(self._float_value_to_int(value))
        self.value_changed.emit(self.value())


class LogDoubleSliderWrapper(DoubleSliderWrapper):

    def __init__(self, vmin: float = 1., vmax: float = 100., steps: int = 128, parent=None):
        super().__init__(vmin, vmax, steps, parent)
        self._log_vmin = math.log(vmin)
        self._log_vmax = math.log(vmax)

    def _int_value_to_float(self, value: int):
        return math.exp((value / self._steps * (self._log_vmax - self._log_vmin)) + self._log_vmin)

    def _float_value_to_int(self, value: float):
        return int(round((math.log(value) - self._log_vmin) / (self._log_vmax - self._log_vmin) * self._steps))


class DoubleSliderSpinner(QWidget):

    def __init__(self, vmin: float = 0., vmax: float = 1., steps: int = 128, width: int = 2, parent=None) -> None:
        super().__init__(parent)
        self.slider_wrapper = DoubleSliderWrapper(vmin, vmax, steps, self)
        self.spinbox = QDoubleSpinBox(value=vmin, minimum=vmin, maximum=vmax, decimals=2)
        layout = QGridLayout()
        layout.addWidget(self.spinbox, 0, 0, 1, 1)
        layout.addWidget(self.slider_wrapper, 0, 1, width - 1, 1)
        self.setLayout(layout)
        self.slider_wrapper.slider.valueChanged.connect(self.update_spinbox)
        self.spinbox.valueChanged.connect(self.update_value)

    def update_spinbox(self, value: float) -> None:
        self.spinbox.setValue(self.slider_wrapper.value())

    def update_value(self, value: float) -> None:
        self.slider_wrapper.slider.blockSignals(True)
        self.slider_wrapper.set_value(self.spinbox.value())
        self.slider_wrapper.slider.blockSignals(False)

    def value(self) -> float:
        return self.spinbox.value()

    def set_value(self, new_value: float) -> None:
        self.slider_wrapper.set_value(new_value)


class LogDoubleSliderSpinner(QWidget):

    def __init__(self, vmin: float = 1., vmax: float = 100., steps: int = 128, width: int = 2, parent=None) -> None:
        super().__init__(parent)
        self.slider_wrapper = DoubleSliderWrapper(vmin, vmax, steps, self)
        self.spinbox = QDoubleSpinBox(value=vmin, minimum=vmin, maximum=vmax, decimals=2)
        layout = QGridLayout()
        layout.addWidget(self.spinbox, 0, 0, 1, 1)
        layout.addWidget(self.slider_wrapper, 0, 1, width - 1, 1)
        self.setLayout(layout)
        self.slider_wrapper.slider.valueChanged.connect(self.update_spinbox)
        self.spinbox.valueChanged.connect(self.update_value)

    def update_spinbox(self, value: float) -> None:
        self.spinbox.setValue(self.slider_wrapper.value())

    def update_value(self, value: float) -> None:
        self.slider_wrapper.slider.blockSignals(True)
        self.slider_wrapper.set_value(self.spinbox.value())
        self.slider_wrapper.slider.blockSignals(False)

    def value(self) -> float:
        return self.spinbox.value()

    def set_value(self, new_value: float) -> None:
        self.slider_wrapper.set_value(new_value)


class RangeSpinner(object):

    def __init__(
            self,
            parent: QtWidgets.QWidget,
            default_min: float, default_max: float,
            global_min: float, global_max: float,
            step=0.5,
    ):
        self.global_min = global_min
        self.global_max = global_max
        self.step = step
        self.min_spinner = QtWidgets.QDoubleSpinBox(parent)
        self.min_spinner.setValue(default_min)
        self.min_spinner.setRange(self.global_min, self.global_max - self.step)
        self.min_spinner.setPrefix('min: ')
        self.max_spinner = QtWidgets.QDoubleSpinBox(parent)
        self.max_spinner.setValue(default_max)
        self.max_spinner.setRange(self.global_min + self.step, self.global_max)
        self.max_spinner.setPrefix('max: ')
        self.min_spinner.valueChanged.connect(self._update_max_spinner)
        self.max_spinner.valueChanged.connect(self._update_min_spinner)

    def _update_max_spinner(self):
        value = self.max_spinner.value()
        new_min_value = self.min_spinner.value()
        if value <= new_min_value:
            self.max_spinner.setValue(min(new_min_value + self.step, self.global_max))

    def _update_min_spinner(self):
        value = self.min_spinner.value()
        new_max_value = self.max_spinner.value()
        if value >= new_max_value:
            self.min_spinner.setValue(max(new_max_value - self.step, self.global_min))

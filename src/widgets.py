import math

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QLineEdit, QSlider


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


class LogSlider(QtWidgets.QWidget):

    def __init__(self, min: float, max: float, steps: int = 128, parent=None):
        super().__init__(parent)
        self.min = float(min)
        self.max = float(max)
        self.steps = int(steps)
        self._log_min = math.log(self.min)
        self._log_max = math.log(self.max)
        self.display = QLineEdit()
        validator = QDoubleValidator()
        validator.setRange(self.min, self.max)
        self.display.setValidator(validator)
        self.display.setAlignment(QtCore.Qt.AlignRight)
        self.display.setMaximumWidth(64)
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.steps)
        self.slider.sliderMoved.connect(self._update_display_on_slider_change)
        self.display.editingFinished.connect(self._update_slider_on_display_change)
        self.display.setText('4000.00')
        self._update_slider_on_display_change()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.display, 0, 0, 1, 1)
        layout.addWidget(self.slider, 0, 1, 1, 4)
        self.setLayout(layout)

    def _get_slider_value(self):
        return math.exp(self.slider.value() / self.steps * (self._log_max - self._log_min) + self._log_min)

    def _update_display_on_slider_change(self):
        self.display.setText(f'{self._get_slider_value():.2f}')

    def _update_slider_on_display_change(self):
        slider_value = (math.log(float(self.display.text())) - self._log_min) / (self._log_max - self._log_min)
        slider_value = round(self.steps * slider_value)
        self.slider.setValue(int(slider_value))

    def get_value(self):
        return float(self.display.text())

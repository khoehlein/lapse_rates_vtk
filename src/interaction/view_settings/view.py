from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QSlider, QFormLayout, QLabel

from src.interaction.background_color.view import SelectColorButton


class ViewSettingsView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider_z_scale = QSlider(Qt.Horizontal, parent=self)
        self.slider_z_scale.setMinimum(0)
        self.slider_z_scale.setMinimum(128)
        self.slider_z_scale.setValue(64)
        self.button_background_color = SelectColorButton(QColor(0, 0, 0), self)
        self.button_background_color.setText(' Select background color')
        layout = QFormLayout(self)
        layout.addRow(QLabel('Vertical scale:'), self.slider_z_scale)
        layout.addRow(QLabel('Background color:'), self.button_background_color)
        self.setLayout(layout)

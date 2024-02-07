from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QStackedLayout, QPushButton, QVBoxLayout

from src.interaction.visualizations.interface import VisualizationSettingsView


class SceneSettingsView(QWidget):

    new_interface_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.combo_visual_selection = QComboBox()
        self.interface_stack = QStackedLayout()
        self.interfaces = {}
        self.combo_visual_selection.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.button_new = QPushButton('New visualization')
        self.button_new.clicked.connect(self.new_interface_requested.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_visual_selection)
        layout.addLayout(self.interface_stack)
        layout.addWidget(self.button_new)
        self.setLayout(layout)

    def register_settings_view(self, widget: VisualizationSettingsView, label=None):
        key = widget.key
        if label is None:
            label = key
        self.combo_visual_selection.addItem(label)
        self.interface_stack.addWidget(widget)
        self.interfaces[key] = widget
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QStackedLayout, QPushButton, QVBoxLayout, QHBoxLayout

from src.interaction.visualizations.interface import VisualizationSettingsView


class SceneSettingsView(QWidget):

    new_interface_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.combo_visual_selection = QComboBox()
        self.combo_visual_selection.setEnabled(False)
        self._combo_index_by_key = []
        self.interface_stack = QStackedLayout()
        self.interfaces = {}
        self.combo_visual_selection.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.button_new = QPushButton('New')
        self.button_reset = QPushButton('Reset')
        self.button_new.clicked.connect(self.new_interface_requested.emit)
        self.button_reset.clicked.connect(self.reset_requested.emit)
        self._set_layout()

    def _set_layout(self):
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.button_new)
        hlayout.addWidget(self.button_reset)
        layout = QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.combo_visual_selection)
        layout.addLayout(self.interface_stack)
        self.setLayout(layout)

    def register_settings_view(self, widget: VisualizationSettingsView, label=None):
        self.combo_visual_selection.setEnabled(True)
        key = widget.key
        if label is None:
            label = key
        self.combo_visual_selection.addItem(label)
        self._combo_index_by_key.append(key)
        self.interface_stack.addWidget(widget)
        self.interfaces[key] = widget
        self.combo_visual_selection.setCurrentIndex(len(self._combo_index_by_key) - 1)
        return self

    def remove_settings_view(self, key: str):
        item_index = self._combo_index_by_key.index(key)
        self.combo_visual_selection.removeItem(item_index)
        self._combo_index_by_key.pop(item_index)
        widget = self.interfaces[key]
        self.interface_stack.removeWidget(widget)
        del self.interfaces[key]
        if not len(self.interfaces):
            self.combo_visual_selection.setEnabled(False)
        return self

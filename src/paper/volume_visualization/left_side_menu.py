from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QLabel

from src.paper.volume_visualization.color_lookup import ADCLSettingsView
from src.paper.volume_visualization.volume import VolumeVisualSettingsView, SceneScalingSettingsView


class LeftSideMenu(QDockWidget):

    def __init__(self, parent: QObject = None):
        super().__init__()
        self.setWindowTitle('Settings')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._build_scroll_area()
        self._populate_scroll_area()

    def _build_scroll_area(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_contents = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_area_contents)
        self.setWidget(self.scroll_area)

    def _populate_scroll_area(self):
        self.colormap_settings = ADCLSettingsView(self.scroll_area_contents)
        self.volume_vis_settings = VolumeVisualSettingsView(self.scroll_area_contents)
        self.scaling_settings = SceneScalingSettingsView(self.scroll_area_contents)
        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(self.colormap_settings)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(self.volume_vis_settings)
        layout.addWidget(QLabel('Scale settings'))
        layout.addWidget(self.scaling_settings)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)
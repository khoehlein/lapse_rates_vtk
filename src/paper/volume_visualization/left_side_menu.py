from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QLabel, QTabWidget

from src.paper.volume_visualization.color_lookup import ADCLSettingsView
from src.paper.volume_visualization.reference_grid import ReferenceGridSettingsView
from src.paper.volume_visualization.scaling import SceneScalingSettingsView
from src.paper.volume_visualization.volume import ScalarVolumeSettingsView


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
        self.scaling_settings = SceneScalingSettingsView(self.scroll_area_contents)
        self.vis_settings = QTabWidget(self.scroll_area_contents)
        self._build_gradient_tab()
        self._build_volume_grid_tab()
        self._build_surface_o1280_tab()
        self._build_surface_o8000_tab()
        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(QLabel('Scale settings'))
        layout.addWidget(self.scaling_settings)
        layout.addWidget(QLabel('Visualization settings'))
        layout.addWidget(self.vis_settings)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)

    def _build_gradient_tab(self):
        self.gradient_color_settings = ADCLSettingsView(self.scroll_area_contents)
        self.gradient_representation_settings = ScalarVolumeSettingsView(self.scroll_area_contents)
        gradient_tab = QWidget(self.vis_settings)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(self.gradient_color_settings)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(self.gradient_representation_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.vis_settings.addTab(gradient_tab, 'Gradients')

    def _build_volume_grid_tab(self):
        self.volume_mesh_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.vis_settings)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.volume_mesh_settings)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.vis_settings.addTab(grid_tab, 'Reference mesh (volume)')

    def _build_surface_o1280_tab(self):
        self.surface_settings_o1280 = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.vis_settings)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.surface_settings_o1280)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.vis_settings.addTab(grid_tab, 'Surface reference (O1280)')

    def _build_surface_o8000_tab(self):
        self.surface_settings_o8000 = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.vis_settings)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.surface_settings_o8000)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.vis_settings.addTab(grid_tab, 'Surface reference (O8000)')

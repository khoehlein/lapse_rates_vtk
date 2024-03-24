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
        self._build_model_data_tabs()
        self._build_reference_data_tabs()
        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(QLabel('Scale settings'))
        layout.addWidget(self.scaling_settings)
        layout.addWidget(QLabel('Visualization settings'))
        layout.addWidget(self.vis_settings)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)

    def _build_model_data_tabs(self):
        container = QWidget(self.vis_settings)
        self.model_data_tabs = QTabWidget(container)
        self._build_gradient_volume_tab()
        self._build_temperature_volume_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.model_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings.addTab(container, 'Model data')

    def _build_reference_data_tabs(self):
        container = QWidget(self.vis_settings)
        self.reference_data_tabs = QTabWidget(container)
        self._build_volume_grid_tab()
        self._build_surface_o1280_tab()
        self._build_surface_o8000_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.reference_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings.addTab(container, 'Reference data')

    def _build_gradient_volume_tab(self):
        self.gradient_color_settings = ADCLSettingsView(self.scroll_area_contents)
        self.gradient_volume_settings = ScalarVolumeSettingsView(self.scroll_area_contents)
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(self.gradient_color_settings)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(self.gradient_volume_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.model_data_tabs.addTab(gradient_tab, 'Gradients')

    def _build_temperature_volume_tab(self):
        self.temperature_volume_settings = ScalarVolumeSettingsView(self.scroll_area_contents)
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(self.temperature_volume_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.model_data_tabs.addTab(gradient_tab, 'Temperature')

    def _build_volume_grid_tab(self):
        self.volume_mesh_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.volume_mesh_settings)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Volume (model levels)')

    def _build_surface_o1280_tab(self):
        self.surface_settings_o1280 = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.surface_settings_o1280)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Surface (O1280)')

    def _build_surface_o8000_tab(self):
        self.surface_settings_o8000 = ReferenceGridSettingsView(self.scroll_area_contents)
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(self.surface_settings_o8000)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Surface (O8000)')

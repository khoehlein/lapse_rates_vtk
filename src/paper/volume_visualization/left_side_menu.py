from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QLabel, QTabWidget

from src.paper.volume_visualization.color_lookup import ADCLSettingsView
from src.paper.volume_visualization.station import StationScalarSettingsView
from src.paper.volume_visualization.volume_reference_grid import ReferenceGridSettingsView
from src.paper.volume_visualization.scaling import SceneScalingSettingsView
from src.paper.volume_visualization.volume import VolumeScalarSettingsView


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
        self.vis_settings_tabs = QTabWidget(self.scroll_area_contents)
        self.color_settings_views = {}
        self.vis_settings_views = {}
        self._build_model_data_tabs()
        self._build_station_data_tabs()
        self._build_terrain_data_tabs()
        self._build_reference_data_tabs()
        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(QLabel('Scale settings'))
        layout.addWidget(self.scaling_settings)
        layout.addWidget(QLabel('Visualization settings'))
        layout.addWidget(self.vis_settings_tabs)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)

    def _build_model_data_tabs(self):
        container = QWidget(self.vis_settings_tabs)
        self.model_data_tabs = QTabWidget(container)
        self._build_gradient_volume_tab()
        self._build_temperature_volume_tab()
        self._build_t2m_volume_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.model_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Model')

    def _build_station_data_tabs(self):
        container = QWidget(self.vis_settings_tabs)
        self.station_data_tabs = QTabWidget(container)
        self._build_observation_tab()
        self._build_prediction_tab()
        self._build_error_tab()
        self._build_station_gradient_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.station_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Station')

    def _build_terrain_data_tabs(self):
        container = QWidget(self.vis_settings_tabs)
        self.terrain_data_tabs = QTabWidget(container)
        self._build_lsm_o1280_tab()
        self._build_lsm_o8000_tab()
        self._build_z_o1280_tab()
        self._build_z_o8000_tab()
        self._build_station_offset_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.terrain_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Terrain')

    def _build_reference_data_tabs(self):
        container = QWidget(self.vis_settings_tabs)
        self.reference_data_tabs = QTabWidget(container)
        self._build_volume_grid_tab()
        self._build_surface_o1280_tab()
        self._build_surface_o8000_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.reference_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Mesh')

    def _build_gradient_volume_tab(self):
        key = 'model_grad_t'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.model_data_tabs.addTab(gradient_tab, 'Gradients')

    def _build_temperature_volume_tab(self):
        key = 'model_t'
        volume_settings = VolumeScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = volume_settings
        temperature_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(temperature_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addStretch()
        temperature_tab.setLayout(layout)
        self.model_data_tabs.addTab(temperature_tab, 'T')

    def _build_observation_tab(self):
        key = 'station_t_obs'
        scalar_settings = StationScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = scalar_settings
        observation_tab = QWidget(self.station_data_tabs)
        layout = QVBoxLayout(observation_tab)
        layout.addWidget(QLabel('Marker properties'))
        layout.addWidget(scalar_settings)
        layout.addStretch()
        observation_tab.setLayout(layout)
        self.station_data_tabs.addTab(observation_tab, 'Observation')

    def _build_prediction_tab(self):
        key = 'station_t_pred'
        scalar_settings = StationScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = scalar_settings
        observation_tab = QWidget(self.station_data_tabs)
        layout = QVBoxLayout(observation_tab)
        layout.addWidget(QLabel('Marker properties'))
        layout.addWidget(scalar_settings)
        layout.addStretch()
        observation_tab.setLayout(layout)
        self.station_data_tabs.addTab(observation_tab, 'Prediction')

    def _build_error_tab(self):
        key = 'station_t_diff'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        scalar_settings = StationScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = scalar_settings
        self.color_settings_views[key] = color_settings
        observation_tab = QWidget(self.station_data_tabs)
        layout = QVBoxLayout(observation_tab)
        layout.addWidget(QLabel('Marker properties'))
        layout.addWidget(scalar_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        observation_tab.setLayout(layout)
        self.station_data_tabs.addTab(observation_tab, 'T diff.')

    def _build_station_offset_tab(self):
        key = 'station_offset'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        scalar_settings = StationScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = scalar_settings
        self.color_settings_views[key] = color_settings
        observation_tab = QWidget(self.terrain_data_tabs)
        layout = QVBoxLayout(observation_tab)
        layout.addWidget(QLabel('Marker properties'))
        layout.addWidget(scalar_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        observation_tab.setLayout(layout)
        self.terrain_data_tabs.addTab(observation_tab, 'Station offset')

    def _build_station_gradient_tab(self):
        key = 'station_grad_t'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = StationScalarSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.station_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Marker properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.station_data_tabs.addTab(gradient_tab, 'Gradients')

    def _build_t2m_volume_tab(self):
        key = 'model_t2m'
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self.vis_settings_views[key] = volume_settings
        t2m_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(t2m_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addStretch()
        t2m_tab.setLayout(layout)
        self.model_data_tabs.addTab(t2m_tab, 'T2m')

    def _build_volume_grid_tab(self):
        key = 'model_grid'
        mesh_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = mesh_settings
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(mesh_settings)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Volume (model levels)')

    def _build_surface_o1280_tab(self):
        key = 'surface_grid_o1280'
        surface_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = surface_settings
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(surface_settings)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Surface (O1280)')

    def _build_surface_o8000_tab(self):
        key = 'surface_grid_o8000'
        surface_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self.vis_settings_views[key] = surface_settings
        grid_tab = QWidget(self.reference_data_tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Mesh settings'))
        layout.addWidget(surface_settings)
        layout.addStretch()
        grid_tab.setLayout(layout)
        self.reference_data_tabs.addTab(grid_tab, 'Surface (O8000)')

    def _build_lsm_o1280_tab(self):
        key = 'lsm_o1280'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.terrain_data_tabs.addTab(gradient_tab, 'LSM (O1280)')

    def _build_lsm_o8000_tab(self):
        key = 'lsm_o8000'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.terrain_data_tabs.addTab(gradient_tab, 'LSM (O8000)')

    def _build_z_o1280_tab(self):
        key = 'z_o1280'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.terrain_data_tabs.addTab(gradient_tab, 'Z (O1280)')

    def _build_z_o8000_tab(self):
        key = 'z_o8000'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self.vis_settings_views[key] = volume_settings
        self.color_settings_views[key] = color_settings
        gradient_tab = QWidget(self.model_data_tabs)
        layout = QVBoxLayout(gradient_tab)
        layout.addWidget(QLabel('Volume properties'))
        layout.addWidget(volume_settings)
        layout.addWidget(QLabel('Transfer function'))
        layout.addWidget(color_settings)
        layout.addStretch()
        gradient_tab.setLayout(layout)
        self.terrain_data_tabs.addTab(gradient_tab, 'Z (O8000)')

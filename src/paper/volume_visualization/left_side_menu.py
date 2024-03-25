from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QLabel, QTabWidget

from src.interaction.plotter_controls.view import PlotterSettingsView
from src.paper.volume_visualization.color_lookup import ADCLSettingsView
from src.paper.volume_visualization.scaling import SceneScalingSettingsView
from src.paper.volume_visualization.station import StationScalarSettingsView
from src.paper.volume_visualization.station_reference import StationSiteReferenceSettingsView, \
    StationOnTerrainReferenceSettingsView
from src.paper.volume_visualization.volume_reference_grid import ReferenceGridSettingsView
from src.paper.volume_visualization.volume import VolumeScalarSettingsView


class RightDockMenu(QDockWidget):

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._build_scroll_area()
        self._populate_scroll_area()

    def _build_scroll_area(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_contents = QTabWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_area_contents)
        self.setWidget(self.scroll_area)

    def _populate_scroll_area(self):
        self._build_vis_settings_tab()
        self._build_plotter_settings_tab()

    def _build_vis_settings_tab(self):
        self.vis_settings_tabs = QTabWidget(self.scroll_area_contents)
        self.color_settings_views = {}
        self.vis_settings_views = {}
        self._build_model_data_tabs()
        self._build_station_data_tabs()
        self._build_terrain_data_tabs()
        self._build_reference_data_tabs()
        container = QWidget(self.scroll_area_contents)
        layout = QVBoxLayout()
        layout.addWidget(self.vis_settings_tabs)
        layout.addStretch(2)
        container.setLayout(layout)
        self.scroll_area_contents.addTab(container, 'Visualization settings')

    def _build_plotter_settings_tab(self):
        container = QWidget(self.scroll_area_contents)
        self.plotter_settings = PlotterSettingsView(container)
        self.scaling_settings = SceneScalingSettingsView(container)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Plotter settings'))
        layout.addWidget(self.plotter_settings)
        layout.addWidget(QLabel('Scale settings'))
        layout.addWidget(self.scaling_settings)
        layout.addStretch(2)
        container.setLayout(layout)
        self.scroll_area_contents.addTab(container, 'Scene settings')

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
        self._build_station_gradient_tab()
        self._build_error_tab()
        layout = QVBoxLayout()
        layout.addWidget(self.station_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Station')

    def _build_terrain_data_tabs(self):
        container = QWidget(self.vis_settings_tabs)
        self.terrain_data_tabs = QTabWidget(container)
        self._build_station_offset_tab()
        self._build_z_o1280_tab()
        self._build_lsm_o1280_tab()
        self._build_z_o8000_tab()
        self._build_lsm_o8000_tab()
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
        self._build_station_ref_tabs()
        layout = QVBoxLayout()
        layout.addWidget(self.reference_data_tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Mesh')

    def _build_station_ref_tabs(self):
        self._make_tab_without_color_lookup(
            'station_sites',
            StationSiteReferenceSettingsView(self),
            'Mesh properties',
            self.reference_data_tabs,
            'Station sites'
        )
        self._make_tab_without_color_lookup(
            'station_on_terrain',
            StationOnTerrainReferenceSettingsView(self),
            'Mesh properties',
            self.reference_data_tabs,
            'Station on terrain'
        )

    def _make_tab_with_color_lookup(
            self,
            key: str,
            vis_settings_view: QWidget, color_settings_view: QWidget,
            vis_settings_label: str,
            tabs: QTabWidget, tab_label: str
    ):
        self.vis_settings_views[key] = vis_settings_view
        self.color_settings_views[key] = color_settings_view
        tab_widget = QTabWidget(tabs)
        w1 = QWidget(tab_widget)
        l1 = QVBoxLayout()
        l1.addWidget(vis_settings_view)
        l1.addStretch(2)
        w1.setLayout(l1)
        tab_widget.addTab(w1, vis_settings_label)
        w2 = QWidget(tab_widget)
        l2 = QVBoxLayout()
        l2.addWidget(color_settings_view)
        l2.addStretch(2)
        w2.setLayout(l2)
        tab_widget.addTab(w2, 'Transfer function')
        container = QWidget(tabs)
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        layout.addStretch()
        container.setLayout(layout)
        tabs.addTab(container, tab_label)

    def _make_tab_without_color_lookup(
            self,
            key: str,
            vis_settings_view: QWidget,
            vis_settings_label: str,
            tabs: QTabWidget, tab_label: str
    ):
        self.vis_settings_views[key] = vis_settings_view
        tab_widget = QWidget(tabs)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(vis_settings_label))
        layout.addWidget(vis_settings_view)
        layout.addStretch()
        tab_widget.setLayout(layout)
        tabs.addTab(tab_widget, tab_label)

    def _build_gradient_volume_tab(self):
        key = 'model_grad_t'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        vis_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_with_color_lookup(
            key, vis_settings, color_settings,
            'Volume properties',
            self.model_data_tabs,
            'Gradients'
        )

    def _build_temperature_volume_tab(self):
        key = 'model_t'
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, volume_settings,
            'Volume properties',
            self.model_data_tabs, 'T'
        )

    def _build_observation_tab(self):
        key = 'station_t_obs'
        scalar_settings = StationScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, scalar_settings,
            'Marker properties',
            self.station_data_tabs, 'Observations'
        )

    def _build_prediction_tab(self):
        key = 'station_t_pred'
        scalar_settings = StationScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, scalar_settings,
            'Marker properties',
            self.station_data_tabs, 'Predictions'
        )

    def _build_error_tab(self):
        key = 'station_t_diff'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        scalar_settings = StationScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_with_color_lookup(
            key, scalar_settings, color_settings,
            'Marker properties',
            self.station_data_tabs, 'T diff.'
        )

    def _build_station_offset_tab(self):
        key = 'station_offset'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        scalar_settings = StationScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_with_color_lookup(
            key, scalar_settings, color_settings,
            'Marker properties',
            self.terrain_data_tabs, 'Station offset'
        )

    def _build_station_gradient_tab(self):
        key = 'station_grad_t'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = StationScalarSettingsView(parent=self.scroll_area_contents)
        self._make_tab_with_color_lookup(
            key, volume_settings, color_settings,
            'Marker properties',
            self.station_data_tabs, 'Gradients'
        )

    def _build_t2m_volume_tab(self):
        key = 'model_t2m'
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self._make_tab_without_color_lookup(
            key, volume_settings,
            'Surface properties',
            self.model_data_tabs, 'T2m'
        )

    def _build_volume_grid_tab(self):
        key = 'model_grid'
        mesh_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, mesh_settings,
            'Mesh properties',
            self.reference_data_tabs, 'Volume (model levels)'
        )

    def _build_surface_o1280_tab(self):
        key = 'surface_grid_o1280'
        surface_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, surface_settings,
            'Mesh properties',
            self.reference_data_tabs, 'Surface (O1280)'
        )

    def _build_surface_o8000_tab(self):
        key = 'surface_grid_o8000'
        surface_settings = ReferenceGridSettingsView(self.scroll_area_contents)
        self._make_tab_without_color_lookup(
            key, surface_settings,
            'Mesh properties',
            self.reference_data_tabs, 'Surface (O8000)'
        )

    def _build_lsm_o1280_tab(self):
        key = 'lsm_o1280'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self._make_tab_with_color_lookup(
            key, volume_settings, color_settings,
            'Surface properties',
            self.terrain_data_tabs, 'LSM (O1280)'
        )

    def _build_lsm_o8000_tab(self):
        key = 'lsm_o8000'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self._make_tab_with_color_lookup(
            key, volume_settings, color_settings,
            'Surface properties',
            self.terrain_data_tabs, 'LSM (O8000)'
        )

    def _build_z_o1280_tab(self):
        key = 'z_o1280'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self._make_tab_with_color_lookup(
            key, volume_settings, color_settings,
            'Surface properties',
            self.terrain_data_tabs, 'Z (O1280)'
        )

    def _build_z_o8000_tab(self):
        key = 'z_o8000'
        color_settings = ADCLSettingsView(self.scroll_area_contents)
        volume_settings = VolumeScalarSettingsView(parent=self.scroll_area_contents, use_dvr=False, use_contours=False)
        self._make_tab_with_color_lookup(
            key, volume_settings, color_settings,
            'Surface properties',
            self.terrain_data_tabs, 'Z (O8000)'
        )
import copy
import dataclasses
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from PyQt5.QtCore import QObject, pyqtSignal
import pyvista as pv
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QSpinBox, QPushButton, QFormLayout, QHBoxLayout, QVBoxLayout, \
    QLabel, QTabWidget

from src.model.geometry import Coordinates
from src.paper.volume_visualization.color_lookup import make_elevation_lookup, make_temperature_lookup, \
    make_count_lookup, make_lapse_rate_lookup, ADCLSettingsView, make_lsm_lookup, make_score_lookup, \
    make_temperature_difference_lookup
from src.paper.volume_visualization.lapse_rates.algorithm import LapseRateData
from src.paper.volume_visualization.lapse_rates.clipping import RampClipProperties, RampMinClip, RampClipSettingsView, \
    RampMaxClip, DEFAULT_CLIP_MIN, DEFAULT_CLIP_MAX
from src.paper.volume_visualization.multi_method_visualization import MultiMethodScalarVisualization, \
    MultiMethodVisualizationController
from src.paper.volume_visualization.plotter_slot import PlotterSlot, SurfaceProperties, StationSiteProperties
from src.paper.volume_visualization.scaling import VolumeVisual, ScalingParameters
from src.paper.volume_visualization.station import StationScalarVisualization, StationScalarSettingsView
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.volume import VolumeScalarVisualization, VolumeScalarSettingsView, \
    VolumeRepresentationMode
from src.paper.volume_visualization.volume_data import VolumeData


class LapseRateComponent(Enum):
    LAPSE_RATE_RAW = 'lapse_rate_raw'
    LOWER_CLIPPING = 'lower_clipping'
    UPPER_CLIPPING = 'upper_clipping'
    LAPSE_RATE = 'lapse_rate'
    SCORE = 'score'
    Z_MIN = 'z_min'
    Z_MAX = 'z_max'
    Z_MEAN = 'z_mean'
    Z_NEAREST = 'z_nearest'
    Z_RANGE = 'z_range'
    T2M_MIN = 't2m_min'
    T2M_MAX = 't2m_max'
    T2M_MEAN = 't2m_mean'
    T2M_NEAREST = 't2m_nearest'
    T2M_RANGE = 't2m_range'
    NEIGHBOR_COUNT = 'neighbor_count'
    PREDICTION = 'prediction'
    DIFFERENCE = 'difference'
    CORRECTION = 'correction'


@dataclasses.dataclass
class LapseRateProperties(object):
    radius_km: float = 60
    weight_scale_km: float = 30
    default_lapse_rate: float = -6.5
    min_samples: int = 20
    min_elevation: float = 100
    min_clip: RampClipProperties = DEFAULT_CLIP_MIN
    max_clip: RampClipProperties = DEFAULT_CLIP_MAX
    lsm_threshold: float = 0.5


class SiteMode(Enum):
    STATION = 'station'
    SURFACE = 'surface'


class LapseRateVisualization(VolumeVisual):

    def __init__(
            self,
            lapse_rate_data: LapseRateData,
            site_data: Union[pd.DataFrame, xr.Dataset], station_data: pd.DataFrame,
            plotter: pv.Plotter, properties: LapseRateProperties, scaling: ScalingParameters,
            parent=None
    ):
        super().__init__(parent)

        self.site_data = site_data.copy()
        self.station_data = station_data.copy()
        self.plotter = plotter

        self.properties = properties
        self.scaling = scaling

        self.min_clip = RampMinClip(properties.min_clip)
        self.max_clip = RampMaxClip(properties.max_clip)

        self._coords_sites = Coordinates.from_dataframe(self.site_data).as_xyz().values
        self._coords_stations = Coordinates.from_dataframe(self.station_data).as_xyz().values

        self.lapse_rate_data = lapse_rate_data.update(self.properties)
        self._update_visualization_data()

        self.representations = {}
        self._build_representations()

    def get_plotter(self) -> pv.Plotter:
        return self.plotter

    def _update_visualization_data(self):
        self._update_raw_data()
        self._update_postprocessed_data()

    def _update_raw_data(self):
        self._lapse_rates_at_sites = self.lapse_rate_data.get_data_at_closest_site(self._coords_sites)
        self._lapse_rates_at_stations = self.lapse_rate_data.get_data_at_closest_site(self._coords_stations)
        data = self._lapse_rates_at_sites
        variables = {
            key.value: data[key.value].values
            for key in [
                LapseRateComponent.Z_MIN,
                LapseRateComponent.Z_MAX,
                LapseRateComponent.Z_MEAN,
                LapseRateComponent.Z_NEAREST,
                LapseRateComponent.Z_RANGE,
                LapseRateComponent.T2M_MIN,
                LapseRateComponent.T2M_MAX,
                LapseRateComponent.T2M_MEAN,
                LapseRateComponent.T2M_NEAREST,
                LapseRateComponent.T2M_RANGE,
                LapseRateComponent.NEIGHBOR_COUNT
            ]
        }
        variables['latitude'] = self.site_data['latitude'].values
        variables['longitude'] = self.site_data['longitude'].values
        self._site_data = pd.DataFrame(variables)
        self._station_data = pd.DataFrame({
            'latitude': self.station_data['latitude'].values,
            'longitude': self.station_data['longitude'].values,
            'elevation': self.station_data['elevation'].values,
            'elevation_difference': self.station_data['elevation_difference'].values,
        })

    def _update_postprocessed_data(self):
        data = self._lapse_rates_at_sites
        lapse_rates_raw = data['lapse_rate'].values
        scores = data['score'].values
        lapse_rates, lower_clipping = self.min_clip.clip(lapse_rates_raw, scores, return_thresholds=True)
        lapse_rates, upper_clipping = self.max_clip.clip(lapse_rates, scores, return_thresholds=True)
        num_neighbors = data['neighbor_count']
        z_range = data['z_range'].values
        lapse_rates[np.logical_or(num_neighbors < self.properties.min_samples, z_range < self.properties.min_elevation)] = self.properties.default_lapse_rate
        # dz = (self.site_data['z_surf'].values - data['z_nearest'].values) * 0.001
        # predictions = data['t2m_nearest'].values + lapse_rates * dz
        variables = {
            LapseRateComponent.LAPSE_RATE_RAW.value: lapse_rates_raw,
            LapseRateComponent.LAPSE_RATE.value: lapse_rates,
            LapseRateComponent.LOWER_CLIPPING.value: lower_clipping,
            LapseRateComponent.UPPER_CLIPPING.value: upper_clipping,
            LapseRateComponent.SCORE.value: scores,
        }
        for key, values in variables.items():
            self._site_data[key] = values
        data = self._lapse_rates_at_stations
        lapse_rates = data['lapse_rate'].values
        scores = data['score'].values
        lapse_rates = self.min_clip.clip(lapse_rates, scores, return_thresholds=False)
        lapse_rates = self.max_clip.clip(lapse_rates, scores, return_thresholds=False)
        num_neighbors = data['neighbor_count']
        z_range = data['z_range'].values
        lapse_rates[np.logical_or(num_neighbors < self.properties.min_samples, z_range < self.properties.min_elevation)] = self.properties.default_lapse_rate
        dz = (self.station_data['elevation'].values - data['z_nearest'].values) * 0.001
        corrections = lapse_rates * dz
        predictions = data['t2m_nearest'].values + corrections
        difference = self.station_data['observation'].values - predictions
        self._station_data[LapseRateComponent.PREDICTION.value] = predictions
        self._station_data[LapseRateComponent.DIFFERENCE.value] = difference
        self._station_data[LapseRateComponent.CORRECTION.value] = corrections

    def _build_representation(self, key: LapseRateComponent, visual: VolumeVisual):
        self.representations[key] = visual
        return self

    def _build_representations(self):
        label_suffix = 'surface'
        for key, label in {
            LapseRateComponent.Z_MAX: f'Z (max, {label_suffix})',
            LapseRateComponent.Z_MIN: f'Z (min, {label_suffix})',
            LapseRateComponent.Z_MEAN: f'Z (mean, {label_suffix})',
            LapseRateComponent.Z_RANGE: f'Z range ({label_suffix})',
            LapseRateComponent.Z_NEAREST: f'Z (nearest, {label_suffix})'
        }.items():
            self._build_representation(
                key,
                VolumeScalarVisualization(
                    PlotterSlot(self.plotter, label),
                    VolumeData(
                        self._site_data, self.site_data, scalar_key=key.value,
                        model_level_key=(key.value if key != LapseRateComponent.Z_RANGE else 'z_surf_o1280')
                    ),
                    make_elevation_lookup(),
                    SurfaceProperties(),
                    self.scaling,
                )
            )
        for key, label in {
            LapseRateComponent.T2M_MAX: f'T2m (max, {label_suffix})',
            LapseRateComponent.T2M_MIN: f'T2m (min, {label_suffix})',
            LapseRateComponent.T2M_MEAN: f'T2m (mean, {label_suffix})',
            LapseRateComponent.T2M_RANGE: f'T2m range ({label_suffix})',
            LapseRateComponent.T2M_NEAREST: f'T2m (nearest, {label_suffix})',
        }.items():
            self._build_representation(
                key,
                VolumeScalarVisualization(
                    PlotterSlot(self.plotter, label),
                    VolumeData(
                        self._site_data, self.site_data,
                        scalar_key=key.value, model_level_key='z_surf_o1280'
                    ),
                    make_temperature_lookup(),
                    SurfaceProperties(),
                    self.scaling,
                )
            )
        for key, label in {
            LapseRateComponent.LAPSE_RATE_RAW: f'Lapse rate (estimate, {label_suffix}, K/km)',
            LapseRateComponent.LAPSE_RATE: f'Lapse rate (clipped, {label_suffix}, K/km)',
            LapseRateComponent.LOWER_CLIPPING: f'Lower clip ({label_suffix}, K/km)',
            LapseRateComponent.UPPER_CLIPPING: f'Upper clip ({label_suffix}, K/km)',
        }.items():
            self._build_representation(
                key,
                VolumeScalarVisualization(
                    PlotterSlot(self.plotter,label),
                    VolumeData(
                        self._site_data, self.site_data,
                        scalar_key=key.value,
                        model_level_key='z_surf_o1280'
                    ),
                    make_lapse_rate_lookup(),
                    SurfaceProperties(),
                    self.scaling,
                )
            )
        self._build_representation(
            LapseRateComponent.NEIGHBOR_COUNT,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, f'Neighbor count ({label_suffix})'),
                VolumeData(
                    self._site_data, self.site_data,
                    scalar_key=LapseRateComponent.NEIGHBOR_COUNT.value,
                    model_level_key='z_surf_o1280'
                ),
                make_count_lookup(),
                SurfaceProperties(),
                self.scaling,
            )
        )
        self._build_representation(
            LapseRateComponent.SCORE,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, f'R2 score ({label_suffix})'),
                VolumeData(
                    self._site_data, self.site_data,
                    scalar_key=LapseRateComponent.SCORE.value,
                    model_level_key='z_surf_o1280'
                ),
                make_score_lookup(),
                SurfaceProperties(),
                self.scaling,
            )
        )
        self._build_representation(
            LapseRateComponent.PREDICTION,
            StationScalarVisualization(
                PlotterSlot(self.plotter, f'T2m (adaptive, station)'),
                StationData(
                    self._station_data, self.site_data,
                    scalar_key=LapseRateComponent.PREDICTION.value,
                    compute_gradient=False
                ),
                make_temperature_lookup(),
                StationSiteProperties(),
                self.scaling,
            )
        )
        self._build_representation(
            LapseRateComponent.DIFFERENCE,
            StationScalarVisualization(
                PlotterSlot(self.plotter, f'T2m difference (adaptive, station)'),
                StationData(
                    self._station_data, self.site_data,
                    scalar_key=LapseRateComponent.DIFFERENCE.value,
                    compute_gradient=False
                ),
                make_temperature_difference_lookup(),
                StationSiteProperties(),
                self.scaling,
            )
        )
        self._build_representation(
            LapseRateComponent.CORRECTION,
            StationScalarVisualization(
                PlotterSlot(self.plotter, f'T2m correction (adaptive, station)'),
                StationData(
                    self._station_data, self.site_data,
                    scalar_key=LapseRateComponent.CORRECTION.value,
                    compute_gradient=False
                ),
                make_temperature_difference_lookup(),
                StationSiteProperties(),
                self.scaling,
            )
        )

    def is_visible(self):
        for visual in self.representations.values():
            if visual.is_visible():
                return True
        return False

    def show(self, renderer: bool = True):
        pass

    def clear(self, renderer: bool = True):
        for visual in self.representations.values():
            visual.clear()
        return self

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        for visual in self.representations.values():
            visual.set_scaling(self.scaling, render=render)
        return self

    def set_properties(self, properties: LapseRateProperties, render: bool = True):
        self.properties = properties
        self.lapse_rate_data.update(properties)
        self.min_clip.set_properties(properties.min_clip)
        self.max_clip.set_properties(properties.max_clip)
        self._update_visualization_data()
        self._update_representations()
        if render:
            self.plotter.render()
        return self

    def _update_representations(self):
        station_keys = [LapseRateComponent.PREDICTION, LapseRateComponent.DIFFERENCE, LapseRateComponent.CORRECTION]
        for key, visual in self.representations.items():
            new_data = self._station_data if key in station_keys else self._site_data
            visual.update_data(new_data, render=False)

    def update_postprocessed_data(self, properties: LapseRateProperties, render: bool = True):
        self.properties = properties
        self.min_clip.set_properties(properties.min_clip)
        self.max_clip.set_properties(properties.max_clip)
        self._update_postprocessed_data()
        self._update_representations()
        if render:
            self.plotter.render()
        return self


class LapseRateSettingsView(QWidget):

    settings_changed = pyqtSignal()
    cutoff_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_radius = QDoubleSpinBox(self)
        self.spinner_radius.setRange(20, 180)
        self.spinner_radius.setValue(60)
        self.spinner_radius.setSingleStep(1)
        self.spinner_radius.setSuffix(' km')
        self.spinner_weight_scale = QDoubleSpinBox(self)
        self.spinner_weight_scale.setRange(10, 1000)
        self.spinner_weight_scale.setValue(30)
        self.spinner_weight_scale.setSingleStep(1)
        self.spinner_weight_scale.setSuffix(' km')
        self.spinner_default_lapse_rate = QDoubleSpinBox(self)
        self.spinner_default_lapse_rate.setRange(-12., 100.)
        self.spinner_default_lapse_rate.setSingleStep(0.05)
        self.spinner_default_lapse_rate.setValue(-6.5)
        self.spinner_min_samples = QSpinBox(self)
        self.spinner_min_samples.setRange(3, 128)
        self.spinner_min_samples.setValue(20)
        self.spinner_min_range = QSpinBox(self)
        self.spinner_min_range.setRange(10, 1000)
        self.spinner_min_range.setValue(100)
        self.min_clip_settings = RampClipSettingsView(self)
        self.max_clip_settings = RampClipSettingsView(self)
        self.button_apply = QPushButton(self)
        self.button_apply.setText('Apply')
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_apply.clicked.connect(self.settings_changed)
        self.min_clip_settings.settings_changed.connect(self.cutoff_changed)
        self.max_clip_settings.settings_changed.connect(self.cutoff_changed)

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(QLabel('Algorithm settings:'))
        layout = QFormLayout()
        layout.addRow('Radius:', self.spinner_radius)
        layout.addRow('Weight scale:', self.spinner_weight_scale)
        layout.addRow('Default lapse rate:', self.spinner_default_lapse_rate)
        layout.addRow('Min. neighbors:', self.spinner_min_samples)
        layout.addRow('Min. vertical range:', self.spinner_min_range)
        outer_layout.addLayout(layout)
        outer_layout.addWidget(self.button_apply)
        outer_layout.addWidget(QLabel('Max clipping:'))
        outer_layout.addLayout(self.max_clip_settings.get_layout())
        outer_layout.addWidget(QLabel('Min clipping:'))
        outer_layout.addLayout(self.min_clip_settings.get_layout())
        outer_layout.addStretch()
        self.setLayout(outer_layout)

    def apply_settings(self, settings: LapseRateProperties):
        self.spinner_radius.setValue(settings.radius_km)
        self.spinner_min_samples.setValue(settings.min_samples)
        self.spinner_min_range.setValue(settings.min_elevation)
        self.spinner_default_lapse_rate.setValue(settings.default_lapse_rate)
        self.min_clip_settings.apply_settings(settings.min_clip)
        self.max_clip_settings.apply_settings(settings.max_clip)
        return self

    def get_settings(self):
        return LapseRateProperties(
            radius_km=self.spinner_radius.value(),
            min_samples=self.spinner_min_samples.value(),
            min_elevation=self.spinner_min_range.value(),
            default_lapse_rate=self.spinner_default_lapse_rate.value(),
            min_clip=self.min_clip_settings.get_settings(),
            max_clip=self.max_clip_settings.get_settings(),
        )


CONTOUR_KEYS = ['latitude', 'longitude', 'z_surf', 'lapse_rate', 'lapse_rate_raw', 'score']


class LapseRateRepresentationSettings(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lapse_rate_settings = LapseRateSettingsView(self)
        self.vis_settings_views =  {}
        self.color_settings_views = {}
        self.vis_settings_tabs = QTabWidget(self)
        self._build_algorithm_tab()
        self._build_stations_tab()
        self._build_elevation_tab()
        self._build_temperature_tab()

    def get_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.vis_settings_tabs)
        layout.addWidget(self.lapse_rate_settings)
        layout.addStretch()
        return layout

    def _make_tab_with_color_lookup(
            self,
            key: LapseRateComponent,
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
            key: LapseRateComponent,
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

    def _build_elevation_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        for key, label in {
            LapseRateComponent.Z_MAX: 'Max',
            LapseRateComponent.Z_MIN: 'Min',
            LapseRateComponent.Z_MEAN: 'Mean',
            LapseRateComponent.Z_RANGE: 'Range',
            LapseRateComponent.Z_NEAREST: 'Nearest'
        }.items():
            vis_settings = VolumeScalarSettingsView(parent=self, use_dvr=False)
            vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS)
            color_settings = ADCLSettingsView(self)
            self._make_tab_with_color_lookup(
                key, vis_settings, color_settings,
                'Surface properties',
                tabs, label
            )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Elevation')

    def _build_temperature_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        for key, label in {
            LapseRateComponent.T2M_MAX: 'Max',
            LapseRateComponent.T2M_MIN: 'Min',
            LapseRateComponent.T2M_MEAN: 'Mean',
            LapseRateComponent.T2M_RANGE: 'Range',
            LapseRateComponent.T2M_NEAREST: 'Nearest',
        }.items():
            vis_settings = VolumeScalarSettingsView(parent=self, use_dvr=False)
            vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS)
            self._make_tab_without_color_lookup(
                key, vis_settings,
                'Surface properties',
                tabs, label
            )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Temperature')

    def _build_stations_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        key = LapseRateComponent.PREDICTION
        vis_settings = StationScalarSettingsView(parent=self)
        self._make_tab_without_color_lookup(
            key, vis_settings,
            'Marker properties',
            tabs, 'Prediction (adaptive)'
        )
        for key, label in {
            LapseRateComponent.DIFFERENCE: 'Difference (adaptive)',
            LapseRateComponent.CORRECTION: 'Correction (adaptive)',
        }.items():
            vis_settings = StationScalarSettingsView(parent=self)
            color_settings = ADCLSettingsView(self)
            self._make_tab_with_color_lookup(
                key, vis_settings, color_settings,
                'Marker properties',
                tabs, label
            )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Stations')

    def _build_algorithm_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        for key, label in {
            LapseRateComponent.LAPSE_RATE_RAW: 'Lapse rate (estimate)',
            LapseRateComponent.LAPSE_RATE: 'Lapse rate (clipped)',
            LapseRateComponent.LOWER_CLIPPING: 'Lower clipping',
            LapseRateComponent.UPPER_CLIPPING: 'Upper clipping',
            LapseRateComponent.SCORE: 'R2',
            LapseRateComponent.NEIGHBOR_COUNT: 'Num. neighbors',
        }.items():
            vis_settings = VolumeScalarSettingsView(parent=self, use_dvr=False)
            vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS)
            color_settings = ADCLSettingsView(self)
            self._make_tab_with_color_lookup(
                key, vis_settings, color_settings,
                'Surface properties',
                tabs, label
            )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Algorithm')


class LapseRateController(QWidget):

    def __init__(self, view: LapseRateRepresentationSettings, model: LapseRateVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self._synchronize_lapse_rate_settings()
        self.vis_controls = {}
        self.color_controls = {}
        self._connect_settings_views()
        self._connect_lapse_rate_settings()

    def _synchronize_lapse_rate_settings(self):
        self.view.lapse_rate_settings.apply_settings(self.model.properties)

    def _connect_lapse_rate_settings(self):
        lapse_rate_settings = self.view.lapse_rate_settings
        lapse_rate_settings.settings_changed.connect(self.on_settings_changed)
        lapse_rate_settings.cutoff_changed.connect(self.on_cutoff_changed)

    def on_settings_changed(self):
        settings = self.view.lapse_rate_settings.get_settings()
        self.model.set_properties(settings)

    def on_cutoff_changed(self):
        settings = self.view.lapse_rate_settings.get_settings()
        self.model.update_postprocessed_data(settings)

    def _connect_settings_views(self):
        for key in LapseRateComponent:
            visual = self.model.representations[key]
            settings_view = self.view.vis_settings_views[key]
            self._register_vis_controls(key, visual, settings_view)
            if key in self.view.color_settings_views:
                color_settings = self.view.color_settings_views[key]
                controller = visual.color_lookup.get_controller(color_settings)
                self.color_controls[key] = controller
        return self

    def _register_vis_controls(self, key: LapseRateComponent, visual: VolumeVisual, settings_view: QWidget):
        if isinstance(visual, MultiMethodScalarVisualization):
            controller = MultiMethodVisualizationController(settings_view, visual, self)
        else:
            raise NotImplementedError()
        self.vis_controls[key] = controller

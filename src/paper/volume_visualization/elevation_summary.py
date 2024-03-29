from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QSpinBox, QFormLayout, QTabWidget, QVBoxLayout, QLabel, QPushButton
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import Coordinates
from src.paper.volume_visualization.color_lookup import make_elevation_lookup, make_elevation_offset_lookup, \
    make_quantile_lookup, make_quantile_difference_lookup, ADCLSettingsView
from src.paper.volume_visualization.multi_method_visualization import MultiMethodScalarVisualization, \
    MultiMethodVisualizationController
from src.paper.volume_visualization.plotter_slot import PlotterSlot, VolumeProperties, MeshProperties, \
    SurfaceReferenceProperties
from src.paper.volume_visualization.scaling import VolumeVisual, ScalingParameters
from src.paper.volume_visualization.volume import VolumeScalarVisualization, VolumeScalarSettingsView, \
    VolumeRepresentationMode
from src.paper.volume_visualization.volume_data import VolumeData
from src.paper.volume_visualization.volume_reference_grid import ReferenceGridVisualization, ReferenceGridSettingsView, \
    ReferenceGridController, ReferenceLevelVisualization


class ElevationSummaryComponent(Enum):
    Z_OVERVIEW = 'z_overview'
    DZ_FROM_MED_OVERVIEW = "dz_from_med_overview"
    QLEV_OVERVIEW = "qlev_overview"
    DQLEV_FROM_MED_OVERVIEW = "dqlev_from_med_overview"
    EXCEEDANCE_OVERVIEW = "exceedance_overview"
    Z_IN_IQR = "z_in_iqr"
    DZ_FROM_MED_IN_IQR = "dz_in_iqr"
    QLEV_IN_IQR = "qlev_in_iqr"
    DQLEV_FROM_MED_IN_IQR = "dqlev_in_iqr"
    EXCEEDANCE_IN_IQR = "exceedance_in_iqr"
    Z_IQR_BOUNDS = "z_iqr_bounds"
    Z_FLIERS = "z_fliers"
    Z_MEDIAN = "z_med"
    IQR = "iqr"


CONTOUR_KEYS_OVERVIEW = [
    ElevationSummaryComponent.Z_OVERVIEW.value,
    ElevationSummaryComponent.DZ_FROM_MED_OVERVIEW.value,
    ElevationSummaryComponent.QLEV_OVERVIEW.value,
    ElevationSummaryComponent.DQLEV_FROM_MED_OVERVIEW.value,
    ElevationSummaryComponent.EXCEEDANCE_OVERVIEW.value,
    'latitude_3d', 'longitude_3d'
]

CONTOUR_KEYS_IN_IQR = [
    ElevationSummaryComponent.Z_IN_IQR.value,
    ElevationSummaryComponent.DZ_FROM_MED_IN_IQR.value,
    ElevationSummaryComponent.QLEV_IN_IQR.value,
    ElevationSummaryComponent.DQLEV_FROM_MED_IN_IQR.value,
    ElevationSummaryComponent.EXCEEDANCE_IN_IQR.value,
    'latitude_3d', 'longitude_3d'
]


@dataclass
class ElevationSummaryProperties(object):
    radius_km: float = 30.
    num_quantiles: int = 8
    flier_factor: float = 1.5
    lookup_leaf_size: int = 100


class ElevationSummary(VolumeVisual):

    def __init__(
            self,
            terrain_data_lr: xr.Dataset, terrain_data_hr: xr.Dataset, plotter: pv.Plotter,
            properties: ElevationSummaryProperties, scaling: ScalingParameters,
            terrain_level_key: str = 'z_surf_o1280',
            parent=None
    ):
        super().__init__(parent)
        self.terrain_level_key = terrain_level_key
        self.terrain_data_lr = terrain_data_lr
        self.terrain_data_hr = terrain_data_hr
        self.plotter = plotter

        self.properties = properties
        self.scaling = scaling

        self.representations = {}

        self._lookup_lr, self._coords_lr = self._build_lookup_for_data(self.terrain_data_lr)
        self._lookup_hr, self._coords_hr = self._build_lookup_for_data(self.terrain_data_hr)

        self._compute_summary()
        self._build_representations()

    def get_plotter(self) -> pv.Plotter:
        return self.plotter

    def _build_lookup_for_data(self, data: xr.Dataset):
        coords = Coordinates.from_xarray(data).as_xyz().values
        lookup = NearestNeighbors(leaf_size=self.properties.lookup_leaf_size, n_neighbors=1)
        lookup.fit(coords)
        return lookup, coords

    def _compute_summary(self):
        distances = self._lookup_hr.radius_neighbors_graph(
            self._coords_lr, radius=self.properties.radius_km * 1000.,
            mode='distance', sort_results=True
        )
        indices = distances.indices
        site_id = np.zeros_like(indices)
        site_id[distances.indptr[1:-1]] = 1
        site_id = np.cumsum(site_id)
        z_sel = self.terrain_data_hr['z_surf'].values[indices]

        data = pd.DataFrame({'site_id': site_id, 'z': z_sel})

        groups = data.groupby('site_id')['z']

        p_all = np.linspace(0., 1., self.properties.num_quantiles + 1)
        q_all_ = groups.quantile(p_all)
        q_all = np.reshape(q_all_.values, (-1, len(p_all))).T

        p_iqr = np.linspace(0.25, 0.75, self.properties.num_quantiles + 1)
        q_iqr = np.reshape(groups.quantile(p_iqr).values, (-1, len(p_iqr))).T

        q_med = groups.median().values

        iqr_bounds = q_iqr[[0, -1]]
        iqr_bounds[0] = np.minimum(iqr_bounds[0] - q_med, 1.) + q_med
        iqr_bounds[-1] = np.maximum(iqr_bounds[-1] - q_med, 1.) + q_med

        iqr = iqr_bounds[-1] - iqr_bounds[0]
        upper_whisker_max = iqr_bounds[-1] + self.properties.flier_factor * iqr
        lower_whisker_max = iqr_bounds[0] - self.properties.flier_factor * iqr

        data_whiskers = data.loc[np.logical_and(z_sel <= upper_whisker_max[site_id], z_sel >= lower_whisker_max[site_id])]
        groups_whiskers = data_whiskers.groupby('site_id')
        upper_whisker = groups_whiskers['z'].max().sort_index().values
        lower_whisker = groups_whiskers['z'].min().sort_index().values

        fliers = np.stack([lower_whisker, upper_whisker], axis=0)

        num_sites = len(q_med)
        num_quantiles = len(q_all)

        p_med = np.full((num_sites,), 0.5)

        longitude = self.terrain_data_lr['longitude'].values
        latitude = self.terrain_data_lr['latitude'].values

        self.terrain_summary = xr.Dataset(
            data_vars={
                ElevationSummaryComponent.Z_OVERVIEW.value: (('quantiles', 'values'), q_all),
                ElevationSummaryComponent.DZ_FROM_MED_OVERVIEW.value: (('quantiles', 'values'), q_all - q_med),
                ElevationSummaryComponent.QLEV_OVERVIEW.value: (('quantiles', 'values'), np.tile(p_all[:, None], (1, num_sites))),
                ElevationSummaryComponent.EXCEEDANCE_OVERVIEW.value: (('quantiles', 'values'), np.tile(1. - p_all[:, None], (1, num_sites))),
                ElevationSummaryComponent.DQLEV_FROM_MED_OVERVIEW.value: (('quantiles', 'values'), p_all[:, None] - p_med),
                ElevationSummaryComponent.Z_IN_IQR.value: (('quantiles', 'values'), q_iqr),
                ElevationSummaryComponent.DZ_FROM_MED_IN_IQR.value: (('quantiles', 'values'), q_iqr - q_med),
                ElevationSummaryComponent.QLEV_IN_IQR.value: (('quantiles', 'values'), np.tile(p_iqr[:, None], (1, num_sites))),
                ElevationSummaryComponent.DQLEV_FROM_MED_IN_IQR.value: (('quantiles', 'values'), p_iqr[:, None] - p_med),
                ElevationSummaryComponent.EXCEEDANCE_IN_IQR.value: (('quantiles', 'values'), np.tile(1. - p_iqr[:, None], (1, num_sites))),
                ElevationSummaryComponent.Z_IQR_BOUNDS.value: (('lower_upper', 'values'), iqr_bounds),
                ElevationSummaryComponent.Z_MEDIAN.value: (('values',), q_med),
                ElevationSummaryComponent.IQR.value: (('values',), iqr_bounds[-1] - iqr_bounds[0]),
                ElevationSummaryComponent.Z_FLIERS.value: (('lower_upper', 'values'), fliers),
                'longitude_3d': (('quantiles', 'values'), np.tile(longitude[None, :], (num_quantiles, 1))),
                'latitude_3d': (('quantiles', 'values'), np.tile(latitude[None, :], (num_quantiles, 1))),
            },
            coords={
                'latitude': (('values',), latitude),
                'longitude': (('values',), latitude),
                'p_overview': (('quantiles',), p_all),
                'p_iqr': (('quantiles',), p_iqr),
            }
        )

    def _build_representations(self):
        self._build_overview_representations()
        self._build_in_iqr_representations()
        self._build_surface_representations()
        self._build_flier_representations()
        return self

    def _build_representation(self, key: ElevationSummaryComponent, visual: VolumeVisual):
        self.representations[key] = visual
        return self

    def _build_overview_representations(self):
        model_level_key = ElevationSummaryComponent.Z_OVERVIEW.value
        self._build_representation(
            ElevationSummaryComponent.Z_OVERVIEW,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Elevation (m, ov)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=model_level_key,
                    model_level_key=model_level_key,
                ),
                make_elevation_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.DZ_FROM_MED_OVERVIEW,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Elevation from median (m, ov)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.DZ_FROM_MED_OVERVIEW.value,
                    model_level_key=model_level_key,
                ),
                make_elevation_offset_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.QLEV_OVERVIEW,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Quantile level (ov)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.QLEV_OVERVIEW.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.DQLEV_FROM_MED_OVERVIEW,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Quantile from median (ov)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.DQLEV_FROM_MED_OVERVIEW.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_difference_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.EXCEEDANCE_OVERVIEW,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Exceedance probability (ov)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.EXCEEDANCE_OVERVIEW.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )

    def _build_in_iqr_representations(self):
        model_level_key = ElevationSummaryComponent.Z_IN_IQR.value
        self._build_representation(
            ElevationSummaryComponent.Z_IN_IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Elevation (m, iqr)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=model_level_key,
                    model_level_key=model_level_key,
                ),
                make_elevation_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.DZ_FROM_MED_IN_IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Elevation from median (m, iqr)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.DZ_FROM_MED_IN_IQR.value,
                    model_level_key=model_level_key,
                ),
                make_elevation_offset_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.QLEV_IN_IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Quantile level (iqr)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.QLEV_IN_IQR.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.DQLEV_FROM_MED_IN_IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Quantile from median (iqr)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.DQLEV_FROM_MED_IN_IQR.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_difference_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.EXCEEDANCE_IN_IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'Exceedance probability (iqr)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.EXCEEDANCE_IN_IQR.value,
                    model_level_key=model_level_key,
                ),
                make_quantile_lookup(),
                VolumeProperties(),
                self.scaling
            )
        )

    def _build_surface_representations(self):
        self._build_representation(
            ElevationSummaryComponent.Z_IQR_BOUNDS,
            ReferenceLevelVisualization(
                PlotterSlot(self.plotter),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    model_level_key=ElevationSummaryComponent.Z_IQR_BOUNDS.value,
                ),
                SurfaceReferenceProperties(),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.Z_MEDIAN,
            ReferenceGridVisualization(
                PlotterSlot(self.plotter),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    model_level_key=ElevationSummaryComponent.Z_MEDIAN.value,
                ),
                SurfaceReferenceProperties(color=(255, 0, 0)),
                self.scaling
            )
        )
        self._build_representation(
            ElevationSummaryComponent.IQR,
            VolumeScalarVisualization(
                PlotterSlot(self.plotter, 'IQR (m)'),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    scalar_key=ElevationSummaryComponent.IQR.value,
                    model_level_key='z_surf_o1280',
                ),
                make_elevation_lookup(),
                MeshProperties(),
                self.scaling
            )
        )

    def _build_flier_representations(self):
        self._build_representation(
            ElevationSummaryComponent.Z_FLIERS,
            ReferenceLevelVisualization(
                PlotterSlot(self.plotter),
                VolumeData(
                    self.terrain_summary, self.terrain_data_lr,
                    model_level_key=ElevationSummaryComponent.Z_FLIERS.value,
                ),
                SurfaceReferenceProperties(color=(255, 255, 255), opacity=0.5),
                self.scaling
            )
        )

    # def _compute_fliers(self):
    #     iqr_bounds = self.terrain_summary['z_iqr_bounds'].values
    #     iqr = iqr_bounds[-1] - iqr_bounds[0]
    #     fliers = iqr_bounds.copy()
    #     flier_offset = self.properties.flier_factor * iqr
    #     fliers[0] -= flier_offset
    #     fliers[1] += flier_offset

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

    def set_properties(self, properties: ElevationSummaryProperties, render: bool = True):
        self.properties = properties
        self._compute_summary()
        self._update_representations()
        if render:
            self.plotter.render()
        return self

    def _update_representations(self):
        for visual in self.representations.values():
            visual.update_data(self.terrain_summary, render=False)

    # def update_fliers(self, properties: ElevationSummaryProperties, render: bool = True):
    #     self.properties = properties
    #     self._compute_fliers()
    #     self.representations[ElevationSummaryComponent.Z_FLIERS].update_data(self.terrain_summary, render=render)
    #     return self

    def update_lookup(self, properties: ElevationSummaryProperties, render: bool = True):
        self.properties = properties
        self._lookup_lr, self._coords_lr = self._build_lookup_for_data(self.terrain_data_lr)
        self._lookup_hr, self._coords_hr = self._build_lookup_for_data(self.terrain_data_hr)
        return self


class ElevationSummarySettingsView(QWidget):

    settings_changed = pyqtSignal()
    fliers_changed = pyqtSignal()
    lookup_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_radius = QDoubleSpinBox(self)
        self.spinner_radius.setRange(20, 180)
        self.spinner_radius.setValue(30)
        self.spinner_radius.setSingleStep(1)
        self.spinner_radius.setSuffix(' km')
        self.spinner_num_quantiles = QSpinBox(self)
        self.spinner_num_quantiles.setRange(0, 128)
        self.spinner_num_quantiles.setValue(32)
        self.spinner_flier_factor = QDoubleSpinBox(self)
        self.spinner_flier_factor.setRange(0., 5.)
        self.spinner_flier_factor.setSingleStep(0.05)
        self.spinner_flier_factor.setValue(1.5)
        self.spinner_lookup_leaf_size = QSpinBox(self)
        self.spinner_lookup_leaf_size.setRange(10, 1000)
        self.spinner_lookup_leaf_size.setSingleStep(10)
        self.spinner_lookup_leaf_size.setValue(100)
        self.button_apply = QPushButton(self)
        self.button_apply.setText('Apply')
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_apply.clicked.connect(self.settings_changed)
        # self.spinner_radius.valueChanged.connect(self.settings_changed)
        # self.spinner_num_quantiles.valueChanged.connect(self.settings_changed)
        self.spinner_lookup_leaf_size.valueChanged.connect(self.lookup_changed)
        self.spinner_flier_factor.valueChanged.connect(self.fliers_changed)

    def _set_layout(self):
        outer_layout = QVBoxLayout()
        layout = QFormLayout()
        layout.addRow('Radius:', self.spinner_radius)
        layout.addRow('Num. of quantiles:', self.spinner_num_quantiles)
        layout.addRow('Flier factor:', self.spinner_flier_factor)
        layout.addRow('Lookup leaf size:', self.spinner_lookup_leaf_size)
        outer_layout.addLayout(layout)
        outer_layout.addWidget(self.button_apply)
        outer_layout.addStretch()
        self.setLayout(outer_layout)

    def apply_settings(self, settings: ElevationSummaryProperties):
        self.spinner_radius.setValue(settings.radius_km)
        self.spinner_num_quantiles.setValue(settings.num_quantiles)
        self.spinner_flier_factor.setValue(settings.flier_factor)
        self.spinner_lookup_leaf_size.setValue(settings.lookup_leaf_size)
        return self

    def get_settings(self):
        return ElevationSummaryProperties(
            radius_km=self.spinner_radius.value(),
            num_quantiles=self.spinner_num_quantiles.value(),
            flier_factor=self.spinner_flier_factor.value(),
            lookup_leaf_size=self.spinner_lookup_leaf_size.value()
        )


class ElevationSummaryRepresentationSettings(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary_settings = ElevationSummarySettingsView(self)
        self.vis_settings_views =  {}
        self.color_settings_views = {}
        self.vis_settings_tabs = QTabWidget(self)
        self._build_overview_volume_tab()
        self._build_iqr_volume_tab()
        self._build_surface_tab()

    def get_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.vis_settings_tabs)
        layout.addWidget(self.summary_settings)
        layout.addStretch()
        return layout

    def _make_tab_with_color_lookup(
            self,
            key: ElevationSummaryComponent,
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
            key: ElevationSummaryComponent,
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

    def _build_overview_volume_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        # z overview
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_OVERVIEW)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.Z_OVERVIEW,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Elevation'
        )
        # dz from med
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_OVERVIEW)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.DZ_FROM_MED_OVERVIEW,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Elevation from median'
        )
        # qlevel
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_OVERVIEW)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.QLEV_OVERVIEW,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Quantile level'
        )
        # dqlevel
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_OVERVIEW)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.DQLEV_FROM_MED_OVERVIEW,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Quantile from median'
        )
        # exceedance
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_OVERVIEW)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.EXCEEDANCE_OVERVIEW,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Exceedance'
        )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Overview (volume)')

    def _build_iqr_volume_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        # z in iqr
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_IN_IQR)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.Z_IN_IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Elevation'
        )
        # dz from med
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_IN_IQR)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.DZ_FROM_MED_IN_IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Elevation from median'
        )
        # qlevel
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_IN_IQR)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.QLEV_IN_IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Quantile level'
        )
        # dqlevel
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_IN_IQR)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.DQLEV_FROM_MED_IN_IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Quantile from median'
        )
        # exceedance
        vis_settings = VolumeScalarSettingsView(parent=self)
        vis_settings.representation_views[VolumeRepresentationMode.ISO_CONTOURS].set_contour_keys(CONTOUR_KEYS_IN_IQR)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.EXCEEDANCE_IN_IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'Exceedance'
        )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'IQR (volume)')

    def _build_surface_tab(self):
        container = QWidget(self.vis_settings_tabs)
        tabs = QTabWidget(container)
        # iqr on lowres
        vis_settings = VolumeScalarSettingsView(parent=self, use_dvr=False, use_contours=False)
        color_settings = ADCLSettingsView(self)
        self._make_tab_with_color_lookup(
            ElevationSummaryComponent.IQR,
            vis_settings, color_settings,
            'Volume properties',
            tabs, 'IQR'
        )
        # iqr bounds
        vis_settings = ReferenceGridSettingsView(self)
        self._make_tab_without_color_lookup(
            ElevationSummaryComponent.Z_IQR_BOUNDS,
            vis_settings,
            'Surface properties',
            tabs, 'IQR bounds'
        )
        # median surface
        vis_settings = ReferenceGridSettingsView(self)
        self._make_tab_without_color_lookup(
            ElevationSummaryComponent.Z_MEDIAN,
            vis_settings,
            'Surface properties',
            tabs, 'Median'
        )
        # flier surfaces
        vis_settings = ReferenceGridSettingsView(self)
        self._make_tab_without_color_lookup(
            ElevationSummaryComponent.Z_FLIERS,
            vis_settings,
            'Surface properties',
            tabs, 'Fliers'
        )
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addStretch()
        container.setLayout(layout)
        self.vis_settings_tabs.addTab(container, 'Surface')


class ElevationSummaryController(QWidget):

    def __init__(self, view: ElevationSummaryRepresentationSettings, model: ElevationSummary, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self._synchronize_summary_settings()
        self.vis_controls = {}
        self.color_controls = {}
        self._connect_settings_views()
        self._connect_summary_settings()

    def _synchronize_summary_settings(self):
        self.view.summary_settings.apply_settings(self.model.properties)

    def _connect_summary_settings(self):
        summary_settings = self.view.summary_settings
        summary_settings.settings_changed.connect(self.on_settings_changed)
        summary_settings.fliers_changed.connect(self.on_settings_changed)
        summary_settings.lookup_changed.connect(self.on_lookup_changed)

    def on_settings_changed(self):
        settings = self.view.summary_settings.get_settings()
        self.model.set_properties(settings)

    # def on_fliers_changed(self):
    #     settings = self.view.summary_settings.get_settings()
    #     self.model.update_fliers(settings)

    def on_lookup_changed(self):
        settings = self.view.summary_settings.get_settings()
        self.model.update_lookup(settings)

    def _connect_settings_views(self):
        for key in ElevationSummaryComponent:
            visual = self.model.representations[key]
            settings_view = self.view.vis_settings_views[key]
            self._register_vis_controls(key, visual, settings_view)
            if key in self.view.color_settings_views:
                color_settings = self.view.color_settings_views[key]
                controller = visual.color_lookup.get_controller(color_settings)
                self.color_controls[key] = controller
        return self

    def _register_vis_controls(self, key: ElevationSummaryComponent, visual: VolumeVisual, settings_view: QWidget):
        if isinstance(visual, MultiMethodScalarVisualization):
            controller = MultiMethodVisualizationController(settings_view, visual, self)
        elif isinstance(visual, (ReferenceGridVisualization, ReferenceLevelVisualization)):
            controller = ReferenceGridController(settings_view, visual, self)
        else:
            raise NotImplementedError()
        self.vis_controls[key] = controller

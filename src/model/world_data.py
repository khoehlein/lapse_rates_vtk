import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Union, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QPushButton, QComboBox, QStackedLayout
from sklearn.neighbors import KDTree
from cartopy import crs as ccrs

from src.model.geometry import OctahedralGrid, DomainBounds, TriangleMesh, WedgeMesh, Coordinates, LocationBatch
from src.model.level_heights import compute_physical_level_height
from src.widgets import LogDoubleSliderSpinner, DoubleSliderSpinner

N_LOW_RES = 1280
N_HIGH_RES = 8000
N_LEVELS = 20


class ConfigurationKey(object):
    PATH = 'path'
    SELECTION = 'select'
    ENGINE = 'engine'


class DataConfiguration(object):

    DEFAULTS = {
        ConfigurationKey.SELECTION: None,
        ConfigurationKey.ENGINE: 'cfgrib'
    }

    def __init__(self, path: str, selection: Dict[str, Any], engine: str):
        self.path = path
        self.selection = selection
        self.engine = engine

    @classmethod
    def from_config_entry(cls, config_entry: Union[str, Dict[str, Any]]) -> 'DataConfiguration':
        if isinstance(config_entry, str):
            return cls(config_entry, cls.DEFAULTS[ConfigurationKey.SELECTION], cls.DEFAULTS[ConfigurationKey.ENGINE])
        path = config_entry.get(ConfigurationKey.PATH)
        selection = config_entry.get(ConfigurationKey.SELECTION, cls.DEFAULTS[ConfigurationKey.SELECTION])
        engine = config_entry.get(ConfigurationKey.ENGINE, cls.DEFAULTS[ConfigurationKey.ENGINE])
        return cls(path, selection, engine)

    def load_data(self):
        data = xr.open_dataset(self.path, engine=self.engine)
        if self.selection is not None:
            data = data.isel(**self.selection)
        return data


class ConfigReader(object):

    @staticmethod
    def load_json_config(path_to_data_config: str):
        logging.info(f'Loading model from {path_to_data_config}.')
        with open(path_to_data_config, 'r') as f:
            configs = json.load(f)
        return configs

    def __init__(self, config_class):
        self.config_class = config_class

    def load_data(self, config_entry: Union[str, Dict[str, Any]]):
        configuration = self.config_class.from_config_entry(config_entry)
        return configuration.load_data()


class DomainData(object):

    def __init__(
            self,
            bounds: DomainBounds,
            surface_mesh_lr: TriangleMesh, surface_mesh_hr: TriangleMesh,
            volume_mesh_model_levels: WedgeMesh,
            data_lr: xr.Dataset, data_hr: xr.Dataset,
    ):
        self.bounds = bounds
        self.surface_mesh_lr = surface_mesh_lr
        self.surface_mesh_hr = surface_mesh_hr
        self.volume_mesh_model_levels = volume_mesh_model_levels
        self.data_lr = data_lr
        self.data_hr = data_hr
        self._pv_meshes = {}

    def get_orography_mesh_lr(self, z_scale = 1.):
        z_lr = self.data_lr.z.values
        return self.surface_mesh_lr.to_pyvista(z=(z_lr / z_scale))


class UniformNeighborhoodGraph(object):

    def __init__(
            self,
            locations: LocationBatch,
            neighbors: np.ndarray,
            distances: np.ndarray
    ):
        self.locations = locations
        self.neighbors = neighbors
        self.distances = distances


class NeighborhoodGraph(object):

    def __init__(
            self,
            locations: LocationBatch,
            neighbors: List[np.ndarray],
            distances: List[np.ndarray],
    ):
        self.locations = locations
        self.neighbors = neighbors
        self.distances = distances
        self._compute_index_stats()
        self._compute_distance_stats()

    def _compute_links(self):
        flat_locids = np.zeros((self.total_links,), dtype=int)
        counter = 0
        for i, num_links in enumerate(self.num_links):
            flat_locids[counter:(counter + num_links)] = i
            counter += num_links
        return pd.DataFrame({
            'location': flat_locids,
            'neighbor': np.fromiter(self.neighbors, count=self.total_links, dtype=int)
        })

    def _compute_index_stats(self):
        neighbor_props = np.fromiter(
            ((len(nids), nids[0] if len(nids) else -1) for nids in self.neighbors),
            count=len(self.neighbors), dtype=int
        )
        self.num_links = neighbor_props[:, 0]
        self.nearest_neighbor = neighbor_props[:, -1]
        self.total_links = np.sum(self.num_links)
        self.links = self._compute_links()

    def _compute_distance_stats(self):
        extremes = np.fromiter((
            (d[0], d[-1]) if len(d) else (-1., -1.) for d in self.distances)
            , count=len(self.neighbors), dtype=float
        )
        self.min_distance = extremes[:, 0]
        self.max_distance = extremes[:, -1]

    @classmethod
    def from_tree_query(cls, locations: LocationBatch, tree: KDTree, radius_km: float):
        xyz = locations.coords.as_geocentric().values
        neighbors, distances = tree.query_radius(
            xyz, r=1000. * radius_km,
            return_distance=True, sort_results=True
        )
        data = cls(locations, neighbors, distances)
        return data

    def get_subset(self, location_ids: np.ndarray):
        locations = self.locations.get_subset(location_ids)
        neighbors = self.neighbors[location_ids]
        distances = self.distances[location_ids]
        return self.__class__(locations, neighbors, distances)

    def get_uniform_neighborhoods(self):
        groups = pd.DataFrame({
            'location': np.arange(self.num_links),
            'num_links': self.num_links
        }).set_index('location').groupby(by='num_links').groups

        def build_uniform_neighborhood(key):
            location_ids = groups.get(key).values
            locations = self.locations.get_subset(location_ids)
            neighbors = np.asarray(self.neighbors[location_ids])
            distances = np.asarray(self.distances[location_ids])
            return UniformNeighborhoodGraph(locations, neighbors, distances)

        neighborhoods = [build_uniform_neighborhood(key) for key in groups.keys()]
        return neighborhoods


class AboveThresholdFilter(QWidget):

    mask_changed = pyqtSignal()

    def __init__(self, data: np.ndarray, threshold: float, parent=None):
        super().__init__(parent)
        self.data = data
        self.threshold = threshold
        self._compute_mask()

    @pyqtSlot(float)
    def set_threshold(self, threshold: float):
        if threshold != self.threshold:
            self.threshold = float(threshold)
            self.mask_changed.emit(threshold)

    def _compute_mask(self):
        self.mask = self.data >= self.threshold


class NeighborhoodLookup(QWidget):

    tree_lookup_changed = pyqtSignal()

    def __init__(self, data: xr.DataArray, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        if config is None:
            config = {}
        self.locations = Coordinates.from_xarray(data).as_geocentric().values
        self.filter = AboveThresholdFilter(data.values, config.get('lsm_threshold', 0.), parent=self)
        self._tree_kws = config.get('tree_kws', {'leaf_size': 100})
        self._refresh_tree_lookup()
        self.filter.mask_changed.connect(self._refresh_tree_lookup)

    def _refresh_tree_lookup(self):
        self.tree_lookup = KDTree(self.locations[self.filter.mask], **self._tree_kws)
        self.tree_lookup_changed.emit()

    def query_neighborhood_graph(self, locations: LocationBatch, radius_km: float) -> NeighborhoodGraph:
        graph = NeighborhoodGraph.from_tree_query(locations, self.tree_lookup, radius_km)
        return graph

    def query_nearest_neighbors(self, locations: LocationBatch, k: int) -> UniformNeighborhoodGraph:
        raise NotImplementedError()

@dataclass()
class RadialNeighborhoodProperties(object):
    lookup_radius: float
    lsm_threshold: float


class RadialNeighborhoodLookup(NeighborhoodLookup):

    lookup_radius_changed = pyqtSignal()

    def __init__(self, land_sea_mask: xr.DataArray, config: Dict[str, Any] = None, parent=None):
        if config is None:
            config = {}
        super().__init__(land_sea_mask, config, parent)
        self.lookup_radius = config.get('lookup_radius', 30.)

    @pyqtSlot(float)
    def set_lookup_radius(self, radius_km: float):
        if radius_km != self.lookup_radius:
            self.lookup_radius = radius_km
            self.lookup_radius_changed.emit()

    def query_neighborhood(self, locations: LocationBatch) -> NeighborhoodGraph:
        return self.query_neighborhood_graph(locations, self.lookup_radius)


class RadialNeighborhoodLookupHandles(QWidget):

    neighborhood_changed = pyqtSignal()

    def __init__(self, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        if config is None:
            config = {}

        self.radius_slider = LogDoubleSliderSpinner(
            config.get('lookup_radius_min', 10.),
            config.get('lookup_radius_max', 180.),
            config.get('lookup_radius_steps', 128),
            width=4, parent=self
        )
        self.radius_slider.set_value(config.get('lookup_radius', 30.))
        self.mask_slider = DoubleSliderSpinner(width=4, parent=self)
        self.mask_slider.set_value(config.get('lsm_threshold', 0.))
        self.button_apply = QPushButton('Apply')
        self.button_apply.clicked.connect(self._on_button_apply)
        layout = QGridLayout()
        layout.addWidget(QLabel('Radius (km)'), 0, 0, 1, 1)
        layout.addWidget(self.radius_slider, 0, 1, 1, 4)
        layout.addWidget(QLabel('Land/Sea threshold'), 1, 0, 1, 1)
        layout.addWidget(self.mask_slider, 1, 1, 4, 1)
        layout.addWidget(self.button_apply, 2, 0, 1, 5)
        self.setLayout(layout)

    def _on_button_apply(self):
        self.neighborhood_changed.emit()

    def get_properties(self):
        return RadialNeighborhoodProperties(self.radius_slider.value(), self.mask_slider.value())


class NeighborhoodLookupView(QWidget):

    def __init__(self, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self.combo_lookup_type = QComboBox(parent=self)
        self.combo_lookup_type.addItem('Radius')
        self.radial_interface = RadialNeighborhoodLookupHandles(config=config, parent=self)
        self.interface_stack = QStackedLayout()
        self.interface_stack.addWidget(self.radial_interface)
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Neighborhood lookup', parent=self))
        combo_layout = QGridLayout()
        combo_layout.addWidget(QLabel('Neighborhood type:'), 0, 0, 1, 1)
        combo_layout.addWidget(self.combo_lookup_type, 0, 1, 1, 4)
        layout.addLayout(combo_layout)
        layout.addLayout(self.interface_stack)
        self.setLayout(layout)


class WorldData(object):

    @classmethod
    def from_config_file(cls, path_to_config_file: str):
        config_reader = ConfigReader(DataConfiguration)
        configs = config_reader.load_json_config(path_to_config_file)
        logging.info('Loading low-res model...')
        orography_lr = config_reader.load_data(configs['orography']['low-res']).z
        t2m = config_reader.load_data(configs['temperature']['2m'])
        t3d = config_reader.load_data(configs['temperature']['bulk']).t.transpose('hybrid', 'values')
        lnsp = config_reader.load_data(configs['pressure']).lnsp
        q3d = config_reader.load_data(configs['humidity']).q.transpose('hybrid', 'values')
        logging.info('Computing model level heights...')
        z_model_levels = compute_physical_level_height(
            np.exp(lnsp.values)[None, :], orography_lr.values[None, :],
            t3d.values, q3d.values
        )
        logging.info('Merging low-res model...')
        data_lr = xr.merge([orography_lr, t2m, t3d, lnsp, q3d], compat='override')
        data_lr = data_lr.assign(z_model_levels=(('hybrid', 'values'), z_model_levels))
        logging.info(f'Loading high-res model...')
        data_hr = config_reader.load_data(configs['orography']['high-res'])
        logging.info(f'Loading completed.')
        grid_lr = OctahedralGrid(N_LOW_RES)
        grid_hr = OctahedralGrid(N_HIGH_RES)
        return cls(grid_lr, grid_hr, data_lr, data_hr)

    def __init__(
            self,
            grid_lr: OctahedralGrid, grid_hr: OctahedralGrid,
            data_lr: xr.Dataset, data_hr: xr.Dataset,
            tree_props: Dict[str, Any] = None
    ):
        self.grid_lr = grid_lr
        self.grid_hr = grid_hr
        self.data_lr = data_lr
        self.data_hr = data_hr
        if tree_props is None:
            tree_props = {'metric': 'euclidean', 'leaf_size': 100}
        self.tree_lr = self._build_lr_tree(tree_props)

    def _build_lr_tree(self, tree_props: Dict[str, Any]) -> KDTree:
        """Build LR tree."""
        coords = Coordinates.from_xarray(self.data_lr).as_geocentric().values
        tree = KDTree(coords, **tree_props)
        return tree

    def query_domain_data(self, domain_bounds: DomainBounds) -> DomainData:
        mesh_lr = self.grid_lr.get_subgrid(domain_bounds)
        mesh_hr = self.grid_hr.get_subgrid(domain_bounds)
        z_model_levels = self.data_lr.z_model_levels.values[:, mesh_lr.source_reference]
        mesh_model_levels = WedgeMesh(mesh_lr, z_model_levels)
        return DomainData(
            domain_bounds,
            mesh_lr, mesh_hr, mesh_model_levels,
            self.data_lr.isel(values=mesh_lr.source_reference),
            self.data_hr.isel(values=mesh_hr.source_reference),
        )

    def query_sample_data(self, sample_size: int) -> None:
        pass

    def query_knn_data(self, locations: LocationBatch, k: int):
        pass
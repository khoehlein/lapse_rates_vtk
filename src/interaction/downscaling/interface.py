import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Tuple, Union, Any, Dict

import numpy as np
import pyvista as pv
import cartopy.crs as ccrs
from scipy.special import roots_legendre
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.model.data_store.config_interface import ConfigReader, SourceConfiguration
from src.model.level_heights import compute_physical_level_height
from src.model.neighborhood_lookup.neighborhood_graphs import UniformNeighborhoodGraph, RadialNeighborhoodGraph, \
    NeighborhoodGraph
from src.model.visualization.interface import PropertyModel, PropertyModelUpdateError


geocentric_system = ccrs.Geocentric()
xyz_system = geocentric_system
lat_lon_system = ccrs.PlateCarree()


class Coordinates(object):

    @classmethod
    def from_xarray(cls, data: Union[xr.Dataset, xr.DataArray]):
        return cls(lat_lon_system, data['longitude'].values, data['latitude'].values)

    def __init__(self, system: ccrs.CRS, *values: np.ndarray):
        self.components = values
        self.system = system

    def transform_to(self, new_system: ccrs.CRS):
        return Coordinates(new_system, *new_system.transform_points(self.system, *self.components).T)

    def as_geocentric(self):
        return self.transform_to(geocentric_system)

    def as_lat_lon(self):
        new_coords = self.transform_to(lat_lon_system)
        # restrict lat lon coordinates to 2d coordinates
        if len(new_coords.components) == 3:
            new_coords.components = new_coords.components[:2]
        return new_coords

    def as_xyz(self):
        return self.as_geocentric()

    @property
    def values(self):
        return np.stack(self.components, axis=-1)

    @property
    def x(self):
        return self.components[0]

    @property
    def y(self):
        return self.components[1]

    @property
    def z(self):
        return self.components[2]

    def __len__(self):
        return len(self.values)


class AngularInterval(object):

    def __init__(self, min: float, max: float, period: float = None):
        self.min = float(min)
        self._max = float(max)
        if period is not None:
            period = float(period)
        self.period = period

    def is_periodic(self) -> bool:
        return self.period is not None

    @property
    def max(self) -> float:
        return (self._max - self.min) % self.period if self.is_periodic() else self._max

    def argwhere(self, x: np.ndarray) -> np.ndarray:
        return np.argwhere(self.contains(x))

    def contains(self, x: np.ndarray) -> np.ndarray:
        if self.is_periodic():
            x = (x - self.min) % self.period
            mask = x < self.max
        else:
            mask = np.logical_and(x >= self.min, x <= self.max)
        return mask


class LocationBatch(object):

    def __init__(self, coords: Coordinates, elevation: np.ndarray = None, source_reference: np.ndarray = None):
        self.coords = coords
        self.elevation = elevation
        self.source_reference = source_reference

    @property
    def x(self):
        return self.coords.x

    @property
    def y(self):
        return self.coords.y

    @property
    def z(self):
        return self.elevation

    def get_subset(self, location_ids: np.ndarray) -> 'LocationBatch':
        coords = Coordinates(self.coords.system, *[c[location_ids] for c in self.coords.components])
        source_reference = self.source_reference[location_ids]
        return self.__class__(coords, source_reference)

    def __len__(self):
        return self.coords.__len__()


class TriangleMesh(object):

    def __init__(self, nodes: LocationBatch, vertices: np.ndarray):
        self.nodes = nodes
        self.vertices = vertices
        self._x = None
        self._y = None

    @property
    def x(self):
        if self._x is None:
            self._x = self.nodes.x[self.vertices]
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = self.nodes.y[self.vertices]
        return self._y

    @property
    def z(self):
        return self.nodes.z

    @property
    def left(self) -> np.ndarray:
        return np.amin(self.x, axis=-1)

    @property
    def right(self) -> np.ndarray:
        return np.amax(self.x, axis=-1)

    @property
    def bottom(self) -> np.ndarray:
        return np.amin(self.y, axis=-1)

    @property
    def top(self) -> np.ndarray:
        return np.amax(self.y, axis=-1)


    @property
    def source_reference(self):
        return self.nodes.source_reference

    @property
    def node_coordinates(self):
        return self.nodes.coords

    @property
    def num_nodes(self):
        return len(self.x)

    def get_node_positions(self) -> np.ndarray:
        z = self.z
        if z is None:
            z = np.zeros_like(self.x)
        coordinates = np.stack([self.x, self.y, z], axis=-1)
        return coordinates

    def get_faces(self, add_prefix: bool = False) -> np.ndarray:
        prefix_offset = int(add_prefix)
        faces = np.zeros((len(self.vertices), 3 + prefix_offset), dtype=int)
        if add_prefix:
            faces[:, 0] = 3
        faces[:, prefix_offset:] = self.vertices
        return faces

    def to_polydata(self) -> pv.PolyData:
        faces = self.get_faces()
        points = self.get_node_positions()
        return pv.PolyData(points, faces)


@dataclass
class DomainLimits(object):
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


class DomainBoundingBox(object):

    def __init__(self, bounds: DomainLimits):
        self.latitude = AngularInterval(bounds.min_latitude, bounds.max_latitude)
        self.longitude = AngularInterval(bounds.min_longitude, bounds.max_longitude, period=360.)

    def contains(self, locations: LocationBatch) -> np.ndarray:
        coords = locations.coords.as_lat_lon()
        longitude, latitude = coords.components
        return np.logical_and(
            self.latitude.contains(latitude),
            self.longitude.contains(longitude)
        )

    def intersects(self, triangles: TriangleMesh) -> np.ndarray:
        # approximate triangles as rectangles
        overlaps_longitude = np.logical_or(
            self.longitude.contains(triangles.left),
            self.longitude.contains(triangles.right)
        )
        overlaps_latitude = np.logical_or(
            self.latitude.contains(triangles.top),
            self.latitude.contains(triangles.bottom)
        )
        return np.logical_and(overlaps_latitude, overlaps_longitude)


class GridCircle(object):

    def __init__(self, degree: int, index_from_north: int):
        self.degree = int(degree)
        self.index_from_north = int(index_from_north)
        self.index_from_pole = min(self.index_from_north, 2 * self.degree - 1 - self.index_from_north)

    @property
    def num_nodes(self):
        return 4 * self.index_from_pole + 20

    @property
    def first_index(self):
        n = self.index_from_pole
        if self.index_from_north < self.degree:
            first_index = 20 * n + 2 * n * (n - 1)
        else:
            d = self.degree
            total_on_hemisphere = 20 * d + 2 * d * (d - 1)
            total_from_pole = 20 * (n + 1) + 2 * n * (n + 1)
            first_index = 2 * total_on_hemisphere - total_from_pole
        return first_index

    @property
    def vertices(self):
        return self.first_index + np.arange(self.num_nodes)

    @property
    def longitudes(self):
        num_nodes = self.num_nodes
        return (360 / num_nodes) * np.arange(num_nodes)

    def _get_offset_for_smaller_circle(self):
        num_nodes = self.num_nodes
        offset = np.arange(num_nodes)
        offset_on_smaller = (offset - offset // (num_nodes / 4)) % (num_nodes - 4)
        return offset_on_smaller

    def _get_offset_for_larger_circle(self):
        num_nodes = self.num_nodes
        offset = np.arange(num_nodes)
        offset_on_larger = offset + offset // (num_nodes / 4) + 1
        return offset_on_larger

    def _get_triangles(self, index_on_other, pointing_southward):
        num_nodes = self.num_nodes
        first_index = self.first_index
        vertices = np.zeros((num_nodes, 3), dtype=int)
        offset = np.arange(num_nodes)
        if pointing_southward:
            vertices[:, 2] = first_index + offset
            vertices[:, 1] = first_index + (1 + offset) % num_nodes
            vertices[:, 0] = index_on_other
        else:
            vertices[:, 0] = first_index + offset
            vertices[:, 1] = first_index + (1 + offset) % num_nodes
            vertices[:, 2] = index_on_other
        return vertices

    def triangles_to_northern_circle(self):
        on_northern_hemisphere = self.index_from_north < self.degree
        if on_northern_hemisphere:
            if self.index_from_pole == 0:
                return None
            offset_on_other = self._get_offset_for_smaller_circle()
            first_index_other = self.first_index - (self.num_nodes - 4)
        else:
            offset_on_other = self._get_offset_for_larger_circle()
            first_index_other = self.first_index - (self.num_nodes + 4)
        return self._get_triangles(first_index_other + offset_on_other, False)

    def triangles_to_southern_circle(self):
        on_northern_hemisphere = self.index_from_north < self.degree
        first_index_other = self.first_index + self.num_nodes
        if on_northern_hemisphere:
            offset_on_other = self._get_offset_for_larger_circle()
        else:
            if self.index_from_pole == 0:
                return None
            offset_on_other = self._get_offset_for_smaller_circle()
        return self._get_triangles(first_index_other + offset_on_other, True)


class OctahedralGrid(object):

    def __init__(self, degree: int):
        self.degree = int(degree)
        self._circle_latitudes = - np.rad2deg(np.arcsin(roots_legendre(2 * self.degree)[0]))

    @property
    def num_triangles(self):
        nodes_per_hemisphere = self.num_nodes // 2
        triangles_per_hemisphere = 2 * nodes_per_hemisphere - (20 + (20 + 4 * (self.degree - 1)))
        return 2 * triangles_per_hemisphere

    @property
    def num_nodes(self):
        n = self.degree
        nodes_per_hemisphere = 20 * n + 2 * n * (n - 1)
        return 2 * nodes_per_hemisphere

    @property
    def longitudes(self):
        longitudes = np.zeros(self.num_nodes)
        counter = 0
        for n in range(2 * self.degree):
            circle = GridCircle(self.degree, n)
            num_nodes = circle.num_nodes
            longitudes[counter:(counter + num_nodes)] = circle.longitudes
            counter += num_nodes
        return longitudes

    @property
    def latitudes(self):
        latitudes = np.empty(self.num_nodes)
        counter = 0
        for n in range(2 * self.degree):
            num_nodes = GridCircle(self.degree, n).num_nodes
            latitudes[counter:(counter + num_nodes)] = self._circle_latitudes[n]
            counter = counter + num_nodes
        return latitudes

    @property
    def coordinates(self):
        return Coordinates(lat_lon_system, self.longitudes, self.latitudes)

    @property
    def triangles(self):
        vertices = - np.ones((self.num_triangles, 3), dtype=int)
        circle = GridCircle(self.degree, 0)
        counter = circle.num_nodes
        vertices[:counter] = circle.triangles_to_southern_circle()
        for n in range(1, self.degree - 1):
            circle = GridCircle(self.degree, n)
            num_nodes = circle.num_nodes
            vertices[counter:(counter + num_nodes)] = circle.triangles_to_northern_circle()
            counter += num_nodes
            vertices[counter:(counter + num_nodes)] = circle.triangles_to_southern_circle()
            counter += num_nodes
        circle = GridCircle(self.degree, self.degree - 1)
        num_nodes = circle.num_nodes
        vertices[counter:(counter + num_nodes)] = circle.triangles_to_northern_circle()
        counter += num_nodes
        circle = GridCircle(self.degree, self.degree)
        num_nodes = circle.num_nodes
        vertices[counter:(counter + num_nodes)] = circle.triangles_to_southern_circle()
        counter += num_nodes
        for n in range(self.degree + 1, 2 * self.degree - 1):
            circle = GridCircle(self.degree, n)
            num_nodes = circle.num_nodes
            vertices[counter:(counter + num_nodes)] = circle.triangles_to_northern_circle()
            counter += num_nodes
            vertices[counter:(counter + num_nodes)] = circle.triangles_to_southern_circle()
            counter += num_nodes
        circle = GridCircle(self.degree, 2 * self.degree - 1)
        num_nodes = circle.num_nodes
        vertices[counter:(counter + num_nodes)] = circle.triangles_to_northern_circle()
        counter += num_nodes
        return vertices

    def get_mesh_for_subdomain(self, bounds: DomainBoundingBox) -> TriangleMesh:
        latitudes_in_bounds = bounds.latitude.argwhere(self._circle_latitudes)
        first_latitude = max(0, latitudes_in_bounds[0] - 1)
        last_latitude = min(2 * self.degree - 1, latitudes_in_bounds[-1] + 1)
        all_triangles = []
        all_latitudes = []
        all_longitudes = []

        def get_coordinates(northern_circle, southern_circle):
            longitudes = np.concatenate([northern_circle.longitudes, southern_circle.longitudes])
            latitudes = np.zeros_like(longitudes)
            num_northern = northern_circle.num_nodes
            latitudes[:num_northern] = self._circle_latitudes[northern_circle.index_from_north]
            latitudes[num_northern:] = self._circle_latitudes[southern_circle.index_from_north]
            return Coordinates(lat_lon_system, longitudes, latitudes)

        for n in range(first_latitude, last_latitude):
            northern_circle = GridCircle(self.degree, n)
            first_index = northern_circle.first_index
            southern_circle = GridCircle(self.degree, n + 1)
            coords = get_coordinates(northern_circle, southern_circle)
            locations = LocationBatch(coords)
            vertices = northern_circle.triangles_to_southern_circle() - first_index
            triangles = TriangleMesh(locations, vertices)
            valid = bounds.intersects(triangles)
            all_triangles.append(vertices[valid] + first_index)
            all_longitudes.append(triangles.x[valid])
            all_latitudes.append(triangles.y[valid])
            vertices = southern_circle.triangles_to_northern_circle() - first_index
            triangles = TriangleMesh(locations, vertices)
            valid = bounds.intersects(triangles)
            all_triangles.append(vertices[valid] + first_index)
            all_longitudes.append(triangles.x[valid])
            all_latitudes.append(triangles.y[valid])

        all_triangles = np.concatenate(all_triangles, axis=0)
        unique, indices, inverse = np.unique(all_triangles.ravel(), return_inverse=True, return_index=True)
        all_triangles = np.reshape(inverse, all_triangles.shape)
        all_latitudes = np.concatenate(all_latitudes, axis=0).ravel()[indices]
        all_longitudes = np.concatenate(all_longitudes, axis=0).ravel()[indices]
        locations = LocationBatch(
            Coordinates(lat_lon_system, all_longitudes, all_latitudes),
            source_reference=unique
        )
        return TriangleMesh(locations, all_triangles)


class DataStore(object):

    def __init__(self, grid: OctahedralGrid, data: xr.Dataset):
        assert data.dims['values'] == grid.num_nodes
        self.grid = grid
        self.data = data

    @classmethod
    def from_config(cls, configs: Dict[str, Any], grid: OctahedralGrid, compute_level_heights: bool = False) -> 'DataStore':
        config_reader = ConfigReader(SourceConfiguration)
        logging.info('Loading data...')
        data = [
            config_reader.load_data(configs[key])
            for key in configs
        ]
        data = xr.merge(data, compat='override')

        if compute_level_heights:
            logging.info('Computing model level heights...')
            keys = set(data.data_vars.keys())
            assert 'z' in keys
            assert 'lnsp' in keys
            assert 't' in keys
            assert 'q' in keys
            z_model_levels = compute_physical_level_height(
                np.exp(data.lnsp.values)[None, :], data.z.values[None, :],
                data.t.values, data.q.values
            )
            logging.info('Merging data...')
            data = data.assign(z_model_levels=(('hybrid', 'values'), z_model_levels))

        return cls(grid, data)

    def get_lsm(self):
        if 'lsm' in self.data.data_vars.keys():
            return self.data['lsm']
        return None

    def query_site_data(self, domain_bounds):
        raise NotImplementedError()


class DomainData(object):

    def __init__(self, grid: OctahedralGrid, data_store: DataStore):
        self.grid = grid
        self.data_store = data_store
        self.bounding_box = None
        self.mesh = None
        self.site_data: xr.Dataset = None

    @property
    def sites(self):
        if self.mesh is None:
            return None
        return self.mesh.nodes

    def set_bounds(self, bounds: DomainLimits):
        if bounds is not None:
            self.bounding_box = DomainBoundingBox(bounds)
        else:
            self.bounding_box = None
        self._update()
        return self

    def _update(self):
        if self.bounding_box is not None:
            self.mesh = self.grid.get_mesh_for_subdomain(self.bounding_box)
            self.site_data = self.data_store.query_site_data(self.mesh.nodes)
        else:
            self.reset()
        return self

    def reset(self):
        return self.set_bounds(None)


class DomainSettingsView(object):
    domain_limits_changed = None

    def get_domain_settings(self):
        raise NotImplementedError()


class DomainController(object):
    domain_data_changed = None

    def __init__(
            self,
            settings_view: DomainSettingsView,
            domain_lr: DomainData, domain_hr: DomainData
    ):
        self.settings_view = settings_view
        self.domain_lr = domain_lr
        self.domain_hr = domain_hr
        self.settings_view.domain_limits_changed.connect(self._on_domain_changed)

    def _on_domain_changed(self):
        self._synchronize_domain_settings()
        self.domain_data_changed.emit()

    def _synchronize_domain_settings(self):
        bounds = self.settings_view.get_domain_settings()
        self.domain_lr.set_bounds(bounds)
        self.domain_hr.set_bounds(bounds)


class NeighborhoodType(Enum):
    RADIAL = 'radial'
    NEAREST_NEIGHBORS = 'nearest_neighbors'


class TreeType(Enum):
    AUTO = 'auto'
    KD_TREE = 'kd_tree'
    BALL_TREE = 'ball_tree'
    BRUTE = 'brute'


class NeighborhoodLookup(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        neighborhood_type: NeighborhoodType
        neighborhood_size: Union[int, float]
        tree_type: TreeType
        num_jobs: int # 1 for single-process, -1 for all processors
        lsm_threshold: float

    def __init__(self, data_store: DataStore):
        super().__init__(None)
        self.data_store = data_store
        self.search_structure: NearestNeighbors = None
        self._actions = {
            NeighborhoodType.NEAREST_NEIGHBORS: self._query_k_nearest_neighbors,
            NeighborhoodType.RADIAL: self._query_k_radial_neighbors,
        }

    @property
    def neighborhood_type(self) -> NeighborhoodType:
        return self._properties.neighborhood_type

    @property
    def neighborhood_size(self) -> Union[int, float]:
        return self._properties.neighborhood_size

    @property
    def tree_type(self) -> TreeType:
        return self._properties.tree_type

    @property
    def num_jobs(self) -> int:
        return self._properties.num_jobs

    @property
    def lsm_threshold(self) -> float:
        return self._properties.lsm_threshold

    def _refresh_search_structure(self):
        properties = self._properties
        self.search_structure = NearestNeighbors(algorithm=properties.tree_type.value, leaf_size=100,
                                            n_jobs=properties.num_jobs)
        data = self.data_store.get_lsm()
        if data is None:
            raise RuntimeError('[ERROR] Error while building neighborhood lookup from properties: LSM unavailable.')
        mask = np.argwhere(data.values >= properties.lsm_threshold)
        data = data.isel(values=mask)
        coords = Coordinates.from_xarray(data).as_geocentric().values
        self.search_structure.fit(coords)

    def set_properties(self, properties: 'NeighborhoodLookup.Properties'):
        if self.tree_update_required(properties):
            super().set_properties(properties)
            self._refresh_search_structure()
        else:
            super().set_properties(properties)
        return self

    def tree_update_required(self, properties: 'NeighborhoodLookup.Properties') -> bool:
        if self._properties is None:
            return True
        if properties.lsm_threshold != self.lsm_threshold:
            return True
        if properties.tree_type != self.tree_type:
            return True
        if properties.num_jobs != self.num_jobs:
            return True
        return False

    def query_neighbor_graph(self, locations: LocationBatch) -> NeighborhoodGraph:
        action = self._actions.get(self.neighborhood_type, None)
        if action is None:
            raise RuntimeError()
        return action(locations)

    def _query_k_nearest_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)

    def _query_radial_neighbors(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return RadialNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)


class NeighborhoodData(object):

    def __init__(self, domain_model: DomainData):
        self.domain_model = domain_model
        self.lookup = NeighborhoodLookup(self.data_store)
        self._neighborhood_graph = None
        self.data = None

    @property
    def data_store(self):
        return self.domain_model.data_store

    def set_neighborhood_properties(self, properties: NeighborhoodLookup.Properties):
        self.lookup.set_properties(properties)
        self.update()
        return self

    def update(self):
        domain_sites = self.domain_model.sites
        self._neighborhood_graph = self.lookup.query_neighbor_graph(domain_sites)
        self.data = self.data_store.query_site_data(self._neighborhood_graph.links['neighbors'])

    def query_neighbor_graph(self, locations: LocationBatch) -> NeighborhoodGraph:
        if self.lookup is None:
            raise RuntimeError('[ERROR]  Error while querying neighborhood graph: lookup not found.')
        return self.lookup.query_neighbor_graph(locations)


class NeighborhoodSettingsView(object):
    neighborhood_settings_changed = None

    def get_neighborhood_settings(self):
        raise NotImplementedError()


class NeighborhoodController(object):
    neighborhood_data_changed = None

    def __init__(
            self,
            settings_view: NeighborhoodSettingsView,
            model: NeighborhoodData,
    ):
        self.settings_view = settings_view
        self.settings_view.neighborhood_settings_changed.connect(self._on_neighborhood_settings_changed)
        self.model = model
        self._synchronize_neighborhood_settings()

    def _on_neighborhood_settings_changed(self):
        self._synchronize_neighborhood_settings()
        self.neighborhood_data_changed.emit()

    def _synchronize_neighborhood_settings(self):
        settings = self.settings_view.get_neighborhood_settings()
        self.model.set_neighborhood_properties(settings)


class DownscalingMethod(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    NETWORK = 'network'


class Downscaler(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        pass

    def new_properties_valid(self, properties: 'Downscaler.Properties') -> bool:
        if not isinstance(properties, self.Properties):
            return False
        return True

    @classmethod
    def from_properties(cls, properties):
        raise NotImplementedError()

    def downscale(self, site_data: xr.Dataset, neighbor_data: xr.Dataset, target_domain: DomainData):
        raise NotImplementedError()


class DownscalerSettingsView(object):
    downscaler_settings_changed = None

    def get_settings(self):
        raise NotImplementedError()


class DownscalerController(object):
    downscaler_changed = None

    def __init__(self, settings_view: DownscalerSettingsView, downscaler: Downscaler):
        self.settings_view = settings_view
        self.model = downscaler
        self.settings_view.downscaler_settings_changed.connect(self._on_downscaler_settings_changed)

    def _on_downscaler_settings_changed(self):
        settings = self.settings_view.get_settings()
        self.model.set_properties(settings)
        self.downscaler_changed.emit()


class DownscalingPipeline(object):

    def __init__(
            self,
            site_data: DomainData, neighborhood_data: NeighborhoodData,
            target_data: DomainData,
            downscaler_model: DownscalingMethod,
    ):
        self.site_data = site_data
        self.neighborhood_data = neighborhood_data
        self.target_data = target_data
        self.downscaler_model = downscaler_model
        self.output = None

    def query_neighborhoods(self):
        domain_sites = self.source_data.sites
        self._neighboorhood_graph = self.neighborhood_model.query_neighbor_graph(domain_sites)

    def _update(self):
        if self.downscaler is None:
            return
        self.output = self.downscaler.process(self.target_domain_data, self.neighborhood_model.neighbor_samples)


class DownscalingPipelineController(object):
    output_data_changed = None

    def __init__(
            self,
            downscaling_pipeline: DownscalingPipeline,
            neighborhood_controller: NeighborhoodController,
            downscaler_controller: DownscalerController,
            domain_controller: DomainController
    ):
        self.neighborhood_controller = neighborhood_controller
        self.downscaler_controller = downscaler_controller
        self.domain_controller = domain_controller
        self.domain_controller.domain_data_changed.connect(self._on_domain_data_changed)

    def _on_domain_data_changed(self):
        self.neighborhood_controller.model.update()
        self.downscaler_controller.model.update()






from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import pyvista as pv
from scipy.special import roots_legendre
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import AngularInterval, LocationBatch, Coordinates, TriangleBatch, lat_lon_system
from src.model.neighborhood_lookup.neighborhood_graphs import UniformNeighborhoodGraph, RadialNeighborhoodGraph
from src.model.visualization.interface import PropertyModel, PropertyModelUpdateError

N_LOWRES = 1280
N_HIGHRES = 8000


class TriangleMesh(object):

    def __init__(self, locations: LocationBatch, vertices: np.ndarray):
        self.locations = locations
        self.vertices = vertices

    @property
    def x(self):
        return self.locations.x

    @property
    def y(self):
        return self.locations.y

    @property
    def source_reference(self):
        return self.locations.source_reference

    @property
    def coordinates(self):
        return self.locations.coords

    @property
    def num_nodes(self):
        return len(self.x)

    def get_horizontal_coordinates(self, transform=None) -> Tuple[np.ndarray, np.ndarray]:
        if transform is not None:
            return transform(self.x, self.y)
        return self.x, self.y

    def get_vertex_positions(self, z: np.ndarray = None, transform=None) -> np.ndarray:
        coords = self.coordinates
        if z is None:
            z = np.zeros_like(coords.x)
        coordinates = np.stack([*coords.components, z], axis=-1)
        return coordinates

    def get_triangle_vertices(self, add_prefix: bool = False) -> np.ndarray:
        prefix_offset = int(add_prefix)
        faces = np.zeros((len(self.vertices), 3 + prefix_offset), dtype=int)
        if add_prefix:
            faces[:, 0] = 3
        faces[:, prefix_offset:] = self.vertices
        return faces

    def to_polydata(self, z: np.ndarray = None, transform=None) -> pv.PolyData:
        faces = np.concatenate([np.full((len(self.vertices), 1), 3, dtype=int), self.vertices], axis=-1)
        points = self.get_vertex_positions(z, transform)
        return pv.PolyData(points, faces)


class DataStore(object):

    def query_site_data(self, domain_bounds):
        raise NotImplementedError()



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

    def contains(self, coords: Coordinates) -> np.ndarray:
        longitude, latitude = coords.as_lat_lon().components
        return np.logical_and(
            self.latitude.contains(latitude),
            self.longitude.contains(longitude)
        )

    def intersects(self, triangles: TriangleBatch) -> np.ndarray:
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


@lru_cache(maxsize=2)
def _get_legendre_latitudes(n: int):
    return - np.rad2deg(np.arcsin(roots_legendre(2 * n)[0]))


class OctahedralGrid(object):

    def __init__(self, degree: int):
        self.degree = int(degree)

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
        circle_latitudes = _get_legendre_latitudes(self.degree)
        latitudes = np.empty(self.num_nodes)
        counter = 0
        for n in range(2 * self.degree):
            num_nodes = GridCircle(self.degree, n).num_nodes
            latitudes[counter:(counter + num_nodes)] = circle_latitudes[n]
            counter = counter + num_nodes
        return latitudes

    @property
    def coordinates(self):
        return (self.latitudes, self.longitudes)

    @property
    def triangles(self):
        vertices = -np.ones((self.num_triangles, 3), dtype=int)
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

    def get_subgrid(self, bounds: DomainBoundingBox) -> TriangleMesh:
        circle_latitudes = _get_legendre_latitudes(self.degree)
        latitudes_in_bounds = bounds._latitude_interval.argwhere(circle_latitudes)
        first_latitude = max(0, latitudes_in_bounds[0] - 1)
        last_latitude = min(2 * self.degree - 1, latitudes_in_bounds[-1] + 1)
        all_triangles = []
        all_latitudes = []
        all_longitudes = []

        def get_coordinates(northern_circle, southern_circle):
            longitudes = np.concatenate([northern_circle.longitudes, southern_circle.longitudes])
            latitudes = np.zeros_like(longitudes)
            num_northern = northern_circle.num_nodes
            latitudes[:num_northern] = circle_latitudes[northern_circle.index_from_north]
            latitudes[num_northern:] = circle_latitudes[southern_circle.index_from_north]
            return Coordinates(lat_lon_system, longitudes, latitudes)

        for n in range(first_latitude, last_latitude):
            northern_circle = GridCircle(self.degree, n)
            first_index = northern_circle.first_index
            southern_circle = GridCircle(self.degree, n + 1)
            coords = get_coordinates(northern_circle, southern_circle)
            locations = LocationBatch(coords)
            vertices = northern_circle.triangles_to_southern_circle() - first_index
            triangles = TriangleBatch(locations, vertices)
            valid = bounds.intersects(triangles)
            all_triangles.append(vertices[valid] + first_index)
            all_longitudes.append(triangles.x[valid])
            all_latitudes.append(triangles.y[valid])
            vertices = southern_circle.triangles_to_northern_circle() - first_index
            triangles = TriangleBatch(locations, vertices)
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


class SubDomainData(object):
    GRID_LOWRES = OctahedralGrid(N_LOWRES)
    GRID_HIGHRES = OctahedralGrid(N_HIGHRES)

    def __init__(self, grid: OctahedralGrid, data_store: DataStore):
        self.grid = grid
        self.data_store = data_store
        self.bounding_box = None
        self.mesh = None
        self.site_data: xr.Dataset = None

    def set_bounds(self, bounds: DomainLimits):
        if bounds is not None:
            self.bounding_box = DomainBoundingBox(bounds)
        else:
            self.bounding_box = None
        self._update()
        return self

    def _update(self):
        if self.bounding_box is not None:
            self.mesh = self.grid.get_subgrid(self.bounding_box)
            self.site_data = self.data_store.query_site_data(self.mesh.locations)
        else:
            self.reset()
        return self

    def reset(self):
        self.mesh = None
        self.site_data = None


class DomainSettingsView(object):
    domain_limits_changed = None

    def get_domain_settings(self):
        raise NotImplementedError()


class DomainController(object):
    domain_data_changed = None

    def __init__(
            self,
            settings_view: DomainSettingsView,
            domain_lr: SubDomainData, domain_hr: SubDomainData
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

    def __init__(self, search_structure: NearestNeighbors, properties: 'NeighborhoodLookup.Properties'):
        super().__init__(properties)
        self.search_structure = search_structure
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

    @classmethod
    def from_properties(cls, properties: 'NeighborhoodLookup.Properties', data_store: DataStore):
        search_structure = NearestNeighbors(algorithm=properties.tree_type.value, leaf_size=100, n_jobs=properties.num_jobs)
        data = data_store.get_lsm_data()
        mask = np.argwhere(data.values >= properties.lsm_threshold)
        data = data.isel(values=mask)
        coords = Coordinates.from_xarray(data).as_geocentric().values
        search_structure.fit(coords)
        return cls(search_structure, properties)

    def set_properties(self, properties) -> 'NeighborhoodLookup':
        if not self._new_properties_valid(properties):
            raise PropertyModelUpdateError()
        return super().set_properties(properties)

    def new_properties_valid(self, properties: 'NeighborhoodLookup.Properties') -> bool:
        if properties.lsm_threshold != self.lsm_threshold:
            return False
        if properties.tree_type != self.tree_type:
            return False
        if properties.num_jobs != self.num_jobs:
            return False
        return True

    def query_neighbor_graph(self, locations: LocationBatch):
        action = self._actions.get(self.neighborhood_type, None)
        if action is not None:
            raise RuntimeError()
        return action(locations)

    def _query_k_nearest_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)

    def _query_radial_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return RadialNeighborhoodGraph.from_tree_query(locations, self.search_structure, self.neighborhood_size)


class NeighborhoodModel(object):

    def __init__(self, domain_model: SubDomainData, data_store: DataStore):
        self.domain_model = domain_model
        self.data_store = data_store
        self.lookup = None
        self.neighbor_graph = None
        self.neighbor_samples = None

    def _update(self):
        if self.lookup is None:
            self.neighbor_graph = None
            self.neighbor_samples = None
            return self
        locations = self.domain_model.mesh.locations
        self.neighbor_graph = self.lookup.query_neighbor_graph(locations)
        self.update_neighbor_samples()
        return self

    def update_neighbor_samples(self):
        self.neighbor_samples = self.data_store.query_site_data(self.neighbor_graph.links['neighbor'])
        return self

    def set_neighborhood_properties(self, properties: NeighborhoodLookup.Properties):
        update_successful = False
        if self.lookup is not None:
            try:
                self.lookup.set_properties(properties)
            except PropertyModelUpdateError:
                pass
            else:
                update_successful = True
        if not update_successful:
            self.lookup = NeighborhoodLookup.from_properties(properties, self.domain_model.data_store)
        self._update()
        return self


class NeighborhoodSettingsView(object):
    neighborhood_settings_changed = None

    def get_neighborhood_settings(self):
        raise NotImplementedError()


class NeighborhoodController(object):
    neighborhood_data_changed = None

    def __init__(
            self,
            settings_view: NeighborhoodSettingsView,
            model: NeighborhoodModel,
            domain_controller: DomainController
    ):
        self.settings_view = settings_view
        self.settings_view.neighborhood_settings_changed.connect(self._on_neighborhood_settings_changed)
        self.model = model
        self._synchronize_neighborhood_settings()
        self.domain_controller = domain_controller
        self.domain_controller.domain_data_changed.connect(self._on_domain_data_changed)

    def _on_neighborhood_settings_changed(self):
        self._synchronize_neighborhood_settings()
        self.neighborhood_data_changed.emit()

    def _synchronize_neighborhood_settings(self):
        settings = self.settings_view.get_neighborhood_settings()
        self.model.set_neighborhood_properties(settings)

    def _on_domain_data_changed(self):
        self.model.update_neighbor_samples()
        self.neighborhood_data_changed.emit()


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

    def downscale(self, site_data: xr.Dataset, neighbor_data: xr.Dataset, target_domain: SubDomainData):
        raise NotImplementedError()


class DownscalingModel(object):

    def __init__(self, target_domain_data: SubDomainData, neighborhood_model: NeighborhoodModel):
        self.target_domain_data = target_domain_data
        self.neighborhood_model = neighborhood_model
        self.downscaler = None
        self.output = None

    def set_downscaler_properties(self, properties):
        update_successful = False
        if self.downscaler is not None:
            try:
                self.downscaler.update_properties(properties)
            except PropertyModelUpdateError:
                pass
            else:
                update_successful = True
        if not update_successful:
            self.downscaler = Downscaler.from_properties(properties)
        self._update()
        return self

    def _update(self):
        if self.downscaler is None:
            return
        self.output = self.downscaler.process(self.target_domain_data, self.neighborhood_model.neighbor_samples)






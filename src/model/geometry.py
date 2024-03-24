from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import pyvista as pv
import cartopy.crs as ccrs
from scipy.special import roots_legendre
import xarray as xr


geocentric_system = ccrs.Geocentric()
xyz_system = geocentric_system
lat_lon_system = ccrs.PlateCarree()


class Coordinates(object):

    @classmethod
    def from_xarray(cls, data: Union[xr.Dataset, xr.DataArray]):
        return cls(lat_lon_system, data['longitude'].values, data['latitude'].values)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame):
        return cls(lat_lon_system, data['longitude'].values, data['latitude'].values)

    @classmethod
    def from_lat_lon(cls, latitude: np.ndarray, longitude: np.ndarray):
        return cls(lat_lon_system, longitude, latitude)

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
        return np.argwhere(self.contains(x)).ravel()

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
        if self.source_reference is not None:
            source_reference = self.source_reference[location_ids]
        else:
            source_reference = None
        if self.elevation is not None:
            elevation = self.elevation[location_ids]
        else:
            elevation = None
        return self.__class__(coords, elevation=elevation, source_reference=source_reference)

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
            self._x = self.nodes.x#[self.vertices]
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = self.nodes.y#[self.vertices]
        return self._y

    @property
    def z(self):
        return self.nodes.z

    @property
    def left(self) -> np.ndarray:
        return np.amin(self.x[self.vertices], axis=-1)

    @property
    def right(self) -> np.ndarray:
        return np.amax(self.x[self.vertices], axis=-1)

    @property
    def bottom(self) -> np.ndarray:
        return np.amin(self.y[self.vertices], axis=-1)

    @property
    def top(self) -> np.ndarray:
        return np.amax(self.y[self.vertices], axis=-1)


    @property
    def source_reference(self):
        return self.nodes.source_reference

    @property
    def node_coordinates(self):
        return self.nodes.coords

    @property
    def num_nodes(self):
        return len(self.nodes)

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
        faces = self.get_faces(add_prefix=True)
        points = self.get_node_positions()
        return pv.PolyData(points, faces)


class WedgeMesh(object):

    def __init__(self, base_mesh: TriangleMesh, z: np.ndarray):
        self.base_mesh = base_mesh
        assert len(z.shape) == 2 and z.shape[-1] == self.base_mesh.num_nodes
        self.z = z

    @property
    def num_levels(self):
        return len(self.z)

    def get_node_positions(self) -> np.ndarray:
        level_coordinates = self.base_mesh.get_node_positions()
        coordinates = np.tile(level_coordinates, (self.num_levels, 1))
        coordinates[:, -1] = self.z.ravel()
        return coordinates

    def get_wedges(self, add_prefix: bool = False):
        triangles = self.base_mesh.vertices
        num_triangles = len(triangles)
        num_nodes = self.base_mesh.num_nodes
        prefix_offset = int(add_prefix)
        j_lower = prefix_offset
        j_upper = prefix_offset + 3
        wedges = np.zeros(((self.num_levels - 1) * num_triangles, 6 + prefix_offset), dtype=int)
        if add_prefix:
            wedges[:, 0] = 6
        current_triangles = triangles.copy()
        for level in range(self.num_levels - 1):
            i_lower = level * num_triangles
            i_upper = (level + 1) * num_triangles
            wedges[i_lower:i_upper, j_lower:j_upper] = current_triangles
            current_triangles += num_nodes
            wedges[i_lower:i_upper, j_upper:] = current_triangles
        return wedges

    def to_wedge_grid(self) -> pv.UnstructuredGrid:
        coords = self.get_node_positions()
        wedges = self.get_wedges(add_prefix=True)
        cell_types = [pv.CellType.WEDGE] * len(wedges)
        return pv.UnstructuredGrid(wedges, cell_types, coords)


@dataclass
class DomainLimits(object):
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float

    def plus_safety_margin(self):
        return DomainLimits(self.min_latitude - 0.5, self.max_latitude + 0.5, self.min_longitude - 0.5, self.max_latitude + 0.5)


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
        self.index_from_pole = int(self.index_from_north_to_index_from_pole(self.index_from_north, self.degree))

    @staticmethod
    def index_from_north_to_index_from_pole(index_from_north: np.ndarray, degree: np.ndarray) -> np.ndarray:
        return np.fmin(index_from_north, 2 * degree - 1 - index_from_north)

    @property
    def num_nodes(self):
        return 4 * self.index_from_pole + 20

    @property
    def first_index(self):
        n = self.index_from_pole
        return self.first_index_on_circle(n, self.degree)

    @classmethod
    def first_index_on_circle(cls, index_from_north: np.ndarray, degree: np.ndarray) -> np.ndarray:
        n = cls.index_from_north_to_index_from_pole(index_from_north, degree)
        total_on_hemisphere = 20 * degree + 2 * degree * (degree - 1)
        total_from_pole = 20 * (n + 1) + 2 * n * (n + 1)
        first_index = np.where(
            index_from_north < degree,
            20 * n + 2 * n * (n - 1),
            2 * total_on_hemisphere - total_from_pole,
        )
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
        self.circle_latitudes = - np.rad2deg(np.arcsin(roots_legendre(2 * self.degree)[0]))

    def find_nearest_neighbor(self, sites: LocationBatch) -> LocationBatch:
        coords = sites.coords.as_lat_lon()
        longitudes, latitudes = coords.components
        latitude_index = np.searchsorted(-self.circle_latitudes, -latitudes)
        n_upper_circle = np.clip(latitude_index - 1, a_min=0, a_max=(2 * self.degree - 1))
        latitudes_upper_circle = self.circle_latitudes[n_upper_circle]
        n_lower_circle = np.clip(latitude_index, a_min=0, a_max=(2 * self.degree - 1))
        latitudes_lower_circle = self.circle_latitudes[n_lower_circle]

        def get_index_on_circle(n_circle):
            nodes_on_circle = 4 * GridCircle.index_from_north_to_index_from_pole(n_circle, self.degree) + 20
            first_index_on_circle = GridCircle.first_index_on_circle(n_circle, self.degree)
            lons_rescaled = (longitudes % 360) * nodes_on_circle / 360
            index_left_on_circle = np.floor(lons_rescaled) % nodes_on_circle
            longitude_left_on_circle = ((360 * index_left_on_circle / nodes_on_circle) + 180) % 360 - 180
            index_right_on_circle = np.ceil(lons_rescaled) % nodes_on_circle
            longitude_right_on_circle = ((360 * index_right_on_circle / nodes_on_circle) + 180) % 360 - 180
            output = (
                (index_left_on_circle.astype(int) + first_index_on_circle,
                 index_right_on_circle.astype(int) + first_index_on_circle),
                (longitude_left_on_circle, longitude_right_on_circle),
            )
            return output

        indices_upper, lons_upper = get_index_on_circle(n_upper_circle)
        indices_lower, lons_lower = get_index_on_circle(n_lower_circle)

        longitudes_test = np.stack(lons_upper + lons_lower, axis=0)
        latitudes_test = np.stack(
            (
                latitudes_upper_circle, latitudes_upper_circle,
                latitudes_lower_circle, latitudes_lower_circle
            ), axis=0
        )
        xyz_test = Coordinates.from_lat_lon(
            latitudes_test.ravel(), longitudes_test.ravel()
        ).as_xyz().values
        xyz_test = np.reshape(xyz_test, (*longitudes_test.shape, -1))

        xyz_sites = coords.as_xyz().values

        d = np.sum(np.square(xyz_test - xyz_sites[None, ...]), axis=-1)
        min_distance_index = np.argmin(d, axis=0)
        site_index = np.arange(len(min_distance_index))

        indices_test = np.stack(indices_upper + indices_lower, axis=0)

        nearest_index = indices_test[min_distance_index, np.arange(len(min_distance_index))]
        coords_nearest = Coordinates.from_lat_lon(
            latitudes_test[min_distance_index, site_index],
            longitudes_test[min_distance_index, site_index]
        )
        locations_nearest = LocationBatch(coords_nearest, source_reference=nearest_index)

        return locations_nearest

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
            latitudes[counter:(counter + num_nodes)] = self.circle_latitudes[n]
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

        latitudes_in_bounds = bounds.latitude.argwhere(self.circle_latitudes)
        first_latitude = max(0, latitudes_in_bounds[0] - 1)
        last_latitude = min(2 * self.degree - 1, latitudes_in_bounds[-1] + 1)
        all_triangles = []
        all_latitudes = []
        all_longitudes = []

        def get_coordinates(northern_circle, southern_circle):
            longitudes = np.concatenate([northern_circle.longitudes, southern_circle.longitudes])
            latitudes = np.zeros_like(longitudes)
            num_northern = northern_circle.num_nodes
            latitudes[:num_northern] = self.circle_latitudes[northern_circle.index_from_north]
            latitudes[num_northern:] = self.circle_latitudes[southern_circle.index_from_north]
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
            tvv_ = triangles.vertices[valid]
            all_longitudes.append(triangles.x[tvv_])
            all_latitudes.append(triangles.y[tvv_])
            vertices = southern_circle.triangles_to_northern_circle() - first_index
            triangles = TriangleMesh(locations, vertices)
            valid = bounds.intersects(triangles)
            all_triangles.append(vertices[valid] + first_index)
            tvv_ = triangles.vertices[valid]
            all_longitudes.append(triangles.x[tvv_])
            all_latitudes.append(triangles.y[tvv_])

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


def _verify_neighbor_lookup():
    import time
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import pyplot as plt

    num_samples = 1000000
    lons = np.random.random(size=(num_samples,)) * 360 - 180
    lats = np.random.random(size=(num_samples,)) * 180 - 90
    coords = Coordinates.from_lat_lon(lats, lons)

    grid = OctahedralGrid(2560)
    coords_grid = grid.coordinates
    xyz = coords_grid.as_xyz().values

    t1 = time.time()
    tree = NearestNeighbors()
    tree.fit(xyz)
    t2 = time.time()
    xyz_query = coords.as_xyz().values
    _, neighbor_tree = tree.kneighbors(xyz_query, n_neighbors=1)
    t3 = time.time()

    neighbor_tree = neighbor_tree.ravel()
    locations = LocationBatch(coords)

    t4 = time.time()
    neighbor_manual = grid.nearest_neighbor(locations)
    t5 = time.time()

    print(f"""
    Fitting tree: {t2 - t1} seconds
    Query tree: {t3 - t2} seconds
    Manual computation: {t5 - t4} seconds
    """)
    mask = neighbor_tree != neighbor_manual

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(lons[mask], lats[mask], label='samples')
    ax.scatter((coords_grid.x + 180) % 360 - 180, coords_grid.y, label='grid')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(np.any(mask))

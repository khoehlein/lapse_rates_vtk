import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre
import cartopy.crs as ccrs


class AxisBounds(object):

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


class RegionBounds(object):

    def __init__(self, min_latitude: float, max_latitude: float, min_longitude: float, max_longitude: float, longitude_period: float = 360.):
        self.latitude = AxisBounds(min_latitude, max_latitude)
        self.longitude = AxisBounds(min_longitude, max_longitude, period=longitude_period)

    def contains(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        return np.logical_and(self.latitude.contains(latitude), self.longitude.contains(longitude))

    def intersects(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        # approximate triangles as rectangles
        left = np.amin(longitude, axis=-1)
        right = np.amax(longitude, axis=-1)
        overlaps_longitude = np.logical_or(self.longitude.contains(left), self.longitude.contains(right))
        top = np.amax(latitude, axis=-1)
        bottom = np.amin(latitude, axis=-1)
        overlaps_latitude = np.logical_or(self.latitude.contains(top), self.latitude.contains(bottom))
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

    def get_subgrid(self, bounds: RegionBounds, rescale_indices=True, return_coordinates=True):
        circle_latitudes = _get_legendre_latitudes(self.degree)
        latitudes_in_bounds = np.argwhere(np.logical_and(circle_latitudes >= bounds.latitude.min, circle_latitudes <= bounds.latitude.max)).ravel()
        first_latitude = max(0, latitudes_in_bounds[0] - 1)
        last_latitude = min(2 * self.degree - 1, latitudes_in_bounds[-1] + 1)
        all_triangles = []
        all_latitudes = []
        all_longitudes = []
        for n in range(first_latitude, last_latitude):
            northern_circle = GridCircle(self.degree, n)
            southern_circle = GridCircle(self.degree, n + 1)
            longitudes_north = northern_circle.longitudes
            longitudes_south = southern_circle.longitudes
            triangles = northern_circle.triangles_to_southern_circle()
            longitudes = np.zeros(triangles.shape)
            longitudes[:, 0] = longitudes_south[triangles[:, 0] - southern_circle.first_index]
            longitudes[:, 1:] = longitudes_north[triangles[:, 1:] - northern_circle.first_index]
            latitudes = np.zeros(triangles.shape)
            latitudes[:, 0] = circle_latitudes[n + 1]
            latitudes[:, 1:] = circle_latitudes[n]
            valid = bounds.intersects(latitudes, longitudes)
            all_triangles.append(triangles[valid])
            if return_coordinates:
                all_latitudes.append(latitudes[valid])
                all_longitudes.append(longitudes[valid])
            triangles = southern_circle.triangles_to_northern_circle()
            longitudes = np.zeros(triangles.shape)
            longitudes[:, :-1] = longitudes_south[triangles[:, :-1] - southern_circle.first_index]
            longitudes[:, -1] = longitudes_north[triangles[:, -1] - northern_circle.first_index]
            latitudes = np.zeros(triangles.shape)
            latitudes[:, :2] = circle_latitudes[n + 1]
            latitudes[:, -1] = circle_latitudes[n]
            valid = bounds.intersects(latitudes, longitudes)
            all_triangles.append(triangles[valid])
            if return_coordinates:
                all_latitudes.append(latitudes[valid])
                all_longitudes.append(longitudes[valid])
        all_triangles = np.concatenate(all_triangles, axis=0)
        if not rescale_indices and not return_coordinates:
            return all_triangles
        unique, indices, inverse = np.unique(all_triangles.ravel(), return_inverse=True, return_index=True)
        if rescale_indices:
            all_triangles = np.reshape(inverse, all_triangles.shape)
        if return_coordinates:
            all_latitudes = np.concatenate(all_latitudes, axis=0).ravel()[indices]
            all_longitudes = np.concatenate(all_longitudes, axis=0).ravel()[indices]
            return all_triangles, unique, (all_latitudes, all_longitudes) 
        return all_triangles, unique


def get_xyz(latitude, longitude):
    return ccrs.Geocentric().transform_points(ccrs.Geodetic(), longitude, latitude)

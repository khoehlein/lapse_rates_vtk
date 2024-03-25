import pandas as pd
import numpy as np
import xarray as xr
import pyvista as pv

from src.model.geometry import TriangleMesh, Coordinates, LocationBatch
from src.paper.volume_visualization.scaling import ScalingParameters


class StationData(object):

    def __init__(
            self,
            station_data: pd.DataFrame, terrain_data: xr.Dataset,
            scalar_key=None
    ):
        self.scalar_key = scalar_key
        self.station_data = station_data
        self.terrain_data = terrain_data

        self._points = np.zeros((len(self.station_data), 3))
        self._points[:, 0] = self.station_data['longitude'].values
        self._points[:, 1] = self.station_data['latitude'].values
        self._compute_terrain_altitude()
        self._compute_effective_gradients()

    def _compute_terrain_altitude(self):
        terrain_mesh = TriangleMesh(LocationBatch(Coordinates.from_xarray(self.terrain_data)), self.terrain_data['triangles'].values).to_polydata()
        terrain_mesh['elevation'] = self.terrain_data['z_surf'].values.ravel()
        station_sites = pv.PolyData(self._points.copy())
        self._terrain_elevation = np.asarray(station_sites.sample(terrain_mesh)['elevation'])
        self._relative_elevation = self.station_data['elevation'].values - self._terrain_elevation
        self.station_data['elevation_difference'] = self._relative_elevation.ravel()

    def _compute_effective_gradients(self):
        z_diff = self.station_data['elevation_difference'].values
        z_diff = np.sign(z_diff) * (np.abs(z_diff) + 1)
        self.station_data['grad_t'] = 1000. * self.station_data['difference'] / z_diff

    def compute_station_elevation(self, scale_params: ScalingParameters):
        inv_scale = 1.0 / scale_params.scale
        z = self._relative_elevation.copy()
        if scale_params.offset_scale != 1.:
            z *= scale_params.offset_scale
        z += self._terrain_elevation
        z *= inv_scale
        return z

    def compute_terrain_elevation(self, scale_params: ScalingParameters):
        inv_scale = 1.0 / scale_params.scale
        return self._terrain_elevation * inv_scale

    def get_station_sites(self, scale_params: ScalingParameters):
        points = self._points.copy()
        z = self.compute_station_elevation(scale_params)
        points[:, -1] = z
        mesh = pv.PolyData(points)
        if self.scalar_key is not None:
            mesh[self.scalar_key] = self.station_data[self.scalar_key].values
        return mesh

    def get_station_reference(self, scale_parameters: ScalingParameters):
        points = np.tile(self._points, (2, 1))
        n = len(points) // 2
        points[:n, -1] = self.compute_station_elevation(scale_parameters)
        points[n:, -1] = self.compute_terrain_elevation(scale_parameters)
        lines = np.zeros((n, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(n)
        lines[:, 2] = lines[:, 1] + n
        mesh = pv.PolyData(points, lines.ravel())
        if self.scalar_key is not None:
            mesh[self.scalar_key] = self.station_data[self.scalar_key].values
        return mesh

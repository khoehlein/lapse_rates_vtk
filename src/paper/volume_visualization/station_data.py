import pandas as pd
import numpy as np
import xarray as xr
import pyvista as pv

from src.model.geometry import TriangleMesh, Coordinates
from src.paper.volume_visualization.scaling import ScalingParameters


class StationData(object):

    def __init__(
            self,
            station_data: pd.DataFrame, terrain_data: xr.Dataset
    ):
        self.station_data = station_data
        self.terrain_data = terrain_data

        self._points = np.zeros((len(self.station_data), 3))
        self._points[:, 0] = self.station_data['longitude'].values
        self._points[:, 1] = self.station_data['latitude'].values

        self._compute_terrain_altitude()

    def _compute_terrain_altitude(self):
        longitude = self.terrain_data['longitude'].values
        latitude = self.terrain_data['latitude'].values
        terrain_coords = np.stack([longitude, latitude, np.zeros_like(longitude)], axis=-1)
        terrain_mesh = TriangleMesh(terrain_coords, self.terrain_data['triangles'].values).to_polydata()
        terrain_mesh['elevation'] = self.terrain_data['z_surf'].values.ravel()
        station_sites = pv.PolyData(self._points.copy())
        self._terrain_elevation = np.asarray(station_sites.sample(terrain_mesh)['elevation'])
        self._relative_elevation = self.station_data['elevation'].values - self._terrain_elevation

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
        return pv.Polydata(points)

    def get_station_reference(self, scale_parameters: ScalingParameters):
        points = np.tile(self._points, (2, 1))
        n = len(points) // 2
        points[:n, -1] = self.compute_station_elevation(scale_parameters)
        points[n:, -1] = self.compute_terrain_elevation(scale_parameters)
        lines = np.zeros((n, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(n)
        lines[:, 2] = lines[:, 1] + n
        return pv.PolyData(points, lines.ravel())

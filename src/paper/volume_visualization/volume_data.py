import numpy as np
import xarray as xr
import pyvista as pv

from src.model.geometry import TriangleMesh, LocationBatch, Coordinates, WedgeMesh
from src.paper.volume_visualization.plotter_slot import ContourParameters
from src.paper.volume_visualization.scaling import ScalingParameters


class VolumeData(object):

    def __init__(
            self,
            field_data: xr.Dataset, terrain_data: xr.Dataset,
            scalar_key: str = None, terrain_level_key: str = 'z_surf', model_level_key: str = 'z_model_levels'
    ):
        self.field_data = field_data
        self.terrain_data = terrain_data

        self.scalar_key = scalar_key
        self.terrain_level_key = terrain_level_key
        self.model_level_key = model_level_key

        self._terrain_level_elevation = self.terrain_data[self.terrain_level_key].values
        z = None
        try:
            z = self.field_data[self.model_level_key].values
        except KeyError:
            pass
        if z is None:
            z = self.terrain_data[self.model_level_key].values[None, :]
        assert z is not None
        self._relative_elevation = z - self._terrain_level_elevation

    def get_volume_mesh(self, scale_params: ScalingParameters, use_scalar_key: bool = True) -> pv.UnstructuredGrid:
        surface_mesh = TriangleMesh(
            LocationBatch(Coordinates.from_xarray(self.terrain_data)),
            self.terrain_data['triangles'].values
        )
        z = self.compute_elevation_coordinate(scale_params)
        mesh = WedgeMesh(surface_mesh, z)
        mesh = mesh.to_wedge_grid()
        if use_scalar_key and self.scalar_key is not None:
            mesh[self.scalar_key] = self.field_data[self.scalar_key].values.ravel()
        return mesh

    def get_level_mesh(self, scale_params: ScalingParameters, use_scalar_key: bool = True) -> pv.PolyData:
        coords = Coordinates.from_xarray(self.terrain_data)
        num_nodes = len(coords)
        triangles_base = self.terrain_data['triangles'].values
        num_triangles = len(triangles_base)
        z = self.compute_elevation_coordinate(scale_params)
        num_levels = z.shape[0]
        faces = np.zeros((num_triangles * num_levels, 4), dtype=int)
        faces[:, 0] = 3
        for i in range(num_levels):
            faces[(i * num_triangles): ((i + 1) * num_triangles), 1:] = triangles_base + i * num_nodes
        coords = np.stack([coords.x, coords.y], axis=-1)
        coords = np.tile(coords, (num_levels, 1))
        coords = np.concatenate([coords, np.reshape(z, (-1, 1))], axis=-1)
        mesh = pv.PolyData(coords, faces)
        if use_scalar_key and self.scalar_key is not None:
            mesh[self.scalar_key] = self.field_data[self.scalar_key].values.ravel()
        return mesh

    def compute_elevation_coordinate(self, scale_params):
        inv_scale = 1.0 / scale_params.scale
        z = self._relative_elevation.copy()
        if scale_params.offset_scale != 1.:
            z *= scale_params.offset_scale
        z += self._terrain_level_elevation
        z *= inv_scale
        return z

    def get_contour_mesh(self, contour_params: ContourParameters, scale_params: ScalingParameters) -> pv.PolyData:
        mesh = self.get_volume_mesh(scale_params)
        contour_key = contour_params.contour_key
        if contour_key is None:
            contour_key = self.scalar_key
        if contour_key is None:
            raise ValueError('Isocontours cannot be computed without specified contour key.')
        if contour_key != self.scalar_key:
            mesh[self.scalar_key] = self.field_data[self.scalar_key].values.ravel()
        if contour_key != self.model_level_key:
            mesh[contour_key] = self.field_data[contour_key].values.ravel()
        else:
            mesh[contour_key] = self.compute_elevation_coordinate(scale_params).ravel()
        iso_mesh = mesh.contour(
            scalars=contour_key, isosurfaces=contour_params.num_levels, method='contour', compute_scalars=True,
        )
        iso_mesh.set_active_scalars(self.scalar_key)
        return iso_mesh

    def get_reference_mesh(self, scale_params: ScalingParameters):
        if self._relative_elevation.shape[0] == 1:
            return self.get_level_mesh(scale_params, use_scalar_key=False)
        else:
            return self.get_volume_mesh(scale_params, use_scalar_key=False)
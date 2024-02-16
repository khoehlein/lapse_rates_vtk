import uuid

import numpy as np
import pyvista as pv

from src.interaction.downscaling.geometry import TriangleMesh


class SurfaceDataset(object):

    def __init__(self, mesh: TriangleMesh, z: np.ndarray):
        self.mesh = mesh
        self.z = z
        self.scalars = {}

    def add_scalar_field(self, field_data: np.ndarray, name: str = None) -> str:
        if name is None:
            name = str(uuid.uuid4())
        self.scalars[name] = field_data
        return name

    def get_polydata(self) -> pv.PolyData:
        polydata = self.mesh.to_polydata(self.z)
        for key in self.scalars:
            polydata[key] = self.scalars[key]
        return polydata


class WedgeMesh(object):

    def __init__(self, base_mesh: TriangleMesh, num_levels: int):
        self.base_mesh = base_mesh
        self.num_levels = num_levels

    def get_coordinates(self, z: np.ndarray = None, transform=None) -> np.ndarray:
        level_coordinates = self.base_mesh.get_node_positions(z=None, transform=transform)
        coordinates = np.tile(level_coordinates, (self.num_levels, 1))
        if z is not None:
            assert len(z) == self.num_levels
            coordinates[:, -1] = z.ravel()
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

    def to_pyvista(self, z: np.ndarray = None, transform=None) -> pv.UnstructuredGrid:
        coords = self.get_coordinates(z, transform)
        wedges = self.get_wedges(add_prefix=True)
        cell_types = [pv.CellType.WEDGE] * len(wedges)
        return pv.UnstructuredGrid(wedges, cell_types, coords)

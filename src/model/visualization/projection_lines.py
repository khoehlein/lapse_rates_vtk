import logging
from typing import Dict, Any

import numpy as np
import pyvista as pv

from src.model._legacy.geometry import SurfaceDataset
from src.model.visualization.colors import ColorModel
from src.model.visualization.interface import VisualizationType
from src.model.visualization.mesh_geometry import MeshGeometryModel
from src.model.visualization.mesh_visualization import MeshVisualization


DIFFERENCE_TOLERANCE = 100 # meters


class ProjectionLines(MeshVisualization):

    def __init__(self, geometry: MeshGeometryModel, coloring: ColorModel, visual_key: str = None, parent=None):
        super().__init__(VisualizationType.PROJECTION_LINES, geometry, coloring, visual_key, parent)

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key: str = None) -> 'ProjectionLines':
        logging.info('Building surface scalar visualization')
        color_properties = properties['coloring']
        color_model = ColorModel.from_properties(color_properties)
        lines = cls._compute_lines(dataset['surface_o1280'], dataset['surface_o8000'])
        geometry_model = MeshGeometryModel(lines, properties=properties['geometry'], scalar_preference='cell')
        return cls(geometry_model, color_model, visual_key=key)

    @staticmethod
    def _compute_lines(data_lr: SurfaceDataset, data_hr: SurfaceDataset) -> pv.PolyData:
        mesh_lr = data_lr.get_polydata()
        mesh_hr = data_hr.get_polydata()
        origins = mesh_hr.points.copy()
        origins[:, -1] = np.max(data_lr.z) + 1.
        directions = np.zeros_like(origins)
        directions[:, -1] = -1.
        points_lr, rays, cells = mesh_lr.multi_ray_trace(origins, directions, first_point=True)
        points_hr = mesh_hr.points[rays]
        z_difference = points_hr[:, -1] - points_lr[:, -1]
        mask = np.abs(z_difference) > DIFFERENCE_TOLERANCE
        points_lr = points_lr[mask]
        points_lr[:, -1] += np.sign(z_difference[mask]) * DIFFERENCE_TOLERANCE
        points_hr = points_hr[mask]
        points = np.concatenate([points_lr, points_hr], axis=0)
        rays = rays[mask]
        num_lines = np.sum(mask)
        lines = np.zeros((num_lines, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(num_lines, dtype=int)
        lines[:, 2] = np.arange(num_lines, 2 * num_lines, dtype=int)
        polydata = pv.PolyData(points, lines=lines.ravel())
        for key in data_hr.scalars:
            polydata.cell_data[key] = data_hr.scalars[key][rays]
        return polydata

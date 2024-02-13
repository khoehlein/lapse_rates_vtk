import math

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.neighbors import KDTree

from src.model.geometry import SurfaceDataset, OctahedralGrid, DomainBounds


class DummyPipeline(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_outputs()

    def get_output(self):
        return {
            'surface_o8000': self._surface_o8000,
            'surface_o1280': self._surface_o1280,
        }

    def _build_outputs(self):
        domain_bounds = DomainBounds(10, 10.5, 5, 5.5)
        grid_highres = OctahedralGrid(8000).get_mesh_for_subdomain(domain_bounds)
        grid_lowres = OctahedralGrid(1280).get_mesh_for_subdomain(domain_bounds)
        np.random.seed(1234)
        z_highres = np.exp(np.random.randn(grid_highres.num_nodes) * 0.5) * 400.
        z_lowres = np.random.randn(grid_lowres.num_nodes) * (np.std(z_highres) / math.sqrt(10)) + np.mean(z_highres)
        tree_lowres = KDTree(grid_lowres.coordinates.as_geocentric().values, leaf_size=100)
        nearest = tree_lowres.query(grid_highres.coordinates.as_geocentric().values, k=1, return_distance=False).ravel()
        z_highres = z_lowres[nearest] + 0.2 * (z_highres - z_lowres[nearest])
        z_difference = z_highres - z_lowres[nearest]

        lapse_rate_lowres = - 6.5 + np.random.randn(grid_lowres.num_nodes)
        t_lowres = 273.15 - (6.5 / 1000.) * z_lowres
        t_difference = lapse_rate_lowres[nearest] * z_difference / 1000.
        t_highres = t_lowres[nearest] + t_difference

        self._surface_o1280 = SurfaceDataset(grid_lowres, z_lowres)
        self._surface_o1280.add_scalar_field(lapse_rate_lowres, 'lapse_rate')
        self._surface_o1280.add_scalar_field(t_lowres, 't2m_o1280')
        self._surface_o1280.add_scalar_field(z_lowres, 'z_o1280')

        self._surface_o8000 = SurfaceDataset(grid_highres, z_highres)
        self._surface_o8000.add_scalar_field(lapse_rate_lowres[nearest], 'lapse_rate')
        self._surface_o8000.add_scalar_field(t_lowres[nearest], 't2m_o1280')
        self._surface_o8000.add_scalar_field(t_highres, 't2m_o8000')
        self._surface_o8000.add_scalar_field(t_difference, 't2m_difference')
        self._surface_o8000.add_scalar_field(z_lowres[nearest], 'z_o1280')
        self._surface_o8000.add_scalar_field(z_highres, 'z_o8000')
        self._surface_o8000.add_scalar_field(z_difference, 'z_difference')


class DummyController(QObject):

    domain_changed = pyqtSignal()
    data_changed = pyqtSignal()

    def __init__(self, pipeline: DummyPipeline, parent=None):
        super().__init__(parent=parent)
        self.pipeline = pipeline


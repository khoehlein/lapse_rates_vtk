import logging
from typing import Dict, Any

import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.model.geometry import OctahedralGrid, DomainLimits, DomainBoundingBox, LocationBatch, \
    Coordinates, TriangleMesh
from src.model.data.config_interface import ConfigReader, SourceConfiguration
from src.model.level_heights import compute_physical_level_height


class _GridDataStore(object):

    def __init__(self, grid: OctahedralGrid, data: xr.Dataset):
        self.grid = grid
        self.data = data
        coords = Coordinates.from_xarray(data)
        elevation = data['z'] if 'z' in data.data_vars else None
        self.sites = LocationBatch(coords, elevation=elevation, source_reference=None)
        self._grid_lookup = None

    def scalar_names(self):
        return list(self.data.data_vars.keys())

    def get_grid_lookup(self, mask=None):
        if mask is None:
            if self._grid_lookup is None:
                self._grid_lookup = self._build_lookup()
            return self._grid_lookup
        return self._build_lookup(mask)

    def _build_lookup(self, mask=None):
        xyz = self.sites.coords.as_xyz().values
        if mask is not None:
            xyz = xyz[mask]
        lookup = NearestNeighbors(n_neighbors=1)
        lookup.fit(xyz)
        return lookup


class DomainData(_GridDataStore):

    def __init__(self, bounding_box: DomainBoundingBox, mesh: TriangleMesh, grid: OctahedralGrid, data: xr.Dataset):
        super().__init__(grid, data)
        self.bounding_box = bounding_box
        self.mesh = mesh


class GlobalData(_GridDataStore):

    def __init__(self, grid: OctahedralGrid, data: xr.Dataset):
        super().__init__(grid, data)
        assert grid.num_nodes == len(self.sites)

    @classmethod
    def from_config(cls, configs: Dict[str, Any], grid: OctahedralGrid, compute_level_heights: bool = False) -> 'GlobalData':
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

    def get_domain_dataset(self, domain_bounds: DomainLimits):
        bounding_box = DomainBoundingBox(domain_bounds)
        mesh = self.grid.get_mesh_for_subdomain(bounding_box)
        data = self.data.isel(values=mesh.source_reference)
        return DomainData(bounding_box, mesh, self.grid, data)

    def query_site_data(self, domain_bounds):
        raise NotImplementedError()

    def query_link_data(self, x):
        raise NotImplementedError()

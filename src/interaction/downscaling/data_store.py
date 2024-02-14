import logging
from typing import Dict, Any

import numpy as np
import xarray as xr

from src.interaction.downscaling.geometry import OctahedralGrid
from src.model.data_store.config_interface import ConfigReader, SourceConfiguration
from src.model.level_heights import compute_physical_level_height


class DataStore(object):

    def __init__(self, grid: OctahedralGrid, data: xr.Dataset):
        assert data.dims['values'] == grid.num_nodes
        self.grid = grid
        self.data = data

    @classmethod
    def from_config(cls, configs: Dict[str, Any], grid: OctahedralGrid, compute_level_heights: bool = False) -> 'DataStore':
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

    def query_site_data(self, domain_bounds):
        raise NotImplementedError()

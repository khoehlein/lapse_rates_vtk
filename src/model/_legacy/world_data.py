import logging
from typing import Union

import numpy as np
import xarray as xr

from src.model.data.config_interface import ConfigReader, SourceConfiguration
from src.model._legacy.geometry import OctahedralGrid, DomainBounds, TriangleMesh, WedgeMesh, LocationBatch, SurfaceDataset
from src.model.level_heights import compute_physical_level_height
from src.model.downscaling.neighborhood_graphs import NeighborhoodGraph

N_LOW_RES = 1280
N_HIGH_RES = 8000
N_LEVELS = 20


class DomainData(object):

    def __init__(
            self,
            bounds: DomainBounds,
            surface_mesh_lr: TriangleMesh, surface_mesh_hr: TriangleMesh,
            volume_mesh_model_levels: WedgeMesh,
            data_lr: xr.Dataset, data_hr: xr.Dataset,
    ):
        self.bounds = bounds
        self.surface_mesh_lr = surface_mesh_lr
        self.surface_mesh_hr = surface_mesh_hr
        self.volume_mesh_model_levels = volume_mesh_model_levels
        self.data_lr = data_lr
        self.data_hr = data_hr

    def get_highres_orography(self) -> SurfaceDataset:
        return SurfaceDataset(self.surface_mesh_hr, self.data_hr.z.values)

    def get_lowres_orography(self) -> SurfaceDataset:
        return SurfaceDataset(self.surface_mesh_lr, self.data_lr.z.values)


class SampleBatch(object):

    def __init__(self, locations: LocationBatch, data: xr.Dataset, source_reference: Union[np.ndarray, NeighborhoodGraph] = None):
        self.locations = locations
        self.data = data
        self.source_reference = source_reference


class WorldData(object):

    @classmethod
    def from_config_file(cls, path_to_config_file: str):
        config_reader = ConfigReader(SourceConfiguration)
        configs = config_reader.load_json_config(path_to_config_file)
        logging.info('Loading low-res model...')
        orography_lr = config_reader.load_data(configs['orography']['low-res']).z
        lsm_lr = config_reader.load_data(configs['lsm']['low-res']).lsm
        t2m = config_reader.load_data(configs['temperature']['2m'])
        t3d = config_reader.load_data(configs['temperature']['bulk']).t.transpose('hybrid', 'values')
        lnsp = config_reader.load_data(configs['pressure']).lnsp
        q3d = config_reader.load_data(configs['humidity']).q.transpose('hybrid', 'values')
        logging.info('Computing model level heights...')
        z_model_levels = compute_physical_level_height(
            np.exp(lnsp.values)[None, :], orography_lr.values[None, :],
            t3d.values, q3d.values
        )
        logging.info('Merging low-res model...')
        data_lr = xr.merge([orography_lr, lsm_lr, t2m, t3d, lnsp, q3d], compat='override')
        data_lr = data_lr.assign(z_model_levels=(('hybrid', 'values'), z_model_levels))
        logging.info(f'Loading high-res model...')
        data_hr = config_reader.load_data(configs['orography']['high-res'])
        logging.info(f'Loading completed.')
        grid_lr = OctahedralGrid(N_LOW_RES)
        grid_hr = OctahedralGrid(N_HIGH_RES)
        return cls(grid_lr, grid_hr, data_lr, data_hr)

    def __init__(
            self,
            grid_lr: OctahedralGrid, grid_hr: OctahedralGrid,
            data_lr: xr.Dataset, data_hr: xr.Dataset,
    ):
        self.grid_lr = grid_lr
        self.grid_hr = grid_hr
        self.data_lr = data_lr
        self.data_hr = data_hr

    def query_domain_data(self, domain_bounds: DomainBounds) -> DomainData:
        mesh_lr = self.grid_lr.get_mesh_for_subdomain(domain_bounds)
        mesh_hr = self.grid_hr.get_mesh_for_subdomain(domain_bounds)
        z_model_levels = self.data_lr.z_model_levels.values[:, mesh_lr.source_reference]
        mesh_model_levels = WedgeMesh(mesh_lr, z_model_levels)
        return DomainData(
            domain_bounds,
            mesh_lr, mesh_hr, mesh_model_levels,
            self.data_lr.isel(values=mesh_lr.source_reference),
            self.data_hr.isel(values=mesh_hr.source_reference),
        )

    def query_sample_data(self, neighborhood: NeighborhoodGraph):
        site_samples = self.data_lr.isel(values=(('site',), neighborhood.locations.source_reference))
        neighbor_samples = self.data_lr.isel(values=(('link',), neighborhood.links['neighbor'].values))
        return SampleBatch(neighborhood.locations, (site_samples, neighbor_samples), neighborhood)

    def get_lowres_land_sea_data(self):
        return self.data_lr.lsm


def _test():
    world_data = WorldData.from_config_file('/cfg/data/2021121906_ubuntu.json')
    bounds = DomainBounds(45, 50, 20,25)
    world_data.query_domain_data(bounds)


if __name__ == '__main__':
    _test()

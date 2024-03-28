import os

import numpy as np
import pandas as pd
import xarray as xr

from src.model.geometry import OctahedralGrid, DomainLimits, DomainBoundingBox, LocationBatch, Coordinates, TriangleMesh
from src.model.level_heights import compute_standard_surface_pressure, compute_full_level_pressure, \
    compute_approximate_level_height
import networkx as nx


DOMAIN_NAME = 'central_europe'
DEFAULT_DOMAIN = DomainLimits(43., 47., 6., 12.)


def compute_level_heights(z_surf, t2m):
    z_2m = z_surf.values + 2
    p_surf = compute_standard_surface_pressure(
        z_surf.values,
        base_temperature=t2m.values,
        base_temperature_height=z_2m
    )
    p = compute_full_level_pressure(p_surf)
    z_level = compute_approximate_level_height(
        p, p_surf, z_surf.values,
        base_temperature=t2m.values,
        base_temperature_height=z_2m
    )
    return z_level


def compute_gradients(z_surf, z_model_levels, t2m, t):
    t_ = np.concatenate([t.values, t2m.values[None, :]], axis=0)
    z_ = np.concatenate([z_model_levels.values, z_surf.values[None, :] + 2], axis=0)
    grad = (t_[:-1] - t_[1:]) / (z_[:-1] - z_[1:])
    return grad


def extract_station_data(bbox: DomainBoundingBox, timestamp):
    station_data = pd.read_parquet('/mnt/ssd4tb/ECMWF/Vis/station_data_europe_hres-const-lapse.parquet')
    station_metadata = pd.read_csv('/mnt/ssd4tb/ECMWF/Obs/station_locations_nearest.csv', index_col=0)
    station_metadata = station_metadata.set_index('stnid')
    locations = LocationBatch(Coordinates.from_xarray(station_metadata))
    valid_stnids = station_metadata.index.values[bbox.contains(locations)]
    mask = np.logical_and(
        station_data['timestamp'].values == timestamp,
        station_data['stnid'].isin(valid_stnids)
    )
    station_data = station_data.loc[mask]
    for field in ['prediction', 'observation']:
        station_data[field] = station_data[field].values - 273.15
    stnids = station_data['stnid'].values
    station_data['latitude'] = station_metadata['latitude'].loc[stnids].values
    station_data['longitude'] = station_metadata['longitude'].loc[stnids].values
    station_data['elevation'] = station_metadata['elevation'].loc[stnids].values
    return station_data


def extract_terrain_data(bbox, mesh, paths):
    node_ids = mesh.source_reference
    triangles = mesh.vertices
    terrain_data = xr.open_dataset(paths['lsm']).isel(values=node_ids)
    lsm = terrain_data.lsm
    watermass_id = - np.ones((len(lsm.values),), dtype=int)
    is_sea = lsm.values < 0.5
    is_sea_triangle = np.any(lsm.values[triangles] < 0.5, axis=-1)
    sea_triangles = triangles[is_sea_triangle]
    graph = nx.from_edgelist(
        np.concatenate([sea_triangles[:, [0, 1]], sea_triangles[:, [0, 2]], sea_triangles[:, [1, 2]]], axis=0).tolist())
    cc = list(nx.connected_components(graph))

    counter = 0
    for component in cc:
        component_nodes = list(component)
        land_fraction = lsm.values[component_nodes].mean()
        if land_fraction >= 0.5:
            watermass_id[component_nodes] = counter
            counter += 1

    z_surf = xr.open_dataset(paths['z']).z.isel(values=node_ids)
    df = pd.DataFrame({
        'id': watermass_id[is_sea],
        'z': z_surf.values[is_sea]
    })
    z_wm_med = df.groupby('id')['z'].median()
    z_watermass = z_wm_med.loc[watermass_id].values

    bbox_safe = DomainBoundingBox(bbox.bounds.plus_safety_margin())
    mesh_o1280 = OctahedralGrid(1280).get_mesh_for_subdomain(bbox_safe)
    nodes_o1280 = mesh_o1280.source_reference
    z_surf_o1280 = xr.open_dataset("/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib").z.isel(values=nodes_o1280)
    mesh_o1280 = mesh_o1280.to_polydata()
    mesh_o1280['elevation'] = z_surf_o1280.values.ravel()

    mesh_other = TriangleMesh(LocationBatch(Coordinates.from_xarray(z_surf)), triangles).to_polydata()

    interpolated = mesh_other.sample(mesh_o1280)
    interpolated = np.asarray(interpolated['elevation'])

    terrain_data = terrain_data.assign({
        'node_id': ('values', node_ids),
        'watermass_id': ('values', watermass_id),
        'z_surf': ('values', z_surf.values),
        'z_surf_o1280': ('values', interpolated),
        'z_watermass': ('values', z_watermass),
        'triangles': (['mesh_cell', 'vertex'], triangles)
    })
    return terrain_data


def extract_model_data(mesh, date, time, step):
    node_ids = mesh.source_reference
    path_to_t2m = "/mnt/ssd4tb/ECMWF/HRES_2m_temp_{}.grib".format(date.replace('-', ''))
    path_to_t = "/mnt/ssd4tb/ECMWF/HRES_Model_Level_temp_{}.grib".format(date.replace('-', ''))
    t2m = xr.open_dataset(path_to_t2m).t2m.isel(time=time, step=step, values=node_ids)
    z_surf = xr.open_dataset("/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib").z.isel(values=node_ids)
    z_model_levels = compute_level_heights(z_surf, t2m)
    model_data = xr.open_dataset(path_to_t).isel(time=time, step=step, values=node_ids)
    model_data = model_data.assign({
        'z_model_levels': (['hybrid', 'values'], z_model_levels),
        'latitude_3d': (['hybrid', 'values'], np.tile(model_data['latitude'].values[None, :], (20, 1))),
        'longitude_3d': (['hybrid', 'values'], np.tile(model_data['longitude'].values[None, :], (20, 1))),
    })
    z_model_levels = model_data['z_model_levels']
    t = model_data['t']
    grad_t = compute_gradients(z_surf, z_model_levels, t2m, t)
    model_data = model_data.assign({
        'grad_t': (['hybrid', 'values'], grad_t * 1000),
        't2m': ('values', t2m.values - 273.15),
    })
    model_data['t'] = model_data['t'] - 273.15
    return model_data


def export(name, bbox, date, time, step):

    output_path = '/mnt/ssd4tb/ECMWF/Vis/{}'.format(name)
    os.makedirs(output_path, exist_ok=True)

    print('Exporting regular lowres')
    model_data, surface_data = _extract_from_lowres(bbox, date, step, time)
    surface_data.to_netcdf(os.path.join(output_path, 'terrain_data_o1280.nc'))
    model_data.to_netcdf(os.path.join(output_path, 'model_data_o1280.nc'))

    print('Exporting safe lowres')
    bbox_safe = DomainBoundingBox(bbox.bounds.plus_safety_margin())
    model_data, surface_data = _extract_from_lowres(bbox_safe, date, step, time)
    surface_data.to_netcdf(os.path.join(output_path, 'terrain_data_o1280_safe.nc'))
    model_data.to_netcdf(os.path.join(output_path, 'model_data_o1280_safe.nc'))

    print('Exporting highres')
    surface_data = _extract_from_highres(bbox)
    surface_data.to_netcdf(os.path.join(output_path, 'terrain_data_o8000.nc'))

    print('Exporting stations')
    timestamp = np.datetime64('{}T{:02d}:00'.format(date, 12 * time + step))
    station_data = extract_station_data(bbox, timestamp)
    station_data.to_parquet(os.path.join(output_path, 'station_data.parquet'))

    print('Done')


def _extract_from_highres(bbox):
    paths_o8000 = {
        'lsm': "/mnt/ssd4tb/ECMWF/lsm_from_watermask.nc",
        'z': "/mnt/ssd4tb/ECMWF/orog_reduced_gaussian_grid_1km.grib"
    }
    mesh = OctahedralGrid(8000).get_mesh_for_subdomain(bbox)
    surface_data = extract_terrain_data(bbox, mesh, paths_o8000)
    return surface_data


def _extract_from_lowres(bbox, date, step, time):
    paths_o1280 = {
        'lsm': "/mnt/ssd4tb/ECMWF/LSM_HRES_Sep2022.grib",
        'z': "/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib"
    }
    mesh = OctahedralGrid(1280).get_mesh_for_subdomain(bbox)
    surface_data = extract_terrain_data(bbox, mesh, paths_o1280)
    model_data = extract_model_data(mesh, date, time, step)
    return model_data, surface_data


def export_winter(bbox):
    export(f'{DOMAIN_NAME}_winter', bbox, '2021-12-19', 0, 6)


def export_summer(bbox):
    export(f'{DOMAIN_NAME}_summer', bbox, '2021-07-12', 1, 3)


def main():
    bbox = DomainBoundingBox(DEFAULT_DOMAIN)
    export_summer(bbox)
    export_winter(bbox)


if __name__ == "__main__":
    main()

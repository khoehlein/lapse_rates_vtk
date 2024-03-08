import argparse
import os.path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.model.data.config_interface import ConfigReader, SourceConfiguration
from src.model.downscaling.neighborhood_graphs import RadialNeighborhoodGraph
from src.model.geometry import LocationBatch, Coordinates, OctahedralGrid


parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str)
parser.add_argument('--data-path', type=str)
args = vars(parser.parse_args())

PARQUET_PATH = args['data_path']
CONFIG_FILE_PATH = args['config_path']

THRESHOLD_ANGLE = 0.
THRESHOLD_ELEVATION = 0.
MAX_FRACTION_MISSING = 0.5
RADIUS_LSM = 18
THRESHOLD_LSM = 0.1


# PARQUET_PATH = '/mnt/data2/ECMWF/Obs/observations.parquet'
# CONFIG_FILE_PATH = '/home/hoehlein/PycharmProjects/production/lapse_rates_vtk/cfg/data/2021121906_ubuntu.json'

data = pd.read_parquet(PARQUET_PATH)

print('Raw observations:', len(data))

elevation = data['elevation'].values
missing_elevation = np.isin(elevation, [99999, 999.9, 999.99])
print('Missing elevation values:', np.sum(missing_elevation))
invalid_italy_parts = np.logical_and(data['stnid'].str.startswith('3900').values, np.isin(elevation, [0., -1.]))
print('Invalid italian parts:' , np.sum(invalid_italy_parts))

has_valid_elevation = np.logical_and(~missing_elevation, ~invalid_italy_parts)
data = data.loc[has_valid_elevation]

print('After elevation filter:', len(data))


def _compute_fraction_of_days_missing():
    grouped = data.groupby(by=['stnid', 'date'])
    counts = grouped['latitude'].count()
    da = counts.to_xarray()
    fraction_of_days_missing = da.isnull().mean(dim='date')
    return fraction_of_days_missing


def _find_selfconsistent_stations(grouped):
    lon_range = grouped['longitude'].max().values - grouped['longitude'].min().values
    lat_range = grouped['latitude'].max().values - grouped['latitude'].min().values
    elev_range = grouped['elevation'].max().values - grouped['elevation'].min().values
    mask = np.logical_and(lat_range <= THRESHOLD_ANGLE, lon_range <= THRESHOLD_ANGLE)
    mask = np.logical_and(mask, elev_range <= THRESHOLD_ELEVATION)
    return mask


def _find_land_stations(grouped):
    longitudes = grouped['longitude'].max()
    latitudes = grouped['latitude'].max()
    sites = LocationBatch(Coordinates.from_lat_lon(latitudes, longitudes))
    config_reader = ConfigReader(SourceConfiguration)
    configs = config_reader.load_json_config(CONFIG_FILE_PATH)

    nearest_1km = OctahedralGrid(8000).find_nearest_neighbor(sites)
    data_1km = config_reader.load_data(configs['o8000']['lsm'], key='lsm')
    lsm_1km = data_1km.isel(values=nearest_1km.source_reference).values

    data_hres = config_reader.load_data(configs['o1280']['lsm'], key='lsm')
    tree = NearestNeighbors()
    tree.fit(Coordinates.from_xarray(data_hres).as_xyz().values)
    links = RadialNeighborhoodGraph.from_tree_query(sites, tree, RADIUS_LSM).links
    links['lsm'] = data_hres.isel(values=links['neighbor'])
    lsm_hres = links.groupby('location')['lsm'].max().values

    mask_land = np.logical_or(lsm_hres > THRESHOLD_LSM, lsm_1km > THRESHOLD_LSM)

    return mask_land


def export_valid_stations():

    # Compute fraction of missing dates
    fraction_of_days_missing = _compute_fraction_of_days_missing()
    mask_completeness = fraction_of_days_missing.values < MAX_FRACTION_MISSING
    print('Fraction of persistent stations:', np.mean(mask_completeness))

    grouped = data.groupby(by='stnid', as_index=True)
    mask_sc = _find_selfconsistent_stations(grouped)
    print('Fraction of self-consistent stations:', np.mean(mask_sc))

    mask_lsm = _find_land_stations(grouped)
    print('Fraction of land stations:', np.mean(mask_lsm))

    valid = np.all(np.stack([mask_sc, mask_lsm, mask_completeness], axis=-1), axis=-1)
    print('Fraction of valid stations:', np.mean(valid))

    valid = pd.Series(valid, index=grouped['longitude'].min().index)

    metadata = grouped[['latitude', 'longitude', 'elevation']].min()
    metadata['num_obs'] = grouped['latitude'].count()
    metadata = metadata.loc[valid.values]
    print('Valid stations remaining:', len(metadata))
    print('Writing meta data')
    metadata.to_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations.csv'))
    print('Done')

    valid_extended = valid.loc[data.stnid.values]
    print('Fraction of valid observations:', np.mean(valid_extended.values))

    data_ = data.loc[valid_extended.values]
    print('Valid observations remaining:', len(data_))

    out_path, ext = os.path.splitext(PARQUET_PATH)
    out_path = out_path + '_filtered' + ext

    print('Writing filtered data')
    data_.to_parquet(out_path)
    print('Done')


def compute_grid_heights():
    print('Computing nearest grid cells')

    metadata = pd.read_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations.csv'))

    station_latitudes = metadata['latitude']
    station_longitudes = metadata['longitude']
    elevation = metadata['elevation']

    sites = LocationBatch(Coordinates.from_lat_lon(station_latitudes, station_longitudes), elevation=elevation)

    grid_hres = OctahedralGrid(1280)
    nearest_hres = grid_hres.find_nearest_neighbor(sites)

    grid_1km = OctahedralGrid(8000)
    nearest_1km = grid_1km.find_nearest_neighbor(sites)

    config_reader = ConfigReader(SourceConfiguration)
    configs = config_reader.load_json_config(CONFIG_FILE_PATH)

    data_hres = config_reader.load_data(configs['o1280']['z'], key='z')
    z_hres = data_hres.isel(values=nearest_hres.source_reference)

    data_1km = config_reader.load_data(configs['o8000']['z'], key='z')
    z_1km = data_1km.isel(values=nearest_1km.source_reference)

    metadata['elevation_o1280'] = z_hres.values
    metadata['elevation_o8000'] = z_1km.values
    metadata['latitude_o1280'] = z_hres.latitude.values
    metadata['latitude_o8000'] = z_1km.latitude.values
    metadata['longitude_o1280'] = (z_hres.longitude.values + 180) % 360 - 180
    metadata['longitude_o8000'] = (z_1km.longitude.values + 180) % 360 - 180
    metadata['nearest_node_o1280'] = nearest_hres.source_reference
    metadata['nearest_node_o8000'] = nearest_1km.source_reference

    metadata.to_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations_nearest.csv'))

    print('Done')


if __name__ == '__main__':
    export_valid_stations()
    compute_grid_heights()

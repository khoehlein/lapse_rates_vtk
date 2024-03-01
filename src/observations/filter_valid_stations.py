import os.path

import numpy as np
import pandas as pd

from src.model.data.config_interface import ConfigReader, SourceConfiguration
from src.model.geometry import LocationBatch, Coordinates, OctahedralGrid

PARQUET_PATH = '/mnt/ssd4tb/ECMWF/Obs/observations.parquet'
CONFIG_FILE_PATH = '/home/hoehlein/PycharmProjects/local/lapse_rates_vtk/cfg/data/2021121906_ubuntu.json'

data = pd.read_parquet(PARQUET_PATH)

print('Raw observations:', len(data))

elevation = data['elevation'].values
missing_elevation = elevation == 99999
print('Missing elevation values:', np.sum(missing_elevation))
invalid_italy_parts = np.logical_and(data['stnid'].str.startswith('3900').values, np.logical_or(elevation == 0, elevation == -1))
print('Invalid italian parts:' , np.sum(invalid_italy_parts))

has_valid_elevation = np.logical_and(~missing_elevation, ~invalid_italy_parts)
data = data.loc[has_valid_elevation]

print('After elevation filter:', len(data))

THRESHOLD_ANGLE = 0.
THRESHOLD_ELEVATION = 0.
MAX_FRACTION_MISSING = 0.5


def export_valid_stations():
    grouped = data.groupby(by=['stnid', 'date'])
    counts = grouped['latitude'].count()
    da = counts.to_xarray()
    fraction_of_days_missing = da.isnull().mean(dim='date')
    print('Number of stations:', len(fraction_of_days_missing))

    grouped = data.groupby(by='stnid')
    lon_range = grouped['longitude'].max() - grouped['longitude'].min()
    lat_range = grouped['latitude'].max() - grouped['latitude'].min()
    elev_range = grouped['elevation'].max() - grouped['elevation'].min()

    mask = np.logical_and(lat_range <= THRESHOLD_ANGLE, lon_range <= THRESHOLD_ANGLE)
    mask = np.logical_and(mask, elev_range <= THRESHOLD_ELEVATION)
    print('Fraction of self-consistent stations:', np.mean(mask))

    assert np.all(lon_range.index.values == da.stnid.values)
    valid = np.logical_and(mask, fraction_of_days_missing.values < MAX_FRACTION_MISSING)

    print('Fraction of valid stations:', np.mean(valid))

    valid = pd.Series(valid, index=lon_range.index)
    valid_extended = valid.loc[data.stnid.values]

    print('Fraction of valid observations:', np.mean(valid_extended.values))

    metadata = grouped[['latitude', 'longitude', 'elevation']].min()
    metadata['num_obs'] = grouped['latitude'].count()
    metadata = metadata.loc[valid.values]
    print('Valid stations remaining:', len(metadata))
    print('Writing meta data')
    metadata.to_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations.csv'))
    print('Done')

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
    # export_valid_stations()
    compute_grid_heights()

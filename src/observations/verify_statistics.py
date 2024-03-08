import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from matplotlib.axes import Axes

from src.model.domain_selection import DEFAULT_DOMAIN
from src.model.geometry import DomainBoundingBox, LocationBatch, Coordinates

PARQUET_PATH = '/mnt/ssd4tb/ECMWF/Obs/observations_filtered.parquet'
CONFIG_FILE_PATH = '/home/hoehlein/PycharmProjects/local/lapse_rates_vtk/cfg/data/2021121906_ubuntu.json'
_data = None
_parquet_path = None
_metadata = None
metadata_keys = [
    'stnid', 'latitude', 'longitude', 'elevation', 'num_obs',
    'elevation_o1280', 'elevation_o8000',
    'latitude_o1280', 'latitude_o8000',
    'longitude_o1280', 'longitude_o8000',
    'nearest_node_o1280', 'nearest_node_o8000'
]


def load_data(path=None):
    global _data
    global _parquet_path
    if path is None:
        path = PARQUET_PATH
    if _data is None or path != _parquet_path:
        _data = pd.read_parquet(path)
        _parquet_path = path
    return _data


def load_metadata():
    global _metadata
    global _parquet_path
    if _parquet_path is None:
        _parquet_path = PARQUET_PATH
    if _metadata is None:
        _metadata = pd.read_csv(os.path.join(os.path.dirname(_parquet_path), 'station_locations_nearest.csv'))
    return _metadata


def plot_count_distribution():
    data = load_data()
    grouped = data.groupby(by=['stnid', 'date'])
    counts = grouped['latitude'].count()
    da = counts.to_xarray()
    fraction_of_days = da.isnull().mean(dim='date')
    print(fraction_of_days)
    print(len(fraction_of_days))

    grouped = data.groupby(by='stnid')
    lon_range = grouped['longitude'].max() - grouped['longitude'].min()
    lat_range = grouped['latitude'].max() - grouped['latitude'].min()
    mask = np.logical_and(lat_range == 0, lon_range == 0)
    print(np.mean(mask))

    plt.figure(dpi=300)
    plt.hist(fraction_of_days.values[mask], bins=50, label='exact')
    plt.hist(fraction_of_days.values[~mask], bins=50, label='variable')
    plt.legend()
    plt.show()
    plt.close()
    valid_ids = counts.loc[counts.values >= 365]
    print(len(valid_ids))
    print('Done')


def plot_station_locations():
    metadata = load_metadata()
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax in axs:
        ax.scatter(metadata['longitude'], metadata['latitude'], c=metadata['num_obs'], vmin=0, vmax=8760, s=2, alpha=0.5, cmap='tab10')
        ax.gridlines()
        ax.coastlines()
    axs[0].set(xlim=(-180, 180), ylim=(-90, 90))
    axs[1].set(xlim=(-15, 35), ylim=(35, 60))
    plt.tight_layout()
    plt.show()
    plt.close()


class LinearFit(object):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_range = (np.min(x), np.max(x))
        self.y_range = (np.min(y), np.max(y))
        self.slope, self.bias = np.polyfit(x, y, deg=1)
        self.rho = np.corrcoef(x, y)[0, 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.slope * x + self.bias

    def paint(self, ax: Axes):
        xrange = np.asarray(self.x_range)
        y_hat = self.predict(xrange)
        ax.plot(self.x_range, y_hat, color='r', linestyle='--')
        label = f"""
        y = {self.slope:.2f} * x + {self.bias:.2f}
        r = {self.rho:.2f}
        """
        ax.annotate(label, (xrange.mean(), y_hat.mean()), ha='left', va='top')


def plot_elevation_differences():
    metadata = load_metadata()
    z_station = metadata['elevation']
    z_o1280 = metadata['elevation_o1280']
    z_o8000 = metadata['elevation_o8000']
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex='all', sharey='all')
    axs[0].scatter(z_o1280, z_station, alpha=0.05)
    axs[0].set(title='O1280', ylabel='station altitude [m]', xlabel='grid altitude (O1280) [m]')
    LinearFit(z_o1280, z_station).paint(axs[0])
    axs[1].scatter(z_o8000, z_station, alpha=0.05)
    axs[1].set(title='O8000', xlabel='grid altitude (O8000) [m]')
    LinearFit(z_o8000, z_station).paint(axs[1])
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_m1_stations():
    metadata = load_metadata()
    metadata = metadata[metadata['elevation'] == -1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    p = ax.scatter(metadata['longitude'], metadata['latitude'], c=metadata['elevation'] - metadata['elevation_o8000'], cmap='magma')
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label('Elevation difference (m)')
    gl = ax.gridlines()
    ax.coastlines()
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    ax.set(xlim=(10, 20), ylim=(40, 45))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_stations_in_default_domain():
    domain_limits = DEFAULT_DOMAIN.get_domain_limits()
    bounding_box = DomainBoundingBox(domain_limits)
    metadata = load_metadata()
    locations = LocationBatch(Coordinates.from_lat_lon(metadata['latitude'].values, metadata['longitude'].values))
    mask = bounding_box.contains(locations)

    selected_data = metadata.loc[mask]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    p = ax.scatter(selected_data['longitude'], selected_data['latitude'], c=selected_data['elevation'])
    cbar = plt.colorbar(p, ax=ax)
    ax.coastlines()
    gl = ax.gridlines()
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_elevation_differences()

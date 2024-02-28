import os

import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from cartopy import crs as ccrs

from src.model.geometry import OctahedralGrid

PARQUET_PATH = '/mnt/ssd4tb/ECMWF/Obs/observations_filtered.parquet'
data = pd.read_parquet(PARQUET_PATH)
metadata = pd.read_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations.csv'))

print(len(data))


def plot_count_distribution():
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


def compute_grid_heights():
    station_latitudes = metadata['latitude']
    station_longitudes = metadata['longitude']

    def compute_heights_for_grid(grid: OctahedralGrid, data: xr.Dataset):
        circle_latitudes = grid.circle_latitudes
        latitude_index = np.searchsorted(-circle_latitudes, -station_latitudes)
        print(np.sum(latitude_index == 0), np.sum(latitude_index == 2 * grid.degree))
        print(np.sum(station_latitudes < circle_latitudes.min()))

        n_upper_circle = np.clip(latitude_index - 1, a_min=0, a_max=(2 * grid.degree - 1))
        condition_upper_circle = circle_latitudes[n_upper_circle] > station_latitudes

        n_lower_circle = np.clip(latitude_index, a_min=0, a_max=(2 * grid.degree - 1))
        condition_lower_circle = circle_latitudes[n_lower_circle] < station_latitudes

        condition_between_circles = np.logical_and(condition_upper_circle, condition_lower_circle)
        condition_equal = n_lower_circle == n_upper_circle

        located = np.logical_or(condition_equal, condition_between_circles)

        num_nodes_upper = 4 * np.fmin(n_upper_circle + 1, 2 * grid.degree - n_upper_circle) + 16
        num_nodes_lower = 4 * np.fmin(n_lower_circle + 1, 2 * grid.degree - n_lower_circle) + 16


    compute_heights_for_grid(OctahedralGrid(1280), None)



if __name__ == '__main__':
    compute_grid_heights()
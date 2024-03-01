import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import rankdata

from src.observations.verify_statistics import load_data

pred_path = '/mnt/ssd4tb/ECMWF/Predictions'
eval_path = '/mnt/ssd4tb/ECMWF/Evaluation'

experiment = 'predictions_hres-const-lapse'

data = pd.read_csv(os.path.join(eval_path, experiment, 'scores.csv'))
elevation_difference = data['elevation_difference'].values
distance = data['model_station_distance'].values
count = data['count'].values
alphas = count / 8760

latitude = data['station_latitude'].values
longitude = data['station_longitude'].values

plot_path = os.path.join(eval_path, experiment, 'plots')
os.makedirs(plot_path, exist_ok=True)


def plot_overview():
    for metric in ['rmse', 'mae', 'max', 'bias']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        fig.suptitle(metric)
        p = ax.scatter(elevation_difference, data[metric].values, alpha=alphas, c=distance / 1000, vmin=0, cmap='magma')
        cbar = plt.colorbar(p, ax=ax)
        cbar.set_label('Distance [km]')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'overview_{metric}.png'))
        plt.show()
        plt.close()


def plot_extreme_cases():
    for metric in ['rmse', 'mae', 'max', 'bias']:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300, figsize=(16, 10))
        fig.suptitle(f'{metric}-extremes')
        values = data[metric].values
        if metric == 'bias':
            order = np.argsort(np.abs(values))
            quantiles = rankdata(np.abs(values)) / len(values)
        else:
            order = np.argsort(values)
            quantiles = rankdata(values) / len(values)
        values = values[order]
        quantiles = quantiles[order]
        p = ax.scatter(longitude[order], latitude[order], c=values, alpha=quantiles ** 4, cmap='gist_rainbow', s=2)
        ax.set(xlim=(-180, 180), ylim=(-90, 90))
        ax.coastlines()
        gl = ax.gridlines()
        gl.left_labels = True
        gl.bottom_labels = True
        cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
        cbar.set_label(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'extremes_{metric}.png'))
        plt.show()
        plt.close()


def _get_timestamp(date, time):
    dt = datetime.datetime.strptime((date + time), '%Y%m%d%H%M')
    return np.datetime64(dt)


def plot_time_series_for_extremes(top_n=20):
    observations = load_data()
    predictions = pd.read_parquet(os.path.join(pred_path, f'{experiment}.parquet'))

    for metric in ['rmse', 'mae', 'max', 'bias']:
        data_ = data.sort_values(metric, ascending=False).iloc[:top_n]
        print(metric, data_.stnid.values, data_[metric].values)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300, figsize=(16, 10))
        fig.suptitle(f'{metric}-top-{top_n}')
        p = ax.scatter(data_.station_longitude.values, data_.station_latitude.values, c=data_[metric].values)
        cbar = plt.colorbar(p, ax=ax, orientation='horizontal')
        cbar.set_label(metric)
        ax.set(xlim=(-180, 180), ylim=(-90, 90))
        gl = ax.gridlines()
        gl.left_labels = True
        gl.bottom_labels = True
        ax.coastlines()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'top-{top_n}_{metric}.png'))
        plt.show()
        plt.close()

        ts_plot_path = os.path.join(plot_path, f'time-series_top-{top_n}_{metric}')
        os.makedirs(ts_plot_path, exist_ok=True)
        for stnid in data_.stnid.unique():
            obs_ = observations.loc[observations.stnid == stnid]
            time_stamps_ = np.asarray([_get_timestamp(date, time) for date, time in zip(obs_.date.values, obs_.time.values)])
            pred_ = predictions.loc[predictions.stnid == stnid]
            time_stamps__ = np.asarray([_get_timestamp(date, time) for date, time in zip(pred_.date.values, pred_.time.values)])
            assert np.all(time_stamps_ == time_stamps__)
            fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=300)
            fig.suptitle(
                'Station ID: {}, Lat: {:.2f}°, Lon: {:.2f}°, Elev(Stn): {:d} m, Elev(Mod): {:d} m'.format(
                    stnid,obs_.latitude.iloc[0],obs_.longitude.iloc[0],
                    int(np.round(obs_.elevation.iloc[0])), int(np.round(pred_.elevation.iloc[0]))
                )
            )
            ax.plot(time_stamps_, obs_.value_0.values, label='observations', c='r')
            ax.plot(time_stamps_, pred_.value_0.values, linestyle='--', label='predictions', c='k')
            ax.set(xlabel='time', ylabel='T [K]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ts_plot_path, f'{stnid}.png'), dpi=300)
            plt.show()
            plt.close()





if __name__ == '__main__':
    # plot_overview()
    # plot_extreme_cases()
    plot_time_series_for_extremes()

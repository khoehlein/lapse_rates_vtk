import datetime
import gc
import os
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

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
    for metric in ['rmse', 'mae', 'max', 'rmse_deb', 'mae_deb', 'max_deb', 'bias']:
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
    for metric in ['rmse', 'mae', 'max', 'rmse_deb', 'mae_deb', 'max_deb', 'bias']:
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

    for metric in ['rmse', 'mae', 'max', 'rmse_deb', 'mae_deb', 'max_deb', 'bias']:
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


def plot_time_series_for_stnid(stnid: str):
    observations = load_data()
    predictions = pd.read_parquet(os.path.join(pred_path, f'{experiment}.parquet'))

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
    # plt.savefig(os.path.join(ts_plot_path, f'{stnid}.png'), dpi=300)
    plt.show()
    plt.close()

    env = LocalOutlierFactor(n_neighbors=10)
    classification = env.fit_predict(np.stack([pred_.value_0.values, obs_.value_0.values], axis=-1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    fig.suptitle(
        'Station ID: {}, Lat: {:.2f}°, Lon: {:.2f}°, Elev(Stn): {:d} m, Elev(Mod): {:d} m'.format(
            stnid, obs_.latitude.iloc[0], obs_.longitude.iloc[0],
            int(np.round(obs_.elevation.iloc[0])), int(np.round(pred_.elevation.iloc[0]))
        )
    )
    ax.scatter(pred_.value_0.values, obs_.value_0.values, c=classification, alpha=0.05)
    ax.set(xlabel='Prediction [K]', ylabel='Observation [K]')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(ts_plot_path, f'{stnid}.png'), dpi=300)
    plt.show()
    plt.close()


def plot_correlations():

    corr_data = pd.read_csv(os.path.join(eval_path, 'reliability_metrics.csv'))

    fig, ax = plt.subplots(1, 1)
    ax.hist(corr_data['pearson_statistic'].values, bins=np.linspace(-1, 1, 51), alpha=0.5, label='Pearson')
    ax.hist(corr_data['spearman_statistic'].values, bins=np.linspace(-1, 1, 51), alpha=0.5, label='Spearman')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(corr_data['pearson_statistic'].values, corr_data['spearman_statistic'].values, alpha=0.05)
    ax.set(xlabel='pearson', ylabel='spearman')
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(corr_data['intercept'].values, corr_data['scale'].values, alpha=0.05)
    ax.set(xlabel='intercept', ylabel='scale')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_complete_statistics():
    observations = load_data()
    predictions = pd.read_parquet(os.path.join(pred_path, f'{experiment}.parquet'))
    x = predictions.value_0.values
    y = observations.value_0.values
    corr_data = pd.read_csv(os.path.join(eval_path, 'reliability_metrics.csv')).set_index('stnid')
    intercept = corr_data['intercept'].loc[predictions.stnid.values]
    scale = corr_data['scale'].loc[predictions.stnid.values]
    x_corrected = x * scale + intercept

    print(np.mean(x - y), np.mean(np.abs(x - y)))
    print(np.mean(x_corrected - y), np.mean(np.abs(x_corrected  - y)))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey='all', sharex='all')
    ax=axs[0]
    ax.hist2d(x, y, bins=(128, 128), norm=mpl.colors.LogNorm())
    ax.set(xlabel='prediction', ylabel='observation')

    ax = axs[1]
    ax.hist2d(x_corrected, y, bins=(128, 128), norm=mpl.colors.LogNorm())
    ax.set(xlabel='prediction (debiased)')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_metrics_pca():

    metric_data = data.sort_index().loc[:, ['rmse', 'mae', 'max', 'bias']].values

    pca = PCA(n_components=2, whiten=True)
    transformed = pca.fit_transform(metric_data)

    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.05)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_all_time_series():
    observations = load_data()
    predictions = pd.read_parquet(os.path.join(pred_path, f'{experiment}.parquet'))

    ts_plot_path = os.path.join(plot_path, f'time-series_ALL')
    os.makedirs(ts_plot_path, exist_ok=True)

    stnids = observations.stnid.unique()

    for stnid in tqdm(stnids):
        obs_ = observations.loc[observations.stnid == stnid]
        time_stamps_ = np.asarray([_get_timestamp(date, time) for date, time in zip(obs_.date.values, obs_.time.values)])
        pred_ = predictions.loc[predictions.stnid == stnid]
        time_stamps__ = np.asarray([_get_timestamp(date, time) for date, time in zip(pred_.date.values, pred_.time.values)])
        assert np.all(time_stamps_ == time_stamps__)
        fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=100)
        fig.suptitle(
            'Station ID: {}, Lat: {:.2f}°, Lon: {:.2f}°, Elev(Stn): {:d} m, Elev(Mod): {:d} m'.format(
                stnid, obs_.latitude.iloc[0], obs_.longitude.iloc[0],
                int(np.round(obs_.elevation.iloc[0])), int(np.round(pred_.elevation.iloc[0]))
            )
        )
        ax.plot(time_stamps_, obs_.value_0.values, label='observations', c='r')
        ax.plot(time_stamps_, pred_.value_0.values, linestyle='--', label='predictions', c='k')
        # ax.set(xlabel='time', ylabel='T [K]')
        # plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ts_plot_path, f'{stnid}.png'), dpi=100)
        # plt.show()
        plt.close()
        gc.collect()


if __name__ == '__main__':
    plot_all_time_series()
    # plot_overview()
    # plot_extreme_cases()
    # plot_time_series_for_extremes()
    # plot_correlations()
    # plot_time_series_for_stnid('78896')
    # plot_complete_statistics()
    # plot_metrics_pca()

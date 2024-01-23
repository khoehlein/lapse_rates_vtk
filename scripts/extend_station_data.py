import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree


short_names= {
        '2021072314': 'Summer',
        '2021121906': 'Winter',
        'MISTRAL_202311_meta_table_IFS-1': 'MISTRAL',
    }

def extend_station_data():
    # date = '2021072314'
    date = 'MISTRAL_202311_meta_table_IFS-1'
    path_to_station_data = f'/mnt/ssd4tb/ECMWF/Observations/{date}.csv'
    station_data = pd.read_csv(path_to_station_data)
    station_coordinates = ccrs.Geocentric().transform_points(ccrs.PlateCarree(), station_data['longitude'].values, station_data['latitude'].values)

    path_to_orography = '/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib'
    orography = xr.open_dataset(path_to_orography, engine='cfgrib').z
    mesh_coordinates = ccrs.Geocentric().transform_points(ccrs.PlateCarree(), orography['longitude'].values, orography['latitude'].values)

    tree = KDTree(mesh_coordinates)
    nearest_index = tree.query(station_coordinates, k=1, return_distance=False)
    nearest_index = nearest_index.ravel()
    station_data['grid_elevation_nn'] = orography.values[nearest_index]

    station_data.to_csv(f'/mnt/ssd4tb/ECMWF/Observations/{date}_extended.csv')
    print('Done')


def plot_height_relation():
    dates = ['2021072314', '2021121906', 'MISTRAL_202311_meta_table_IFS-1']
    fig, axs = plt.subplots(1, len(dates), figsize=(4 * len(dates), 4), sharex='all', sharey='all')
    for i, date in enumerate(dates):
        path_to_station_data = f'/mnt/ssd4tb/ECMWF/Observations/{date}_extended.csv'
        station_data = pd.read_csv(path_to_station_data, index_col=0)
        fraction_missing = np.mean((station_data['elevation'] == 99999) * 100)
        fraction_zero = np.mean((station_data['elevation'] == 0) * 100)
        station_data = station_data.loc[station_data['elevation'] < 99999]
        axs[i].scatter(station_data['grid_elevation_nn'], station_data['elevation'], alpha=0.05)
        axs[i].set(xlabel='grid elevation [m]', title='{}: {:.2f}% missing, {:.2f}% zero'.format(short_names.get(date), fraction_missing, fraction_zero))
    axs[0].set(ylabel='station elevation [m]')
    plt.tight_layout()
    plt.show()


def compare_old_new():
    new_data = pd.read_csv(f'/mnt/ssd4tb/ECMWF/Observations/MISTRAL_202311_meta_table_IFS-1_extended.csv')

    dates = ['2021072314', '2021121906']
    fig, axs = plt.subplots(nrows=len(dates), ncols=3, figsize=(12, 4 * len(dates)))

    for i, date in enumerate(dates):
        old_data = pd.read_csv(f'/mnt/ssd4tb/ECMWF/Observations/{date}_extended.csv')
        overlap = set(old_data['stnid'].values.tolist()).intersection(set(new_data['station_id'].values.tolist()))
        print(len(overlap))
        overlap = list(overlap)
        old_data = old_data.set_index('stnid').loc[overlap]
        fraction_missing = np.mean(old_data['elevation'] == 99999) * 100
        fraction_zero = np.mean(old_data['elevation'] == 0) * 100
        print(fraction_zero)
        old_data = old_data.loc[old_data['elevation'] != 99999]
        new_data_ = new_data.set_index('station_id').loc[old_data.index.values]

        for j, column in enumerate(['latitude', 'longitude', 'elevation']):
            axs[i, j].scatter(new_data_[column], old_data[column], alpha=0.05)
            axs[i, j].set(ylabel=f'{column} (old, {short_names.get(date)}, {fraction_missing:.2f}% missing)', xlabel=f'{column} (new)')
    plt.tight_layout()
    plt.show()

    print('done')


def plot_zero_stations():
    dates = ['2021121906', '2021072314']
    fig, axs = plt.subplots(1, len(dates), figsize=(8 * len(dates), 8),subplot_kw={'projection': ccrs.PlateCarree()})

    new_data = pd.read_csv(f'/mnt/ssd4tb/ECMWF/Observations/MISTRAL_202311_meta_table_IFS-1_extended.csv')
    station_coordinates_new = ccrs.Geocentric().transform_points(ccrs.PlateCarree(), new_data['longitude'].values,
                                                                 new_data['latitude'].values)
    tree_new = KDTree(station_coordinates_new)

    for i, date in enumerate(dates):
        old_data = pd.read_csv(f'/mnt/ssd4tb/ECMWF/Observations/{date}_extended.csv')
        old_data = old_data.loc[old_data['elevation'] == 0]
        station_coordinates_old = ccrs.Geocentric().transform_points(ccrs.PlateCarree(), old_data['longitude'].values, old_data['latitude'].values)
        distances, nearest_to_old = tree_new.query(station_coordinates_old, k=1)
        #
        # plt.figure()
        # plt.hist(np.log10(distances.ravel() + 1.e-6), bins=50)
        # plt.show()
        # plt.close()
        #
        mask = distances.ravel() < 100

        old_data_ = old_data.loc[mask]
        new_data_ = new_data.iloc[nearest_to_old.ravel()[mask]]
        ax = axs[i]
        ax.scatter(new_data['longitude'], new_data['latitude'], alpha=0.5, label='MISTRAL stations')
        ax.scatter(old_data['longitude'], old_data['latitude'], alpha=0.5, label='old stations, z == 0')
        mask_mistral = new_data_['elevation'].values != 0
        ax.scatter(old_data_['longitude'].loc[mask_mistral], old_data_['latitude'].loc[mask_mistral], alpha=0.8, label='both, z(MISTRAL) != 0')
        ax.scatter(old_data_['longitude'].loc[~mask_mistral], old_data_['latitude'].loc[~mask_mistral], alpha=0.8, label='both, z(MISTRAL) == 0', c='red')
        ax.set(xlim=(5, 20), ylim=(35,50), title=f'{short_names.get(date)}')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.legend()

        new_data_.loc[~mask_mistral, ['station_id', 'station_name', 'elevation', 'grid_elevation_nn']].to_csv(f'/mnt/ssd4tb/ECMWF/Observations/MISTRAL_zero_elevation_{date}.csv')

        print(list(zip(new_data_['station_id'].loc[~mask_mistral].values.tolist(), new_data_['station_name'].loc[~mask_mistral].values.tolist())))

        fraction_matching = mask.mean() * 100
        fraction_new_zeros = (new_data_['elevation'] == 0).mean() * 100
        print(f'Fraction zeros: {fraction_new_zeros:.2f}%')
        print(list(zip(old_data_['stnid'], new_data_['station_id'])))

        # fig, axs = plt.subplots(1, 1)
        # axs.scatter(old_data_['elevation'], new_data_['elevation'], alpha=0.1)
        # axs.set(xlabel='elevation (old) [m]', ylabel='elevation (new) [m]', title=f'{date}, {fraction_matching:.2f}% matching')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    # extend_station_data()
    # plot_height_relation()
    # compare_old_new()
    plot_zero_stations()
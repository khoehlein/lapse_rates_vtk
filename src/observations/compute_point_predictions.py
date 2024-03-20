import os

import pandas as pd
import tqdm
import xarray as xr

from src.observations.verify_statistics import load_data, load_metadata


temp_data_root_path = '/mnt/data2/ECMWF/Temp_Data'
temp_file_pattern = 'HRES_2m_temp_{}.grib'
raw_elevation_path = '/mnt/data2/ECMWF/Orog_Data/HRES_orog_o1279_2021-2022.grib'
output_path = '/mnt/data2/ECMWF/Predictions'
observation_path = '/mnt/data2/ECMWF/Obs/observations_filtered.parquet'


if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)
    observation_data = pd.read_parquet(observation_path)
    metadata = pd.read_csv(os.path.join(os.path.dirname(observation_path), 'station_locations_nearest.csv')).set_index('stnid')
    elevation_data = xr.open_dataset(raw_elevation_path)['z']
else:
    elevation_data = None
    metadata = None
    observation_data = None


def model_predictions_plain():
    site_predictions = []

    grouped = observation_data.groupby('date')
    groups_date = grouped

    for date, observations_per_date in groups_date:
        path_to_predictions = os.path.join(temp_data_root_path, temp_file_pattern.format(date))
        predictions_per_date = xr.open_dataset(path_to_predictions)['t2m'].stack(hour=('time', 'step'))
        groups_hour = observations_per_date.groupby('time')
        assert len(groups_hour) == 24

        output_per_day = []
        with tqdm.tqdm(total=24, desc=date) as pbar:
            for hour, observations_per_hour in groups_hour:
                station_ids = observations_per_hour['stnid'].values
                if len(station_ids):
                    site_metadata = metadata.loc[station_ids]
                    nearest_node = site_metadata['nearest_node_o1280'].values
                    predictions_per_site = predictions_per_date.isel(hour=int(hour) // 100, values=nearest_node)
                    elevation_per_site = elevation_data.isel(values=nearest_node)
                    data = pd.DataFrame({
                        'stnid': station_ids,
                        'latitude': site_metadata['latitude'].values,
                        'longitude': site_metadata['longitude'].values,
                        'date': [date] * len(station_ids),
                        'time': [hour] * len(station_ids),
                        'elevation': elevation_per_site.values,
                        'value_0': predictions_per_site.values
                    }, index=observations_per_hour.index)
                    output_per_day.append(data)
                pbar.update(1)

        output_per_day = pd.concat(output_per_day, axis=0)
        output_per_day.sort_index(inplace=True)
        site_predictions.append(output_per_day)

    site_predictions = pd.concat(site_predictions, axis=0)
    site_predictions.to_parquet(os.path.join(output_path, 'predictions_hres.parquet'))


def model_predictions_constant_lapse():
    site_predictions = []

    grouped = observation_data.groupby('date')
    groups_date = grouped

    for date, observations_per_date in groups_date:
        path_to_predictions = os.path.join(temp_data_root_path, temp_file_pattern.format(date))
        predictions_per_date = xr.open_dataset(path_to_predictions)['t2m'].stack(hour=('time', 'step'))
        groups_hour = observations_per_date.groupby('time')
        output_per_day = []
        with tqdm.tqdm(total=24, desc=date) as pbar:
            for hour, observations_per_hour in groups_hour:
                station_ids = observations_per_hour['stnid'].values
                if len(station_ids):
                    site_metadata = metadata.loc[station_ids]
                    nearest_node = site_metadata['nearest_node_o1280'].values
                    predictions_per_site = predictions_per_date.isel(hour=int(hour) // 100, values=nearest_node)
                    elevation_per_site = elevation_data.isel(values=nearest_node)
                    dz = site_metadata['elevation'].values - elevation_per_site
                    t_pred = predictions_per_site.values - 0.0065 * dz
                    data = pd.DataFrame({
                        'stnid': station_ids,
                        'latitude': site_metadata['latitude'].values,
                        'longitude': site_metadata['longitude'].values,
                        'date': [date] * len(station_ids),
                        'time': [hour] * len(station_ids),
                        'elevation': site_metadata['elevation'].values,
                        'value_0': t_pred
                    }, index=observations_per_hour.index)
                    output_per_day.append(data)
                pbar.update(1)

        output_per_day = pd.concat(output_per_day, axis=0)
        output_per_day.sort_index(inplace=True)
        site_predictions.append(output_per_day)

    site_predictions = pd.concat(site_predictions, axis=0)
    site_predictions.to_parquet(os.path.join(output_path, 'predictions_hres-const-lapse.parquet'))


def load_predictions(experiment: str):
    path = os.path.join(output_path, f'{experiment}.parquet')
    return pd.read_parquet(path)


if __name__ == '__main__':
    model_predictions_plain()
    model_predictions_constant_lapse()

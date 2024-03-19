import os
import time
from multiprocessing import Pool

import pandas as pd
import xarray as xr
from src.model.level_heights import (
    compute_approximate_level_height,
    compute_full_level_pressure,
    compute_standard_surface_pressure,
    silly_interpolation_np
)


temp_data_root_path = '/mnt/data2/ECMWF/Temp_Data'
t2m_file_pattern = 'HRES_2m_temp_{}.grib'
t3d_file_pattern = 'HRES_Model_Level_temp_{}.grib'

raw_elevation_path = '/mnt/data2/ECMWF/Orog_Data/HRES_orog_o1279_2021-2022.grib'
raw_lsm_path = '/mnt/data2/ECMWF/LSM_Data/LSM_HRES_Sep2022.grib'
output_path = '/mnt/data2/ECMWF/Predictions'
observation_path = '/mnt/data2/ECMWF/Obs/observations_masked.parquet'

observation_data = pd.read_parquet(observation_path, columns=['stnid', 'date', 'time'])
metadata = pd.read_csv(os.path.join(os.path.dirname(observation_path), 'station_locations_nearest.csv')).set_index('stnid')
elevation_data = xr.open_dataset(raw_elevation_path)['z']


def compute_volumetric_temperature(
        t3d_per_site, t2m_per_site,
        elevation_per_site, z_station
):
    z_surf = elevation_per_site.values
    z_2m = z_surf + 2
    p_surf = compute_standard_surface_pressure(
        z_surf,
        base_temperature=t2m_per_site.values,
        base_temperature_height=z_2m
    )
    p = compute_full_level_pressure(p_surf)
    z_level = compute_approximate_level_height(
        p, p_surf, z_surf,
        base_temperature=t2m_per_site.values,
        base_temperature_height=z_2m
    )
    t_site = silly_interpolation_np(
        z_station,
        z_level, t3d_per_site.values
    )
    return t_site, z_level[-1], z_level[0]


def process_day(x):
    date, observations_per_date = x
    t1 = time.time()

    path_to_t2m = os.path.join(temp_data_root_path, t2m_file_pattern.format(date))
    path_to_t3d = os.path.join(temp_data_root_path, t3d_file_pattern.format(date))
    t2m_per_date = xr.open_dataset(path_to_t2m)['t2m'].stack(hour=('time', 'step'))
    t3d_per_date = xr.open_dataset(path_to_t3d, engine='cfgrib')['t'].stack(hour=('time', 'step'))

    output_per_day = []
    for hour, observations_per_hour in observations_per_date.groupby('time'):
        station_ids = observations_per_hour['stnid'].values
        if not len(station_ids):
            continue
        site_metadata = metadata.loc[station_ids]
        nearest_node = site_metadata['nearest_node_o1280'].values
        t2m_per_site = t2m_per_date.isel(hour=int(hour) // 100, values=nearest_node)
        t3d_per_site = t3d_per_date.isel(hour=int(hour) // 100, values=nearest_node)
        elevation_per_site = elevation_data.isel(values=nearest_node)
        z_station = site_metadata['elevation'].values
        dz = z_station - elevation_per_site

        t_pred, elevation_min, elevation_max = compute_volumetric_temperature(
            t3d_per_site, t2m_per_site,
            elevation_per_site, z_station
        )

        data = pd.DataFrame({
            'stnid': station_ids,
            'latitude': site_metadata['latitude'].values,
            'longitude': site_metadata['longitude'].values,
            'date': [date] * len(station_ids),
            'time': [hour] * len(station_ids),
            'elevation': site_metadata['elevation'].values,
            'hres': t2m_per_site.values,
            'elevation_difference': dz,
            'elevation_min': elevation_min,
            'elevation_max': elevation_max,
            'value_0': t_pred
        }, index=observations_per_hour.index)
        output_per_day.append(data)

    output_per_day = pd.concat(output_per_day, axis=0)
    output_per_day.sort_index(inplace=True)
    t2 = time.time()
    print(f'Date {date} completed in {t2 - t1} sec')
    return output_per_day


def model_predictions_interpolated():
    dates = list(observation_data.groupby('date'))
    print('Running predictions')
    with Pool(3) as p:
        site_predictions = p.map(process_day, dates)
    # site_predictions = [process_day(k) for k in dates]
    site_predictions = pd.concat(site_predictions, axis=0)
    site_predictions.to_parquet(os.path.join(output_path, 'predictions_hres-3d.parquet'))


if __name__ == '__main__':
    model_predictions_interpolated()
import os

import numpy as np
import pandas as pd
import xarray as xr
from src.observations.verify_statistics import load_data, load_metadata

temp_data_root_path = '/mnt/data2/ECMWF/Temp_Data'
temp_file_pattern = 'HRES_2m_temp_{}.grib'
raw_elevation_path = '/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib'
cache_path = None

if cache_path is not None:
    os.makedirs(cache_path, exist_ok=True)

observation_data = load_data()
metadata = load_metadata().set_index('stnid')
elevation_data = xr.open_dataset(raw_elevation_path)['z']

site_predictions = []

t_pred = np.ones((len(observation_data),))

grouped = observation_data.groupby('date')
groups = grouped.groups

for date in groups.keys():
    indices = groups.get_group(date)
    observations_per_date = observation_data.loc[indices]

    path_to_predictions = os.path.join(temp_data_root_path, temp_file_pattern.format(date))
    predictions_per_date = xr.open_dataset(path_to_predictions)['t2m']

    groups_hour = observations_per_date.groupby('time')
    assert len(groups_hour) == 24

    output_per_day = []

    for i, hour in enumerate(sorted(groups_hour.keys())):
        indices_hour = groups_hour.get_group(hour)
        observations_per_hour = observations_per_date.loc[indices_hour]
        station_ids = observations_per_hour['stnid'].values
        if len(station_ids):
            site_metadata = metadata.loc[station_ids.values]
            nearest_node = site_metadata['nearest_node_o1280'].values
            predictions_per_site = predictions_per_date.isel(time=i, values=nearest_node)
            elevation_per_site = elevation_data.isel(values=nearest_node)
            data = pd.DataFrame({
                'stnid': station_ids,
                'latitude': site_metadata['latitude'],
                'longitude': site_metadata['longitude'],
                'date': [date] * len(station_ids),
                'time': [hour] * len(station_ids)
                'elevation': elevation_per_site.values,
                'value_0': predictions_per_site.t2m.values
            })
            output_per_day.append(data)

    output_per_day = pd.concat(output_per_day, axis=0, ignore_index=True)
    if cache_path is not None:
        output_per_day.to_parquet(os.path.join(cache_path, f'predictions_{date}.parquet'))
    site_predictions.append(output_per_day)
    output_per_day = []

site_predictions = pd.concat(site_predictions, axis=0, ignore_index=True)




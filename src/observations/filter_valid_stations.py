import os.path

import numpy as np
import pandas as pd


PARQUET_PATH = '/mnt/ssd4tb/ECMWF/Obs/observations.parquet'
data = pd.read_parquet(PARQUET_PATH)
data = data.loc[data['elevation'] < 99999]

THRESHOLD_ANGLE = 0.
THRESHOLD_ELEVATION = 0.
MAX_FRACTION_MISSING = 0.5


def export_valid_stations():
    grouped = data.groupby(by=['stnid', 'date'])
    counts = grouped['latitude'].count()
    da = counts.to_xarray()
    fraction_of_days_missing = da.isnull().mean(dim='date')
    print(len(fraction_of_days_missing))

    grouped = data.groupby(by='stnid')
    lon_range = grouped['longitude'].max() - grouped['longitude'].min()
    lat_range = grouped['latitude'].max() - grouped['latitude'].min()
    elev_range = grouped['elevation'].max() - grouped['elevation'].min()

    mask = np.logical_and(lat_range <= THRESHOLD_ANGLE, lon_range <= THRESHOLD_ANGLE)
    mask = np.logical_and(mask, elev_range <= THRESHOLD_ELEVATION)
    print(np.mean(mask))

    assert np.all(lon_range.index.values == da.stnid.values)

    valid = np.logical_and(mask, fraction_of_days_missing.values < MAX_FRACTION_MISSING)

    print(np.mean(valid))

    valid = pd.Series(valid, index=lon_range.index)
    valid_extended = valid.loc[data.stnid.values]

    metadata = grouped[['latitude', 'longitude', 'elevation']].min()
    metadata['num_obs'] = grouped['latitude'].count()
    metadata = metadata.loc[valid.values]
    print(len(metadata))
    print('Writing meta data')
    metadata.to_csv(os.path.join(os.path.dirname(PARQUET_PATH), 'station_locations.csv'))
    print('Done')

    data_ = data.loc[valid_extended.values]

    print(len(data), len(data_))
    out_path, ext = os.path.splitext(PARQUET_PATH)
    out_path = out_path + '_filtered' + ext
    print('Writing filtered data')
    data_.to_parquet(out_path)


if __name__ == '__main__':
    export_valid_stations()

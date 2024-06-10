import os

import numpy as np
import pandas as pd

from src.model.geometry import Coordinates
from src.observations.verify_statistics import load_metadata, load_data

observations = load_data()
metadata = load_metadata().set_index('stnid')

eval_path = '/path/to/data/Evaluation'
os.makedirs(eval_path, exist_ok=True)


def agg_bias(data):
    return np.mean(data)


def agg_rmse(data):
    return np.sqrt(np.mean(np.square(data)))


def agg_mae(data):
    return np.mean(np.abs(data))


def agg_max_abs(data):
    return np.max(np.abs(data))


def agg_rmse_debiased(data):
    mu = np.mean(data)
    return np.sqrt(np.mean(np.square(data - mu)))


def agg_mae_debiased(data):
    mu = np.mean(data)
    return np.mean(np.abs(data - mu))


def agg_max_abs_debiased(data):
    mu = np.mean(data)
    return np.max(np.abs(data - mu))


def evaluate(predictions: pd.DataFrame, label: str):
    difference = predictions['value_0'] - observations['value_0']
    difference = difference.to_frame()
    difference['stnid'] = predictions['stnid']
    grouped = difference.groupby('stnid')
    stats = grouped['value_0'].aggregate(func=[agg_bias, agg_rmse, agg_mae, agg_max_abs, agg_rmse_debiased, agg_mae_debiased, agg_max_abs_debiased, 'count'])
    stats.columns = ['bias', 'rmse', 'mae', 'max', 'rmse_deb', 'mae_deb', 'max_deb', 'count']
    elevation = metadata['elevation'].loc[stats.index.values]
    elevation_model = metadata['elevation_o1280'].loc[stats.index.values]
    stats['elevation'] = elevation
    stats['elevation_difference'] = elevation - elevation_model
    station_longitude = metadata['longitude'].loc[stats.index.values]
    model_longitude = metadata['longitude_o1280'].loc[stats.index.values]
    station_latitude = metadata['latitude'].loc[stats.index.values]
    model_latitude = metadata['latitude_o1280'].loc[stats.index.values]
    coords_station = Coordinates.from_lat_lon(station_latitude, station_longitude).as_xyz().values
    coords_model = Coordinates.from_lat_lon(model_latitude, model_longitude).as_xyz().values
    distance = np.sqrt(np.sum(np.square(coords_station - coords_model), axis=-1))
    stats['station_longitude'] = station_longitude
    stats['station_latitude'] = station_latitude
    stats['model_longitude'] = model_longitude
    stats['model_latitude'] = model_latitude
    stats['model_station_distance'] = distance
    output_path = os.path.join(eval_path, label)
    os.makedirs(output_path, exist_ok=True)
    stats.to_csv(os.path.join(output_path, 'scores.csv'))


if __name__ == '__main__':
    prediction_path = '/path/to/data/Predictions/predictions_hres.parquet'
    predictions = pd.read_parquet(prediction_path)
    evaluate(predictions, 'predictions_hres')

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from src.model.downscaling.neighborhood_graphs import RadialNeighborhoodGraph
from src.model.geometry import Coordinates, LocationBatch


def load_station_temperatures(path: str):
    data = pd.read_parquet(path, columns=['date', 'time', 'value_0', 'stnid'])
    dates = pd.to_datetime(data['date'], format='%Y%m%d')
    hours = pd.to_timedelta(data['time'].values.astype(int) // 100, unit='hours')
    data['timestamp'] = dates + hours
    data = data.set_index(['timestamp', 'stnid'])
    return data


def get_link_data(data, radius_km):
    coords = Coordinates.from_xarray(data)
    lookup = NearestNeighbors()
    lookup.fit(coords.as_xyz().values)

    neighbor_graph = RadialNeighborhoodGraph.from_tree_query(LocationBatch(coords), lookup, radius_km)
    links = neighbor_graph.links
    links['location_id'] = data.index.values[links['location'].values]
    links['neighbor_id'] = data.index.values[links['neighbor'].values]
    return links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--timestamp', type=str, required=True)
    parser.add_argument('--radius', type=float, default=40)
    args = vars(parser.parse_args())

    with open(args['config_file'], 'r') as f:
        configs = json.load(f)

    station_obs = load_station_temperatures(configs['stations']['observations'])
    station_preds = load_station_temperatures(configs['stations']['predictions'])

    # orography = xr.open_dataset(configs['grid_data']['o1280']['z']['path'])[configs['grid_data']['o1280']['z']['key']]
    station_meta_data = pd.read_csv(configs['stations']['meta_data'], index_col=0).set_index('stnid').sort_index()
    active_links = get_link_data(station_meta_data, args['radius'])

    timestamp = np.datetime64(args['timestamp'])

    selected_obs = station_obs.loc[(timestamp, slice(None))]
    selected_stations = selected_obs.index.values
    valid_links = active_links.loc[np.logical_and(
        active_links['location_id'].isin(selected_stations),
        active_links['neighbor_id'].isin(selected_stations)
    )]
    selected_preds = station_preds.loc[(timestamp, slice(None))]

    print('Done')





if __name__ == '__main__':
    main()
import argparse
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.model.geometry import Coordinates
from src.paper.context_cache import DiskCache, CachingItemAdapter, CachingDataItem

logging.basicConfig(level=logging.DEBUG)


class StationSnapshot(object):

    def __init__(self, timestamp: np.datetime64, data: pd.DataFrame, active_links: pd.DataFrame):
        self.timestamp = timestamp
        self.data = data
        self.active_links = active_links

    def distance_weighted_average(self, weights_scale_km: float, min_count: int = 1, return_counts=False):
        if self.active_links is None:
            raise RuntimeError()
        weights_scale_km = float(weights_scale_km)
        distance = self.active_links['distance'].values
        weights = np.exp(-0.5 * (distance / weights_scale_km) ** 2)
        stnids = self.active_links['location_id'].values
        grouped = pd.DataFrame({
            'stnid': stnids,
            'weight': weights
        }).groupby('stnid')
        counts = grouped['weight'].count().rename('count')
        normalization = grouped['weight'].mean().loc[stnids].values
        weights /= normalization
        neighbor_ids = self.active_links['neighbor_id'].values
        data = pd.DataFrame({
            c: self.data[c].loc[neighbor_ids].values * weights
            for c in self.data.columns
        })
        data['stnid'] = stnids
        data = data.groupby('stnid').mean()
        if min_count > 1:
            counts = counts.loc[counts >= min_count]
            data = data.loc[counts.index.values]
            active_links = self.active_links.loc[
                np.logical_and(
                    self.active_links['location_id'].isin(data.index),
                    self.active_links['neighbor_id'].isin(data.index)
                )
            ]
        else:
            active_links = self.active_links
        data = data.sort_index()
        data = StationSnapshot(self.timestamp, data, active_links)
        if return_counts:
            return data, counts
        return data

    def histogram(self, **kwargs):
        columns = self.data.columns
        num_columns = len(columns)
        fig, axs = plt.subplots(1, num_columns, figsize=(num_columns * 3, 3))
        for i, c in enumerate(columns):
            axs[i].hist(self.data[c], bins=50, **kwargs)
            axs[i].set(title=str(c))
        plt.tight_layout()
        plt.show()
        plt.close()


class StationDatabase(object):

    @classmethod
    def from_files(cls, data_path: str, metadata_path: str):
        logging.debug('Loading station data from {}'.format(data_path))
        data = pd.read_parquet(data_path)
        logging.debug('Loading completed')
        logging.debug('Loading metadata from {}'.format(metadata_path))
        metadata = pd.read_csv(metadata_path, index_col=0)
        logging.debug('Loading completed')
        return cls(data, metadata)

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data.set_index(['timestamp', 'stnid'])
        self.metadata = metadata.set_index('stnid').sort_index()
        self.lookup = None
        self.link_cutoff_km = None
        self.active_links = None
        self._build_lookup()

    def get_timestamps(self):
        timestamps = self.data.index.get_level_values(0).unique().values
        return timestamps

    def _build_lookup(self):
        logging.debug('Indexing station locations')
        self._station_coords = Coordinates.from_xarray(self.metadata).as_xyz().values
        self.lookup = NearestNeighbors()
        self.lookup.fit(self._station_coords)
        logging.debug('Indexing completed')

    def set_link_cutoff(self, cutoff_radius_km: float):
        cutoff_radius_km = float(cutoff_radius_km)
        if self.link_cutoff_km != cutoff_radius_km:
            logging.debug('Setting link cutoff to {} km'.format(cutoff_radius_km))
            self.link_cutoff_km = cutoff_radius_km
            self.active_links = None
        if self.active_links is None:
            self._update_active_links()
        return self

    def _update_active_links(self):
        if self.link_cutoff_km is None:
            raise RuntimeError()
        logging.debug('Updating active links')
        distances, neighbor_ids = self.lookup.radius_neighbors(
            self._station_coords, self.link_cutoff_km * 1000., return_distance=True, sort_results=True
        )
        n_neighbors = np.fromiter((len(a) for a in neighbor_ids), int, count=len(neighbor_ids))
        indptr = np.cumsum(n_neighbors)
        neighbor_ids = np.concatenate(neighbor_ids)
        distances_km = np.concatenate(distances) / 1000.
        location_indicator = np.zeros((indptr[-1],), dtype=int)
        location_indicator[indptr[:-1]] = 1
        location_ids = np.cumsum(location_indicator)
        stnids = self.metadata.index.values
        self.active_links = pd.DataFrame({
            'location_id': stnids[location_ids],
            'neighbor_id': stnids[neighbor_ids],
            'distance': distances_km,
        })
        logging.debug('Update completed')

    def get_snapshot(self, timestamp: np.datetime64, columns: List[str] = None):
        logging.debug(f'Selecting valid data for timestamp {timestamp}')
        selected_data = self.data.loc[(timestamp, slice(None))]
        if columns is not None:
            selected_data = selected_data[columns]
        selected_stations = selected_data.index
        if self.active_links is not None:
            active_links = self.active_links.loc[np.logical_and(
                self.active_links['location_id'].isin(selected_stations),
                self.active_links['neighbor_id'].isin(selected_stations)
            )]
        else:
            active_links = None
        return StationSnapshot(timestamp, selected_data, active_links)

    def distance_weighted_average(self, weight_scale: float, columns: List[str] = None, min_count: int = 1):
        if self.active_links is None:
            raise RuntimeError()
        weight_scale = float(weight_scale)
        min_count = int(min_count)
        data = []
        for timestamp in tqdm(self.get_timestamps()):
            snapshot = self.get_snapshot(timestamp, columns=columns)
            average = snapshot.distance_weighted_average(
                weight_scale, min_count=min_count, return_counts=False
            )
            average.data['timestamp'] = [timestamp] * len(average.data)
            data.append(average.data.reset_index())
        data = pd.concat(data, axis=0, ignore_index=True)
        stnids = data['stnid'].unique()
        metadata = self.metadata.loc[stnids].reset_index()
        return StationDatabase(data, metadata)

    def rename(self, columns: Dict[str, str] = None, inplace: bool = False):
        data = self.data.rename(columns=columns)
        if inplace:
            self.data = data
            return self
        data = StationDatabase(data, self.metadata.copy())
        data.lookup = self.lookup.copy()
        if self.link_cutoff_km is not None:
            data.link_cutoff_km = self.link_cutoff_km
            data.active_links = self.active_links.copy()
        return data


class StationContextData(CachingDataItem):

    def __init__(self, data_path: str, metadata_path: str, link_cutoff_km: float, data: StationDatabase = None):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.link_cutoff = link_cutoff_km
        self.data = data

    def hash(self) -> str:
        items = (self.data_path, self.metadata_path, self.link_cutoff)
        m = hashlib.md5()
        for item in items:
            m.update(str(item).encode())
        return str(m.hexdigest())

    def to_disk(self, path: str) -> str:
        if self.data is not None:
            data_file = os.path.join(path, 'data.parquet')
            self.data.data.reset_index().to_parquet(data_file)
            metadata_file = os.path.join(path, 'metadata.csv')
            self.data.metadata.reset_index().to_csv(metadata_file)
        info_file = os.path.join(path, 'source_info.json')
        with open(info_file, 'w') as f:
            json.dump({
                'data_path': self.data_path,
                'metadata_path': self.metadata_path,
                'link_cutoff': self.link_cutoff,
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            }, f, indent=4, sort_keys=True)
        return path

    @classmethod
    def from_disk(cls, path: str) -> 'StationContextData':
        info_file = os.path.join(path, 'source_info.json')
        with open(info_file, 'r') as f:
            info = json.load(f)
        data_file = os.path.join(path, 'data.parquet')
        if not os.path.exists(data_file):
            raise FileNotFoundError()
        data = pd.read_parquet(data_file)
        metadata_file = os.path.join(path, 'metadata.csv')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError()
        metadata = pd.read_csv(metadata_file, index_col=0)
        data = StationDatabase(data, metadata)
        return cls(info['data_path'], info['metadata_path'], info['link_cutoff'], data=data)


def compute_context_and_write_to_cache(item: StationContextData, context_cache: DiskCache, station_db: StationDatabase):
    context = station_db.distance_weighted_average(
        item.link_cutoff / 2.,
        min_count=4,
        columns=['difference', 'abs_difference']
    ).rename(columns={'difference': 'bias', 'abs_difference': 'mae'}, inplace=True)
    item.data = context
    logging.debug('Writing data to cache')
    context_cache.write(item)
    logging.debug('Caching completed')
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--timestamp', type=str, required=True)
    parser.add_argument('--link-cutoff', type=float, default=40)
    args = vars(parser.parse_args())

    with open(args['config_file'], 'r') as f:
        logging.debug('Loading configs from {}'.format(args['config_file']))
        configs = json.load(f)

    station_db = StationDatabase.from_files(
        configs['stations']['historic'],
        configs['stations']['metadata']
    )
    station_db.set_link_cutoff(args['link_cutoff'])

    path_to_cache = configs['cache']
    logging.debug('Preparing context cache at {}'.format(path_to_cache))
    context_cache = DiskCache(
        path_to_cache, adapter=CachingItemAdapter(StationContextData), make_dirs=True
    )
    context_item = StationContextData(
        configs['stations']['historic'],
        configs['stations']['metadata'],
        args['link_cutoff'],
    )
    logging.debug('Requesting context data from cache')
    required_hash = context_item.hash()
    response = context_cache.request(required_hash)
    if response is None:
        logging.debug('Context data was not found. Computing data from scratch.')
        response = compute_context_and_write_to_cache(context_item, context_cache, station_db)
    context = response.data


    # logging.debug('Loading terrain data')
    # orography = xr.open_dataset(configs['terrain']['o1280']['z']['path'])[configs['grid_data']['o1280']['z']['key']]
    # logging.debug('Loading completed')

    print('Done')




if __name__ == '__main__':
    main()
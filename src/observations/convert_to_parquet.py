import argparse
import gc
import os

import pandas as pd
from tqdm import tqdm

from src.observations.geopoints import GeopointsData


def convert_to_parquet(path: str):
    files = sorted([f for f in os.listdir(path) if f.endswith('.geo')])
    data = []
    print('Reading')
    with tqdm(total=len(files)) as pbar:
        day_data = []
        for f in files:
            day_data.append(GeopointsData.from_geo(os.path.join(path, f)).to_pandas())
            pbar.update(1)
            if len(day_data) == 24:
                day_data = pd.concat(day_data, ignore_index=True)
                data.append(day_data)
                day_data = []
                gc.collect()
        if len(day_data):
            day_data = pd.concat(day_data, ignore_index=True)
            data.append(day_data)
        gc.collect()
    print('Concatenating')
    data = pd.concat(data, axis=0, ignore_index=True)
    print('Writing')
    data.to_parquet(os.path.join(path, '..', 'observations.parquet'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root_path', type=str, help='Data root directory')
    args = vars(parser.parse_args())
    convert_to_parquet(args['data_root_path'])


if __name__ == '__main__':
    main()

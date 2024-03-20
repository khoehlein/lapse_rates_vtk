import argparse
import json

import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, required=True)
args = vars(parser.parse_args())

with open(args['config_file'], 'r') as f:
    configs = json.load(f)

station_obs = pd.read_parquet(configs['stations']['observations'])
station_preds = pd.read_parquet(configs['stations']['predictions'])
station_meta = pd.read_csv(configs['stations']['meta_data'], index_col=0)
orography = xr.open_dataset(configs['grid_data']['o1280']['z']['path'])[configs['grid_data']['o1280']['z']['key']]

lookup = NearestNeighbors()

print('Done')

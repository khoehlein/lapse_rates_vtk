import argparse
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, required=True)
args = vars(parser.parse_args())

with open(args['config_file'], 'r') as f:
    configs = json.load(f)

station_obs = pd.read_parquet(configs['stations']['observations'])
station_preds = pd.read_parquet(configs['stations']['predictions'])
station_meta = pd.read_parquet(configs['stations']['meta_data'])



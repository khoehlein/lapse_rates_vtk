import datetime
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from blacklist_reader import read_blacklist, build_mask
from src.observations.helpers import compute_outlier_threshold

SHORT_FILTER_WINDOW = 7
LONG_FILTER_WINDOW = 31
RANSAC_VERSION = 'ransac-95.00'
MIN_RESIDUAL = 20.
THRESHOLD_PROBABILITY = 0.999
# STEP_THRESHOLD = 2.
output_path = '/mnt/data2/ECMWF/Cache'


def main():
    observations = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_filtered.parquet')
    residuals = pd.read_parquet(f'/mnt/data2/ECMWF/Predictions/predictions_hres-{RANSAC_VERSION}.parquet', columns=['residual', 'stnid'])
    filters = pd.read_parquet(os.path.join(output_path, f'filter_outputs_{RANSAC_VERSION}_{LONG_FILTER_WINDOW}_{SHORT_FILTER_WINDOW}.parquet'))
    blacklist = read_blacklist()

    grouped = filters.groupby('stnid')

    mask_data = []

    for stnid, group in tqdm(grouped):
        stn_obs = observations.loc[group.index.values]
        stn_res = residuals.loc[group.index.values]
        assert np.all(stn_obs.stnid.values == stnid)

        if stnid in blacklist:
            stn_mask = build_mask(stn_obs, blacklist.get(stnid))
        else:
            stn_mask = np.full(len(stn_obs), False)
        
        if np.any(~stn_mask):
            stn_residuals = stn_res['residual']
            residual_threshold = compute_outlier_threshold(stn_residuals.loc[~stn_mask], THRESHOLD_PROBABILITY, MIN_RESIDUAL)
            stn_mask = np.logical_or(stn_mask, stn_residuals.values > residual_threshold)

        stn_obs['valid'] = ~stn_mask
        mask_data.append(stn_obs)
    
    print('Concat')
    mask_data = pd.concat(mask_data, axis=0).sort_index()
    valid = mask_data['valid'].values
    print('Number valid observations: {} ({:.2f}%)'.format(np.sum(valid), np.mean(valid)))

    print('Writing')
    mask_data.to_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet')


if __name__ == '__main__':
    main()



import os

import numpy as np
import pandas as pd
from blacklist_reader import read_blacklist, build_mask
from src.observations.helpers import compute_outlier_threshold


RANSAC_VERSION = 'ransac-95.00'
MIN_RESIDUAL = 20.
THRESHOLD_PROBABILITY = 0.999
output_path = '/mnt/data2/ECMWF/Cache'


def main():
    print('Loading data')
    observations = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_filtered.parquet')
    residuals = pd.read_parquet(f'/mnt/data2/ECMWF/Predictions/predictions_hres-{RANSAC_VERSION}.parquet', columns=['residual', 'stnid'])
    blacklist = read_blacklist()

    print(f'Stations on blacklist: {len(blacklist)}')

    grouped = observations.groupby('stnid')
    mask_data = []

    for stnid, stn_obs in grouped:
        stn_res = residuals.loc[stn_obs.index.values]
        if stnid in blacklist:
            stn_mask = build_mask(stn_obs, blacklist.get(stnid))
            invalid = np.mean(stn_mask)
            if invalid == 0.:
                print('All valid after blacklist: {}'.format(stnid))
        else:
            stn_mask = np.full(len(stn_obs), False)
        if not np.all(stn_mask):
            stn_residuals = stn_res['residual']
            residual_threshold = compute_outlier_threshold(stn_residuals.loc[~stn_mask], THRESHOLD_PROBABILITY, MIN_RESIDUAL)
            print('Filtering outliers with threshold {}'.format(residual_threshold))
            stn_mask = np.logical_or(stn_mask, stn_residuals.values > residual_threshold)

        # print('Invalid after outlilers: {}'.format(np.mean(stn_mask)))

        stn_obs['valid'] = ~stn_mask
        mask_data.append(stn_obs)

    print('Concat')
    mask_data = pd.concat(mask_data, axis=0).sort_index()
    valid = mask_data['valid'].values
    print('Number valid observations: {} of {} ({:.2f}%)'.format(np.sum(valid), len(valid), np.mean(valid) * 100))

    print('Writing')
    mask_data.to_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet')


if __name__ == '__main__':
    main()



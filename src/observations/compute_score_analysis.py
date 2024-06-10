import argparse
import os.path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import product
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    args = vars(parser.parse_args())
    export_scores(args['input_file'], train=True)
    export_scores(args['input_file'], train=False)


def export_scores(input_file: str, train=False):
    print('Loading')
    obs, pred = load_predictions(input_file, train)

    print('Computing')
    scores = pred['score'].values
    score_rank = rankdata(scores)
    # score_bin = np.floor(score_rank * (100 / len(score_rank)))
    score_bin = np.digitize(np.fmax(scores, 0.), np.linspace(0, 1, 11))

    dz = pred['elevation_difference'].values
    dz_bin = np.digitize(dz, np.array([np.min(dz) - 1, -50., 50., np.max(dz) + 1]))

    res_default = np.abs(obs['value_0'].values - (pred['hres'].values - 0.0065 * dz))

    scores_per_bin = pd.DataFrame({
        'score_bin': score_bin,
        'dz_bin': dz_bin,
        'score': scores,
        'dz': dz,
    }).groupby(['score_bin', 'dz_bin']).agg(['mean', 'min', 'max', 'count'])
    scores_per_bin.columns = ['_'.join(c) for c in scores_per_bin.columns]

    cutoffs_lower = np.arange(-15, -6)
    cutoffs_upper = np.arange(-7, 51, 3)

    all = []
    for cutoff_min, cutoff_max in tqdm(list(product(cutoffs_lower, cutoffs_upper))):
        pred_adaptive = pred['hres'].values + np.clip(pred['lapse_rate'].values, cutoff_min, cutoff_max) / 1000 * dz
        res_adaptive = np.abs(obs['value_0'].values - pred_adaptive)
        df_mse = pd.DataFrame({
            'score_bin': score_bin,
            'dz_bin': dz_bin,
            'adaptive': res_adaptive**2,
            'default': res_default**2,
        }).groupby(['score_bin', 'dz_bin']).mean()
        df_mse.columns = [f'{x}_mse' for x in df_mse.columns]
        df_max = pd.DataFrame({
            'score_bin': score_bin,
            'dz_bin': dz_bin,
            'adaptive': res_adaptive,
            'default': res_default,
        }).groupby(['score_bin', 'dz_bin']).max()
        df_max.columns = [f'{x}_max' for x in df_max.columns]
        metrics = pd.concat([df_mse, df_max, scores_per_bin], axis='columns')
        metrics = metrics.reset_index()
        metrics['min_cutoff'] = [cutoff_min] * len(metrics)
        metrics['max_cutoff'] = [cutoff_max] * len(metrics)

        all.append(metrics)

    all = pd.concat(all, axis=0, ignore_index=True)
    label = 'train' if train else 'eval'
    all.to_csv(os.path.join(os.path.dirname(input_file), f'score_analysis_{label}.csv'))


def load_predictions(input_file, train=False):
    train_split = pd.read_csv('/path/to/data/Obs/train_stations.csv')
    obs = pd.read_parquet('/path/to/data/Obs/observations_masked.parquet',
                          columns=['value_0', 'valid', 'elevation', 'stnid'])
    pred = pd.read_parquet(input_file)
    split_mask = obs['stnid'].isin(train_split['stnid'].values)
    if not train:
        split_mask = ~split_mask
    mask = np.all(np.stack([
        split_mask,
        obs['valid'].values,
        obs['elevation'].values <= pred['elevation_max'].values,
        ~np.isnan(pred['score'].values)
    ], axis=0), axis=0)
    obs = obs.loc[mask]
    pred = pred.loc[mask]
    return obs, pred


if __name__ == '__main__':
    main()

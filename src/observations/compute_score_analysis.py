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
    export_scores(args['input_file'])


def export_scores(input_file: str):
    print('Loading')
    obs, pred = load_predictions(input_file)

    print('Computing')
    scores = pred['score'].values
    score_rank = rankdata(scores)
    # score_bin = np.floor(score_rank * (100 / len(score_rank)))
    score_bin = np.digitize(np.fmax(scores, 0.), np.linspace(0, 1, 11))

    dz = pred['elevation_difference'].values
    dz_bin = np.digitize(dz, np.array([np.min(dz) - 1, -100., 100., np.max(dz) + 1]))

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
    for cutoff_min, cutoff_max in tqdm(product(cutoffs_lower, cutoffs_upper)):
        pred_adaptive = pred['hres'].values + np.clip(pred['lapse_rate'].values, cutoff_min, cutoff_max) / 1000 * dz
        res_adaptive = np.abs(obs['value_0'].values - pred_adaptive)
        df = pd.DataFrame({
            'score_bin': score_bin,
            'dz_bin': dz_bin,
            'adaptive': res_adaptive**2,
            'default': res_default**2,
        })
        metrics = pd.concat([df.groupby(['score_bin', 'dz_bin']).mean(), scores_per_bin], axis='columns')
        metrics = metrics.reset_index()
        metrics['min_cutoff'] = [cutoff_min] * len(metrics)
        metrics['max_cutoff'] = [cutoff_max] * len(metrics)

        all.append(metrics)

    all = pd.concat(all, axis=0, ignore_index=True)
    all.to_csv(os.path.join(os.path.dirname(input_file), 'score_analysis.csv'))


def load_predictions(input_file):
    train_split = pd.read_csv('/mnt/data2/ECMWF/Obs/train_stations.csv')
    obs = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet',
                          columns=['value_0', 'valid', 'elevation', 'stnid'])
    pred = pd.read_parquet(input_file)
    mask = np.all(np.stack([
        ~obs['stnid'].isin(train_split['stnid'].values),
        obs['valid'].values,
        obs['elevation'].values <= pred['elevation_max'].values,
        ~np.isnan(pred['score'].values)
    ], axis=0), axis=0)
    obs = obs.loc[mask]
    pred = pred.loc[mask]
    return obs, pred


if __name__ == '__main__':
    main()

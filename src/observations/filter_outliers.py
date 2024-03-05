import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from src.observations.analyze_evaluation import pred_path
from src.observations.verify_statistics import load_data


def agg_relative_norm(x):
    norm = x.iloc[:-1].mean()
    return x.iloc[-1] / norm


def detect_peaks(obs: pd.Series, pred: pd.Series):
    diff = np.abs(obs - pred)
    scores = diff.rolling(window=24, center=False).agg(agg_relative_norm)
    print(scores)


experiment = 'predictions_hres-const-lapse'


def main():
    observations = load_data()
    predictions = pd.read_parquet(os.path.join(pred_path, f'{experiment}.parquet'))
    diff = np.abs(observations.value_0.values - predictions.value_0.values)
    station_ids = observations.stnid.unique()
    thresholds = np.arange(0.5, 100, 0.5)

    def metrics(threshold):
        mask = diff > threshold
        loss_fraction = mask.mean()
        affected_stations = len(observations.stnid.loc[mask].unique())
        return loss_fraction, affected_stations

    metric_data = np.asarray([metrics(threshold) for threshold in thresholds])

    fg, ax = plt.subplots(1, 1 )
    ax.plot(thresholds, metric_data[:, 0], label='affected observations')
    ax.plot(thresholds, metric_data[:, 1] / len(station_ids), label='affected stations')
    ax.set(yscale='log')
    ax.axvline(25, linestyle='--', color='k')
    print(metric_data[int(25 / 0.5) - 1])
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    # for stnid in tqdm(stnids):
    #     obs_ = observations.value_0.loc[observations.stnid == stnid]
    #     pred_ = predictions.value_0.loc[predictions.stnid == stnid]
    #     diff = np.abs(obs_ - pred_)
    #     outlier_model = IsolationForest()
    #     is_outlier = outlier_model.fit_predict(diff.values[:, None])

    # print(np.mean(is_outlier))


if __name__ == '__main__':
    main()

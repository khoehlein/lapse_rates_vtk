import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_timestamps(df: pd.DataFrame):
    dates = pd.to_datetime(df['date'].values, format='%Y%m%d').values
    hours = pd.to_timedelta(df['time'].values.astype(int) // 100, unit='h').values
    timestamps = dates + hours
    return timestamps


def mad(x: pd.Series):
    x_vals = x.values
    x_med = np.median(x_vals)
    return 1.4826 * np.median(np.abs(x_vals - x_med))


def compute_outlier_threshold(x: pd.Series, p: float=0.95, tmin=20.):
    mad_ = mad(x)
    f = norm.ppf(p)
    return max(f * mad_, tmin)


def get_step_filter(timestamps, filters, window_size):
    score_backward = filters['median_short_forward'].values - filters['median_long_backward'].values
    score_forward = filters['median_short_backward'].values - filters['median_long_forward'].values
    values = np.nanmin(np.stack([np.abs(score_backward), np.abs(score_forward)], axis=0), axis=0)
    values[timestamps < np.datetime64('2021-04-08T00:00')] = 0
    values[timestamps > np.datetime64('2022-03-24T00:00')] = 0
    series = pd.Series(values, index=pd.DatetimeIndex(timestamps)).fillna(0.)
    step_filter = series.rolling(datetime.timedelta(hours=window_size*24), center=True).max()
    return step_filter.values
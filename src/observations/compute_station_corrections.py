import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge
from tqdm import tqdm

from src.utils.weighted_ransac import WeightedRANSACRegressor

parser = argparse.ArgumentParser(description='Compute station corrections')
parser.add_argument('--confidence', type=float, default=0.95)
args = vars(parser.parse_args())

CONFIDENCE = args['confidence']
MAD_SCALE = np.abs(norm.interval(CONFIDENCE)[-1])


def fourier_features(x: np.ndarray, period: float) -> np.ndarray:
    z = x * ((2. * np.pi) / period)
    return np.stack([np.sin(z), np.cos(z)], axis=-1)


def detrend(y, predictors):
    timestamps = pd.to_datetime(predictors.date, format='%Y%m%d')
    day_of_year = timestamps.dt.dayofyear
    hour_of_day = predictors.time.values.astype(int) // 100
    timestamps = timestamps.values + hour_of_day.astype('timedelta64[h]')
    x = np.concatenate([
        predictors.value_0.values[:, None],
        fourier_features(day_of_year, 365),
        fourier_features(hour_of_day, 24)
    ], axis=-1)
    residuals_plain = y - predictors.value_0.values
    mad = 1.482 * np.median(np.abs(residuals_plain - np.median(residuals_plain)))
    threshold = MAD_SCALE * mad # 95 % confidence level for Gaussian distribution
    model = WeightedRANSACRegressor(
        Ridge(fit_intercept=True, alpha=1.e-4),
        residual_threshold=threshold,
        max_trials=500, min_samples=20,
        random_state=42
    )
    model.fit(x, residuals_plain)
    predicted = model.predict(x)
    estimator = model.estimator_
    stats = {
        'fraction_outlier': np.mean(~model.inlier_mask_),
        'mad': mad, 'threshold': threshold,
        'confidence': CONFIDENCE,
        'bias': estimator.intercept_,
        **{f'c{i}': c for i, c in enumerate(estimator.coef_)}
    }
    return predicted, timestamps, stats, ~model.inlier_mask_


obs = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_filtered.parquet')
pred = pd.read_parquet('/mnt/data2/ECMWF/Predictions/predictions_hres.parquet')

grouped = obs.groupby('stnid')

data = []
outliers = []
postprocessed = []

for stnid in tqdm(grouped.groups.keys()):
    idx = grouped.groups.get(stnid)
    obs_ = obs.loc[idx]
    pred_ = pred.loc[idx]
    predictions, timestamps, stats, mask = detrend(obs_.value_0.values, pred_)
    data.append({'stnid': stnid, **stats})
    outliers.append(obs_.loc[mask])
    pred_['value_0'] = predictions
    pred_['residual'] = obs_.value_0.values - predictions
    pred_['outlier'] = mask
    postprocessed.append(pred_)

label = 'ransac-{:.2f}'.format(CONFIDENCE * 100)

data = pd.DataFrame(data)
data.to_csv(f'/mnt/data2/ECMWF/Predictions/{label}_statistics.csv')

outliers = pd.concat(outliers, axis=0).sort_index()
outliers.to_parquet(f'/mnt/data2/ECMWF/Predictions/{label}_outliers.parquet')

postprocessed = pd.concat(postprocessed, axis=0).sort_index()
postprocessed.to_parquet(f'/mnt/data2/ECMWF/Predictions/predictions_hres-{label}.parquet')


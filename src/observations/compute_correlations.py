import time
from multiprocessing import Pool

import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

from src.observations.compute_point_predictions import load_predictions
from src.observations.verify_statistics import load_data, load_metadata

observations = load_data()
metadata = load_metadata()

predictions = load_predictions('predictions_hres')


def process_station(stnid):
    obs_ = observations[observations['stnid'] == stnid]
    pred_ = predictions[predictions['stnid'] == stnid]
    y = obs_.value_0.values

    model = SGDRegressor(loss='huber')
    model.fit(pred_.value_0.values[:, None], y[:, None])

    pred_pp = model.predict(pred_.value_0.values[:, None])
    residuals = pred_pp - y

    stats = {'intercept': model.intercept_, 'scale': model.coef_, 'stnid': stnid, 'count': len(y)}

    res_shapiro = shapiro(residuals)
    stats.update({'shapiro_statistic': res_shapiro.statistic, 'shapiro_pvalue': res_shapiro.pvalue})

    res_pearson = pearsonr(pred_pp, y)
    stats.update({'pearson_statistic': res_pearson.statistic, 'pearson_pvalue': res_pearson.pvalue})

    res_spearman = spearmanr(pred_pp, y)
    stats.update({'spearman_statistic': res_spearman.statistic, 'spearman_pvalue': res_spearman.pvalue})

    print(stnid)
    return stats


t1 = time.time()
with Pool(8) as p:
    outputs = p.map(process_station, metadata['stnid'].values)
t2 = time.time()

print('Time required: {} seconds per station'.format((t2-t1) / len(metadata)))

outputs = pd.DataFrame(outputs)
outputs.to_csv('/path/to/data/Evaluation/reliability_metrics.csv')
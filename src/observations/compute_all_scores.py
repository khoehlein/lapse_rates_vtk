import os
import gc

from src.observations.compute_score_analysis import export_scores
from src.observations.plot_score_histograms import export_histograms
from src.observations.plot_score_analysis import export_plots

root_path = '/mnt/data2/ECMWF/Predictions'
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

for folder in folders:
    pred_file = os.path.join(root_path, folder, 'predictions.parquet')
    for train in [True, False]:
        export_scores(pred_file, train)
        export_histograms(pred_file, train)
        label = 'train' if train else 'eval'
        score_file = os.path.join(root_path, folder, f'score_analysis_{label}.csv')
        export_plots(score_file, train)
        gc.collect()
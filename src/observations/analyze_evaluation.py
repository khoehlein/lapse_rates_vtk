import pandas as pd
from matplotlib import pyplot as plt

eval_path = '/mnt/ssd4tb/ECMWF/Evaluation/predictions_hres.csv'
data = pd.read_csv(eval_path)
elevation_difference = data['elevation_difference'].values
distance = data['model_station_distance'].values
count = data['count'].values
alphas = count / 8760

for metric in ['rmse', 'mae', 'max', 'bias']:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(metric)
    p = ax.scatter(elevation_difference, data[metric].values, alpha=alphas, c=distance / 1000, vmin=0, cmap='magma')
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label('Distance [km]')
    plt.tight_layout()
    plt.show()
    plt.close()

import numpy as np
import pandas as pd


obs = pd.read_parquet('/mnt/data2/ECMWF/Obs/observations_masked.parquet', columns=['valid', 'stnid'])
obs_counts = obs.groupby('stnid')['valid'].sum()
obs_counts = obs_counts.loc[obs_counts.values > 0].sort_values(ascending=False)

gen = np.random.Generator(np.random.PCG64(42))
selected = np.arange(0, len(obs_counts), 5)
selected = selected + gen.integers(0, 5, size=(len(selected),))

obs_train = obs_counts.iloc[selected].reset_index()

obs_train.to_csv('/mnt/data2/ECMWF/Obs/train_stations.csv')

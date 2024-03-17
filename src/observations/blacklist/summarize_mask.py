import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

observation_path = '/mnt/data2/ECMWF/Obs/observations_masked.parquet'

observations = pd.read_parquet(observation_path, columns=['stnid', 'valid'])

observation_count = observations.groupby('stnid').sum()

fig, ax = plt.subplots(1, 1)
ax.hist(np.log10(observation_count.values[observation_count.values > 0]), bins=50, log=True)
plt.tight_layout()
plt.show()
plt.close()

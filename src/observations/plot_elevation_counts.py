import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('C:\\Users\\kevin\\Downloads\\score_analysis.csv')
data = data.groupby(['score_bin', 'dz_bin'])['dz_count'].min().to_xarray()

fig, ax = plt.subplots(1, 1)
p = ax.pcolor((np.log10(data.values).T))
plt.colorbar(p, ax=ax)
plt.show()
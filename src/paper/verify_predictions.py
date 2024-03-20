import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

p1 = pd.read_parquet('/mnt/ssd4tb/ECMWF/Predictions/20240320035637/predictions.parquet')
p2 = pd.read_parquet('/mnt/ssd4tb/ECMWF/Predictions/predictions_hres-const-lapse.parquet')

var_name = 'elevation'
x1 = p1['hres'].values - 0.0065 * p1['elevation_difference'].values
x2 = p2['value_0'].values

plt.figure()
plt.scatter(x1, x2, alpha=0.05)
plt.tight_layout()
plt.show()
plt.close()

print(np.max(np.abs(x1 -x2)))
print('Done')


import numpy as np
import pandas as pd
import pvlib

date = np.datetime64('2020-01-01T00:00', 'h')
index = pd.date_range(date, periods=24, freq='H')
output = pvlib.solarposition.get_solarposition(index, 45, 0)

print(output)


from typing import Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import pvlib


class SolarLightingModel(object):

    def __init__(self, timestamp: np.datetime64, location: Tuple[float, float], method='nrel_numpy'):
        self.light = pv.Light(color='white', )
        self.light.set_scene_light()
        self.light.set_direction_angle(90., 0.)
        self.timestamp = timestamp.astype('datetime64[h]')
        self.location = location
        self.method = method
        self.elevation, self.azimuth = None, None
        self._update_solar_positions()
        self._update_light()

    def set_location(self, location: Tuple[float, float]):
        self.location = location
        self._update_solar_positions()
        self._update_light()
        return self

    def set_timestamp(self, date: np.datetime64):
        self.timestamp = date
        self._update_solar_positions()
        self._update_light()
        return self

    def _update_solar_positions(self):
        datetime_index = pd.date_range(self.timestamp, periods=1, freq='H')
        output = pvlib.solarposition.get_solarposition(datetime_index, self.location[1], self.location[0], method=self.method)
        self.elevation = output['elevation'][0]
        self.azimuth = (90 - output['azimuth'][0]) % 360

    def set_method(self, method: str):
        self.method = method
        self._update_solar_positions()
        self._update_light()
        return self

    def _update_light(self):
        self.light.set_direction_angle(self.elevation, self.azimuth)


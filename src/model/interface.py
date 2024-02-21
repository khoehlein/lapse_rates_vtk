from dataclasses import dataclass
from enum import Enum


class GridConfiguration(Enum):
    O1280 = 'o1280'
    O8000 = 'o8000'


class SurfaceFieldType(Enum):
    T2M = 't2m'
    T2M_INTERPOLATION = 't2m_interpolation'
    T2M_DIFFERENCE = 't2m_difference'
    LAPSE_RATE = 'lapse_rate'
    LAPSE_RATE_INTERPOLATION = 'lapse_rate_interpolation'
    Z = 'z'
    Z_INTERPOLATION = 'z_interpolation'
    Z_DIFFERENCE = 'z_difference'
    LSM = 'lsm'


class VolumeFieldType(Enum):
    Z_QUANTILES = 'z_quantiles'
    T2M_VOLUME = 't2m_volume'


class PropertyModelUpdateError(Exception):
    pass


class PropertyModel(object):

    @dataclass
    class Properties(object):
        pass

    def __init__(self, properties: 'PropertyModel.Properties' = None):
        self.set_properties(properties)

    def set_properties(self, properties) -> 'PropertyModel':
        if properties == self.properties:
            return self
        self.properties = properties
        return self

    def get_kws(self):
        raise NotImplementedError()

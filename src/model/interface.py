import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


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
        self.properties = None
        self.set_properties(properties)
        self._properties_changed = False

    def set_properties(self, properties) -> 'PropertyModel':
        if properties == self.properties:
            return self
        self.properties = properties
        self._properties_changed = True
        return self

    def properties_changed(self):
        return self._properties_changed

    def update(self):
        self._properties_changed = False
        return self

    def get_kws(self):
        raise NotImplementedError()


class DataNodeModel(object):
    """
    Base class for objects from which data is retrieved
    """

    def __init__(self, name: str = None):
        if name is not None:
            name = str(name)
        self._name = name
        self.uid = str(uuid.uuid4())

    def name(self) -> str:
        return self._name or self.uid

    def data(self) -> Any:
        raise NotImplementedError()

    def is_valid(self) -> bool:
        raise NotImplementedError()

    def update(self) -> 'DataNodeModel':
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class FilterNodeModel(PropertyModel):

    def __init__(self):
        super(FilterNodeModel, self).__init__()
        self.__inputs = {}
        self.__outputs = {}

    def __setattr__(self, key, value):
        if value is not None:
            if key in self.__inputs:
                assert isinstance(value, self.__inputs[key])
            if key in self.__outputs:
                assert isinstance(value, self.__outputs[key])
        return super().__setattr__(key, value)

    def register_input(self, key: str, cls, instance: DataNodeModel = None):
        self.__inputs[key] = cls
        self._setattr_or_check_existing(cls, instance, key)
        return self

    def register_output(self, key: str, cls, instance: DataNodeModel = None):
        self.__outputs[key] = cls
        self._setattr_or_check_existing(cls, instance, key)
        return self

    def _setattr_or_check_existing(self, cls, instance, key):
        if not hasattr(self, key) or instance is not None:
            self.__setattr__(key, instance)
        else:
            existing = getattr(self, key)
            assert existing is None or isinstance(existing, cls)

    def all_inputs_valid(self):
        for key in self.__inputs:
            if not hasattr(self, key):
                return False
            instance = getattr(self, key)
            if instance is None:
                return False
            if not instance.is_valid():
                return False
        return True

    def all_outputs_valid(self):
        for key in self.__outputs:
            if not hasattr(self, key):
                return False
            instance = getattr(self, key)
            if instance is None:
                return False
            if not instance.is_valid():
                return False
        return True

    def set_outputs_valid(self, valid: bool):
        for key in self.__outputs:
            try:
                instance = getattr(self, key)
            except AttributeError:
                continue
            if instance is not None:
                instance.set_valid(valid)
        return self

    def clear_outputs(self):
        for key in self.__outputs:
            try:
                instance = getattr(self, key)
            except AttributeError:
                continue
            if instance is not None:
                instance.clear()
        return self

    def update(self):
        super().update()
        if self.all_outputs_valid():
            return self
        if self.properties is None or not self.all_inputs_valid():
            self.clear_outputs()
        else:
            self.update_outputs()
        self.set_outputs_valid(True)
        return self

    def update_outputs(self):
        raise NotImplementedError()

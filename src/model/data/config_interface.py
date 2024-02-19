import json
import logging
from typing import Dict, Any, Union, List
import xarray as xr


class SourceConfigKey(object):
    PATH = 'path'
    ATTRIBUTE_NAME = 'attribute'
    DIMENSIONS = 'dims'
    SELECTION = 'select'
    ENGINE = 'engine'


class SourceConfiguration(object):

    DEFAULTS = {
        SourceConfigKey.SELECTION: None,
        SourceConfigKey.ENGINE: 'cfgrib'
    }

    def __init__(
            self,
            path: str,
            attribute: str,
            dims: List[str],
            selection: Dict[str, Any],
            engine: str
    ):
        self.path = path
        self.attribute = attribute
        self.dims = dims
        self.selection = selection
        self.engine = engine

    @classmethod
    def from_config_entry(cls, config_entry: Union[str, Dict[str, Any]], entry_key: str) -> 'SourceConfiguration':
        if isinstance(config_entry, str):
            return cls(config_entry, entry_key, cls.DEFAULTS[SourceConfigKey.SELECTION], cls.DEFAULTS[SourceConfigKey.ENGINE])
        path = config_entry.get(SourceConfigKey.PATH)
        assert path is not None
        attribute_name = config_entry.get(SourceConfigKey.ATTRIBUTE_NAME, entry_key)
        dims = config_entry.get(SourceConfigKey.DIMENSIONS, None)
        selection = config_entry.get(SourceConfigKey.SELECTION, None)
        engine = config_entry.get(SourceConfigKey.ENGINE, cls.DEFAULTS[SourceConfigKey.ENGINE])
        return cls(path, attribute_name, dims, selection, engine)

    def load_data(self):
        data = xr.open_dataset(self.path, engine=self.engine)[self.attribute]
        if self.dims is not None:
            data = data.transpose(*self.dims)
        if self.selection is not None:
            data = data.isel(**self.selection)
        return data


class ConfigReader(object):

    @staticmethod
    def load_json_config(path_to_data_config: str):
        logging.info(f'Loading model from {path_to_data_config}.')
        with open(path_to_data_config, 'r') as f:
            configs = json.load(f)
        return configs

    def __init__(self, config_class):
        self.config_class = config_class

    def load_data(self, config_entry: Union[str, Dict[str, Any]]):
        configuration = self.config_class.from_config_entry(config_entry)
        return configuration.load_data()

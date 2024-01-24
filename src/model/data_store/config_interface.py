import json
import logging
from typing import Dict, Any, Union
import xarray as xr


class ConfigurationKey(object):
    PATH = 'path'
    SELECTION = 'select'
    ENGINE = 'engine'


class DataConfiguration(object):

    DEFAULTS = {
        ConfigurationKey.SELECTION: None,
        ConfigurationKey.ENGINE: 'cfgrib'
    }

    def __init__(self, path: str, selection: Dict[str, Any], engine: str):
        self.path = path
        self.selection = selection
        self.engine = engine

    @classmethod
    def from_config_entry(cls, config_entry: Union[str, Dict[str, Any]]) -> 'DataConfiguration':
        if isinstance(config_entry, str):
            return cls(config_entry, cls.DEFAULTS[ConfigurationKey.SELECTION], cls.DEFAULTS[ConfigurationKey.ENGINE])
        path = config_entry.get(ConfigurationKey.PATH)
        selection = config_entry.get(ConfigurationKey.SELECTION, cls.DEFAULTS[ConfigurationKey.SELECTION])
        engine = config_entry.get(ConfigurationKey.ENGINE, cls.DEFAULTS[ConfigurationKey.ENGINE])
        return cls(path, selection, engine)

    def load_data(self):
        data = xr.open_dataset(self.path, engine=self.engine)
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

import json
import os
import shutil
from typing import Any, Type

import pandas as pd


class CachingDataItem(object):

    def hash(self) -> str:
        raise NotImplementedError()

    def to_disk(self, path: str) -> 'str':
        raise NotImplementedError()

    @classmethod
    def from_disk(cls, path: str) -> 'CachingDataItem':
        raise NotImplementedError()


class CacheAdapter(object):

    def read(self, entry_path: str) -> Any:
        raise NotImplementedError()

    def write(self, data: Any, entry_path: str) -> 'CacheAdapter':
        raise NotImplementedError()


class DiskCache(object):

    def __init__(self, path: str, make_dirs: bool = False, adapter: CacheAdapter = None):
        self.path = os.path.abspath(path)
        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        elif make_dirs:
            os.makedirs(self.path)
        else:
            raise RuntimeError('[ERROR] Cache directory does not exist and cannot be created')
        self.default_adapter = adapter

    def entries(self, sort=False):
        entries = os.listdir(self.path)
        if sort:
            entries = sorted(entries)
        return entries

    def exists(self, hash: str) -> bool:
        return hash in set(self.entries())

    def request(self, hash: str, adapter: CacheAdapter = None) -> Any:
        if self.exists(hash):
            adapter = self._parse_adapter_argument(adapter)
            entry_path = self._path_for_hash(hash)
            return adapter.read(entry_path)
        return None

    def _path_for_hash(self, hash):
        return os.path.join(self.path, hash)

    def _prepare_directory_for(self, hash: str) -> str:
        entry_path = self._path_for_hash(hash)
        os.makedirs(entry_path)
        return entry_path

    def write(self, data: CachingDataItem, adapter: CacheAdapter = None, overwrite: bool = False) -> str:
        adapter = self._parse_adapter_argument(adapter)
        data_hash = data.hash()
        if self.exists(data_hash):
            if not overwrite:
                raise RuntimeError('[ERROR] Cache entry exists and may not be overwritten')
            shutil.rmtree(self.path)
        entry_path = self._prepare_directory_for(data_hash)
        adapter.write(data, entry_path)
        return entry_path

    def _parse_adapter_argument(self, adapter):
        if adapter is None:
            if self.default_adapter is None:
                raise RuntimeError('[ERROR] adapter must be set during init or specified during request')
            adapter = self.default_adapter
        return adapter


class CachingItemAdapter(CacheAdapter):

    def __init__(self, item_class: Type[CachingDataItem]):
        self.item_class = item_class

    def read(self, entry_path: str) -> Any:
        return self.item_class.from_disk(entry_path)

    def write(self, data: Any, entry_path: str) -> str:
        if not isinstance(data, self.item_class):
            raise TypeError('[ERROR] ContextCacheAdapter expects data in form of ContextDataItem')
        return data.to_disk(entry_path)

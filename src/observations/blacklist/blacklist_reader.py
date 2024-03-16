import json
import os
from collections import namedtuple
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd

from src.observations.helpers import get_timestamps

MIN_DATE = np.datetime64('2021-04-01T00:00')
MAX_DATE = np.datetime64('2022-04-01T00:00')


MaskConfig = namedtuple('MaskConfig', ['type', 'start', 'end', 'data'])


def read_entry(entry: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[MaskConfig, List[MaskConfig]]:
    if isinstance(entry, list):
        return [read_entry(e) for e in entry]
    min_date = entry.get('from')
    if min_date is None:
        min_date = MIN_DATE
    else:
        min_date = np.datetime64(min_date + 'T00:00')
    max_date = entry.get('to')
    if max_date is None:
        max_date = MAX_DATE
    else:
        max_date = np.datetime64(max_date + 'T00:00')
    assert min_date >= MIN_DATE
    assert max_date <= MAX_DATE
    assert min_date < max_date
    data = None
    if 'default' in entry:
        type_ = 'default'
        data = entry['default']
    elif 'below' in entry:
        type_ = 'below'
        data = entry['below']
    else:
        type_ = 'all'
    return MaskConfig(type_, min_date, max_date, data)


def read_blacklist():
    bl_path = os.path.join(os.path.dirname(__file__), 'blacklist.json')
    with open(bl_path, 'r') as f:
        blacklist = json.load(f)
    return {key: read_entry(entry) for key, entry in blacklist.items()}


def build_mask(data: pd.DataFrame, mask_config: Union[MaskConfig, List[MaskConfig]], key=None) -> np.ndarray:
    if isinstance(mask_config, list):
        masks = [build_mask(data, cfg) for cfg in mask_config]
        return np.any(np.stack(masks, axis=0), axis=0)
    timestamps = get_timestamps(data)
    time_mask = np.logical_and(timestamps >= mask_config.start, timestamps < mask_config.end)
    if mask_config.type == 'all':
        return time_mask
    if key is None:
        key = 'value_0'
    values = data[key].values
    if mask_config.type == 'default':
        value_mask = values == mask_config.data
    else:
        value_mask = values <= mask_config.data
    return np.logical_and(time_mask, value_mask)


def _test():
    blacklist = read_blacklist()
    print(blacklist)


if __name__ == '__main__':
    _test()

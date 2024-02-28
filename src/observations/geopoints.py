import pandas as pd

path = ''


class GeopointsData(object):

    @classmethod
    def from_geo(cls, path):
        reader = GeopointsReader()
        return reader.read(path)

    def __init__(self):
        self.columns = None
        self.data = []
        self.metadata = {}

    def to_pandas(self):
        return pd.DataFrame(self.data, columns=self.columns)

    def to_netcdf(self):
        raise NotImplementedError()


class GeopointsReader(object):

    KEY_MARKER = '#'

    def __init__(self):
        self.current_geo_key = None
        self.dataset: GeopointsData = None
        self._conversions = {
            'stnid': str,
            'date': str,
            'time': (lambda x: '{:02}'.format(x)),
            'level': int,
            'period': int,
            'step': int,
        }
        self._read_actions = {
            '#COLUMNS': self.read_column_names,
            '#METADATA': self.read_meta_data,
            '#DATA': self.read_data_line,
        }
        self._current_read_action = None

    def read_column_names(self, line: str):
        columns = line.split('\t')
        self.dataset.columns = columns

    def read_meta_data(self, line: str):
        key, value = line.split('=')
        self.dataset.metadata[key] = self._conversions.get(key, str)(value)

    def read_data_line(self, line: str):
        columns = self.dataset.columns
        new_data = {
            key: self._conversions.get(key, float)(value)
            for key, value in zip(columns, line.split('\t'))
        }
        self.dataset.data.append(new_data)

    def read(self, path, dataset: GeopointsData = None):
        if dataset is None:
            dataset = GeopointsData()
        self.dataset = dataset
        with open(path, 'r') as f:
            self.read_file_contents(f)
        self.dataset = None
        return dataset

    def read_file_contents(self, f):
        for line in f:
            if line.endswith('\n'):
                line = line[:-1]
            if line.startswith(self.KEY_MARKER):
                self.update_current_key(line)
            else:
                self.parse_contents(line)

    def update_current_key(self, line):
        self.current_geo_key = str(line)
        self._current_read_action = self._read_actions.get(self.current_geo_key)

    def parse_contents(self, line: str):
        if self._current_read_action:
            self._current_read_action(line)
        else:
            raise RuntimeError('[ERROR] Current read action not set')


def _test():
    path = '/mnt/ssd4tb/ECMWF/Observations/2t_obs_2021121906_elev.geo'
    reader = GeopointsReader()
    data = reader.read(path)

    print(data)

if __name__ == '__main__':
    _test()

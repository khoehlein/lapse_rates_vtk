import dataclasses
from typing import Dict, Any


class KeywordAdapter(object):

    def __init__(self, aliases: Dict[str, str], transforms: Dict[str, Any]):
        self._aliases = aliases
        self._transforms = transforms

    def _field_value(self, name: str, source):
        value = getattr(source, name)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def _field_name(self, name: str):
        return self._aliases.get(name, name)

    def read(self, properties):
        return {
            self._field_name(field.name): self._field_value(field.name, properties)
            for field in dataclasses.fields(properties)
        }

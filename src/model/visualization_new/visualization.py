import uuid
from dataclasses import dataclass
from enum import Enum

from src.model.interface import PropertyModel, DataNodeModel


class MappingTarget(object):
    pass


class SurfaceColor(MappingTarget):
    pass


class MappingModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        pass

    def __init__(self, source: DataNodeModel, target: MappingTarget):
        super().__init__()
        self.source = source
        self.target = target

class VisualizationModel(object):

    class MappingSlot(Enum):
        pass

    def __init__(self):
        self.uid = str(uuid.uuid4())
        self.mappers = {slot: None for slot in self.MapperSlot}

    def set_mapper(self, slot: 'VisualizationModel.MapperSlot')
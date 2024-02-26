import uuid
from dataclasses import dataclass
from enum import Enum

import pyvista as pv
from src.model.interface import PropertyModel
from src.model.visualization_new.colormaps import DEFAULT_ADAPTER, MeshStyleModel
from src.model.visualization_new.utils import KeywordAdapter


class DisplayItem(PropertyModel):

    def __init__(self, name: str = None):
        super().__init__()
        self.uid = str(uuid.uuid4())
        if name is not None:
            self.name = str(name)
        self.name = name
        self._is_valid = True

    def name(self) -> str:
        return self.name or self.uid

    def is_valid(self) -> bool:
        return not self._properties_changed

    def paint(self, plotter: pv.Plotter, adapter: KeywordAdapter = None):
        self._properties_changed = False
        return self


class MeshStyle(Enum):
    WIREFRAME = 'wireframe'
    SURFACE = 'surface'
    POINTS = 'points'


class MeshDisplayModel(DisplayItem):

    class Properties(DisplayItem.Properties):
        style: MeshStyle
        properties: PropertyModel.Properties

    def __init__(self):
        super().__init__()
        self.mesh_source = None
        self.style = MeshStyleModel()

    def set_mesh(self):

    def paint(self, host: pv.Plotter, adapter: KeywordAdapter = None):
        super().paint(host, adapter)
        if adapter is None:
            adapter = DEFAULT_ADAPTER
        actor = host.add_mesh()
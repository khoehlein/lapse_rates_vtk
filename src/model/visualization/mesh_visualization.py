from typing import Dict, Any

from src.model._legacy.geometry import SurfaceDataset
from src.model.visualization.colors import ColorModel
from src.model.visualization.interface import VisualizationModel, VisualizationType, DataConfiguration
from src.model.visualization.mesh_geometry import MeshGeometryModel


class MeshVisualization(VisualizationModel):

    def __init__(
            self,
            visual_type: VisualizationType,
            geometry: MeshGeometryModel, coloring: ColorModel,
            visual_key: str = None, parent=None,
    ):
        super().__init__(visual_type, visual_key, parent)
        self.geometry = geometry
        self.coloring = coloring

    def clear_host(self):
        if 'scalar_bar' in self.host_actors:
            scalar_bar_title = self.coloring.get_scalar_bar_title(self.gui_label)
            self.host.remove_scalar_bar(scalar_bar_title)
            self.host.remove_actor(self.host_actors['scalar_bar'])
            del self.host_actors['scalar_bar']
        return super().clear_host()

    def _write_to_host(self):
        color_kws = self.coloring.get_kws()
        actors = self.geometry.write_to_host(self.host, **color_kws, name=self.key)
        self.host_actors.update(actors)
        scalar_title = self.coloring.get_scalar_bar_title(self.gui_label)
        self.coloring.update_actors(self.host_actors, self.host, scalar_title, self.gui_label)

    def update_geometry(self, properties: MeshGeometryModel.Properties) -> 'MeshVisualization':
        self.geometry.set_properties(properties)
        if self.host_actors:
            self.geometry.update_actors(self.host_actors)
        return self

    def update_color(self, properties: ColorModel.Properties) -> 'MeshVisualization':
        if self.coloring.supports_update(properties):
            scalar_bar_old = self.coloring.get_scalar_bar_title(self.gui_label)
            self.coloring.set_properties(properties)
            if self.host_actors:
                self.coloring.update_actors(self.host_actors, self.host, scalar_bar_old, self.gui_label)
        else:
            host = self.clear_host()
            self.coloring = ColorModel.from_properties(properties)
            self.set_host(host)
        return self

    def set_vertical_scale(self, scale: float) -> 'MeshVisualization':
        self.geometry.set_vertical_scale(scale)
        if self.host_actors:
            self.geometry.update_actors(self.host_actors)
        return self

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key: str = None) -> 'MeshVisualization':
        raise NotImplementedError()

    @staticmethod
    def _select_source_data(dataset: Dict[str, SurfaceDataset], source_properties):
        selection = {
            DataConfiguration.SURFACE_O1280: ['surface_o1280'],
            DataConfiguration.SURFACE_O8000: ['surface_o8000'],
        }[source_properties]
        return [dataset[key] for key in selection]

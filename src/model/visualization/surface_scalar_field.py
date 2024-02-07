import logging
from typing import Dict, Any

from src.model.geometry import SurfaceDataset
from src.model.visualization.colors import ColorModel
from src.model.visualization.interface import VisualizationModel, VisualizationType, DataConfiguration
from src.model.visualization.mesh_geometry import MeshGeometryModel


class SurfaceScalarField(VisualizationModel):

    def __init__(
            self,
            geometry: MeshGeometryModel, coloring: ColorModel,
            visual_key: str = None, parent=None
    ):
        super().__init__(VisualizationType.SURFACE_SCALAR_FIELD, visual_key, parent)
        self.geometry = geometry
        self.coloring = coloring

    def _write_to_host(self):
        color_kws = self.coloring.get_kws()
        actors = self.geometry.write_to_host(self.host, **color_kws, name=self.key)
        self.host_actors.update(actors)
        scalar_title = self.coloring.scalar_bar_title
        if scalar_title is not None:
            actor = self.host.add_scalar_bar(mapper=self.host_actors['mesh'].mapper, title=scalar_title)
            self.host_actors['scalar_bar'] = actor

    def update_geometry(self, properties: MeshGeometryModel.Properties) -> 'SurfaceScalarField':
        self.geometry.set_properties(properties)
        if self.host_actors:
            self.geometry.update_actors(self.host_actors)
        return self

    def update_color(self, properties: ColorModel.Properties) -> 'SurfaceScalarField':
        if self.coloring.supports_update(properties):
            scalar_bar_old = self.coloring.scalar_bar_title
            self.coloring.set_properties(properties)
            if self.host_actors:
                self.coloring.update_actors(self.host_actors, self.host, scalar_bar_old)
        else:
            self.coloring = ColorModel.from_properties(properties)
            self._write_to_host()
        return self

    def set_vertical_scale(self, scale: float) -> 'SurfaceScalarField':
        self.geometry.set_vertical_scale(scale)
        if self.host_actors:
            self.geometry.update_actors(self.host_actors)
        return self

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key: str = None) -> 'SurfaceScalarField':
        logging.info('Building surface scalar visualization')
        color_properties = properties['coloring']
        color_model = ColorModel.from_properties(color_properties)
        source_properties = properties['source_data']
        surface_data = cls._select_source_data(dataset, source_properties)
        geometry_model = MeshGeometryModel(surface_data[0].get_polydata())
        geometry_properties = properties['geometry']
        geometry_model.set_properties(geometry_properties)
        return cls(geometry_model, color_model, visual_key=key)

    @staticmethod
    def _select_source_data(dataset: Dict[str, SurfaceDataset], source_properties):
        selection = {
            DataConfiguration.SURFACE_O1280: ['surface_o1280'],
            DataConfiguration.SURFACE_O8000: ['surface_o8000'],
        }[source_properties]
        return [dataset[key] for key in selection]

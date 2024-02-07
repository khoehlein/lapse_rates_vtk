import logging
from typing import Dict, Any

from src.model.geometry import SurfaceDataset
from src.model.visualization.colors import ColorModel
from src.model.visualization.interface import VisualizationType
from src.model.visualization.mesh_geometry import MeshGeometryModel
from src.model.visualization.mesh_visualization import MeshVisualization


class SurfaceScalarField(MeshVisualization):

    def __init__(
            self,
            geometry: MeshGeometryModel, coloring: ColorModel,
            visual_key: str = None, parent=None
    ):
        super().__init__(VisualizationType.SURFACE_SCALAR_FIELD, geometry, coloring, visual_key, parent)

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key: str = None) -> 'SurfaceScalarField':
        logging.info('Building surface scalar visualization')
        color_properties = properties['coloring']
        color_model = ColorModel.from_properties(color_properties)
        source_properties = properties['source_data']
        surface_data = cls._select_source_data(dataset, source_properties)
        mesh = surface_data[0].get_polydata()
        geometry_model = MeshGeometryModel(mesh, properties=properties['geometry'])
        return cls(geometry_model, color_model, visual_key=key)

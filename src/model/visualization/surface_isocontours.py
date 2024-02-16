import logging
from dataclasses import dataclass
from typing import Dict, Any
import pyvista as pv

from src.model._legacy.geometry import SurfaceDataset
from src.model.visualization.colors import ColorModel
from src.model.visualization.interface import VisualizationType, standard_adapter, ScalarType
from src.model.visualization.mesh_geometry import MeshGeometryModel
from src.model.visualization.mesh_visualization import MeshVisualization
from src.model.visualization.surface_scalar_field import SurfaceScalarField


@dataclass
class ContourProperties(object):
    contour_scalar: ScalarType
    isolevels: int


class SurfaceIsocontours(MeshVisualization):

    def __init__(
            self,
            geometry: MeshGeometryModel, coloring: ColorModel,
            visual_key: str = None, parent=None
    ):
        super().__init__(VisualizationType.SURFACE_ISOCONTOURS, geometry, coloring, visual_key, parent)

    @classmethod
    def from_dataset(cls, dataset: Dict[str, SurfaceDataset], properties: Dict[str, Any], key: str = None) -> 'SurfaceScalarField':
        logging.info('Building surface scalar visualization')
        color_properties = properties['coloring']
        color_model = ColorModel.from_properties(color_properties)
        source_properties = properties['source_data']
        surface_data = cls._select_source_data(dataset, source_properties)
        contour_properties = properties['contours']
        contours = cls._compute_contours(surface_data[0].get_polydata(), contour_properties)
        geometry_model = MeshGeometryModel(contours, properties=properties['geometry'])
        return cls(geometry_model, color_model, visual_key=key)

    @staticmethod
    def _compute_contours(mesh: pv.PolyData, contour_properties: ContourProperties) -> pv.PolyData:
        kws = standard_adapter.read(contour_properties)
        contours = mesh.contour(**kws, compute_scalars=True, method='contour')
        return contours

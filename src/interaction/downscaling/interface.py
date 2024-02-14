from PyQt5.QtCore import QObject, pyqtSignal

from src.interaction.downscaling.domain_selection import DomainData
from src.interaction.downscaling.downscaling import DownscalerModel, InterpolationType, NearestNeighborInterpolation, \
    BarycentricInterpolation
from src.model.visualization.interface import PropertyModel


class DownscalingProcess(PropertyModel):
    
    def __init__(self, source_domain: DomainData, target_domain: DomainData):
        super().__init__(None)
        self.source_domain = source_domain
        self.target_domain = target_domain
        
    def update(self):
        raise NotImplementedError()
    
    
class LowresDownscalingProcess(DownscalingProcess):

    class Properties(DownscalingProcess.Properties):
        interpolation_type = InterpolationType

    def __init__(self, source_domain: DomainData, target_domain: DomainData):
        super().__init__(source_domain, target_domain)
        self.interpolator = None
        self._interpolators = {
            InterpolationType.NEAREST_NEIGHBOR: NearestNeighborInterpolation,
            InterpolationType.BARYCENTRIC: BarycentricInterpolation,
        }

    def set_properties(self, properties) -> 'PropertyModel':
        super().set_properties(properties)
        self.update_interpolator()

    def update_interpolator(self, properties=None):
        self.interpolator = self._get_interpolator(properties)

    def _get_interpolator(self, properties = None):
        if properties is None:
            properties = self.properties
        if properties is None:
            return None
        cls = self._interpolators[properties.interpolation_type]
        return cls(self.source_domain.data)

class DownscalingPipelineModel(object):

    def __init__(
            self,
            neighborhood_lookup: NeighborhoodLookupModel,
            downscaler: DownscalerModel
    ):
        self.neighborhood_lookup = neighborhood_lookup
        self.downscaler = downscaler


class DownscalingPipelineController(QObject):

    def from_settings(
            self,
            neighborhood_settings: NeighborhoodLookupModel.Properties,
            downscaling_settings: DownscalerModel.Properties,
    ):


    def __init__(self, view: DownscalingPipelineView, model: DownscalingPipelineModel, parent=None):
        super(DownscalingPipelineController, self).__init__(parent)
        self.view = view
        self.model = model


class NeighborhoodData(object):

    def __init__(self, domain_model: DomainData):
        self.domain_model = domain_model
        self.lookup = NeighborhoodLookup(self.data_store)
        self._neighborhood_graph = None
        self.data = None

    @property
    def data_store(self):
        return self.domain_model.data_store

    def set_neighborhood_properties(self, properties: NeighborhoodLookup.Properties):
        self.lookup.set_properties(properties)
        self.update()
        return self

    def update(self):
        domain_sites = self.domain_model.sites
        self._neighborhood_graph = self.lookup.query_neighbor_graph(domain_sites)
        self.data = self.data_store.query_site_data(self._neighborhood_graph.links['neighbors'])

    def query_neighbor_graph(self, locations: LocationBatch) -> NeighborhoodGraph:
        if self.lookup is None:
            raise RuntimeError('[ERROR]  Error while querying neighborhood graph: lookup not found.')
        return self.lookup.query_neighbor_graph(locations)

import uuid

from src.model.domain_selection import DomainSelectionModel
from src.model.downscaling.methods import DownscalingMethodModel


class DownscalingPipelineModel(object):

    def __init__(
            self,
            source_domain: DomainSelectionModel,
            target_domain: DomainSelectionModel,
    ):
        self.uid = str(uuid.uuid4())
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.downscaler: DownscalingMethodModel = None

    @property
    def data(self):
        if self.downscaler is None:
            return None
        return self.downscaler.outputs

    def set_downscaler(self, downscaler: DownscalingMethodModel):
        self.downscaler = downscaler
        self.update_downscaler_data()
        return self

    def update_downscaler_data(self):
        # self.downscaler.set_data(self.source_domain.data, self.target_domain.data)
        return self

    def update(self):
        # self.downscaler.update()
        return self

from dataclasses import dataclass

from src.model.geometry import DomainLimits
from src.model.data.data_store import GlobalData
from src.model.interface import PropertyModel


class DomainSelectionModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        min_latitude: float
        max_latitude: float
        min_longitude: float
        max_longitude: float

        def get_domain_limits(self):
            return DomainLimits(
                min_latitude=self.min_latitude, max_latitude=self.max_latitude,
                min_longitude=self.min_longitude, max_longitude=self.max_longitude
            )

    def __init__(self, data_store: GlobalData):
        super().__init__(None)
        self.data_store = data_store
        self.data = None

    def set_properties(self, properties: 'DomainSelectionModel.Properties') -> 'DomainSelectionModel':
        super().set_properties(properties)
        return self

    def update(self):
        domain_bounds = self.properties.get_domain_limits()
        if self.data_store is not None:
            self.data = self.data_store.get_domain_dataset(domain_bounds)
        return self


DEFAULT_DOMAIN = DomainSelectionModel.Properties(43., 47., 6., 12.)

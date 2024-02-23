from dataclasses import dataclass

from src.model.data.data_source import MeshDataSource, MultiFieldSource
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

    def __init__(self, data_store: GlobalData, name: str = None):
        super().__init__()
        self.data_store = data_store
        self.data = None
        mesh_name = 'mesh'
        fields_name = 'source_fields'
        if name is not None:
            mesh_name = '{}_{}'.format(mesh_name, name)
            fields_name = '{}_{}'.format(fields_name, name)
        self.mesh_source = MeshDataSource(name=mesh_name).set_valid(True)
        self.fields = MultiFieldSource(
            self.data_store.scalar_names(),
            self.mesh_source,
            name=fields_name
        ).set_valid(True)

    def set_properties(self, properties: 'DomainSelectionModel.Properties') -> 'DomainSelectionModel':
        super().set_properties(properties)
        if self.properties_changed():
            self.mesh_source.set_valid(False)
            self.fields.set_valid(False)
        return self

    def update(self):
        super().update()
        if self.properties is None or self.data_store is None:
            self.data = None
        else:
            domain_bounds = self.properties.get_domain_limits()
            self.data = self.data_store.get_domain_dataset(domain_bounds)
        self._update_data_sources()
        return self

    def _update_data_sources(self):
        self.mesh_source.set_data(self.data.mesh)
        self.mesh_source.set_valid(True)
        self.fields.set_data(self.data.data)
        self.fields.set_valid(True)


DEFAULT_DOMAIN = DomainSelectionModel.Properties(43., 47., 6., 12.)

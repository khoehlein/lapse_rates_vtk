from dataclasses import dataclass

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QWidget

from src.interaction.downscaling.data_store import GlobalData
from src.interaction.downscaling.geometry import DomainLimits
from src.model.visualization.interface import PropertyModel


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
        domain_bounds = self.properties.get_domain_limits()
        self.data = self.data_store.get_domain_dataset(domain_bounds)
        return self


DEFAULT_DOMAIN = DomainSelectionModel.Properties(43., 47., 6., 12.)


class DomainSelectionView(QWidget):
    domain_limits_changed = None

    def get_settings(self) -> DomainSelectionModel.Properties:
        raise NotImplementedError()

    def update_settings(self, settings: DomainSelectionModel.Properties):
        raise NotImplementedError()


class DomainSelectionController(QObject):
    domain_changed = None

    def __init__(
            self,
            settings_view: DomainSelectionView,
            domain_lr: DomainSelectionModel,
            domain_hr: DomainSelectionModel,
            parent=None
    ):
        super().__init__(parent)
        self.view = settings_view
        self.model_lr = domain_lr
        self.model_hr = domain_hr
        self.view.domain_limits_changed.connect(self._on_domain_changed)
        self.view.update_settings(DEFAULT_DOMAIN)

    def _on_domain_changed(self):
        self._synchronize_domain_settings()
        self.domain_changed.emit()

    def _synchronize_domain_settings(self):
        bounds = self.view.get_settings()
        self.model_lr.set_properties(bounds)
        self.model_hr.set_properties(bounds)

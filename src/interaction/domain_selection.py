from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton

from src.model.domain_selection import DomainSelectionModel, DEFAULT_DOMAIN
from src.widgets import RangeSpinner


class DomainSelectionView(QWidget):

    domain_limits_changed = pyqtSignal(DomainSelectionModel.Properties)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.select_latitude = RangeSpinner(
            self,
            DEFAULT_DOMAIN.min_latitude, DEFAULT_DOMAIN.max_latitude,
            0., 90.
        )
        self.select_longitude = RangeSpinner(
            self,
            DEFAULT_DOMAIN.min_longitude, DEFAULT_DOMAIN.max_longitude,
            -180., 180.
        )
        self.button_apply = QPushButton('Apply', self)
        self.button_apply.clicked.connect(self._on_domain_change_applied)
        self._set_layout()

    def _on_domain_change_applied(self):
        self.domain_limits_changed.emit(self.get_settings())

    def get_settings(self) -> DomainSelectionModel.Properties:
        return DomainSelectionModel.Properties(
            *self.select_latitude.limits(),
            *self.select_longitude.limits(),
        )

    def update_settings(self, settings: DomainSelectionModel.Properties):
        self.select_latitude.set_limits(settings.min_latitude, settings.max_latitude)
        self.select_longitude.set_limits(settings.min_longitude, settings.max_longitude)
        return self


class DomainSelectionController(QObject):
    domain_changed = pyqtSignal()

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

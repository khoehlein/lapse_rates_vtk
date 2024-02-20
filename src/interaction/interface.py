from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget

from src.model.interface import PropertyModel


class PropertyModelView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def get_settings(self) -> PropertyModel.Properties:
        raise NotImplementedError()

    def update_settings(self, settings: PropertyModel.Properties):
        raise NotImplementedError()

    def set_defaults(self):
        raise NotImplementedError()


class PropertyModelController(QObject):

    model_changed = pyqtSignal()

    def __init__(
            self,
            view: PropertyModelView,
            model: PropertyModel,
            parent=None,
            apply_defaults: bool = True,
    ):
        super().__init__(parent)
        self.view = view
        self.model = model
        self.view.settings_changed.connect(self._on_settings_changed)
        if apply_defaults:
            self.set_defaults()
        else:
            self._synchronize_properties()

    def set_defaults(self):
        self.view.update_settings(self.default_settings())

    def default_settings(self):
        raise NotImplementedError()

    def _on_settings_changed(self):
        self._synchronize_properties()
        self.model_changed.emit()

    def _synchronize_properties(self):
        properties = self.view.get_settings()
        self.model.set_properties(properties)

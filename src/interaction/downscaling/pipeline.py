from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget

from src.interaction.domain_selection import DomainSelectionController
from src.interaction.downscaling.methods import DownscalingMethodController
from src.model.downscaling.methods import DownscalerType, DownscalingMethodModel
from src.model.downscaling.pipeline import DownscalingPipelineModel


class DownscalingPipelineView(QWidget):
    downscaler_changed = pyqtSignal(DownscalerType)

    def get_settings(self) -> DownscalingMethodModel.Properties:
        raise NotImplementedError()

    def get_current_downscaler_view(self):
        raise NotImplementedError()


class DownscalingPipelineController(QObject):

    def __init__(
            self,
            view: DownscalingPipelineView,
            model: DownscalingPipelineModel,
            downscaling_controller: DownscalingMethodController,
            data_controller: DomainSelectionController,
            parent=None
    ):
        super().__init__(parent)
        self.view = view
        self.view.downscaler_type_changed.connect(self._on_downscaler_type_changed)
        self.model = model
        self.data_controller = data_controller
        self.data_controller.domain_changed.connect(self._on_domain_changed)
        self.downscaling_controller = downscaling_controller

    def _on_downscaler_type_changed(self, downscaler_type: DownscalerType):
        controller = DownscalerFactory(downscaler_type).build_from_view(self.view.get_current_downscaler_view())
        downscaler = controller.model
        self.model.set_downscaler(downscaler)
        self.downscaling_controller = controller
        self.downscaling_controller.model_changed.connect(self._on_model_changed)
        self.model.downscaler.update()

    def _on_domain_changed(self):
        self.model.update_downscaler_data()
        self.model.downscaler.update()

    def _on_model_changed(self):
        self.model.downscaler.update()

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.model.data.data_store import GlobalData
from src.interaction.domain_selection import DomainSelectionController
from src.interaction.interface import PropertyModelView, PropertyModelController
from src.model.downscaling.methods import DownscalingMethodModel


class NetworkDownscalerView(DownscalingMethodView):
    pass


class NetworkDownscalerController(DownscalingMethodController):
    pass


class DownscalerFactory(object):

    def __init__(self, downscaler_type: DownscalerType):
        self.downscaler_type = downscaler_type
        self.controller_class = {
            DownscalerType.FIXED_LAPSE_RATE: FixedLapseRateDownscalerController,
            DownscalerType.ADAPTIVE_LAPSE_RATE: AdaptiveLapseRateDownscalerController,
            DownscalerType.NETWORK: NetworkDownscalerController,
        }.get(self.downscaler_type)
        self.view_class = {
            DownscalerType.FIXED_LAPSE_RATE: FixedLapseRateDownscalerView,
            DownscalerType.ADAPTIVE_LAPSE_RATE: AdaptiveLapseRateDownscalerView,
            DownscalerType.NETWORK: NetworkDownscalerView,
        }.get(self.downscaler_type)

    def build_from_view(self, downscaler_view: DownscalingMethodView) -> DownscalingMethodController:
        assert isinstance(downscaler_view, self.view_class)
        controller = self.controller_class.from_view(downscaler_view)
        return controller



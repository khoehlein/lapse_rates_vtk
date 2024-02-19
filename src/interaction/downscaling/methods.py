from src.interaction.downscaling.neighborhood import NeighborhoodModelView, NeighborhoodModelController
from src.interaction.interface import PropertyModelView, PropertyModelController
from src.model.data.data_store import GlobalData
from src.model.downscaling.methods import DownscalingMethodModel, FixedLapseRateDownscaler, LapseRateEstimator, \
    AdaptiveLapseRateDownscaler
from src.model.downscaling.pipeline import DownscalingPipelineModel


class DownscalingMethodView(PropertyModelView):
    pass


class DownscalingMethodController(PropertyModelController):

    @classmethod
    def from_view(cls, view: DownscalingMethodView, pipeline: DownscalingPipelineModel) -> 'DownscalingMethodController':
        raise NotImplementedError()

    def __init__(self, view: DownscalingMethodView, model: DownscalingMethodModel, parent=None, apply_defaults=True):
        super().__init__(view, model, parent, apply_defaults)


class FixedLapseRateDownscalerView(DownscalingMethodView):
    pass


class FixedLapseRateDownscalerController(DownscalingMethodController):

    @classmethod
    def from_view(cls, view: FixedLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'FixedLapseRateDownscalerController':
        model = FixedLapseRateDownscaler()
        return cls(view, model)


class LapseRateEstimatorView(PropertyModelView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.neighborhood_view = NeighborhoodModelView(self)


    def get_neighborhood_view(self) -> NeighborhoodModelView:
        return self.neighborhood_view


class LapseRateEstimatorController(PropertyModelController):

    @classmethod
    def from_view(cls, view: LapseRateEstimatorView, data_store: GlobalData):
        neighborhood_view = view.get_neighborhood_view()
        neighborhood_controller = NeighborhoodModelController.from_view(neighborhood_view, data_store)
        model = LapseRateEstimator(neighborhood_controller.model)
        return cls(view, model, neighborhood_controller)

    def __init__(
            self,
            view: LapseRateEstimatorView, model: LapseRateEstimator,
            neighborhood_controller: NeighborhoodModelController,
            parent=None, apply_defaults: bool = True
    ):
        super().__init__(view, model, parent, apply_defaults)
        self.neighborhood_controller = neighborhood_controller
        self.neighborhood_controller.model_changed.connect(self._on_neighborhood_changed)

    def _on_neighborhood_changed(self):
        self.model.synchronize_properties()
        self.model.output = None
        self.model_changed.emit()


class AdaptiveLapseRateDownscalerView(DownscalingMethodView):

    def get_estimator_view(self) -> LapseRateEstimatorView():
        raise NotImplementedError()


class AdaptiveLapseRateDownscalerController(DownscalingMethodController):

    @classmethod
    def from_view(cls, view: AdaptiveLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'AdaptiveLapseRateDownscalerController':
        estimator_view = view.get_estimator_view()
        global_data = pipeline.source_domain.data_store
        estimator_controller = LapseRateEstimatorController.from_view(estimator_view, global_data)
        estimator: LapseRateEstimator = estimator_controller.model
        model = AdaptiveLapseRateDownscaler(estimator)
        return cls(view, model, estimator_controller)

    def __init__(
            self,
            view: AdaptiveLapseRateDownscalerView, model: AdaptiveLapseRateDownscaler,
            estimator_controller: LapseRateEstimatorController,
            parent=None, apply_defaults=True
    ):
        super().__init__(view, model, parent, apply_defaults)
        self.estimator_controller = estimator_controller
        self.estimator_controller.model_changed.connect(self._on_estimator_changed)

    def _on_estimator_changed(self):
        self.model.synchronize_properties()
        self.model.output = None
        self.model_changed.emit()
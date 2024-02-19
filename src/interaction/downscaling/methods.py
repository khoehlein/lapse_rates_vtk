from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QCheckBox, QSpinBox, QVBoxLayout, QLabel, \
    QHBoxLayout

from src.interaction.downscaling.neighborhood import NeighborhoodModelView, NeighborhoodModelController
from src.interaction.interface import PropertyModelView, PropertyModelController
from src.model.data.data_store import GlobalData
from src.model.downscaling.interpolation import InterpolationType
from src.model.downscaling.methods import DownscalingMethodModel, FixedLapseRateDownscaler, LapseRateEstimator, \
    AdaptiveLapseRateDownscaler
from src.model.downscaling.pipeline import DownscalingPipelineModel
from src.model.interface import PropertyModel
from src.widgets import RangeSpinner


class DownscalingMethodView(PropertyModelView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_interpolation_type = QComboBox(self)
        self.combo_interpolation_type.addItem('Nearest neighbor', InterpolationType.NEAREST_NEIGHBOR)
        self.combo_interpolation_type.addItem('Barycentric', InterpolationType.BARYCENTRIC)
        self.combo_interpolation_type.model().item(1).setDisabled(True)
        self.combo_interpolation_type.currentIndexChanged(self.settings_changed.emit)
        self._interpolation_types = {
            InterpolationType.NEAREST_NEIGHBOR: 0,
            InterpolationType.BARYCENTRIC: 1,
        }


class DownscalingMethodController(PropertyModelController):

    @classmethod
    def from_view(cls, view: DownscalingMethodView, pipeline: DownscalingPipelineModel) -> 'DownscalingMethodController':
        raise NotImplementedError()

    def __init__(self, view: DownscalingMethodView, model: DownscalingMethodModel, parent=None, apply_defaults=True):
        super().__init__(view, model, parent, apply_defaults)


class FixedLapseRateDownscalerView(DownscalingMethodView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_lapse_rate = QDoubleSpinBox(self)
        self.spinner_lapse_rate.setMinimum(-14.)
        self.spinner_lapse_rate.setMaximum(100.)
        self.spinner_lapse_rate.setValue(-6.5)
        self.spinner_lapse_rate.valueChanged.connect(self.settings_changed.emit)
        self._set_layout()

    def build_layout(self):
        layout = QFormLayout()
        layout.addRow('Interpolation:', self.combo_interpolation_type)
        layout.addRow('Lapse rate:', self.spinner_lapse_rate)
        return layout

    def _set_layout(self):
        self.setLayout(self.build_layout())

    def get_settings(self) -> FixedLapseRateDownscaler.Properties:
        return FixedLapseRateDownscaler.Properties(
            self.combo_interpolation_type.currentData(),
            self.spinner_lapse_rate.value()
        )

    def update_settings(self, settings: FixedLapseRateDownscaler.Properties):
        self.spinner_lapse_rate.setValue(settings.lapse_rate)
        self.combo_interpolation_type.setCurrentIndex(self._interpolation_types.get(settings.interpolation))
        return self


class FixedLapseRateDownscalerController(DownscalingMethodController):

    def __init__(self, view: FixedLapseRateDownscalerView, model: FixedLapseRateDownscaler, parent=None, apply_defaults=True):
        super().__init__(view, model, parent, apply_defaults)

    @classmethod
    def from_view(cls, view: FixedLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'FixedLapseRateDownscalerController':
        model = FixedLapseRateDownscaler()
        return cls(view, model)


class LapseRateEstimatorView(PropertyModelView):

    def __init__(self, parent=None, set_layout=True):
        super().__init__(parent)
        self.neighborhood_view = NeighborhoodModelView(self, set_layout=False)
        self.neighborhood_view.settings_changed.connect(self.settings_changed.emit)
        self.range_lapse_rates = RangeSpinner(self, -14., 14., -25., 25.)
        self.range_lapse_rates.min_spinner.setEnabled(False)
        self.range_lapse_rates.max_spinner.setEnabled(False)
        self.range_lapse_rates.range_changed.connect(self.settings_changed.emit)
        self.spinner_lapse_rate_default = QDoubleSpinBox(self)
        self.spinner_lapse_rate_default.setMinimum(-14.)
        self.spinner_lapse_rate_default.setMaximum(100.)
        self.spinner_lapse_rate_default.setValue(-6.5)
        self.spinner_lapse_rate_default.valueChanged.connect(self.settings_changed.emit)
        self.spinner_num_neighbors = QSpinBox(self)
        self.spinner_num_neighbors.setMinimum(4)
        self.spinner_num_neighbors.setMaximum(256)
        self.spinner_num_neighbors.setValue(10)
        self.spinner_num_neighbors.valueChanged.connect(self.settings_changed.emit)
        self.spinner_weight_scale = QDoubleSpinBox(self)
        self.spinner_weight_scale.setMinimum(1.)
        self.spinner_weight_scale.setMaximum(256.)
        self.spinner_weight_scale.setValue(30.)
        self.spinner_weight_scale.setSuffix(' km')
        self.spinner_weight_scale.valueChanged.connect(self.settings_changed.emit)
        self.toggle_weighting = QCheckBox(self)
        self.toggle_weighting.setText('use weighting')
        self.toggle_weighting.stateChanged.connect(self._toggle_weight_scale_spinner)
        self.toggle_weighting.stateChanged.connect(self.settings_changed.emit)
        self.toggle_volume_data = QCheckBox(self)
        self.toggle_volume_data.setText('use volume data')
        self.toggle_volume_data.stateChanged.connect(self.settings_changed.emit)
        self.toggle_intercept = QCheckBox(self)
        self.toggle_intercept.setText('fit intercept')
        self.toggle_intercept.stateChanged.connect(self.settings_changed.emit)

        if set_layout:
            self._set_layout()

    def _set_layout(self):
        self.setLayout(self.build_layout())

    def _toggle_weight_scale_spinner(self):
        self.spinner_weight_scale.setEnabled(self.toggle_weighting.isChecked())

    def _build_estimator_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Estimator settings:'))
        form_layout = QFormLayout()
        form_layout.addRow(QLabel('Default lapse rate:'), self.spinner_lapse_rate_default)
        range_layout = QHBoxLayout()
        range_layout.addWidget(self.range_lapse_rates.min_spinner)
        range_layout.addWidget(self.range_lapse_rates.max_spinner)
        form_layout.addRow(QLabel('Lapse rate cut-off:'), range_layout)
        form_layout.addRow(QLabel('Min. neighborhood size:'), self.spinner_num_neighbors)
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(self.spinner_weight_scale)
        weight_layout.addWidget(self.toggle_weighting)
        form_layout.addRow(QLabel('Weight scale:'), weight_layout)
        form_layout.addRow('Volume interpolation:', self.toggle_volume_data)
        form_layout.addRow('Intercept:', self.toggle_intercept)
        layout.addLayout(form_layout)
        layout.addStretch()
        return layout

    def build_layout(self):
        layout = QVBoxLayout()
        layout.addLayout(self.neighborhood_view.build_layout())
        layout.addLayout(self._build_estimator_layout())
        return layout

    def get_neighborhood_view(self) -> NeighborhoodModelView:
        return self.neighborhood_view

    def get_settings(self) -> LapseRateEstimator.Properties:
        return LapseRateEstimator.Properties(
            self.toggle_volume_data.isChecked(),
            self.toggle_weighting.isChecked(),
            self.spinner_weight_scale.value(),
            self.spinner_num_neighbors.value(),
            self.toggle_intercept.isChecked(),
            self.spinner_lapse_rate_default.value(),
            self.neighborhood_view.get_settings()
        )

    def update_settings(self, settings: LapseRateEstimator.Properties):
        self.toggle_volume_data.setChecked(settings.use_volume)
        self.toggle_weighting.setChecked(settings.use_weights)
        self.spinner_weight_scale.setValue(settings.weight_scale_km)
        self.spinner_num_neighbors.setValue(settings.min_num_neighbors)
        self.toggle_intercept.setChecked(settings.fit_intercept)
        self.spinner_lapse_rate_default.setValue(settings.default_lapse_rate)
        self.neighborhood_view.update_settings(settings.neighborhood)
        return self


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.estimator_view = LapseRateEstimatorView(self, set_layout=False)

    def get_estimator_view(self) -> LapseRateEstimatorView():
        return self.estimator_view


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
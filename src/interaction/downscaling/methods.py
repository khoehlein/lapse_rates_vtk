from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QCheckBox, QSpinBox, QVBoxLayout, QLabel, \
    QHBoxLayout, QFrame, QDialog, QStackedLayout, QDialogButtonBox

from src.interaction.downscaling.neighborhood import NeighborhoodModelView
from src.interaction.interface import PropertyModelView, PropertyModelController
from src.model.data.data_store import GlobalData
from src.model.downscaling.interpolation import InterpolationType
from src.model.downscaling.methods import DownscalingMethodModel, FixedLapseRateDownscaler, LapseRateEstimator, \
    AdaptiveLapseRateDownscaler, DownscalerType, DEFAULTS_ADAPTIVE_LAPSE_RATE, DEFAULTS_FIXED_LAPSE_RATE, \
    DEFAULTS_ADAPTIVE_ESTIMATOR
from src.model.downscaling.neighborhood import NeighborhoodModel
from src.model.downscaling.pipeline import DownscalingPipelineModel
from src.widgets import RangeSpinner


class DownscalerModelView(PropertyModelView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_interpolation_type = QComboBox(self)
        self.combo_interpolation_type.addItem('Nearest neighbor', InterpolationType.NEAREST_NEIGHBOR)
        self.combo_interpolation_type.addItem('Barycentric', InterpolationType.BARYCENTRIC)
        self.combo_interpolation_type.model().item(1).setEnabled(False)
        self.combo_interpolation_type.currentIndexChanged.connect(self.settings_changed.emit)
        self._interpolation_types = {
            InterpolationType.NEAREST_NEIGHBOR: 0,
            InterpolationType.BARYCENTRIC: 1,
        }
        # self.button_apply = QPushButton(self)
        # self.button_apply.setText("Apply")
        # self.button_apply.clicked.connect(self.settings_changed.emit)


class DownscalingMethodController(PropertyModelController):

    @classmethod
    def from_view(cls, view: DownscalerModelView, pipeline: DownscalingPipelineModel) -> 'DownscalingMethodController':
        raise NotImplementedError()

    def __init__(self, view: DownscalerModelView, model: DownscalingMethodModel, parent=None, apply_defaults=True):
        super().__init__(view, model, parent, apply_defaults)


class FixedLapseRateDownscalerView(DownscalerModelView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_lapse_rate = QDoubleSpinBox(self)
        self.spinner_lapse_rate.setMinimum(-14.)
        self.spinner_lapse_rate.setMaximum(100.)
        self.spinner_lapse_rate.setValue(-6.5)
        self.spinner_lapse_rate.setSuffix(' K/km')
        self.spinner_lapse_rate.valueChanged.connect(self.settings_changed.emit)
        self._set_layout()

    def build_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Interpolation:'))
        layout.addWidget(self.combo_interpolation_type)
        layout.addWidget(QLabel('Estimator settings:'))
        form_layout = QFormLayout()
        form_layout.addRow('Lapse rate:', self.spinner_lapse_rate)
        form = QFrame(self)
        form.setFrameStyle(QFrame.Box|QFrame.Sunken)
        form.setLayout(form_layout)
        layout.addWidget(form)
        layout.addStretch()
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

    def set_defaults(self):
        self.update_settings(DEFAULTS_FIXED_LAPSE_RATE)
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
        self.range_lapse_rates.min_spinner.setSuffix(' K/km')
        self.range_lapse_rates.max_spinner.setSuffix(' K/km')
        self.range_lapse_rates.range_changed.connect(self.settings_changed.emit)
        self.spinner_lapse_rate_default = QDoubleSpinBox(self)
        self.spinner_lapse_rate_default.setMinimum(-14.)
        self.spinner_lapse_rate_default.setMaximum(100.)
        self.spinner_lapse_rate_default.setValue(-6.5)
        self.spinner_lapse_rate_default.setSuffix(' K/km')
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
        self._toggle_weight_scale_spinner()
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
        form = QFrame(self)
        form.setFrameStyle(QFrame.Box |QFrame.Sunken)
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
        form.setLayout(form_layout)
        layout.addWidget(form)
        layout.addStretch()
        return layout

    def build_layout(self, stretch=True):
        layout = QVBoxLayout()
        layout.addLayout(self.neighborhood_view.build_layout(stretch=False))
        layout.addLayout(self._build_estimator_layout())
        if stretch:
            layout.addStretch()
        return layout

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

    def set_defaults(self):
        self.neighborhood_view.set_defaults()
        self.update_settings(DEFAULTS_ADAPTIVE_ESTIMATOR)
        return self


class LapseRateEstimatorController(PropertyModelController):

    @classmethod
    def from_view(cls, view: LapseRateEstimatorView, data_store: GlobalData):
        neighborhood = NeighborhoodModel(data_store)
        model = LapseRateEstimator(neighborhood)
        return cls(view, model)

    def __init__(
            self,
            view: LapseRateEstimatorView, model: LapseRateEstimator,
            parent=None, apply_defaults: bool = True
    ):
        super().__init__(view, model, parent, apply_defaults)


class AdaptiveLapseRateDownscalerView(DownscalerModelView):

    def __init__(self, parent=None, set_layout=True):
        super().__init__(parent)
        self.estimator_view = LapseRateEstimatorView(self, set_layout=False)
        if set_layout:
            self._set_layout()

    def _set_layout(self):
        self.setLayout(self.build_layout())

    def build_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Interpolation:'))
        layout.addWidget(self.combo_interpolation_type)
        layout.addLayout(self.estimator_view.build_layout())
        # layout.addWidget(self.button_apply)
        layout.addStretch()
        return layout

    def get_settings(self) -> AdaptiveLapseRateDownscaler.Properties:
        return AdaptiveLapseRateDownscaler.Properties(
            self.combo_interpolation_type.currentData(),
            self.estimator_view.get_settings()
        )

    def update_settings(self, settings: AdaptiveLapseRateDownscaler.Properties):
        self.combo_interpolation_type.setCurrentIndex(self._interpolation_types.get(settings.interpolation))
        self.estimator_view.update_settings(settings.estimator)
        return self

    def set_defaults(self):
        self.estimator_view.set_defaults()
        self.update_settings(DEFAULTS_ADAPTIVE_LAPSE_RATE)
        return self


class AdaptiveLapseRateDownscalerController(DownscalingMethodController):

    @classmethod
    def from_view(cls, view: AdaptiveLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'AdaptiveLapseRateDownscalerController':
        data_store = pipeline.source_domain.data_store
        neighborhood = NeighborhoodModel(data_store)
        estimator = LapseRateEstimator(neighborhood)
        model = AdaptiveLapseRateDownscaler(estimator)
        return cls(view, model)

    def __init__(
            self,
            view: AdaptiveLapseRateDownscalerView, model: AdaptiveLapseRateDownscaler,
            parent=None, apply_defaults=True
    ):
        super().__init__(view, model, parent, apply_defaults)


class CreateDownscalerDialog(QDialog):

    settings_accepted = pyqtSignal()

    LABELS = {
        DownscalerType.FIXED_LAPSE_RATE: 'Fixed lapse rate',
        DownscalerType.ADAPTIVE_LAPSE_RATE: 'Adaptive lapse rate',
        DownscalerType.NETWORK: 'Network',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Downscaler properties')
        self.combo_downscaler_type = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self._stack_interface(DownscalerType.FIXED_LAPSE_RATE, FixedLapseRateDownscalerView(self))
        self._stack_interface(DownscalerType.ADAPTIVE_LAPSE_RATE, AdaptiveLapseRateDownscalerView(self))
        self.combo_downscaler_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.RestoreDefaults | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.clicked.connect(self._on_button_clicked)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_downscaler_type)
        layout.addLayout(self.interface_stack)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def _stack_interface(self, downscaler_type: DownscalerType, interface: DownscalerModelView):
        self.combo_downscaler_type.addItem(self.LABELS[downscaler_type], downscaler_type)
        self.interface_stack.addWidget(interface)

    def get_settings(self) -> DownscalingMethodModel.Properties:
        return self.interface_stack.currentWidget().get_settings()

    def update_settings(self, settings: DownscalingMethodModel.Properties):
        type_ = type(settings)
        downscaler_type = {
            AdaptiveLapseRateDownscaler.Properties: 1,
            FixedLapseRateDownscaler.Properties: 0,
        }.get(type_)
        assert downscaler_type is not None
        self.combo_downscaler_type.setCurrentIndex(downscaler_type)
        self.interface_stack.currentWidget().update_settings(settings)
        return self

    def _on_button_clicked(self, button):
        if self.button_box.buttonRole(button) == QDialogButtonBox.ResetRole:
            self._on_defaults_requested()

    def _on_defaults_requested(self):
        self.interface_stack.widget(0).set_defaults()
        self.interface_stack.widget(1).set_defaults()
        self.combo_downscaler_type.setCurrentIndex(0)

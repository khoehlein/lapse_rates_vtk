from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QLabel, QFormLayout, QHBoxLayout, QSpinBox, QComboBox, \
    QStackedLayout, QPushButton, QVBoxLayout, QStackedWidget

from src.model.downscaling import LapseRateDownscalerProperties
from src.model.neighborhood_lookup.knn_lookup import KNNNeighborhoodProperties
from src.model.neighborhood_lookup.radial_lookup import RadialNeighborhoodProperties
from src.widgets import FileSelectionWidget


class NeighborhoodMethodView(QWidget):

    neighborhood_settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

    def _set_defaults(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def get_settings(self):
        raise NotImplementedError()


class RadialNeighborhoodHandles(NeighborhoodMethodView):

    def __init__(self, parent=None, config: Dict[str, Any] = None):
        super().__init__(parent)
        self.spinner_radius = QDoubleSpinBox(self)
        self.spinner_radius.setSuffix(' km')
        self.spinner_threshold = QDoubleSpinBox(self)
        self.spinner_threshold.setMinimum(0.)
        self.spinner_threshold.setMaximum(1.)
        self.spinner_threshold.setSingleStep(0.05)
        self._set_defaults(config)
        self._set_layout()

    def _set_defaults(self, config: Dict[str, Any]) -> None:
        config = config or {}
        self.spinner_radius.setValue(config.get('lookup_radius', 30.))
        self.spinner_radius.setMinimum(config.get('lookup_radius_min', 10.))
        self.spinner_radius.setMaximum(config.get('lookup_radius_max', 180.))
        self.spinner_threshold.setValue(config.get('lsm_threshold', 0.))

    def _set_layout(self):
        layout = QFormLayout(self)
        layout.addRow(QLabel('Radius:'), self.spinner_radius)
        layout.addRow(QLabel('Land/sea threshold:'), self.spinner_threshold)
        self.setLayout(layout)

    def get_settings(self):
        return RadialNeighborhoodProperties(self.spinner_radius.value(), self.spinner_threshold.value())


class KNNNeighborhoodHandles(NeighborhoodMethodView):

    def __init__(self, parent=None, config: Dict[str, Any] = None):
        super().__init__(parent)
        self.spinner_neighborhood_size = QSpinBox(self)
        self.spinner_threshold = QDoubleSpinBox(self)
        self.spinner_threshold.setMinimum(0.)
        self.spinner_threshold.setMaximum(1.)
        self.spinner_threshold.setSingleStep(0.05)
        self._set_defaults(config)
        self._set_layout()

    def _set_defaults(self, config: Dict[str, Any]) -> None:
        config = config or {}
        self.spinner_neighborhood_size.setValue(config.get('neighborhood_size', 32))
        self.spinner_neighborhood_size.setMinimum(config.get('neighborhood_size_min', 8))
        self.spinner_neighborhood_size.setMaximum(config.get('neighborhood_size_max', 256))
        self.spinner_threshold.setValue(config.get('lsm_threshold', 0.))

    def _set_layout(self):
        layout = QFormLayout(self)
        layout.addRow(QLabel('Neighborhood size:'), self.spinner_neighborhood_size)
        layout.addRow(QLabel('Land/sea threshold:'), self.spinner_threshold)
        self.setLayout(layout)

    def get_settings(self):
        return KNNNeighborhoodProperties(self.spinner_neighborhood_size.value(), self.spinner_threshold.value())


class NeighborhoodLookupView(QWidget):

    neighborhood_changed = pyqtSignal()

    def __init__(self, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self.combo_lookup_type = QComboBox()
        self.interface_stack = QStackedLayout()
        self.combo_lookup_type.addItem('Radius')
        self.radial_interface = RadialNeighborhoodHandles(config=config, parent=self)
        self.interface_stack.addWidget(self.radial_interface)
        self.combo_lookup_type.addItem('Nearest neighbors')
        self.knn_interface = KNNNeighborhoodHandles(config=config, parent=self)
        self.interface_stack.addWidget(self.knn_interface)
        self.combo_lookup_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.button_apply = QPushButton('Apply')
        self.button_apply.clicked.connect(self._on_button_apply)
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Neighborhood method:'))
        layout.addWidget(self.combo_lookup_type)
        layout.addLayout(self.interface_stack)
        layout.addWidget(self.button_apply)
        layout.addStretch()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def _on_button_apply(self):
        self.neighborhood_changed.emit()

    def get_neighborhood_properties(self):
        return self.interface_stack.currentWidget().get_settings()


class DownscalerMethodView(QWidget):

    method_settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def get_settings(self):
        raise NotImplementedError()


class LapseRateDownscalerView(DownscalerMethodView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spinner_weight_scale = QDoubleSpinBox(self)
        self.spinner_weight_scale.setMinimum(1.)
        self.spinner_weight_scale.setMaximum(180.)
        self.spinner_weight_scale.setValue(30.)
        self.spinner_weight_scale.setSuffix(' km')
        self.toggle_weighting = QCheckBox()
        self.toggle_volume_data = QCheckBox()
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow('Use volume data:', self.toggle_volume_data)
        layout.addRow(QLabel('Use distance weighting:'), self.toggle_weighting)
        layout.addRow('Weight scale:', self.spinner_weight_scale)
        self.setLayout(layout)

    def get_settings(self):
        return LapseRateDownscalerProperties(
            self.toggle_volume_data.isChecked(),
            self.toggle_weighting.isChecked(),
            self.spinner_weight_scale.value()
        )


class NetworkDownscalerView(DownscalerMethodView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_selection = FileSelectionWidget(self)
        self.file_selection.file_selection_changed.connect(self.method_settings_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel('Checkpoint file:'))
        row_layout.addWidget(self.file_selection.line_edit)
        row_layout.addWidget(self.file_selection.button_select)
        layout.addRow(row_layout)
        self.setLayout(layout)


class DownscalingSettingsView(QWidget):

    method_changed = pyqtSignal()
    method_settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_downscaler_method = QComboBox()
        self.interface_stack = QStackedLayout()
        self.combo_downscaler_method.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_downscaler_method.currentIndexChanged.connect(self.method_changed.emit)
        self._set_method_views()
        self._set_layout()

    def _set_method_views(self):
        self.lapse_rate_settings = LapseRateDownscalerView(self)
        self._register_method_view('Linear', self.lapse_rate_settings)
        self.network_settings = NetworkDownscalerView(self)
        self._register_method_view('Network', self.network_settings)

    def _register_method_view(self, label: str, view_widget: DownscalerMethodView):
        self.combo_downscaler_method.addItem(label)
        self.interface_stack.addWidget(view_widget)
        view_widget.method_settings_changed.connect(self.method_settings_changed.emit)

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Downscaling method:'))
        layout.addWidget(self.combo_downscaler_method)
        layout.addLayout(self.interface_stack)
        self.setLayout(layout)

    def get_downscaler_properties(self):
        return self.interface_stack.currentWidget().get_settings()

from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QCheckBox, QLabel, QFormLayout, QHBoxLayout, QSpinBox, QComboBox, \
    QStackedLayout, QPushButton, QVBoxLayout

from src.model._legacy.downscaling import LapseRateDownscalerProperties
from src.model._legacy.knn_lookup import KNNNeighborhoodProperties
from src.model._legacy.radial_lookup import RadialNeighborhoodProperties
from src.widgets import FileSelectionWidget


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
            self.spinner_weight_scale.value(),
            10, True, -0.0065
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
        self.button_apply = QPushButton('Apply')
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
        layout.addStretch()
        layout.addWidget(self.button_apply)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def get_downscaler_properties(self):
        return self.interface_stack.currentWidget().get_settings()

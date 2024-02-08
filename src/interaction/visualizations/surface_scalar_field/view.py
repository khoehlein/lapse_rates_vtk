import uuid

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QTabWidget, QCheckBox, QVBoxLayout
from src.interaction.visualizations.color_settings_view import MultiScalarColorSettingsView
from src.interaction.visualizations.geometry_settings_view import RepresentationSettingsView, LightingSettingsView
from src.interaction.visualizations.interface import VisualizationSettingsView
from src.model.visualization.interface import DataConfiguration
from src.model.visualization.mesh_geometry import MeshGeometryModel


class SurfaceScalarFieldSettingsView(VisualizationSettingsView):

    geometry_changed = pyqtSignal()
    color_changed = pyqtSignal()

    def __init__(self, key: str = None, parent=None):
        super().__init__(key, parent)

        self.combo_source_data = QComboBox(self)
        for config_type in DataConfiguration:
            self.combo_source_data.addItem(config_type.value, config_type)

        self.combo_source_data.currentTextChanged.connect(self._on_source_data_changed)

        self.tabs = QTabWidget(self)
        self._build_color_tab()
        self._build_lighting_tab()

        self.checkbox_visibility = QCheckBox('Visible')
        self.checkbox_visibility.setChecked(True)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self._set_layout()

    def _toggle_scalar_types(self):
        self.color_settings.blockSignals(True)
        selected_source = self.combo_source_data.currentData()
        self.color_settings.toggle_scalars(selected_source)
        self.color_settings.blockSignals(False)

    def _on_source_data_changed(self, source_type: str) -> None:
        self._toggle_scalar_types()
        self.source_data_changed.emit(self.key)

    def _build_color_tab(self):
        self.color_settings = MultiScalarColorSettingsView(self)
        self.color_settings.color_changed.connect(self.color_changed)
        self._toggle_scalar_types()
        self.representation_settings = RepresentationSettingsView(parent=self)
        self.representation_settings.representation_changed.connect(self.geometry_changed.emit)
        color_stack_widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addLayout(self.color_settings.vbox_layout)
        layout.addLayout(self.representation_settings.vbox_layout)
        layout.addStretch()
        color_stack_widget.setLayout(layout)
        self.tabs.addTab(color_stack_widget, 'Color')

    def _build_lighting_tab(self):
        self.lighting_settings = LightingSettingsView(self)
        self.lighting_settings.lighting_changed.connect(self.geometry_changed.emit)
        self.tabs.addTab(self._to_tab_widget(self.lighting_settings), 'Lighting')

    def _to_tab_widget(self, x):
        widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(x)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.combo_source_data)
        layout.addWidget(self.tabs)
        layout.addWidget(self.checkbox_visibility)
        layout.addStretch()
        self.setLayout(layout)

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()

    def get_properties_summary(self):
        return {
            'geometry': self.get_geometry_properties(),
            'coloring': self.get_color_properties(),
            'source_data': self.get_source_properties(),
        }

    def get_geometry_properties(self):
        rep_settings = self.representation_settings.get_settings()
        lighting_settings = self.lighting_settings.get_settings()
        prop = MeshGeometryModel.Properties(mesh=rep_settings, lighting=lighting_settings)
        return prop

    def get_color_properties(self):
        return self.color_settings.get_settings()

    def get_source_properties(self) -> DataConfiguration:
        return self.combo_source_data.currentData()

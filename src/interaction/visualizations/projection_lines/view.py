from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QTabWidget, QCheckBox, QVBoxLayout, QStackedLayout, QPushButton
from src.interaction.visualizations.color_settings_view import UniformColorSettingsView, cmap_defaults, \
    ColormapSettingsView, MultiScalarColorSettingsView
from src.interaction.visualizations.geometry_settings_view import RepresentationSettingsView, LightingSettingsView, \
    WireframeSettingsView
from src.interaction.visualizations.interface import VisualizationSettingsView
from src.model.visualization.interface import DataConfiguration, ScalarType, available_scalars
from src.model.visualization.mesh_geometry import MeshGeometryModel


class ProjectionLinesSettingsView(VisualizationSettingsView):

    geometry_changed = pyqtSignal()
    color_changed = pyqtSignal()

    def __init__(self, key: str = None, parent=None):
        super().__init__(key, parent)

        self.tabs = QTabWidget(self)
        self._build_color_tab()
        self._build_lighting_tab()

        self.checkbox_visibility = QCheckBox('Visible')
        self.checkbox_visibility.setChecked(True)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self._set_layout()

    def _toggle_scalar_types(self):
        model = self.combo_scalar_type.model()
        selected_source = self.combo_source_data.currentData()
        for i in range(model.rowCount()):
            item = model.item(i)
            item_data = self.combo_scalar_type.itemData(i)
            item.setEnabled(item_data in available_scalars[selected_source])

    def _on_source_data_changed(self, source_type: str) -> None:
        self._toggle_scalar_types()
        self.source_data_changed.emit(self.key)

    def _build_color_tab(self):
        self.color_settings = MultiScalarColorSettingsView(self)
        self.color_settings.toggle_scalars(DataConfiguration.SURFACE_O8000)
        self.color_settings.color_changed.connect(self.color_changed.emit)
        self.representation_settings = WireframeSettingsView(parent=self)
        self.representation_settings.representation_changed.connect(self.geometry_changed.emit)
        color_stack_widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addLayout(self.color_settings.vbox_layout)
        layout.addWidget(self.representation_settings)
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
        }

    def get_geometry_properties(self):
        rep_settings = self.representation_settings.get_settings()
        lighting_settings = self.lighting_settings.get_settings()
        prop = MeshGeometryModel.Properties(mesh=rep_settings, lighting=lighting_settings)
        return prop

    def get_color_properties(self):
        return self.color_settings.get_settings()

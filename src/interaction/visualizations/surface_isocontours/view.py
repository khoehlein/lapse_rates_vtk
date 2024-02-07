import uuid

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QTabWidget, QCheckBox, QVBoxLayout, QStackedLayout, QPushButton, \
    QSpinBox, QFormLayout
from src.interaction.visualizations.color_settings_view import UniformColorSettingsView, cmap_defaults, ColormapSettingsView
from src.interaction.visualizations.geometry_settings_view import RepresentationSettingsView, LightingSettingsView
from src.interaction.visualizations.interface import VisualizationSettingsView
from src.model.visualization.interface import DataConfiguration, ScalarType, available_scalars
from src.model.visualization.mesh_geometry import MeshGeometryModel
from src.model.visualization.surface_isocontours import ContourProperties


class ContourSettingsView(QWidget):

    contours_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.combo_contour_scalar = QComboBox(self)
        for scalar_type in ScalarType:
            if scalar_type != ScalarType.GEOMETRY:
                self.combo_contour_scalar.addItem(scalar_type.value, scalar_type)
        self.combo_contour_scalar.currentTextChanged.connect(self.contours_changed)

        self.spinner_isolevels = QSpinBox(self)
        self.spinner_isolevels.setMinimum(1)
        self.spinner_isolevels.setValue(10)
        self.spinner_isolevels.valueChanged.connect(self.contours_changed)

        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow("Contour scalar:", self.combo_contour_scalar)
        layout.addRow("Isolevels:", self.spinner_isolevels)
        self.setLayout(layout)

    def get_settings(self):
        return ContourProperties(
            self.combo_contour_scalar.currentData(),
            self.spinner_isolevels.value()
        )


class SurfaceIsocontoursSettingsView(VisualizationSettingsView):

    geometry_changed = pyqtSignal()
    color_changed = pyqtSignal()

    def __init__(self, key: str = None, parent=None):
        super().__init__(key, parent)

        self.combo_source_data = QComboBox(self)
        for config_type in DataConfiguration:
            self.combo_source_data.addItem(config_type.value, config_type)

        self.contour_settings = ContourSettingsView(self)
        self.contour_settings.contours_changed.connect(self._on_contours_changed)

        self.combo_color_scalar = QComboBox(self)
        for scalar_type in ScalarType:
            self.combo_color_scalar.addItem(scalar_type.value, scalar_type)


        self._toggle_scalar_types()
        self.combo_source_data.currentTextChanged.connect(self._on_source_data_changed)
        self.combo_color_scalar.currentTextChanged.connect(self.color_changed.emit)

        self.tabs = QTabWidget(self)
        self._build_contours_tab()
        self._build_color_tab()
        self._build_lighting_tab()

        self.checkbox_visibility = QCheckBox('Visible')
        self.checkbox_visibility.setChecked(True)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self._set_layout()

    def _toggle_combo_entries(self, combo):
        model = combo.model()
        selected_source = self.combo_source_data.currentData()
        for i in range(model.rowCount()):
            item = model.item(i)
            item_data = combo.itemData(i)
            item.setEnabled(item_data in available_scalars[selected_source])

    def _toggle_scalar_types(self):
        self.contour_settings.blockSignals(True)
        self._toggle_combo_entries(self.contour_settings.combo_contour_scalar)
        self.contour_settings.blockSignals(False)
        self._toggle_combo_entries(self.combo_color_scalar)

    def _on_source_data_changed(self) -> None:
        self._toggle_scalar_types()
        self.source_data_changed.emit(self.key)

    def _on_contours_changed(self):
        self.source_data_changed.emit(self.key)

    def _build_contours_tab(self):
        self.tabs.addTab(self._to_tab_widget(self.contour_settings), 'Isocontours')

    def _build_color_tab(self):
        self.color_settings_stack = QStackedLayout()
        for scalar_type in ScalarType:
            widget = UniformColorSettingsView(self) if scalar_type == ScalarType.GEOMETRY else ColormapSettingsView(self)
            self.color_settings_stack.addWidget(widget)
            widget.set_defaults(cmap_defaults[scalar_type])
            widget.color_changed.connect(self.color_changed.emit)
        self.combo_color_scalar.currentIndexChanged.connect(self.color_settings_stack.setCurrentIndex)
        self.representation_settings = RepresentationSettingsView(enable_surface=False, parent=self)
        self.representation_settings.representation_changed.connect(self.geometry_changed.emit)
        color_stack_widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(self.combo_color_scalar)
        layout.addLayout(self.color_settings_stack)
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
            'contours': self.get_contour_properties(),
            'coloring': self.get_color_properties(),
            'source_data': self.get_source_properties(),
        }

    def get_geometry_properties(self):
        rep_settings = self.representation_settings.get_settings()
        lighting_settings = self.lighting_settings.get_settings()
        prop = MeshGeometryModel.Properties(mesh=rep_settings, lighting=lighting_settings)
        return prop

    def get_contour_properties(self):
        return self.contour_settings.get_settings()

    def get_color_properties(self):
        scalar_type = self.combo_color_scalar.currentData()
        scalar_name = scalar_type.name.lower()
        return self.color_settings_stack.currentWidget().get_settings(scalar_name)

    def get_source_properties(self) -> DataConfiguration:
        return self.combo_source_data.currentData()

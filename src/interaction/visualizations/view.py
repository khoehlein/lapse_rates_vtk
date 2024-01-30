import logging
from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTabWidget, QWidget, QComboBox, QStackedLayout, QVBoxLayout, QLabel, QDoubleSpinBox, \
    QFormLayout, QCheckBox

from src.interaction.background_color.view import SelectColorButton, ColorSelectionMenu
from src.model.visualization.visualizations import WireframeSurface, TranslucentSurface


class SurfaceVisMethodView(QWidget):

    properties_changed = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

    def _set_defaults(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def get_settings(self):
        raise NotImplementedError()


class WireframeSurfaceHandles(SurfaceVisMethodView):

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.spinner_line_width = QDoubleSpinBox(self)
        self.spinner_line_width.setMinimum(0.25)
        self.spinner_line_width.setMaximum(10.)
        self.spinner_line_width.setValue(1.)
        self.spinner_line_width.setSingleStep(0.25)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setValue(1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.menu_color_selection = ColorSelectionMenu(self)
        self.spinner_line_width.valueChanged.connect(self.properties_changed)
        self.spinner_opacity.valueChanged.connect(self.properties_changed)
        self.menu_color_selection.colormap_changed.connect(self.properties_changed)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Color:'), self.menu_color_selection)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        layout.addRow(QLabel('Line width:'), self.spinner_line_width)
        self.setLayout(layout)

    def get_settings(self):
        return WireframeSurface.Properties(
            self.menu_color_selection.get_colormap(),
            self.spinner_line_width.value(),
            self.spinner_opacity.value(),
        )


class TranslucentSurfaceHandles(SurfaceVisMethodView):

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setMinimum(0.)
        self.spinner_opacity.setMaximum(1.)
        self.spinner_opacity.setValue(0.5)
        self.spinner_opacity.setSingleStep(0.05)
        self.menu_color_selection = ColorSelectionMenu(self)
        self.checkbox_show_edges = QCheckBox()
        self.spinner_opacity.valueChanged.connect(self.properties_changed)
        self.checkbox_show_edges.stateChanged.connect(self.properties_changed)
        self.menu_color_selection.colormap_changed.connect(self.properties_changed)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Color:'), self.menu_color_selection)
        layout.addRow(QLabel('Opacity:'), self.spinner_opacity)
        layout.addRow(QLabel('Show edges:'), self.checkbox_show_edges)
        self.setLayout(layout)

    def get_settings(self):
        logging.info('Reading translucent surface properties')
        return TranslucentSurface.Properties(
            self.menu_color_selection.get_colormap(),
            self.spinner_opacity.value(),
            self.checkbox_show_edges.isChecked()
        )


class SurfaceVisSettingsView(QWidget):

    vis_method_changed = pyqtSignal()
    vis_properties_changed = pyqtSignal()
    visibility_changed = pyqtSignal()

    def __init__(self, parent=None, visible=True):
        super().__init__(parent)
        self.interfaces = {}
        self.combo_vis_type = QComboBox()
        self.interface_stack = QStackedLayout()
        self.register_vis_method('wireframe', WireframeSurfaceHandles(parent=self), 'Wireframe')
        self.register_vis_method('surface', TranslucentSurfaceHandles(parent=self), 'Surface')
        self.combo_vis_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_vis_type.currentIndexChanged.connect(self.vis_method_changed.emit)
        self.checkbox_visibility = QCheckBox()
        self.checkbox_visibility.setChecked(visible)
        self.checkbox_visibility.stateChanged.connect(self.visibility_changed.emit)
        self.checkbox_visibility.setText('Visible')
        self._set_layout()

    def register_vis_method(self, key: str, widget: SurfaceVisMethodView, combo_label: str):
        assert key not in self.interfaces
        self.interfaces[key] = widget
        self.combo_vis_type.addItem(combo_label)
        self.interface_stack.addWidget(widget)
        widget.properties_changed.connect(self.vis_properties_changed.emit)

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Visualization type:'))
        layout.addWidget(self.combo_vis_type)
        layout.addLayout(self.interface_stack)
        layout.addWidget(self.checkbox_visibility)
        layout.addStretch()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def get_vis_properties(self):
        return self.interface_stack.currentWidget().get_settings()

    def get_visibility(self):
        return self.checkbox_visibility.isChecked()


class SceneSettingsView(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vis_settings: Dict[str, SurfaceVisSettingsView] = {}
        self.register_settings_view('surface_o1280', SurfaceVisSettingsView(self, visible=False), 'O1280')
        self.register_settings_view('surface_o8000', SurfaceVisSettingsView(self), 'O8000')

    def keys(self):
        return list(self.vis_settings.keys())

    def register_settings_view(self, key: str, settings_view: SurfaceVisSettingsView, tab_label: str):
        assert key not in self.vis_settings
        self.vis_settings[key] = settings_view
        self.addTab(self._to_tab_widget(settings_view), tab_label)

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper

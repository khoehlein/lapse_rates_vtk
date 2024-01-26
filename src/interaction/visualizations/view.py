from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTabWidget, QWidget, QComboBox, QStackedLayout, QVBoxLayout, QLabel, QDoubleSpinBox, \
    QFormLayout

from src.interaction.background_color.view import SelectColorButton
from src.model.visualization.visualizations import WireframeSurface


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
        self.button_line_color = SelectColorButton(QColor(0, 0, 0), self)
        self.button_line_color.setText(' Select line color')
        self.spinner_line_width.valueChanged.connect(self.properties_changed)
        self.button_line_color.color_changed.connect(self.properties_changed)
        self._set_layout()

    def _set_layout(self):
        layout = QFormLayout()
        layout.addRow(QLabel('Line width:'), self.spinner_line_width)
        layout.addRow(QLabel('Line color:'), self.button_line_color)
        self.setLayout(layout)

    def get_settings(self):
        return WireframeSurface.Properties(
            self.spinner_line_width.value(),
            self.button_line_color.current_color
        )


class SurfaceVisSettingsView(QWidget):

    vis_method_changed = pyqtSignal()
    vis_properties_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_vis_type = QComboBox()
        self.interface_stack = QStackedLayout()
        self.combo_vis_type.addItem('Wireframe')
        self.wireframe_interface = WireframeSurfaceHandles(parent=self)
        self.interface_stack.addWidget(self.wireframe_interface
                                       )
        self.combo_vis_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_vis_type.currentIndexChanged.connect(self.vis_method_changed.emit)
        self.wireframe_interface.properties_changed.connect(self.vis_properties_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Visualization type:'))
        layout.addWidget(self.combo_vis_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def get_vis_properties(self):
        return self.interface_stack.currentWidget().get_settings()


class VisualizationSettingsView(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.surface_o1280_settings = SurfaceVisSettingsView(self)
        self.surface_o8000_settings = SurfaceVisSettingsView(self)
        self.addTab(self._to_tab_widget(self.surface_o1280_settings), 'O1280')
        self.addTab(self._to_tab_widget(self.surface_o8000_settings), 'O8000')

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper
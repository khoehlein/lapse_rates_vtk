from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDockWidget, QLabel, QScrollArea, QWidget, QVBoxLayout, QTabWidget, QFormLayout, \
    QGridLayout, QSlider

from src.interaction.background_color.view import SelectColorButton
from src.interaction.domain_selection.view import DomainSelectionView
from src.model.world_data import NeighborhoodLookupView
from src.widgets import LogDoubleSliderSpinner


class SettingsView(QDockWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Properties')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        scroll.setWidget(content)
        vlayout = QVBoxLayout(content)
        vlayout.addWidget(QLabel('Select region:'))
        self.select_region = DomainSelectionView(content)
        vlayout.addWidget(self.select_region)
        vlayout.addWidget(QLabel('Vertical scale:'))
        self.select_z_scale = LogDoubleSliderSpinner(100., 1000., 128, parent=content)
        vlayout.addWidget(self.select_z_scale)
        vlayout.addWidget(QLabel('Background color:'))
        self.select_background_color = SelectColorButton(QColor(0, 0, 0))
        vlayout.addWidget(self.select_background_color)
        self.select_neighborhood_lookup = NeighborhoodLookupView(parent=self)
        vlayout.addWidget(self.select_neighborhood_lookup)
        vlayout.addStretch()
        content.setLayout(vlayout)
        self.setWidget(scroll)

    @property
    def request_new_region(self):
        return self.select_region.button_apply.clicked

    @property
    def request_scale_change(self):
        return self.select_z_scale.slider.valueChanged


class ViewSettingsView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider_z_scale = QSlider(Qt.Horizontal)
        self.button_background_color = SelectColorButton(QColor(0, 0, 0))
        layout = QFormLayout(self)
        layout.addRow(QLabel('Vertical scale'), self.slider_z_scale)
        layout.addRow(QLabel('Background color'), self.button_background_color)
        self.setLayout(layout)


class SettingsViewTabbed(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Properties')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._build_scroll_area()
        self._populate_scroll_area()

    def _build_scroll_area(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_contents = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_area_contents)
        self.setWidget(self.scroll_area)

    def _populate_scroll_area(self):
        self.tab_widget = QTabWidget(self.scroll_area_contents)
        self.domain_selection = DomainSelectionView(self.tab_widget)
        self.neighborhood_settings = NeighborhoodLookupView(parent=self.tab_widget)
        self.view_settings = ViewSettingsView(self.tab_widget)
        self.tab_widget.addTab(self._to_tab_widget(self.domain_selection), 'Domain')
        self.tab_widget.addTab(self._to_tab_widget(self.neighborhood_settings), 'Neighborhood')
        self.tab_widget.addTab(self._to_tab_widget(self.view_settings), 'View')

        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(self.tab_widget)
        layout.addStretch()
        self.scroll_area_contents.setLayout(layout)

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self.tab_widget)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper




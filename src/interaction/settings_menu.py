from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QTabWidget, QLabel

from src.interaction.domain_selection.view import DomainSelectionView
from src.interaction.model_selection.view import DownscalingSettingsView, NeighborhoodLookupView
from src.interaction.view_settings.view import ViewSettingsView
from src.interaction.visualizations.view import SceneSettingsView


class GeneralSettingsView(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.domain_settings = DomainSelectionView(self)
        self.neighborhood_settings = NeighborhoodLookupView(parent=self)
        self.downscaler_settings = DownscalingSettingsView(parent=self)
        self.view_settings = ViewSettingsView(self)
        self.addTab(self._to_tab_widget(self.domain_settings), 'Domain')
        self.addTab(self._to_tab_widget(self.neighborhood_settings), 'Neighborhood')
        self.addTab(self._to_tab_widget(self.downscaler_settings), 'Downscaling')
        self.addTab(self._to_tab_widget(self.view_settings), 'View')

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper


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
        self.general_settings = GeneralSettingsView(self.scroll_area_contents)
        self.visualization_settings = SceneSettingsView(self.scroll_area_contents)
        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(QLabel('General settings'))
        layout.addWidget(self.general_settings)
        layout.addWidget(QLabel('Visualization settings'))
        layout.addWidget(self.visualization_settings)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)



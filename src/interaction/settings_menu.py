from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QVBoxLayout, QTabWidget

from src.interaction.domain_selection.view import DomainSelectionView
from src.interaction.model_selection.view import DownscalingSettingsView, NeighborhoodLookupView
from src.interaction.view_settings.view import ViewSettingsView


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
        self.domain_settings = DomainSelectionView(self.tab_widget)
        self.neighborhood_settings = NeighborhoodLookupView(parent=self.tab_widget)
        self.downscaler_settings = DownscalingSettingsView(parent=self.tab_widget)
        self.view_settings = ViewSettingsView(self.tab_widget)
        self.tab_widget.addTab(self._to_tab_widget(self.domain_settings), 'Domain')
        self.tab_widget.addTab(self._to_tab_widget(self.neighborhood_settings), 'Neighborhood')
        self.tab_widget.addTab(self._to_tab_widget(self.downscaler_settings), 'Downscaling')
        self.tab_widget.addTab(self._to_tab_widget(self.view_settings), 'View')

        layout = QVBoxLayout(self.scroll_area_contents)
        layout.addWidget(self.tab_widget)
        layout.addStretch(2)
        self.scroll_area_contents.setLayout(layout)

    def _to_tab_widget(self, widget: QWidget):
        wrapper = QWidget(self.tab_widget)
        layout = QVBoxLayout(wrapper)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignTop)
        layout.addStretch()
        wrapper.setLayout(layout)
        return wrapper

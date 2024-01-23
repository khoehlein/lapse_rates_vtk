from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDockWidget, QLabel

from src.interaction.background_color.view import SelectColorMenu
from src.interaction.domain_selection.view import DomainSelectionView
from src.model.world_data import NeighborhoodLookupView
from src.widgets import LogDoubleSliderSpinner


class SettingsView(QDockWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Properties')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        vlayout = QtWidgets.QVBoxLayout(content)
        vlayout.addWidget(QLabel('Select region:'))
        self.select_region = DomainSelectionView(content)
        vlayout.addWidget(self.select_region)
        vlayout.addWidget(QLabel('Vertical scale:'))
        self.select_z_scale = LogDoubleSliderSpinner(100., 1000., 128, parent=content)
        vlayout.addWidget(self.select_z_scale)
        vlayout.addWidget(QLabel('Background color:'))
        self.select_background_color = SelectColorMenu(QColor(0, 0, 0))
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
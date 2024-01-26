import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QLabel, QVBoxLayout

from src.model.geometry import DomainBounds
from src.widgets import RangeSpinner


class DomainSelectionView(QWidget):

    domain_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.select_latitude = RangeSpinner(self, 45., 50., 0., 90.)
        self.select_longitude = RangeSpinner(self, 15., 20., -180., 180.)
        self.button_apply = QPushButton('Apply', self)
        self.button_apply.clicked.connect(self.domain_changed.emit)
        self._set_layout()

    def _set_layout(self):
        layout = QGridLayout()
        layout.addWidget(QLabel('Latitude:', self), 0, 0)
        layout.addWidget(self.select_latitude.min_spinner, 0, 1, 1, 2)
        layout.addWidget(self.select_latitude.max_spinner, 0, 3, 1, 2)
        layout.addWidget(QLabel('Longitude:', self), 1, 0)
        layout.addWidget(self.select_longitude.min_spinner, 1, 1, 1, 2)
        layout.addWidget(self.select_longitude.max_spinner, 1, 3, 1, 2)
        outer = QVBoxLayout()
        outer.addWidget(QLabel('Domain boundaries:'))
        outer.addLayout(layout)
        outer.addStretch(2)
        outer.addWidget(self.button_apply)
        self.setLayout(outer)

    def get_domain_boundaries(self):
        return DomainBounds(
            self.select_latitude.min_spinner.value(),
            self.select_latitude.max_spinner.value(),
            self.select_longitude.min_spinner.value(),
            self.select_longitude.max_spinner.value()
        )

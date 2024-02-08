from PyQt5.QtWidgets import QWidget, QFrame, QVBoxLayout
from pyvistaqt import QtInteractor

from src.model.solar_lighting_model import SolarLightingModel


class PyvistaView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = QFrame(parent=self)
        vlayout = QVBoxLayout()
        self.plotter = QtInteractor(self.frame, )
        # self.plotter.enable_eye_dome_lighting()
        vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(vlayout)

from PyQt5.QtWidgets import QWidget, QFrame, QVBoxLayout

import pyvista as pv
from pyvistaqt import QtInteractor


class PyvistaView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = QFrame(parent=self)
        vlayout = QVBoxLayout()
        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        self.plotter.enable_lightkit()
        vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(vlayout)
        self.vertical_scale = 1.
        self.poly_data = []

    def add_mesh(self, mesh: pv.PolyData):
        self.plotter.add_mesh(mesh)

    # def rescale_z(self, z_scale: float):
    #     self.plotter.suppress_rendering = True
    #     for ref in self.poly_data:
    #         ref.rescale_z(z_scale)
    #     self.plotter.update_bounds_axes()
    #     self.plotter.suppress_rendering = False
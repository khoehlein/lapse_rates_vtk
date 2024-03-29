import sys

# Setting the Qt bindings for QtPy
import os

from src.model.geometry import Coordinates, TriangleMesh, LocationBatch, WedgeMesh
from src.paper.volume_visualization.plotter_slot import ContourParameters
from src.paper.volume_visualization.scaling import ScalingParameters
from src.paper.volume_visualization.volume_data import VolumeData

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow

import xarray as xr


model_data = xr.open_dataset('C:\\Users\\kevin\\data\\ECMWF\\Vis\\detailed_alps_summer\\model_data_o1280.nc')
terrain_data = xr.open_dataset('C:\\Users\\kevin\\data\\ECMWF\\Vis\\detailed_alps_summer\\terrain_data_o1280.nc')

vol_data = VolumeData(model_data,  terrain_data, scalar_key='t')
contours = vol_data.get_contour_mesh(ContourParameters('longitude_3d', 10), ScalingParameters(4000, 1, False, False, False))
grid = vol_data.get_volume_mesh(ScalingParameters(4000, 1, False, False, False))

class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        volume = self.plotter.add_volume(
            grid, opacity=1,
            clim=(-20, 40), cmap='hsv'
        )
        self.plotter.add_volume_clip_plane(
            volume,
            interaction_event='always',
            normal_rotation=False,
        )
        # self.plotter.remove_actor(volume)
        # self.plotter.add_mesh(
        #     contours, clim=(-20, 40), cmap='hsv', lighting=False
        # )
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)



        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())
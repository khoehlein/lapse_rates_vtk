import json
import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QPushButton, QGridLayout, QDoubleSpinBox, QWidget, QHBoxLayout, QLabel, \
    QVBoxLayout, QSlider

from src.widgets import CollapsibleBox, LogSlider

logging.basicConfig(level=logging.INFO)

# Setting the Qt bindings for QtPy
import os
from typing import Union, Dict, Any

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow

import os
import numpy as np
import xarray as xr
from src.geometry import RegionBounds, OctahedralGrid
from src.level_heights import compute_physical_level_height


Z_SCALE = 1

def _load_grib_file(config_entry: Union[str, Dict[str, Any]]):
    if isinstance(config_entry, str):
        return xr.open_dataset(config_entry, engine='cfgrib')
    path_to_data = config_entry.get('path')
    data = xr.open_dataset(path_to_data, engine='cfgrib')
    selectors = config_entry.get('select')
    if selectors is not None:
        data = data.isel(**selectors)
    return data


def load_data(path_to_data_config):
    logging.info(f'Loading data from {path_to_data_config}.')
    with open(path_to_data_config, 'r') as f:
        data_paths = json.load(f)
    orography_lr = _load_grib_file(data_paths['orography']['low-res']).z
    orography_hr = _load_grib_file(data_paths['orography']['high-res']).z
    t2m = _load_grib_file(data_paths['temperature']['2m']).t2m
    t3d = _load_grib_file(data_paths['temperature']['bulk']).t.transpose('hybrid', 'values')
    lnsp = _load_grib_file(data_paths['pressure']).lnsp
    q3d = _load_grib_file(data_paths['humidity']).q.transpose('hybrid', 'values')
    logging.info(f'Computing model level heights...')
    z_model_levels = None #compute_physical_level_height(np.exp(lnsp.values)[None, :], orography_lr.values[None, :], t3d.values, q3d.values)
    logging.info(f'Loading completed.')
    return orography_lr, orography_hr, t2m, t3d, z_model_levels


def build_surface_mesh(grid: OctahedralGrid, bounds: RegionBounds, z: np.ndarray = None):
    triangles, source_reference, (latitude, longitude) = grid.get_subgrid(bounds, rescale_indices=True, return_coordinates=True)
    coords = np.stack([(longitude + 180) % 360 - 180, latitude, np.zeros_like(latitude)], axis=-1)
    if z is not None:
        coords[:, -1] = z[source_reference] / Z_SCALE
    faces = np.concatenate([np.full((len(triangles), 1), 3, dtype=int), triangles], axis=-1)
    mesh = pv.PolyData(coords, faces)
    return mesh, source_reference


def extrude_surface_mesh(mesh: pv.PolyData, z: np.ndarray, source_reference: np.ndarray):
    num_levels = len(z)
    num_nodes = len(source_reference)
    coords_3d = np.tile(mesh.points, (num_levels, 1))
    coords_3d[:, -1] = z[:, source_reference].ravel() / Z_SCALE
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    num_triangles = len(triangles)
    wedges = np.zeros(((num_levels - 1) * num_triangles, 7), dtype=int)
    wedges[:, 0] = 6
    for level in range(num_levels - 1):
        wedges[(level * num_triangles):((level + 1) * num_triangles), 1:4] = triangles + level * num_nodes
        wedges[(level * num_triangles):((level + 1) * num_triangles), 4:] = triangles + (level + 1) * num_nodes
    return pv.UnstructuredGrid(wedges, [pv.CellType.WEDGE] * len(wedges), coords_3d)


# Load data for Dec 19, 2021, 0600 UTC
path_to_data_config = os.path.join(os.path.dirname(__file__), 'cfg', 'data', '2021121906_ubuntu.json')
orography_lr, orography_hr, t2m, t3d, z_model_levels = load_data(path_to_data_config)


class SettingsMenu(QDockWidget):

    class SelectRegionMenu(QWidget):

        class SelectAxis(object):

            def __init__(
                    self,
                    parent: QWidget,
                    default_min: float, default_max: float,
                    global_min: float, global_max: float,
                    step=0.5,
            ):
                self.global_min = global_min
                self.global_max = global_max
                self.step = step
                self.min_spinner = QDoubleSpinBox(parent)
                self.min_spinner.setValue(default_min)
                self.min_spinner.setRange(self.global_min, self.global_max - self.step)
                self.min_spinner.setPrefix('min: ')
                self.max_spinner = QDoubleSpinBox(parent)
                self.max_spinner.setValue(default_max)
                self.max_spinner.setRange(self.global_min + self.step, self.global_max)
                self.max_spinner.setPrefix('max: ')
                self.min_spinner.valueChanged.connect(self._update_max_spinner)
                self.max_spinner.valueChanged.connect(self._update_min_spinner)

            def _update_max_spinner(self):
                value = self.max_spinner.value()
                new_min_value = self.min_spinner.value()
                if value <= new_min_value:
                    self.max_spinner.setValue(min(new_min_value + self.step, self.global_max))

            def _update_min_spinner(self):
                value = self.min_spinner.value()
                new_max_value = self.max_spinner.value()
                if value >= new_max_value:
                    self.min_spinner.setValue(max(new_max_value - self.step, self.global_min))

        def __init__(self, parent=None):
            super().__init__(parent)

            self.select_latitude = self.SelectAxis(
                self,
                45., 50.,
                0., 90.,
            )
            self.select_longitude = self.SelectAxis(
                self,
                15., 20.,
                -180., 180.,
            )
            self.button_apply = QPushButton('Apply', self)

            layout = QGridLayout(self)
            layout.addWidget(QLabel('Latitude:', self), 0, 0)
            layout.addWidget(self.select_latitude.min_spinner, 0, 1, 1, 2)
            layout.addWidget(self.select_latitude.max_spinner, 0, 3, 1, 2)
            layout.addWidget(QLabel('Longitude:', self), 1, 0)
            layout.addWidget(self.select_longitude.min_spinner, 1, 1, 1, 2)
            layout.addWidget(self.select_longitude.max_spinner, 1, 3, 1, 2)
            layout.addWidget(self.button_apply, 2, 0, 1, 5)
            self.setLayout(layout)

        def get_region_boundaries(self):
            return RegionBounds(
                self.select_latitude.min_spinner.value(),
                self.select_latitude.max_spinner.value(),
                self.select_longitude.min_spinner.value(),
                self.select_longitude.max_spinner.value()
            )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        scroll = QtWidgets.QScrollArea(self)
        self.setWidget(scroll)
        content = QtWidgets.QWidget(scroll)
        scroll.setWidgetResizable(True)
        vlayout = QtWidgets.QVBoxLayout(content)
        vlayout.addWidget(QLabel('Select region:'))
        self.select_region = self.SelectRegionMenu(content)
        vlayout.addWidget(self.select_region)
        vlayout.addWidget(QLabel('Vertical scale:'))
        self.select_z_scale = LogSlider(100., 100000., parent=content)
        vlayout.addWidget(self.select_z_scale)
        content.setLayout(vlayout)

    @property
    def request_new_region(self):
        return self.select_region.button_apply.clicked

    @property
    def request_scale_change(self):
        return self.select_z_scale.display.textChanged

class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()
        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.settings_menu = SettingsMenu(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.settings_menu)
        self.settings_menu.request_new_region.connect(self._populate_plotter)
        self.settings_menu.request_scale_change.connect(self._change_vertical_plot_scale)

        self._populate_plotter()
        self._build_main_menu()

        if show:
            self.show()

    def _toggle_orography_visibility(self):
        actor = self.plotter.actors['oro-lr']
        is_visible = actor.visibility
        actor.visibility = not is_visible
        self.settings_menu.hide_orography_button.setText('show orography' if is_visible else 'hide orography')

    def _build_main_menu(self):
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

    def _build_left_side_menu(self):
        dock = QDockWidget("Settings")
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        scroll = QtWidgets.QScrollArea(dock)
        dock.setWidget(scroll)
        content = QtWidgets.QWidget(scroll)
        scroll.setWidgetResizable(True)
        vlay = QtWidgets.QVBoxLayout(content)

    def _populate_plotter(self):
        bounds = self.settings_menu.select_region.get_region_boundaries()
        self.plotter.clear_actors()
        # Build meshes
        logging.info('Building meshes.')
        mesh_lr, source_reference_lr = build_surface_mesh(OctahedralGrid(1280), bounds, z=orography_lr.values)
        mesh_hr, source_reference_hr = build_surface_mesh(OctahedralGrid(8000), bounds, z=orography_hr.values)
        # mesh_model_levels = extrude_surface_mesh(mesh_lr, z_model_levels, source_reference_lr)
        # mesh_model_levels['scalars'] = t3d.values[:, source_reference_lr].ravel()

        self.plotter.background_color = 'w'
        self.plotter.add_mesh(mesh_lr, style='wireframe', color='w', name='oro-lr')
        self.plotter.add_mesh(mesh_hr, style='wireframe', color='r')
        self._change_vertical_plot_scale()
        # self.plotter.add_mesh_slice(mesh_model_levels)

    def _change_vertical_plot_scale(self):
        scale = self.settings_menu.select_z_scale.get_value()
        self.plotter.set_scale(zscale=1. / scale)

    # def add_sphere(self):
    #     """ add a sphere to the pyqt frame """
    #     sphere = pv.Sphere()
    #     self.plotter.add_mesh(sphere, show_edges=True)
    #     self.plotter.reset_camera()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())

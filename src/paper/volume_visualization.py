import sys

# Setting the Qt bindings for QtPy
import os
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt

from src.model.geometry import Coordinates, WedgeMesh, TriangleMesh, LocationBatch
from src.paper.color_lookup import AsymmetricDivergentColorLookup
from src.paper.volume import VolumeVisualization

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow

import argparse
import os

import pandas as pd
import xarray as xr

import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
args = vars(parser.parse_args())

DATA_DIR = args['data_dir']


VERTICAL_SCALE = 4000.


class TerrainVisuals(object):

    def __init__(self, terrain_data: xr.Dataset, z_scale: float = VERTICAL_SCALE):
        self.data = terrain_data
        self.z_scale = z_scale
        self._compute_scene_coordinates()

    def _compute_scene_coordinates(self):
        longitude = self.data['longitude'].values
        latitude = self.data['latitude'].values
        self.scene_coordinates = np.stack([longitude, latitude], axis=-1)

    def land_surface(self):
        is_land = self.data['lsm'].values >= 0.5
        triangles = self.data['triangles'].values
        is_land_triangle = np.any(is_land[triangles], axis=-1)
        faces = np.zeros((int(np.sum(is_land_triangle)), 4), dtype=int)
        faces[:, 0] = 3
        faces[:, 1:] = triangles[is_land_triangle]
        z = self.data['z_surf'].values
        coords = np.concatenate([self.scene_coordinates, z[:, None] / self.z_scale], axis=-1)
        polydata = pv.PolyData(coords, faces)
        return polydata

    def sea_surface(self):
        is_sea = self.data['lsm'].values < 0.5
        triangles = self.data['triangles'].values
        is_land_triangle = np.any(is_sea[triangles], axis=-1)
        faces = np.zeros((int(np.sum(is_land_triangle)), 4), dtype=int)
        faces[:, 0] = 3
        faces[:, 1:] = triangles[is_land_triangle]
        z = self.data['z_surf'].values
        coords = np.concatenate([self.scene_coordinates, z[:, None] / self.z_scale], axis=-1)
        polydata = pv.PolyData(coords, faces)
        return polydata


class StationVisuals(object):

    def __init__(self, station_data: pd.DataFrame, z_scale = VERTICAL_SCALE):
        self.data = station_data
        self.z_scale = z_scale
        self.scene_coordinates = np.stack([self.data['longitude'].values, self.data['latitude'].values], axis=-1)

    def station_sites(self, add_scalars: Union[str, List[str]] = None):
        if add_scalars is None:
            add_scalars = []
        elif isinstance(add_scalars, str):
            add_scalars = [add_scalars]

        z = self.data['elevation'].values
        coords = np.concatenate([self.scene_coordinates, z[:, None] / self.z_scale], axis=-1)
        polydata = pv.PolyData(coords)
        for scalar_name in add_scalars:
            polydata[scalar_name] = self.data[scalar_name].values
        return polydata


class ModelVisuals(object):

    def __init__(self, model_data: xr.Dataset, terrain_data: xr.Dataset, z_scale: float = VERTICAL_SCALE):
        self.model_data = model_data
        self.terrain_data = terrain_data
        self.scene_coordinates = np.stack([
            self.terrain_data['longitude'].values,
            self.terrain_data['latitude'].values,
        ], axis=-1)
        self.z_scale = z_scale

    def surface_temperatures(self):
        coords = np.concatenate([
            self.scene_coordinates,
            (self.terrain_data['z_surf'].values[:, None] + 2) / self.z_scale
        ], axis=-1)
        triangles = self.terrain_data['triangles'].values
        faces = np.concatenate([np.full((len(triangles), 1), 3), triangles], axis=-1)
        polydata = pv.PolyData(coords, faces)
        polydata['t2m'] = self.model_data['t2m'].values
        return polydata

    def _get_model_level_grid(self):
        surface_mesh = TriangleMesh(
            LocationBatch(Coordinates.from_xarray(self.terrain_data)),
            self.terrain_data['triangles'].values
        )
        mesh = WedgeMesh(surface_mesh, self.model_data['z_model_levels'].values / self.z_scale)
        mesh = mesh.to_wedge_grid()
        return mesh

    def volume_temperatures(self):
        mesh = self._get_model_level_grid()
        mesh['t'] = self.model_data['t'].values.ravel()
        return mesh

    def temperature_gradients(self):
        mesh = self._get_model_level_grid()
        plt.figure()
        plt.hist(self.model_data['grad_t'].values.ravel() * 1000., bins=50)
        plt.show()
        plt.close()
        mesh['grad_t'] = np.clip(self.model_data['grad_t'].values.ravel() * 1000., -20, 50)
        return mesh


terrain_data_o1280 = xr.open_dataset(os.path.join(DATA_DIR, 'terrain_data_o1280.nc'))
coords_o1280 = Coordinates.from_xarray(terrain_data_o1280)
model_data = xr.open_dataset(os.path.join(DATA_DIR, 'model_data_2021121906_o1280.nc'))
station_data = pd.read_parquet(os.path.join(DATA_DIR, 'station_data_2021121906.parquet'))

terrain_visuals = TerrainVisuals(terrain_data_o1280)
station_visuals = StationVisuals(station_data)
model_visuals = ModelVisuals(model_data, terrain_data_o1280)


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter: pv.Plotter = QtInteractor(self.frame)
        self.plotter.enable_anti_aliasing(multi_samples=8)
        self.plotter.enable_depth_peeling()
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

        gradient_colors = AsymmetricDivergentColorLookup(
            AsymmetricDivergentColorLookup.Properties(
                'coolwarm', -12, 50, -6.5, 256, 2., 2., 1., 1., 'blue', 'red'
            )
        )
        self.gradient_volume = VolumeVisualization(
            'grad_t', 'Temperature gradient (K/km)',
            VERTICAL_SCALE,
            model_data, terrain_data_o1280,
            gradient_colors, self.plotter,
        )

        # self.plotter.add_mesh(terrain_visuals.land_surface(), color='k', style='wireframe')
        # self.plotter.add_mesh(terrain_visuals.sea_surface(), color='blue', style='wireframe')
        self.plotter.add_mesh(station_visuals.station_sites(), color='red', render_points_as_spheres=True)
        # self.plotter.add_mesh(model_visuals.surface_temperatures(), scalars='t2m', render_points_as_spheres=True)
        # self.plotter.add_mesh_slice(model_visuals.temperature_gradients(), cmap=cmap_gradient)

        if show:
            self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())

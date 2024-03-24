import sys

# Setting the Qt bindings for QtPy
import os
from typing import List, Union

import numpy as np
from PyQt5 import QtCore
from matplotlib import pyplot as plt

from src.model.geometry import Coordinates, WedgeMesh, TriangleMesh, LocationBatch
from src.paper.volume_visualization.color_lookup import AsymmetricDivergentColorLookup, ADCLController, \
    CustomOpacityProperties, ECMWFColors
from src.paper.volume_visualization.left_side_menu import LeftSideMenu
from src.paper.volume_visualization.plotter_slot import ReferenceGridProperties, SurfaceStyle, SurfaceProperties
from src.paper.volume_visualization.reference_grid import ReferenceGridVisualization, ReferenceGridController
from src.paper.volume_visualization.scaling import SceneScalingModel, SceneScalingController
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.station_reference import StationSiteVisualization, StationReferenceVisualization
from src.paper.volume_visualization.volume import ScalarVolumeController, VolumeData, \
    ScalarVolumeVisualization, VolumeProperties, PlotterSlot

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow

import argparse
import os

import pandas as pd
import xarray as xr


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

terrain_data_o8000 = xr.open_dataset(os.path.join(DATA_DIR, 'terrain_data_o8000.nc'))
coords_o8000 = Coordinates.from_xarray(terrain_data_o8000)


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter: pv.Plotter = QtInteractor(self.frame)
        self.plotter.enable_anti_aliasing(multi_samples=4)
        self.plotter.enable_depth_peeling(32)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        clear_scalar_bars_button = QtWidgets.QAction('Clear scalar bars', self)
        clear_scalar_bars_button.triggered.connect(self.clear_scalar_bars)
        fileMenu.addAction(clear_scalar_bars_button)
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        self.left_dock_menu = LeftSideMenu(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.left_dock_menu)

        self.plotter_scene = SceneScalingModel(parent=self)
        self.scaling_controls = SceneScalingController(self.left_dock_menu.scaling_settings, self.plotter_scene)

        self.gradient_colors = AsymmetricDivergentColorLookup(
            AsymmetricDivergentColorLookup.Properties(
                'coolwarm', -12, 50, -6.5, 256, 'blue', 'red',
                CustomOpacityProperties()
            )
        )
        self.gradient_color_controls = ADCLController(self.left_dock_menu.gradient_color_settings, self.gradient_colors)
        gradient_field = VolumeData(model_data, terrain_data_o1280, scalar_key='grad_t')
        plotter_slot = PlotterSlot(self.plotter, 'T gradient (K/km)')
        self.gradient_volume = ScalarVolumeVisualization(
            plotter_slot, gradient_field, self.gradient_colors, VolumeProperties(), visible=False
        )
        self.gradient_volume_controls = ScalarVolumeController(self.left_dock_menu.gradient_volume_settings, self.gradient_volume)
        self.plotter_scene.add_visual(self.gradient_volume)

        self.temperature_colors = ECMWFColors()
        temperature_field = VolumeData(model_data, terrain_data_o1280, scalar_key='t')
        plotter_slot = PlotterSlot(self.plotter, 'T (K)')
        self.temperature_volume = ScalarVolumeVisualization(
            plotter_slot, temperature_field, self.temperature_colors, VolumeProperties(), visible=False
        )
        self.temperature_volume_controls = ScalarVolumeController(self.left_dock_menu.temperature_volume_settings, self.temperature_volume)
        self.plotter_scene.add_visual(self.temperature_volume)

        self.t2m_colors = ECMWFColors()
        temperature_field = VolumeData(model_data, terrain_data_o1280, scalar_key='t2m', model_level_key='z_surf')
        plotter_slot = PlotterSlot(self.plotter, 'T2m (K)')
        self.t2m_volume = ScalarVolumeVisualization(
            plotter_slot, temperature_field, self.t2m_colors, SurfaceProperties(), visible=False
        )
        self.t2m_volume_controls = ScalarVolumeController(self.left_dock_menu.t2m_volume_settings,
                                                                  self.t2m_volume)
        self.plotter_scene.add_visual(self.t2m_volume)

        self.volume_reference_mesh = ReferenceGridVisualization(
            PlotterSlot(self.plotter), VolumeData(model_data, terrain_data_o1280, scalar_key=None),
        )
        self.volume_mesh_controls = ReferenceGridController(self.left_dock_menu.volume_mesh_settings, self.volume_reference_mesh)
        self.plotter_scene.add_visual(self.volume_reference_mesh)

        self.surface_reference_o1280 = ReferenceGridVisualization(
            PlotterSlot(self.plotter), VolumeData(model_data, terrain_data_o1280, scalar_key=None, model_level_key='z_surf'),
        )
        self.surface_controls_o1280 = ReferenceGridController(self.left_dock_menu.surface_settings_o1280, self.surface_reference_o1280)
        self.plotter_scene.add_visual(self.surface_reference_o1280)

        self.surface_reference_o8000 = ReferenceGridVisualization(
            PlotterSlot(self.plotter), VolumeData(model_data, terrain_data_o8000, scalar_key=None, model_level_key='z_surf'),
        )
        self.surface_controls_o8000 = ReferenceGridController(self.left_dock_menu.surface_settings_o8000, self.surface_reference_o8000)
        self.plotter_scene.add_visual(self.surface_reference_o8000)

        station_data_ = StationData(station_data, terrain_data_o1280)
        self.station_sites = StationSiteVisualization(
            PlotterSlot(self.plotter), station_data_, ReferenceGridProperties(point_size=10, render_points_as_spheres=True)
        )
        self.plotter_scene.add_visual(self.station_sites)
        self.station_sites.show()

        self.station_refs = StationReferenceVisualization(
            PlotterSlot(self.plotter), station_data_, ReferenceGridProperties(show_edges=True, style=SurfaceStyle.WIREFRAME)
        )
        self.plotter_scene.add_visual(self.station_refs)
        self.station_refs.show()

        if show:
            self.show()

    def clear_scalar_bars(self):
        self.plotter.scalar_bars.clear()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())

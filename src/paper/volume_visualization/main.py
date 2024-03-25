import sys

# Setting the Qt bindings for QtPy
import os
from typing import List, Union

import numpy as np
from PyQt5 import QtCore
from matplotlib import pyplot as plt

from src.interaction.plotter_controls.controller import PlotterController
from src.model.geometry import Coordinates, WedgeMesh, TriangleMesh, LocationBatch
from src.paper.volume_visualization.color_lookup import AsymmetricDivergentColorLookup, ADCLController, \
    CustomOpacityProperties, ECMWFColors
from src.paper.volume_visualization.left_side_menu import RightDockMenu
from src.paper.volume_visualization.multi_method_visualization import MultiMethodVisualizationController
from src.paper.volume_visualization.plotter_slot import SurfaceProperties, \
    PlotterSlot, VolumeProperties, StationSiteProperties, StationSiteReferenceProperties, \
    StationOnTerrainReferenceProperties
from src.paper.volume_visualization.station import StationScalarVisualization
from src.paper.volume_visualization.volume import VolumeScalarVisualization
from src.paper.volume_visualization.volume_data import VolumeData
from src.paper.volume_visualization.volume_reference_grid import ReferenceGridVisualization, ReferenceGridController
from src.paper.volume_visualization.scaling import SceneScalingModel, SceneScalingController
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.station_reference import StationSiteReferenceVisualization, StationOnTerrainReferenceVisualization

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


terrain_data_o1280 = xr.open_dataset(os.path.join(DATA_DIR, 'terrain_data_o1280.nc'))
coords_o1280 = Coordinates.from_xarray(terrain_data_o1280)
model_data = xr.open_dataset(os.path.join(DATA_DIR, 'model_data_2021121906_o1280.nc'))
station_data = pd.read_parquet(os.path.join(DATA_DIR, 'station_data_2021121906.parquet'))

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
        self.plotter.enable_anti_aliasing('fxaa')
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

        self.right_dock_menu = RightDockMenu(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.right_dock_menu)

        self.plotter_controls = PlotterController(self.right_dock_menu.plotter_settings, self.plotter, self)

        self.plotter_scene = SceneScalingModel(parent=self)
        self.scaling_controls = SceneScalingController(self.right_dock_menu.scaling_settings, self.plotter_scene)
        self.color_controls = {}
        self.visualization_controls = {}

        self._build_model_scalar_visuals()
        self._build_reference_visuals()
        self._build_station_visuals()
        self._build_station_references()

        if show:
            self.show()

    def _build_grid_visual(self, key, plotter_slot, dataset, properties=None, color_lookup=None):
        if color_lookup is None:
           visual = ReferenceGridVisualization(plotter_slot, dataset, properties)
        else:
            visual = VolumeScalarVisualization(plotter_slot, dataset, color_lookup, properties)
            if key in self.right_dock_menu.color_settings_views:
                settings_view = self.right_dock_menu.color_settings_views[key]
                controller = color_lookup.get_controller(settings_view)
                if controller is not None:
                    self.color_controls[key] = controller
        if key in self.right_dock_menu.vis_settings_views:
            settings_view = self.right_dock_menu.vis_settings_views[key]
            if color_lookup is None:
                controller = ReferenceGridController(settings_view, visual)
            else:
                controller = MultiMethodVisualizationController(settings_view, visual)
            self.visualization_controls[key] = controller
        self.plotter_scene.add_visual(visual)

    def _build_station_visual(self, key, plotter_slot, dataset, properties, color_lookup):
        visual = StationScalarVisualization(plotter_slot, dataset, color_lookup, properties)
        if key in self.right_dock_menu.color_settings_views:
            settings_view = self.right_dock_menu.color_settings_views[key]
            controller = color_lookup.get_controller(settings_view)
            if controller is not None:
                self.color_controls[key] = controller
        if key in self.right_dock_menu.vis_settings_views:
            settings_view = self.right_dock_menu.vis_settings_views[key]
            controller = MultiMethodVisualizationController(settings_view, visual)
            self.visualization_controls[key] = controller
        self.plotter_scene.add_visual(visual)

    def _build_station_references(self):
        key = 'station_sites'
        visual = StationSiteReferenceVisualization(
            PlotterSlot(self.plotter), StationData(station_data, terrain_data_o1280),
            StationSiteReferenceProperties()
        )
        settings_view = self.right_dock_menu.vis_settings_views[key]
        controller = ReferenceGridController(settings_view, visual)
        self.visualization_controls[key] = controller
        self.plotter_scene.add_visual(visual)

        key = 'station_on_terrain'
        visual = StationOnTerrainReferenceVisualization(
            PlotterSlot(self.plotter), StationData(station_data, terrain_data_o1280),
            StationOnTerrainReferenceProperties()
        )
        settings_view = self.right_dock_menu.vis_settings_views[key]
        controller = ReferenceGridController(settings_view, visual)
        self.visualization_controls[key] = controller
        self.plotter_scene.add_visual(visual)

    def _build_station_visuals(self):
        self._build_station_visual(
            'station_t_obs',
            PlotterSlot(self.plotter, 'Observation (K)'),
            StationData(station_data, terrain_data_o1280, scalar_key='observation'),
            StationSiteProperties(),
            ECMWFColors()
        )
        self._build_station_visual(
            'station_t_pred',
            PlotterSlot(self.plotter, 'Prediction (K)'),
            StationData(station_data, terrain_data_o1280, scalar_key='prediction'),
            StationSiteProperties(),
            ECMWFColors()
        )
        self._build_station_visual(
            'station_t_diff',
            PlotterSlot(self.plotter, 'T diff. (K)'),
            StationData(station_data, terrain_data_o1280, scalar_key='difference'),
            StationSiteProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'coolwarm', 0.5, -30, 30, 0., 29, 'blue', 'red',
                    CustomOpacityProperties()
                )
            ),
        )
        self._build_station_visual(
            'station_grad_t',
            PlotterSlot(self.plotter, 'Obs. gradient (K/km)'),
            StationData(station_data, terrain_data_o1280, scalar_key='grad_t'),
            StationSiteProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'coolwarm', 0.5, -12, 50, -6.5, 256, 'blue', 'red',
                    CustomOpacityProperties()
                )
            ),
        )

    def _build_reference_visuals(self):
        self._build_grid_visual(
            'model_grid',
            PlotterSlot(self.plotter),
            VolumeData(model_data, terrain_data_o1280, scalar_key=None),
        )
        self._build_grid_visual(
            'surface_grid_o1280',
            PlotterSlot(self.plotter),
            VolumeData(model_data, terrain_data_o1280, scalar_key=None, model_level_key='z_surf'),
        )
        self._build_grid_visual(
            'surface_grid_o8000',
            PlotterSlot(self.plotter),
            VolumeData(model_data, terrain_data_o8000, scalar_key=None, model_level_key='z_surf')
        )

        self._build_grid_visual(
            'lsm_o1280',
            PlotterSlot(self.plotter, 'LSM (O1280)'),
            VolumeData(model_data, terrain_data_o1280, scalar_key='lsm', model_level_key='z_surf'),
            SurfaceProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'gist_earth', 0.5, 0., 1., 0.5, 7, 'black', 'white',
                    CustomOpacityProperties()
                )
            ),
        )
        self._build_grid_visual(
            'lsm_o8000',
            PlotterSlot(self.plotter, 'LSM (O8000)'),
            VolumeData(model_data, terrain_data_o8000, scalar_key='lsm', model_level_key='z_surf'),
            SurfaceProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'gist_earth', 0.5, 0., 1., 0.5, 7, 'black', 'white',
                    CustomOpacityProperties()
                )
            ),
        )
        self._build_grid_visual(
            'z_o1280',
            PlotterSlot(self.plotter, 'Z (O1280)'),
            VolumeData(model_data, terrain_data_o1280, scalar_key='z_surf', model_level_key='z_surf'),
            SurfaceProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'gist_earth', 0.3, -100., 3000., 0., 58, 'black', 'white',
                    CustomOpacityProperties(opacity_center=1.)
                )
            ),
        )
        self._build_grid_visual(
            'z_o8000',
            PlotterSlot(self.plotter, 'Z (O8000)'),
            VolumeData(model_data, terrain_data_o8000, scalar_key='z_surf', model_level_key='z_surf'),
            SurfaceProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'gist_earth', 0.3, -100., 3000., 0., 58, 'black', 'white',
                    CustomOpacityProperties(opacity_center=1.)
                )
            ),
        )
        self._build_station_visual(
            'station_offset',
            PlotterSlot(self.plotter, 'Offset (m)'),
            StationData(station_data, terrain_data_o1280, scalar_key='elevation_difference'),
            StationSiteProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'BrBG_r', 0.5, -1500, 1500, 0., 29, 'green', 'orange',
                    CustomOpacityProperties()
                )
            ),
        )

    def _build_model_scalar_visuals(self):
        self._build_grid_visual(
            'model_grad_t',
            PlotterSlot(self.plotter, 'T gradient (K/km)'),
            VolumeData(model_data, terrain_data_o1280, scalar_key='grad_t'),
            VolumeProperties(),
            AsymmetricDivergentColorLookup(
                AsymmetricDivergentColorLookup.Properties(
                    'coolwarm', 0.5, -12, 50, -6.5, 256, 'blue', 'red',
                    CustomOpacityProperties()
                )
            ),
        )
        self._build_grid_visual(
            'model_t',
            PlotterSlot(self.plotter, 'T (K)'),
            VolumeData(model_data, terrain_data_o1280, scalar_key='t'),
            VolumeProperties(),
            ECMWFColors(),
        )
        self._build_grid_visual(
            'model_t2m',
            PlotterSlot(self.plotter, 'T2m (K)'),
            VolumeData(model_data, terrain_data_o1280, scalar_key='t2m', model_level_key='z_surf'),
            SurfaceProperties(),
            ECMWFColors(),
        )

    def clear_scalar_bars(self):
        self.plotter.scalar_bars.clear()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())

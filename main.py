import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDockWidget
import pyvista as pv

from src.interaction.domain_selection.controller import DownscalingController
from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.interaction.visualizations.controller import SceneController
from src.model.backend_model import DownscalingPipeline
from src.model.data_store.world_data import WorldData
from src.model.visualization.scene_model import SceneModel

logging.basicConfig(level=logging.INFO)

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from pyvistaqt import MainWindow

import os


CONFIG_FILE = os.environ['WORLD_DATA_CONFIG']


class MainView(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        self._build_central_widget()
        self._build_menus()
        self._build_downscaling_pipeline()
        self._build_vis_pipeline()
        self._populate_plotter()

        if show:
            self.show()

    def _build_menus(self):
        self.settings_menu = SettingsViewTabbed(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.settings_menu)
        self._build_main_menu()

    def _build_central_widget(self):
        self.render_view = PyvistaView(self)
        self.setCentralWidget(self.render_view.frame)
        self.signal_close.connect(self.render_view.plotter.close)

    def _build_downscaling_pipeline(self):
        self.data_store = WorldData.from_config_file(CONFIG_FILE)
        self.downscaling_pipeline = DownscalingPipeline(self.data_store)
        self.downscaling_controller = DownscalingController(
            self.settings_menu, self.render_view,
            self.downscaling_pipeline, parent=self
        )

    def _build_vis_pipeline(self):
        self.scene_model = SceneModel(self)
        self.scene_controller = SceneController(
            self.settings_menu, self.render_view,
            self.downscaling_controller,
            self.downscaling_pipeline, self.scene_model,
            parent=self
        )

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
        self.render_view.plotter.add_mesh(pv.Sphere())
        # bounds = self.settings_menu.select_region.get_region_boundaries()
        # self.render_view.plotter.clear_actors()
        # # Build meshes
        # logging.info('Building meshes...')
        # scene_data = world_data.query_scene_data(bounds)
        # mesh_lr = scene_data.get_orography_mesh_lr()
        # self.render_view.add_mesh(mesh_lr)
        # self.render_view.plotter.add_bounding_box()
        # self._change_vertical_plot_scale()
        # mesh_model_levels['scalars'] = t3d.values[:, source_reference_lr].ravel()
        # self.plotter.add_mesh(mesh_lr, style='wireframe', color='w', name='oro-lr')
        # self.plotter.add_mesh(mesh_hr, style='wireframe', color='r')
        # self.plotter.add_volume(mesh_model_levels)
        # self._change_vertical_plot_scale()

    def _change_vertical_plot_scale(self):
        scale = self.settings_menu.select_z_scale.get_value()
        print('rescaling to scale {}'.format(scale))
        self.render_view.rescale_z(scale)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainView()
    sys.exit(app.exec_())

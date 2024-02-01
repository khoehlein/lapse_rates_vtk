import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDockWidget
import pyvista as pv

from src.interaction.domain_selection.controller import DownscalingController
from src.interaction.plotter_controls.controller import PlotterController
from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.interaction.visualizations.controller import VisualizationController, SceneController
from src.model.backend_model import DownscalingPipeline
from src.model.data_store.dummy import DummyPipeline, DummyController
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
        # self.data_store = WorldData.from_config_file(CONFIG_FILE)
        # self.downscaling_pipeline = DownscalingPipeline(self.data_store)
        # self.downscaling_controller = DownscalingController(
        #     self.settings_menu, self.render_view,
        #     self.downscaling_pipeline, parent=self
        # )
        self.downscaling_pipeline = DummyPipeline()
        self.downscaling_controller = DummyController(self.downscaling_pipeline)

    def _build_vis_pipeline(self):
        self.plotter_controller = PlotterController(
            self.settings_menu.general_settings.plotter_settings,
            self.render_view.plotter, parent=self
        )
        self.scene_model = SceneModel(self.render_view.plotter, parent=self)
        self.scene_controller = SceneController(self.downscaling_pipeline, self.scene_model, parent=self)
        self.scene_controller.register_settings_view(self.settings_menu.visualization_settings)
        self.scene_controller.reset_scene()

    def _build_main_menu(self):
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainView()
    sys.exit(app.exec_())

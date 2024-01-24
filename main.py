import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDockWidget
import pyvista as pv

from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed

logging.basicConfig(level=logging.INFO)

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from pyvistaqt import MainWindow

import os
import numpy as np

# Load model for Dec 19, 2021, 0600 UTC
# path_to_data_config = os.path.join(os.path.dirname(__file__), 'cfg', 'model', '2021121906_ubuntu.json')
# world_data = WorldData.from_config_file(path_to_data_config)


class MainView(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.render_view = PyvistaView(self)
        self.setCentralWidget(self.render_view.frame)
        self.signal_close.connect(self.render_view.plotter.close)

        self.settings_menu = SettingsViewTabbed(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.settings_menu)
        # self.settings_menu.request_new_region.connect(self._populate_plotter)
        # self.settings_menu.request_scale_change.connect(self._change_vertical_plot_scale)

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

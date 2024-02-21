import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDockWidget, QScrollArea, QWidget, QFrame, QVBoxLayout

import pyvista as pv

pv.global_theme.allow_empty_mesh = True

logging.basicConfig(level=logging.INFO)

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from pyvistaqt import MainWindow

import os

from src.interaction.downscaling.methods import FixedLapseRateDownscalerView, LapseRateEstimatorView, \
    AdaptiveLapseRateDownscalerView

CONFIG_FILE = os.environ.get('WORLD_DATA_CONFIG')


class MainView(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        self._build_central_widget()
        self._build_menus()
        # self._build_downscaling_pipeline()
        # self._build_vis_pipeline()

        if show:
            self.show()

    def _build_menus(self):
        self.settings_menu = QDockWidget(self)
        self.settings_menu.setWindowTitle('Settings')
        self.settings_menu.setFeatures(QDockWidget.NoDockWidgetFeatures)
        scroll_area = QScrollArea(self.settings_menu)
        scroll_area.setWidgetResizable(True)
        contents = QWidget(scroll_area)
        scroll_area.setWidget(contents)
        self.settings_menu.setWidget(scroll_area)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.settings_menu)

        downscaler_view = FixedLapseRateDownscalerView(contents)
        layout = QVBoxLayout()
        layout.addWidget(downscaler_view)
        layout.addStretch()
        contents.setLayout(layout)

    def _build_central_widget(self):
        self.render_view = QFrame(self) #PyvistaView(self)
        self.setCentralWidget(self.render_view)
        # self.signal_close.connect(self.render_view.plotter.close)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainView()
    sys.exit(app.exec_())

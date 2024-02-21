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
    AdaptiveLapseRateDownscalerView, CreateDownscalerDialog

CONFIG_FILE = os.environ.get('WORLD_DATA_CONFIG')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = CreateDownscalerDialog()
    sys.exit(dialog.exec_())

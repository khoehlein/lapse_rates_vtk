import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDockWidget, QFrame, QVBoxLayout, QPushButton, QWidget, QDialog, QFormLayout
from qtpy.QtCore import Signal
from pyvista import Sphere
from pyvistaqt.dialog import RangeGroup
from pyvistaqt.editor import Editor

logging.basicConfig(level=logging.INFO)

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from pyvistaqt import MainWindow, QtInteractor


class ScaleAxesDialog(QDialog):
    """Dialog to control axes scaling."""

    # pylint: disable=too-few-public-methods

    accepted = Signal(float)
    signal_close = Signal()

    def __init__(
        self, parent: MainWindow, plotter: QtInteractor, show: bool = True
    ) -> None:
        """Initialize the scaling dialog."""
        super().__init__(parent)
        self.setGeometry(300, 300, 50, 50)
        self.setMinimumWidth(500)
        self.signal_close.connect(self.close)
        self.plotter = plotter
        # self.plotter.app_window.signal_close.connect(self.close)

        self.x_slider_group = RangeGroup(
            self, self.update_scale, value=plotter.scale[0]
        )
        self.y_slider_group = RangeGroup(
            self, self.update_scale, value=plotter.scale[1]
        )
        self.z_slider_group = RangeGroup(
            self, self.update_scale, value=plotter.scale[2]
        )

        form_layout = QFormLayout(self)
        form_layout.addRow("X Scale", self.x_slider_group)
        form_layout.addRow("Y Scale", self.y_slider_group)
        form_layout.addRow("Z Scale", self.z_slider_group)

        self.setLayout(form_layout)

        if show:  # pragma: no cover
            self.show()

    def update_scale(self) -> None:
        """Update the scale of all actors in the plotter."""
        self.plotter.set_scale(
            self.x_slider_group.value,
            self.y_slider_group.value,
            self.z_slider_group.value,
        )



class MainView(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        frame = QFrame(parent=self)
        self.plotter = QtInteractor(parent=frame)

        layout = QVBoxLayout()
        layout.addWidget(self.plotter)
        frame.setLayout(layout)
        self.setCentralWidget(frame)

        self.plotter.add_mesh(Sphere(), show_edges=True)
        self.signal_close.connect(self.plotter.close)

        dock_menu = QDockWidget(parent=self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_menu)
        contents = QWidget(parent=dock_menu)
        dock_menu.setWidget(contents)
        dock_layout = QVBoxLayout()
        self.button_editor_dialog = QPushButton('Editor dialog', parent=contents)
        self.button_rescale_dialog = QPushButton('Rescale dialog', parent=contents)
        dock_layout.addWidget(self.button_editor_dialog)
        dock_layout.addWidget(self.button_rescale_dialog)
        contents.setLayout(dock_layout)
        self.button_editor_dialog.clicked.connect(self.open_editor_dialog)
        self.button_rescale_dialog.clicked.connect(self.open_rescale_dialog)

        # create the frame
        # self.render_view = PyvistaView(self)
        # self.setCentralWidget(self.render_view.frame)
        # self.signal_close.connect(self.render_view.plotter.close)
        #
        # self.settings_menu = SettingsView(self)
        # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.settings_menu)
        # self.settings_menu.request_new_region.connect(self._populate_plotter)
        # self.settings_menu.request_scale_change.connect(self._change_vertical_plot_scale)
        #
        # self._populate_plotter()
        # self._build_main_menu()
        #
        if show:
            self.show()

    def open_editor_dialog(self):
        print('render')
        dlg = Editor(self, list(self.plotter.renderers))
        dlg.exec_()

    def open_rescale_dialog(self):
        print('rescale')
        print(self.plotter.scale)
        dlg = ScaleAxesDialog(self, self.plotter)
        dlg.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainView()
    sys.exit(app.exec_())

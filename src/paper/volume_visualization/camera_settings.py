import json
import os

import pyvista as pv
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QWidget, QPushButton, QVBoxLayout, QHBoxLayout


DEFAULT_CAMERA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cameras')
os.makedirs(DEFAULT_CAMERA_DIR, exist_ok=True)


class CameraControlsSettingsView(QWidget):

    load_camera_request = pyqtSignal(str)
    save_camera_request = pyqtSignal(str)
    save_screenshot_request = pyqtSignal(str)

    def __init__(self, default_plotting_dir: str, parent=None):
        super().__init__(parent)
        self.default_plotting_dir = os.path.abspath(default_plotting_dir)
        os.makedirs(default_plotting_dir, exist_ok=True)
        self.button_save_screenshot = QPushButton('Save screenshot')
        self.button_save_camera = QPushButton('Save camera')
        self.button_load_camera = QPushButton('Load camera')
        self.button_load_default_camera = QPushButton('Load default camera')
        self._connect_signals()
        self._set_layout()

    def _connect_signals(self):
        self.button_save_screenshot.clicked.connect(self.on_save_screenshot)
        self.button_save_camera.clicked.connect(self.on_save_camera)
        self.button_load_camera.clicked.connect(self.on_load_camera)
        self.button_load_default_camera.clicked.connect(self.on_load_default_camera)

    def _set_layout(self):
        layout = QVBoxLayout()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.button_load_camera)
        hlayout.addWidget(self.button_save_camera)
        layout.addLayout(hlayout)
        layout.addWidget(self.button_load_default_camera)
        layout.addWidget(self.button_save_screenshot)
        self.setLayout(layout)

    def on_save_screenshot(self):
        fname = QFileDialog.getSaveFileName(self, 'Save screenshot', self.default_plotting_dir, '*.png')
        if fname[0]:
            self.save_screenshot_request.emit(fname[0])

    def on_save_camera(self):
        fname = QFileDialog.getSaveFileName(self, 'Save camera', DEFAULT_CAMERA_DIR, '*.json')
        if fname[0]:
            self.save_camera_request.emit(fname[0])

    def on_load_camera(self):
        fname = QFileDialog.getOpenFileName(self, 'Load camera', DEFAULT_CAMERA_DIR, '*.json')
        if fname[0]:
            self.load_camera_request.emit(fname[0])

    def on_load_default_camera(self):
        self.load_camera_request.emit(os.path.join(DEFAULT_CAMERA_DIR, 'default.json'))


class CameraController(QWidget):

    KEYS = ['position', 'focus', 'viewup']

    def __init__(self, view: CameraControlsSettingsView, model: pv.Plotter, parent=None):
        super().__init__(parent)
        self.view = view
        self.plotter = model
        self.view.save_screenshot_request.connect(self.handle_screenshot)
        self.view.save_camera_request.connect(self.handle_save_camera)
        self.view.load_camera_request.connect(self.handle_load_camera)

    def handle_screenshot(self, path: str):
        self.plotter.screenshot(path)

    def handle_save_camera(self, path: str):
        camera = self.plotter.camera_position
        camera = {key: list(value) for key, value in zip(self.KEYS, camera)}
        with open(path, 'w') as f:
            json.dump(camera, f, sort_keys=True, indent=4)

    def handle_load_camera(self, path: str):
        with open(path, 'r') as f:
            camera = json.load(f)
        camera = pv.CameraPosition(*[camera[key] for key in self.KEYS])
        self.plotter.camera_position = camera
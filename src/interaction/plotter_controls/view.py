import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QCheckBox, QComboBox, QPushButton, QFormLayout, QVBoxLayout, QLabel, QDateTimeEdit, \
    QSpinBox, QDoubleSpinBox, QHBoxLayout

from src.widgets import SelectColorButton


class PlotterSettingsView(QWidget):

    aa_changed = pyqtSignal(str)
    lighting_mode_changed = pyqtSignal(str)
    interaction_style_changed = pyqtSignal(str)
    show_boundary_changed = pyqtSignal(bool)
    show_grid_changed = pyqtSignal(bool)
    show_camera_widget_changed = pyqtSignal(bool)
    hlr_changed = pyqtSignal(bool)
    pp_changed = pyqtSignal(bool)
    ssao_changed = pyqtSignal(bool)
    stereo_render_changed = pyqtSignal(bool)
    background_color_changed = pyqtSignal(QColor)

    solar_timestamp_changed = pyqtSignal(np.datetime64)
    solar_location_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.combo_lighting_mode = QComboBox(self)
        self.combo_lighting_mode.addItems(['None', 'LightKit', '3 lights', 'Solar lighting'])
        self.combo_lighting_mode.currentTextChanged.connect(self.lighting_mode_changed.emit)

        self.combo_anti_aliasing = QComboBox(self)
        self.combo_anti_aliasing.addItems(['None', 'SSAA', 'MSAA-2x', 'MSAA-4x', 'MSAA-8x', 'FXAA'])
        self.combo_anti_aliasing.currentTextChanged.connect(self.aa_changed.emit)

        self.combo_interaction_style = QComboBox(self)
        self.combo_interaction_style.addItems(['Image', 'Joystick', 'Joystick actor', 'Terrain', 'Trackball', 'Trackball actor', 'Zoom'])
        self.combo_interaction_style.currentTextChanged.connect(self.interaction_style_changed.emit)

        self.checkbox_boundary_box = QCheckBox(self)
        self.checkbox_boundary_box.stateChanged.connect(self.show_boundary_changed.emit)

        self.checkbox_grid = QCheckBox(self)
        self.checkbox_grid.stateChanged.connect(self.show_grid_changed.emit)

        self.checkbox_camera_widget = QCheckBox(self)
        self.checkbox_camera_widget.stateChanged.connect(self.show_camera_widget_changed.emit)

        self.checkbox_ssao = QCheckBox(self)
        self.checkbox_ssao.stateChanged.connect(self.ssao_changed.emit)

        self.checkbox_parallel_projection = QCheckBox(self)
        self.checkbox_parallel_projection.stateChanged.connect(self.pp_changed.emit)

        self.checkbox_stereo_rendering = QCheckBox(self)
        self.checkbox_stereo_rendering.stateChanged.connect(self.stereo_render_changed.emit)

        self.checkbox_hidden_line_removal = QCheckBox(self)
        self.checkbox_hidden_line_removal.stateChanged.connect(self.hlr_changed.emit)

        self.button_background_color = SelectColorButton(parent=self)
        self.button_background_color.setText(' Select background color')
        self.button_background_color.color_changed.connect(self.background_color_changed.emit)

        self.button_reset = QPushButton(self)
        self.button_reset.setText('Reset')
        self.button_reset.clicked.connect(self.set_defaults)

        self.utc_day = QSpinBox(self)
        self.utc_day.setMaximum(365)
        self.utc_day.setPrefix('day of year: ')
        self.utc_day.valueChanged.connect(self.on_solar_timestamp_changed)
        self.utc_hour = QSpinBox(self)
        self.utc_hour.setMaximum(24)
        self.utc_hour.setPrefix('hour: ')
        self.utc_hour.valueChanged.connect(self.on_solar_timestamp_changed)
        self.solar_longitude = QDoubleSpinBox(self)
        self.solar_longitude.setMinimum(-180.)
        self.solar_longitude.setMaximum(180.)
        self.solar_longitude.setPrefix('lon: ')
        self.solar_longitude.valueChanged.connect(self.on_solar_location_changed)
        self.solar_latitude = QDoubleSpinBox(self)
        self.solar_latitude.setMinimum(-90.)
        self.solar_latitude.setMaximum(90.)
        self.solar_latitude.setPrefix('lat: ')
        self.solar_latitude.valueChanged.connect(self.on_solar_location_changed)
        self._set_layout()

    def on_solar_timestamp_changed(self):
        date = np.datetime64('2020-01-01T00', 'h')
        date = date + (24 * self.utc_day.value() + self.utc_hour.value()) * np.timedelta64(1, 'h')
        print(date)
        self.solar_timestamp_changed.emit(date)

    def on_solar_location_changed(self):
        latitude = self.solar_latitude.value()
        longitude = self.solar_longitude.value()
        self.solar_location_changed.emit(longitude, latitude)

    def _set_layout(self):
        form = QWidget(self)
        layout = QFormLayout(self)
        layout.addRow(QLabel('Anti-aliasing:'), self.combo_anti_aliasing)
        layout.addRow(QLabel('Lighting mode:'), self.combo_lighting_mode)
        layout.addRow(QLabel('Interaction style:'), self.combo_interaction_style)
        layout.addRow(QLabel('Show boundary box:'), self.checkbox_boundary_box)
        layout.addRow(QLabel('Show axes grid:'), self.checkbox_grid)
        layout.addRow(QLabel('Show camera widget:'), self.checkbox_camera_widget)
        layout.addRow(QLabel('SSAO:'), self.checkbox_ssao)
        layout.addRow(QLabel('Parallel projection:'), self.checkbox_parallel_projection)
        layout.addRow(QLabel('Stereo rendering:'), self.checkbox_stereo_rendering)
        layout.addRow(QLabel('Hidden line removal:'), self.checkbox_hidden_line_removal)
        form.setLayout(layout)
        outer = QVBoxLayout()
        outer.addWidget(form)
        outer.addWidget(self.button_background_color)
        form = QFormLayout()
        timespinners = QHBoxLayout()
        timespinners.addWidget(self.utc_day)
        timespinners.addWidget(self.utc_hour)
        form.addRow(QLabel('Timestamp:'), timespinners)
        location = QHBoxLayout()
        location.addWidget(self.solar_latitude)
        location.addWidget(self.solar_longitude)
        form.addRow(QLabel('Location:'), location)
        outer.addWidget(self.button_reset)
        outer.addLayout(form)
        outer.addStretch()
        self.setLayout(outer)

    def set_defaults(self):
        self.combo_anti_aliasing.setCurrentText('MSAA-8x')
        self.combo_lighting_mode.setCurrentText('LightKit')
        self.combo_interaction_style.setCurrentText('Trackball')
        self.checkbox_boundary_box.setChecked(False)
        self.checkbox_grid.setChecked(False)
        self.checkbox_camera_widget.setChecked(False)
        self.checkbox_ssao.setChecked(False)
        self.checkbox_parallel_projection.setChecked(False)
        self.checkbox_stereo_rendering.setChecked(False)
        self.checkbox_hidden_line_removal.setChecked(False)
        self.button_background_color.set_current_color(QColor(255, 255, 255))
        self.on_solar_location_changed()
        self.on_solar_timestamp_changed()

    def get_settings(self):
        return {
            'aa_mode': self.combo_anti_aliasing.currentText(),
            'lighting_mode': self.combo_lighting_mode.currentText(),
            'interaction_style': self.combo_interaction_style.currentText(),
            'show_boundary': self.checkbox_boundary_box.isChecked(),
            'show_grid': self.checkbox_grid.isChecked(),
            'show_camera_widget': self.checkbox_camera_widget.isChecked(),
            'use_ssao': self.checkbox_ssao.isChecked(),
            'apply_pp': self.checkbox_parallel_projection.isChecked(),
            'apply_stereo_rendering': self.checkbox_stereo_rendering.isChecked(),
            'apply_hlr': self.checkbox_hidden_line_removal.isChecked(),
            'background_color': self.button_background_color.current_color
        }




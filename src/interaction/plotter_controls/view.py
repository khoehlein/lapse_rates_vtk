from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QCheckBox, QComboBox, QPushButton, QGridLayout, QFormLayout, QVBoxLayout, QLabel

from src.interaction.background_color.view import SelectColorButton


class PlotterSettingsView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.checkbox_boundary_box = QCheckBox(self)
        self.checkbox_grid = QCheckBox(self)
        self.checkbox_camera_widget = QCheckBox(self)
        # self.checkbox_depth_of_field = QCheckBox(self)
        # self.checkbox_eye_dome_lighting = QCheckBox(self)
        self.checkbox_hidden_line_removal = QCheckBox(self)
        self.checkbox_parallel_projection = QCheckBox(self)
        self.checkbox_shadows = QCheckBox(self)
        self.checkbox_ssao = QCheckBox(self)
        self.checkbox_stereo_rendering = QCheckBox(self)

        # self.checkbox_ssao.stateChanged.connect(self._toggle_dof_activity)
        # self.checkbox_depth_of_field.stateChanged.connect(self._toggle_ssao_activity)

        # self.combo_lighting_mode = QComboBox(self)
        # self.combo_lighting_mode.addItems(['LightKit', '3 lights'])

        self.combo_anti_aliasing = QComboBox(self)
        self.combo_anti_aliasing.addItems(['None', 'SSAA', 'MSAA-2x', 'MSAA-4x', 'MSAA-8x', 'FXAA'])

        self.combo_interaction_style = QComboBox(self)
        self.combo_interaction_style.addItems(['Image', 'Joystick', 'Joystick actor', 'Terrain', 'Trackball', 'Trackball actor', 'Zoom'])

        self.button_background_color = SelectColorButton(parent=self)
        self.button_background_color.setText(' Select background color')
        self.button_reset = QPushButton(self)
        self.button_reset.setText('Reset')
        self.button_reset.clicked.connect(self._set_defaults)
        self._set_defaults()
        self._set_layout()

    def _set_layout(self):
        form = QWidget(self)
        layout = QFormLayout(self)
        layout.addRow(QLabel('Anti-aliasing:'), self.combo_anti_aliasing)
        # layout.addRow(QLabel('Lighting mode:'), self.combo_lighting_mode)
        layout.addRow(QLabel('Interaction style:'), self.combo_interaction_style)
        layout.addRow(QLabel('Show boundary box:'), self.checkbox_boundary_box)
        layout.addRow(QLabel('Show axes grid:'), self.checkbox_grid)
        layout.addRow(QLabel('Show camera widget:'), self.checkbox_camera_widget)
        # layout.addRow(QLabel('Depth of field:'), self.checkbox_depth_of_field)
        # layout.addRow(QLabel('Eye dome lighting:'), self.checkbox_eye_dome_lighting)
        layout.addRow(QLabel('Hidden line removal:'), self.checkbox_hidden_line_removal)
        layout.addRow(QLabel('Parallel projection:'), self.checkbox_parallel_projection)
        layout.addRow(QLabel('Shadows:'), self.checkbox_shadows)
        layout.addRow(QLabel('SSAO:'), self.checkbox_ssao)
        layout.addRow(QLabel('Stereo rendering:'), self.checkbox_stereo_rendering)
        form.setLayout(layout)
        outer = QVBoxLayout()
        outer.addWidget(form)
        outer.addWidget(self.button_background_color)
        outer.addWidget(self.button_reset)
        outer.addStretch()
        self.setLayout(outer)

    def _set_defaults(self):
        self.checkbox_boundary_box.setChecked(False)
        self.checkbox_grid.setChecked(False)
        self.checkbox_camera_widget.setChecked(False)
        # self.checkbox_depth_of_field.setChecked(False)
        # self.checkbox_eye_dome_lighting.setChecked(False)
        self.checkbox_hidden_line_removal.setChecked(True)
        self.checkbox_parallel_projection.setChecked(False)
        self.checkbox_shadows.setChecked(True)
        self.checkbox_ssao.setChecked(True)
        self.checkbox_stereo_rendering.setChecked(False)
        # self.combo_lighting_mode.setCurrentText('LightKit')
        self.combo_anti_aliasing.setCurrentText('None')
        self.combo_interaction_style.setCurrentText('Trackball')
        self.button_background_color.set_current_color(QColor(255, 255, 255))




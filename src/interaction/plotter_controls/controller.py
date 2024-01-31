from PyQt5.QtCore import QObject
import pyvista as pv
from src.interaction.plotter_controls.view import PlotterSettingsView


class PlotterController(QObject):

    def __init__(self, plotter_settings: PlotterSettingsView, plotter: pv.Plotter, parent=None):
        super().__init__(parent)
        self.plotter_settings = plotter_settings
        self.plotter = plotter
        self._grid_actor = None
        self._link_actions()
        self._update_plotter()

    def _link_actions(self):
        ps = self.plotter_settings
        ps.combo_anti_aliasing.currentTextChanged.connect(self._toggle_anti_aliasing)
        # ps.combo_lighting_mode.currentTextChanged.connect(self._toggle_lighting_mode)
        ps.combo_interaction_style.currentTextChanged.connect(self._toggle_interaction_style)
        ps.checkbox_boundary_box.stateChanged.connect(self._toggle_boundary_box)
        ps.checkbox_grid.stateChanged.connect(self._toggle_grid)
        ps.checkbox_camera_widget.stateChanged.connect(self._toggle_camera_orientation_widget)
        # ps.checkbox_depth_of_field.stateChanged.connect(self._toggle_depth_of_field)
        # ps.checkbox_eye_dome_lighting.stateChanged.connect(self._toggle_eye_dome_lighting)
        ps.checkbox_hidden_line_removal.stateChanged.connect(self._toggle_hidden_line_removal)
        ps.checkbox_parallel_projection.stateChanged.connect(self._toggle_parallel_projection)
        ps.checkbox_shadows.stateChanged.connect(self._toggle_shadows)
        ps.checkbox_ssao.stateChanged.connect(self._toggle_ssao)
        ps.checkbox_stereo_rendering.stateChanged.connect(self._toggle_stereo_rendering)
        ps.button_background_color.color_changed.connect(self._update_background_color)

    def _update_plotter(self):
        self._toggle_anti_aliasing()
        # self._toggle_lighting_mode()
        self._toggle_interaction_style()
        self._toggle_boundary_box()
        self._toggle_grid()
        self._toggle_camera_orientation_widget()
        # self._toggle_depth_of_field()
        # self._toggle_eye_dome_lighting()
        self._toggle_hidden_line_removal()
        self._toggle_parallel_projection()
        self._toggle_shadows()
        self._toggle_ssao()
        self._toggle_stereo_rendering()
        self._update_background_color()

    def _toggle_stereo_rendering(self):
        use_stereo_rendering = self.plotter_settings.checkbox_stereo_rendering.isChecked()
        if use_stereo_rendering:
            self.plotter.enable_stereo_render()
        else:
            self.plotter.disable_stereo_render()

    def _toggle_ssao(self):
        use_ssao = self.plotter_settings.checkbox_ssao.isChecked()
        if use_ssao:
            self.plotter.enable_ssao()
        else:
            self.plotter.disable_ssao()

    def _toggle_shadows(self):
        show_shadows = self.plotter_settings.checkbox_shadows.isChecked()
        if show_shadows:
            self.plotter.enable_shadows()
        else:
            self.plotter.disable_shadows()

    def _toggle_parallel_projection(self):
        use_pp = self.plotter_settings.checkbox_parallel_projection.isChecked()
        if use_pp:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def _toggle_hidden_line_removal(self):
        use_hlr = self.plotter_settings.checkbox_hidden_line_removal.isChecked()
        if use_hlr:
            self.plotter.enable_hidden_line_removal()
        else:
            self.plotter.disable_hidden_line_removal()

    def _toggle_eye_dome_lighting(self):
        use_edl = self.plotter_settings.checkbox_eye_dome_lighting.isChecked()
        if use_edl:
            self.plotter.enable_eye_dome_lighting()
        else:
            self.plotter.disable_eye_dome_lighting()

    def _toggle_depth_of_field(self):
        show_dof = self.plotter_settings.checkbox_depth_of_field.isChecked()
        if show_dof:
            self.plotter.enable_depth_of_field()
        else:
            self.plotter.disable_depth_of_field()

    def _toggle_camera_orientation_widget(self):
        show_widget = self.plotter_settings.checkbox_camera_widget.isChecked()
        if show_widget:
            self.plotter.add_camera_orientation_widget()
        else:
            self.plotter.clear_camera_widgets()

    def _toggle_grid(self):
        show_grid = self.plotter_settings.checkbox_grid.isChecked()
        if show_grid and self._grid_actor is None:
            self._grid_actor = self.plotter.show_bounds(
                xtitle='Longitude', ytitle='Latitude', ztitle='Elevation',
                show_zlabels=False,
                grid='front', location='outer',
                all_edges=True
            )
        if not show_grid and self._grid_actor is not None:
            self.plotter.remove_actor(self._grid_actor)
            self._grid_actor = None
            if not self.plotter_settings.checkbox_boundary_box.isChecked():
                self.plotter.remove_bounding_box()

    def _toggle_boundary_box(self):
        has_grid = self.plotter_settings.checkbox_grid.isChecked()
        if self.plotter_settings.checkbox_boundary_box.isChecked() and not has_grid:
            self.plotter.add_bounding_box()
        elif not has_grid:
            self.plotter.remove_bounding_box()

    def _update_background_color(self):
        color = self.plotter_settings.button_background_color.current_color
        self.plotter.set_background(color.name())

    def _toggle_anti_aliasing(self):
        mode = self.plotter_settings.combo_anti_aliasing.currentText()
        if mode == 'None':
            self.plotter.disable_anti_aliasing()
        else:
            if mode.startswith('MSAA'):
                mode, multi_samples = mode.split('-')
                multi_samples = int(multi_samples[:-1])
            else:
                multi_samples = None
            self.plotter.enable_anti_aliasing(aa_type=mode.lower(), multi_samples=multi_samples)

    def _toggle_lighting_mode(self):
        mode = self.plotter_settings.combo_lighting_mode.currentText()
        if mode == 'LightKit':
            self.plotter.enable_lightkit()
        elif mode == '3 lights':
            self.plotter.enable_3_lights()

    def _toggle_interaction_style(self):
        mode = self.plotter_settings.combo_interaction_style.currentText()
        action = getattr(self.plotter, 'enable_{}_style'.format(mode.lower().replace(' ', '_')))
        action()

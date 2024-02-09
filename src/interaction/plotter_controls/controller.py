import logging

import numpy as np
from PyQt5.QtCore import QObject
import pyvista as pv
from PyQt5.QtGui import QColor

from src.interaction.plotter_controls.view import PlotterSettingsView
from src.model.solar_lighting_model import SolarLightingModel


class SolarLightingController(QObject):

    def __init__(self, plotter_settings, lighting_model, parent=None):
        super().__init__(parent)
        self.plotter_settings = plotter_settings
        self.lighting_model = lighting_model
        self.plotter_settings.solar_timestamp_changed.connect(self.set_solar_timestamp)
        self.plotter_settings.solar_location_changed.connect(self.set_solar_location)

    def set_solar_timestamp(self, timestamp):
        self.lighting_model.set_timestamp(timestamp)

    def set_solar_location(self, longitude, latitude):
        self.lighting_model.set_location((longitude, latitude))


class PlotterController(QObject):

    def __init__(self, plotter_settings: PlotterSettingsView, plotter: pv.Plotter, parent=None):
        super().__init__(parent)
        self.plotter_settings = plotter_settings
        self._plotter = plotter
        self._lighting_model = SolarLightingModel(np.datetime64('2020-01-01T06:00'), (0., 0.))
        self._lighting_controller = SolarLightingController(plotter_settings, self._lighting_model)
        self._grid_actor = None
        self._link_actions()
        self._reset_plotter()
        self.plotter_settings.set_defaults()

    @property
    def plotter(self):
        logging.info('Acting on plotter: PlotterController')
        return self._plotter

    def _link_actions(self):
        ps = self.plotter_settings
        ps.aa_changed.connect(self._toggle_anti_aliasing)
        ps.lighting_mode_changed.connect(self._toggle_lighting_mode)
        ps.interaction_style_changed.connect(self._toggle_interaction_style)
        ps.show_boundary_changed.connect(self._toggle_boundary_box)
        ps.show_grid_changed.connect(self._toggle_grid)
        ps.show_camera_widget_changed.connect(self._toggle_camera_orientation_widget)
        ps.ssao_changed.connect(self._toggle_ssao)
        ps.pp_changed.connect(self._toggle_parallel_projection)
        ps.stereo_render_changed.connect(self._toggle_stereo_render)
        ps.hlr_changed.connect(self._toggle_hidden_line_removal)
        ps.background_color_changed.connect(self._update_background_color)

    def _reset_plotter(self):
        plotter = self.plotter
        plotter.disable_anti_aliasing()
        plotter.remove_all_lights()
        plotter.remove_bounding_box()
        if self._grid_actor is not None:
            plotter.remove_actor(self._grid_actor)
        plotter.clear_camera_widgets()
        plotter.disable_ssao()
        if plotter.parallel_projection:
            plotter.disable_parallel_projection()
        plotter.disable_stereo_render()
        plotter.disable_hidden_line_removal()

    def _toggle_stereo_render(self, apply_stereo_rendering: bool):
        if apply_stereo_rendering:
            logging.info('Enabling stereo render')
            self.plotter.enable_stereo_render()
        else:
            logging.info('Disabling stereo render')
            self.plotter.disable_stereo_render()

    def _toggle_ssao(self, use_ssao: bool):
        if use_ssao:
            logging.info('Enabling SSAO')
            self.plotter.enable_ssao()
        else:
            logging.info('Disabling SSAO')
            self.plotter.disable_ssao()

    def _toggle_parallel_projection(self, apply_pp: bool):
        if apply_pp:
            logging.info('Enabling parallel projection')
            self.plotter.enable_parallel_projection()
        else:
            logging.info('Disabling parallel projection')
            self.plotter.disable_parallel_projection()

    def _toggle_hidden_line_removal(self, apply_hlr: bool):
        if apply_hlr:
            logging.info('Enabling hidden line removal')
            self.plotter.enable_hidden_line_removal()
        else:
            logging.info('Disabling hidden line removal')
            self.plotter.disable_hidden_line_removal()

    def _toggle_camera_orientation_widget(self, show_widget: bool):
        if show_widget:
            logging.info('Enabling camera widget')
            self.plotter.add_camera_orientation_widget()
        else:
            logging.info('Disabling camera widget')
            self.plotter.clear_camera_widgets()

    def _toggle_grid(self, show_grid: bool):
        if show_grid and self._grid_actor is None:
            logging.info('Creating grid actor')
            self._grid_actor = self.plotter.show_bounds(
                xtitle='Longitude', ytitle='Latitude', ztitle='Elevation',
                show_zlabels=False,
                grid='front', location='outer',
                all_edges=True
            )
        if not show_grid and self._grid_actor is not None:
            logging.info('Removing grid actor')
            self.plotter.remove_actor(self._grid_actor)
            self._grid_actor = None
            show_box = self.plotter_settings.checkbox_boundary_box.isChecked()
            self.plotter.remove_bounding_box()
            if show_box:
                self.plotter.add_bounding_box()

    def _toggle_boundary_box(self, show_box: bool):
        has_grid = self.plotter_settings.checkbox_grid.isChecked()
        if show_box and not has_grid:
            logging.info('Adding bunding box')
            color = pv.global_theme.color
            self.plotter.add_bounding_box()
        elif not has_grid:
            logging.info('Removing bounding box')
            self.plotter.remove_bounding_box()

    def _update_background_color(self, color: QColor):
        logging.info('Updating background color: {}'.format(color.name()))
        self.plotter.set_background(color.name())

    def _toggle_anti_aliasing(self, mode: str):
        logging.info('Disabling AA')
        self.plotter.disable_anti_aliasing()
        if mode != 'None':
            logging.info('Enabling AA: {}'.format(mode))
            if mode.startswith('MSAA'):
                mode, multi_samples = mode.split('-')
                multi_samples = int(multi_samples[:-1])
            else:
                multi_samples = None
            self.plotter.enable_anti_aliasing(aa_type=mode.lower(), multi_samples=multi_samples)

    def _toggle_lighting_mode(self, mode: str):
        logging.info('Removing scene lights')
        self.plotter.remove_all_lights()
        action = {
            'LightKit': self.plotter.enable_lightkit,
            '3 lights': self.plotter.enable_3_lights,
            'Solar lighting': self._enable_solar_lighting
        }.get(mode, None)
        if action is not None:
            logging.info('Setting lighting mode: {}'.format(mode))
            action()

    def _enable_solar_lighting(self):
        self.plotter.add_light(self._lighting_model.light)

    def _toggle_interaction_style(self, mode: str):
        logging.info('Setting interaction style: {}'.format(mode))
        action = getattr(self.plotter, 'enable_{}_style'.format(mode.lower().replace(' ', '_')))
        action()

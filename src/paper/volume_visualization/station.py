from enum import Enum
from typing import Union

import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QCheckBox, QDoubleSpinBox, QFormLayout
import xarray as xr
from src.paper.volume_visualization.color_lookup import InteractiveColorLookup
from src.paper.volume_visualization.multi_method_visualization import MultiMethodScalarVisualization, \
    MultiMethodSettingsView
from src.paper.volume_visualization.plotter_slot import PlotterSlot, MeshProperties, StationSiteProperties, \
    StationOnTerrainProperties
from src.paper.volume_visualization.scaling import ScalingParameters
from src.paper.volume_visualization.station_data import StationData
from src.paper.volume_visualization.station_data_representation import StationDataVisualization, \
    StationDataRepresentation


class StationRepresentationMode(Enum):
    STATION_SITE = 'station_site'
    STATION_ON_TERRAIN = 'station_on_terrain'


class _StationScalarRepresentation(StationDataRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData, color_lookup: InteractiveColorLookup,
            properties: MeshProperties, scaling: ScalingParameters = None, parent: QObject = None
    ):
        super().__init__(slot, station_data, properties, scaling, parent)
        self.color_lookup = color_lookup

    def update_scalar_colors(self):
        self.slot.update_scalar_colors(self.color_lookup.lookup_table)
        return self


class StationSiteRepresentation(_StationScalarRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData, color_lookup: InteractiveColorLookup,
            properties: StationSiteProperties = None,
            scaling: ScalingParameters = None,
            parent: QObject = None
    ):
        if properties is None:
            properties = StationSiteProperties()
        super().__init__(slot, station_data, color_lookup, properties, scaling, parent)

    def _update_mesh_scaling(self):
        self.mesh.points[:, -1] = self.station_data.compute_station_elevation(self.scaling).ravel()

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_sites(self.scaling)
        self.slot.show_scalar_mesh(self.mesh, self.color_lookup.lookup_table, self.properties, render=False)


class StationOnTerrainRepresentation(_StationScalarRepresentation):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData, color_lookup: InteractiveColorLookup,
            properties: StationOnTerrainProperties = None,
            scaling: ScalingParameters = None,
            parent: QObject = None
    ):
        if properties is None:
            properties = StationOnTerrainProperties()
        super().__init__(slot, station_data, color_lookup, properties, scaling, parent)

    def _update_mesh_scaling(self):
        z_site = self.station_data.compute_station_elevation(self.scaling).ravel()
        z_surf = self.station_data.compute_terrain_elevation(self.scaling).ravel()
        n = len(z_site)
        self.mesh.points[:n, -1] = z_site
        self.mesh.points[n:, -1] = z_surf

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_reference(self.scaling)
        self.slot.show_scalar_mesh(self.mesh, self.color_lookup.lookup_table, self.properties, render=False)


class StationScalarVisualization(MultiMethodScalarVisualization):

    def __init__(
            self,
            slot: PlotterSlot, station_data: StationData, color_lookup: InteractiveColorLookup,
            properties: MeshProperties, scaling: ScalingParameters = None, parent: QObject = None
    ):
        super().__init__(slot, color_lookup, properties, scaling, parent)
        self.station_data = station_data

    @property
    def representation_mode(self):
        return {
            StationSiteProperties: StationRepresentationMode.STATION_SITE,
            StationOnTerrainProperties: StationRepresentationMode.STATION_ON_TERRAIN
        }.get(type(self.properties))

    def set_properties(self, properties: MeshProperties, render: bool = True):
        return super().set_properties(properties)

    def _build_representation(self):
        properties_type = type(self.properties)
        if properties_type == StationSiteProperties:
            self.representation = StationSiteRepresentation(
                self.slot, self.station_data, self.color_lookup, self.properties, self.scaling
            )
        elif properties_type == StationOnTerrainProperties:
            self.representation = StationOnTerrainRepresentation(
                self.slot, self.station_data, self.color_lookup, self.properties, self.scaling
            )
        else:
            raise NotImplementedError()

    def update_data(self, new_data: Union[xr.Dataset, pd.DataFrame], render: bool = True):
        self.blockSignals(True)
        self.station_data.update_station_data(new_data)
        if self.is_visible():
            self.representation.update(render=render)
        self.blockSignals(False)
        return self


class StationSiteSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.build_handles()
        self._connect_signals()
        self._set_layout()

    def build_handles(self):
        self.spinner_point_size = QDoubleSpinBox(self)
        self.spinner_point_size.setRange(0.25, 100)
        self.spinner_point_size.setSingleStep(0.25)
        self.checkbox_points_as_spheres = QCheckBox(self)
        self.spinner_metallic = QDoubleSpinBox(self)
        self.spinner_metallic.setRange(0., 1.)
        self.spinner_metallic.setSingleStep(0.05)
        self.spinner_roughness = QDoubleSpinBox(self)
        self.spinner_roughness.setRange(0., 1.)
        self.spinner_roughness.setSingleStep(0.05)
        self.spinner_ambient = QDoubleSpinBox(self)
        self.spinner_ambient.setRange(0., 1.)
        self.spinner_ambient.setSingleStep(0.05)
        self.spinner_diffuse = QDoubleSpinBox(self)
        self.spinner_diffuse.setRange(0., 1.)
        self.spinner_diffuse.setSingleStep(0.05)
        self.spinner_specular = QDoubleSpinBox(self)
        self.spinner_specular.setRange(0., 1.)
        self.spinner_specular.setSingleStep(0.05)
        self.spinner_specular_power = QDoubleSpinBox(self)
        self.spinner_specular_power.setRange(0., 128.)
        self.spinner_specular_power.setSingleStep(0.5)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setRange(0., 1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.checkbox_lighting = QCheckBox(self)

    def _connect_signals(self):
        self.spinner_point_size.valueChanged.connect(self.settings_changed)
        self.checkbox_points_as_spheres.stateChanged.connect(self.settings_changed)
        self.spinner_metallic.valueChanged.connect(self.settings_changed)
        self.spinner_roughness.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.spinner_opacity.valueChanged.connect(self.settings_changed)
        self.checkbox_lighting.stateChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        self.setLayout(layout)

    def _build_form_layout(self):
        layout = QFormLayout()
        layout.addRow("Opacity:", self.spinner_opacity)
        layout.addRow("Point size:", self.spinner_point_size)
        layout.addRow("Points as spheres:", self.checkbox_points_as_spheres)
        layout.addRow("Metallic:", self.spinner_metallic)
        layout.addRow("Roughness:", self.spinner_roughness)
        layout.addRow("Ambient:", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power:", self.spinner_specular_power)
        layout.addRow("Lighting:", self.checkbox_lighting)
        return layout

    def apply_settings(self, settings: MeshProperties):
        self.spinner_point_size.setValue(settings.point_size)
        self.checkbox_points_as_spheres.setChecked(settings.render_points_as_spheres)
        self.spinner_metallic.setValue(settings.metallic)
        self.spinner_roughness.setValue(settings.roughness)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.spinner_opacity.setValue(settings.opacity)
        self.checkbox_lighting.setChecked(settings.lighting)
        return self

    def get_settings(self):
        return StationSiteProperties(
            self.spinner_metallic.value(),
            self.spinner_roughness.value(),
            self.spinner_point_size.value(),
            self.checkbox_points_as_spheres.isChecked(),
            self.spinner_opacity.value(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            self.checkbox_lighting.isChecked()
        )


class StationOnTerrainSettingsView(QWidget):

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.build_handles()
        self._connect_signals()
        self._set_layout()

    def build_handles(self):
        self.spinner_line_width = QDoubleSpinBox(self)
        self.spinner_line_width.setRange(0.25, 20)
        self.spinner_line_width.setSingleStep(0.25)
        self.checkbox_lines_as_tubes = QCheckBox('')
        self.spinner_metallic = QDoubleSpinBox(self)
        self.spinner_metallic.setRange(0., 1.)
        self.spinner_metallic.setSingleStep(0.05)
        self.spinner_roughness = QDoubleSpinBox(self)
        self.spinner_roughness.setRange(0., 1.)
        self.spinner_roughness.setSingleStep(0.05)
        self.spinner_ambient = QDoubleSpinBox(self)
        self.spinner_ambient.setRange(0., 1.)
        self.spinner_ambient.setSingleStep(0.05)
        self.spinner_diffuse = QDoubleSpinBox(self)
        self.spinner_diffuse.setRange(0., 1.)
        self.spinner_diffuse.setSingleStep(0.05)
        self.spinner_specular = QDoubleSpinBox(self)
        self.spinner_specular.setRange(0., 1.)
        self.spinner_specular.setSingleStep(0.05)
        self.spinner_specular_power = QDoubleSpinBox(self)
        self.spinner_specular_power.setRange(0., 128.)
        self.spinner_specular_power.setSingleStep(0.5)
        self.spinner_opacity = QDoubleSpinBox(self)
        self.spinner_opacity.setRange(0., 1.)
        self.spinner_opacity.setSingleStep(0.05)
        self.checkbox_lighting = QCheckBox(self)

    def _connect_signals(self):
        self.spinner_line_width.valueChanged.connect(self.settings_changed)
        self.checkbox_lines_as_tubes.stateChanged.connect(self.settings_changed)
        self.spinner_metallic.valueChanged.connect(self.settings_changed)
        self.spinner_roughness.valueChanged.connect(self.settings_changed)
        self.spinner_ambient.valueChanged.connect(self.settings_changed)
        self.spinner_diffuse.valueChanged.connect(self.settings_changed)
        self.spinner_specular.valueChanged.connect(self.settings_changed)
        self.spinner_specular_power.valueChanged.connect(self.settings_changed)
        self.spinner_opacity.valueChanged.connect(self.settings_changed)
        self.checkbox_lighting.stateChanged.connect(self.settings_changed)

    def _set_layout(self):
        layout = self._build_form_layout()
        self.setLayout(layout)

    def _build_form_layout(self):
        layout = QFormLayout()
        layout.addRow("Opacity:", self.spinner_opacity)
        layout.addRow("Line width:", self.spinner_line_width)
        layout.addRow("Lines as tubes:", self.checkbox_lines_as_tubes)
        layout.addRow("Metallic:", self.spinner_metallic)
        layout.addRow("Roughness:", self.spinner_roughness)
        layout.addRow("Ambient:", self.spinner_ambient)
        layout.addRow("Diffuse:", self.spinner_diffuse)
        layout.addRow("Specular:", self.spinner_specular)
        layout.addRow("Specular power:", self.spinner_specular_power)
        layout.addRow("Lighting:", self.checkbox_lighting)
        return layout

    def apply_settings(self, settings: MeshProperties):
        self.spinner_line_width.setValue(settings.line_width)
        self.checkbox_lines_as_tubes.setChecked(settings.render_lines_as_tubes)
        self.spinner_metallic.setValue(settings.metallic)
        self.spinner_roughness.setValue(settings.roughness)
        self.spinner_ambient.setValue(settings.ambient)
        self.spinner_diffuse.setValue(settings.diffuse)
        self.spinner_specular.setValue(settings.specular)
        self.spinner_specular_power.setValue(settings.specular_power)
        self.spinner_opacity.setValue(settings.opacity)
        self.checkbox_lighting.setChecked(settings.lighting)
        return self

    def get_settings(self):
        return StationOnTerrainProperties(
            self.spinner_line_width.value(),
            self.checkbox_lines_as_tubes.isChecked(),
            self.spinner_metallic.value(),
            self.spinner_roughness.value(),
            self.spinner_opacity.value(),
            self.spinner_ambient.value(),
            self.spinner_diffuse.value(),
            self.spinner_specular.value(),
            self.spinner_specular_power.value(),
            self.checkbox_lighting.isChecked()
        )


class StationScalarSettingsView(MultiMethodSettingsView):

    def __init__(self, use_sites=True, use_station_on_terrain=True, parent=None):
        defaults = {
            StationRepresentationMode.STATION_SITE: StationSiteProperties(),
            StationRepresentationMode.STATION_ON_TERRAIN: StationOnTerrainProperties(),
        }
        view_mapping = {}
        labels = {}
        if use_sites:
            view_mapping[StationRepresentationMode.STATION_SITE] = StationSiteSettingsView
            labels[StationRepresentationMode.STATION_SITE] = 'station site'
        if use_station_on_terrain:
            view_mapping[StationRepresentationMode.STATION_ON_TERRAIN] = StationOnTerrainSettingsView
            labels[StationRepresentationMode.STATION_ON_TERRAIN] = 'station on terrain'
        super().__init__(view_mapping, defaults, labels, parent)

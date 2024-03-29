from enum import Enum
from typing import Dict, Type

import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QComboBox, QStackedLayout, QCheckBox, QVBoxLayout

from src.paper.volume_visualization.color_lookup import InteractiveColorLookup
from src.paper.volume_visualization.plotter_slot import PlotterSlot, ActorProperties
from src.paper.volume_visualization.scaling import VolumeVisual, ScalingParameters


class MultiMethodScalarVisualization(VolumeVisual):

    def __init__(
            self,
            slot: PlotterSlot, color_lookup: InteractiveColorLookup,
            properties: ActorProperties, scaling: ScalingParameters = None, parent: QObject = None
    ):
        super().__init__(parent)
        self.slot = slot
        if scaling is None:
            scaling = ScalingParameters(1., 1., False, False, False)
        self.scaling = scaling
        self.properties = properties
        self.color_lookup = color_lookup
        self.representation: VolumeVisual = None
        self.color_lookup.lookup_table_changed.connect(self.on_color_lookup_changed)

    @property
    def representation_mode(self):
        raise NotImplementedError()

    def _build_representation(self):
        raise NotImplementedError()

    def get_plotter(self) -> pv.Plotter:
        return self.slot.plotter

    def on_color_lookup_changed(self):
        if self.representation is not None:
            self.representation.update_scalar_colors()

    def clear(self, render: bool = True):
        if self.representation is not None:
            self.representation.clear(render=render)
            self.representation = None
        self.visibility_changed.emit(False)
        return self

    def show(self, render: bool = True):
        if self.representation is not None:
            return self
        self._build_representation()
        self.representation.show(render=render)
        self.visibility_changed.emit(True)
        return self

    def change_representation(self, properties: ActorProperties, render: bool = True):
        self.properties = properties
        if self.representation is not None:
            self.blockSignals(True)
            self.clear(render=False)
            self.show(render=render)
            self.blockSignals(False)
        return self

    def set_properties(self, properties: ActorProperties, render: bool = True):
        self.properties = properties
        if self.representation is not None:
            self.representation.set_properties(properties, render=render)
        return self

    def set_scaling(self, scaling: ScalingParameters, render: bool = True):
        self.scaling = scaling
        if self.representation is not None:
            self.representation.set_scaling(scaling, render=render)
        return self

    def is_visible(self):
        return self.representation is not None


class MultiMethodSettingsView(QWidget):

    settings_changed = pyqtSignal()
    representation_changed = pyqtSignal()
    visibility_changed = pyqtSignal()

    def __init__(self, view_mapping: Dict[Enum, Type[QWidget]], defaults: Dict[Enum, ActorProperties], labels: Dict[Enum, str] = None, parent=None):
        super().__init__(parent)
        self.defaults = defaults
        self.combo_representation_type = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self.representation_views = {}
        if labels is None:
            labels = {}
        for key, view_type in view_mapping.items():
            view = view_type(self)
            label = labels.get(key, key.value)
            self.combo_representation_type.addItem(label, key)
            self.representation_views[key] = view
            self.interface_stack.addWidget(view)
        assert len(self.representation_views)
        self.checkbox_visible = QCheckBox(self)
        self.checkbox_visible.setChecked(True)
        self.checkbox_visible.setText('show')
        self._connect_signals()
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.checkbox_visible)
        layout.addWidget(self.combo_representation_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self):
        self.combo_representation_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)
        self.combo_representation_type.currentIndexChanged.connect(self.representation_changed)
        for key in self.representation_views:
            self.representation_views[key].settings_changed.connect(self.settings_changed)
        self.checkbox_visible.stateChanged.connect(self.visibility_changed)

    def get_representation_mode(self):
        return self.combo_representation_type.currentData()

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

    def apply_settings(self, settings: Dict[Enum, ActorProperties], use_defaults=False):
        for key in self.defaults:
            if key in settings:
                props = settings[key]
            elif use_defaults:
                props = self.defaults.get(key)
            else:
                props = None
            if props is not None and key in self.representation_views:
                self.representation_views[key].apply_settings(props)
        return self

    def apply_visibility(self, visible):
        self.checkbox_visible.setChecked(visible)
        return self

    def get_visibility(self):
        return self.checkbox_visible.isChecked()


class MultiMethodVisualizationController(QObject):

    def __init__(self, view: MultiMethodSettingsView, model: MultiMethodScalarVisualization, parent=None):
        super().__init__(parent)
        self.view = view
        self.model = model
        self._synchronize_settings()
        self.view.settings_changed.connect(self.on_settings_changed)
        self.view.representation_changed.connect(self.on_representation_changed)
        self.view.visibility_changed.connect(self.on_visibility_changed)
        self.model.visibility_changed.connect(self.view.apply_visibility)

    def _synchronize_settings(self):
        self.view.apply_settings({self.model.representation_mode: self.model.properties}, use_defaults=True)
        self.view.apply_visibility(self.model.is_visible())

    def on_visibility_changed(self):
        self.model.set_visible(self.view.get_visibility())

    def on_settings_changed(self):
        settings = self.view.get_settings()
        self.model.set_properties(settings)
        return self

    def on_representation_changed(self):
        settings = self.view.get_settings()
        self.model.change_representation(settings)
        return self

from dataclasses import fields
from enum import Enum
from typing import Union

from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QWidget, QTabWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTreeWidgetItem, \
    QTreeWidget, QMenu, QAction, QDialog

from src.interaction.domain_selection import DomainSelectionController
from src.interaction.downscaling.methods import CreateDownscalerDialog
from src.model.downscaling.methods import AdaptiveLapseRateDownscaler, FixedLapseRateDownscaler
from src.model.downscaling.pipeline import DownscalingPipelineModel
from src.model.interface import PropertyModel, PropertyModelUpdateError


def build_properties_tree(downscaler: DownscalingPipelineModel):

    def read_properties(properties_: PropertyModel.Properties, item: QTreeWidgetItem):
        for field in fields(properties_):
            value = getattr(properties_, field.name)
            if isinstance(value, PropertyModel.Properties):
                field_item = QTreeWidgetItem([field.name])
                field_item.setFlags(Qt.ItemIsEnabled)
                read_properties(getattr(properties_, field.name), field_item)
            else:
                field_item = QTreeWidgetItem([field.name, str(value)])
                field_item.setFlags(Qt.ItemIsEnabled)
            item.addChild(field_item)

    display_name = downscaler.downscaler.__class__.__name__
    properties = downscaler.downscaler.properties
    root_item = QTreeWidgetItem([display_name, downscaler.uid])
    read_properties(properties, root_item)

    return root_item


class DownscalerRegister(object):

    def __init__(self):
        self.pipelines = {}

    def add_downscaling_pipeline(self, pipeline: DownscalingPipelineModel):
        self.pipelines[pipeline.uid] = pipeline
        return self

    def remove_pipeline(self, pipeline: Union[DownscalingPipelineModel, str]):
        if isinstance(pipeline, DownscalingPipelineModel):
            uid = pipeline.uid
        else:
            uid = pipeline
        del self.pipelines[uid]
        return self

    def clear(self):
        self.pipelines.clear()
        return self


class DownscalerRegisterView(QWidget):

    reset_requested = pyqtSignal()
    new_downscaler_requested = pyqtSignal()

    def __init__(self, parent=None):
        super(DownscalerRegisterView, self).__init__(parent)
        self.tabs = QTabWidget(self)
        self.parameter_view = QTreeWidget(self.tabs)
        self.parameter_view.setColumnCount(2)
        self.parameter_view.setHeaderLabels(['Object', 'Value'])
        self.parameter_view.setAlternatingRowColors(True)
        self.parameter_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fields_view = QTreeWidget(self.tabs)
        self.tabs.addTab(self.parameter_view, 'Parameters')
        self.tabs.addTab(self.fields_view, 'Fields')
        self.button_new_downscaler = QPushButton('New', self)
        self.button_reset = QPushButton('Reset', self)
        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        button_box = QHBoxLayout()
        button_box.addWidget(self.button_new_downscaler)
        button_box.addWidget(self.button_reset)
        layout.addLayout(button_box)
        self.setLayout(layout)


class DownscalerRegisterController(QObject):

    register_reset = pyqtSignal()
    downscaler_changed = pyqtSignal(str)

    def __init__(self, view: DownscalerRegisterView, model: DownscalerRegister, domain_selection: DomainSelectionController, parent=None):
        super(DownscalerRegisterController, self).__init__(parent)
        self.view = view
        self.view.parameter_view.customContextMenuRequested.connect(self._open_context_menu)
        self.model = model
        self.domain_controls = domain_selection
        self.view.button_new_downscaler.clicked.connect(self._on_new_downscaler_requested)
        self.view.button_reset.clicked.connect(self._on_reset_requested)

        self.edit_downscaler_action = QAction('Edit downscaler properties', self)
        self.edit_downscaler_action.triggered.connect(self._on_edit_downscaler_requested)

    def _open_context_menu(self, position):
        items = self.view.parameter_view.selectedItems()
        if items:
            menu = QMenu()
            menu.addAction(self.edit_downscaler_action)
            menu.addAction('Delete downscaler')
            menu.exec_(self.view.parameter_view.viewport().mapToGlobal(position))

    def _on_edit_downscaler_requested(self):
        items = self.view.parameter_view.selectedItems()
        if len(items) != 1:
            raise Exception('[ERROR] Multiple selection not supported here.')
        item = items[0]
        while item.parent():
            item = item.parent()
        pipeline_uid = item.data(1, Qt.DisplayRole)
        pipeline = self.model.pipelines[pipeline_uid]
        downscaler = pipeline.downscaler
        assert downscaler is not None
        settings = downscaler.properties
        dialog = CreateDownscalerDialog(parent=self.view)
        dialog.update_settings(settings)
        if dialog.exec_():
            settings = dialog.get_settings()
            if downscaler.supports_update(settings):
                downscaler.set_properties(settings)
            else:
                type_ = type(settings)
                ds_cls = {
                    AdaptiveLapseRateDownscaler.Properties: AdaptiveLapseRateDownscaler,
                    FixedLapseRateDownscaler.Properties: FixedLapseRateDownscaler,
                }[type_]
                downscaler = ds_cls.from_settings(settings, self.domain_controls.model_lr.data_store)
                pipeline.set_downscaler(downscaler)
            pipeline.update()
            self._update_parameter_view()
        self.downscaler_changed.emit(pipeline_uid)

    def _on_new_downscaler_requested(self):
        dialog = CreateDownscalerDialog(parent=self.view)
        if dialog.exec_():
            settings = dialog.get_settings()
            type_ = type(settings)
            pipeline = DownscalingPipelineModel(self.domain_controls.model_lr, self.domain_controls.model_hr)
            ds_cls = {
                AdaptiveLapseRateDownscaler.Properties: AdaptiveLapseRateDownscaler,
                FixedLapseRateDownscaler.Properties: FixedLapseRateDownscaler,
            }[type_]
            downscaler = ds_cls.from_settings(settings, self.domain_controls.model_lr.data_store)
            pipeline.set_downscaler(downscaler)
            pipeline.update()
            self.model.add_downscaling_pipeline(pipeline)
            self._add_tree_items_for_pipeline(pipeline)
        return self

    def _update_parameter_view(self):
        self.view.parameter_view.clear()
        for pipeline in self.model.pipelines.values():
            self._add_tree_items_for_pipeline(pipeline)

    def _add_tree_items_for_pipeline(self, pipeline):
        item = build_properties_tree(pipeline)
        self.view.parameter_view.addTopLevelItem(item)

    def _on_reset_requested(self):
        self.view.parameter_view.clear()
        self.view.fields_view.clear()
        self.model.clear()
        self.register_reset.emit()


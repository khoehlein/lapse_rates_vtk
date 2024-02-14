import uuid
from typing import Dict, Any

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QDialog

from src.interaction.downscaling.interface import NeighborhoodLookup, Downscaler, DomainController, DownscalerController


class DownscalerOutput(object):
    pass


class DownscalingPipeline(object):

    def __init__(
            self,
            uid: str = None,
            lookup: NeighborhoodLookup,
            downscaler: Downscaler
    ):
        if uid is None:
            uid = str(uuid.uuid4())
        self.uid = uid
        self.neighborhood_lookup = neighborhood_lookup
        self._neighbor_data = None
        self.downscaler = downscaler
        self._output_data: DownscalerOutput = None


class NeighborhoodLookupController:

    @classmethod
    def from_settings(cls, settings, data_store):
        raise NotImplementedError()


class PipelineController(object):

    def from_settings(self, settings: Dict[str, Any], data_store):
        lookup_controls = NeighborhoodLookupController.from_settings(settings, data_store)
        downscaler_controls = DownscalerController.from_settings(settings)
        pipeline = DownscalingPipeline()

    def __init__(self, pipeline: DownscalingPipeline):
        self.pipeline = pipeline




class DownscalerRegistry(object):

    def __init__(self):
        self.pipelines = {}

    def add_pipeline(self, pipeline: DownscalingPipeline):
        self.pipelines[pipeline.uid] = pipeline

    def __getattr__(self, uid: str) -> DownscalingPipeline:
        return self.pipelines[uid]


class DownscalerRegistryView(QWidget):
    new_pipeline_requested = pyqtSignal()


class DownscalerCreationDialog(QDialog):

    def get_settings(self) -> Dict[str, Any]:
        raise NotImplementedError()


class DownscalerRegistryController(QObject):

    def __init__(
            self,
            view: DownscalerRegistryView,
            model: DownscalerRegistry,
            domain_controller: DomainController,
            parent=None
    ):
        super().__init__(parent)
        self.view = view
        self.view.new_pipeline_requested.connect(self._on_new_pipeline_requested)
        self.model = model
        self.domain_controller = domain_controller
        self.domain_controller.domain_changed.connect(self._on_domain_changed)

    def _on_new_pipeline_requested(self):
        dialog = DownscalerCreationDialog(self.view)
        if dialog.exec():
            settings = dialog.get_settings()


    def _build_pipeline_interface(self, settings):



class DownscalerDataStore(object):

    def __init__(self):
        self.datasets = {}

    def add_dataset(self, )
    def get_dataset(self, uid):
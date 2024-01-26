import logging

import numpy as np
from PyQt5.QtCore import QObject

from src.interaction.pyvista_display.view import PyvistaView
from src.interaction.settings_menu import SettingsViewTabbed
from src.model.data_store.world_data import WorldData, DomainData
from src.model.downscaling import DownscalerModel, LapseRateDownscalerProperties, LapseRateDownscaler
from src.model.geometry import DomainBounds, SurfaceDataset, Coordinates, lat_lon_system, LocationBatch, TriangleMesh
from src.model.neighborhood_lookup.interface import NeighborhoodLookupModel
from src.model.neighborhood_lookup.knn_lookup import KNNNeighborhoodProperties, KNNNeighborhoodLookup
from src.model.neighborhood_lookup.radial_lookup import RadialNeighborhoodProperties, RadialNeighborhoodLookup


class DownscalingPipeline(QObject):

    def __init__(
            self,
            data_store: WorldData,
            domain_bounds: DomainBounds = None,
            neighborhood_properties=None,
            downscaler_properties=None,
            parent=None
    ):
        super().__init__(parent)

        self.data_store = data_store
        self.domain_bounds = domain_bounds
        self.neighborhood_properties = neighborhood_properties
        self.downscaler_properties = downscaler_properties

        self.neighborhood_lookup: NeighborhoodLookupModel = None
        self.downscaler: DownscalerModel = None

        self._domain_data = None
        self._neighborhood_graph = None
        self._neighborhood_samples = None
        self._downscaler_output = None

    def set_data_store(self, data_store: WorldData):
        self.data_store = data_store
        if self.neighborhood_lookup is not None:
            self.neighborhood_lookup.set_source_data(self.data_store.get_lowres_land_sea_data())
        self._domain_data = None
        self._neighborhood_samples = None

    def set_domain_bounds(self, domain_bounds: DomainBounds):
        self.domain_bounds = domain_bounds
        self._domain_data = None

    def set_neighborhood_properties(self, properties):
        self.neighborhood_properties = properties
        if self.neighborhood_lookup is not None:
            try:
                self.neighborhood_lookup.set_neighborhood_properties(properties)
            except AssertionError:
                self._build_neighborhood_lookup()
        else:
            self._build_neighborhood_lookup()
        self._neighborhood_graph = None

    def _build_neighborhood_lookup(self):
        if isinstance(self.neighborhood_properties, RadialNeighborhoodProperties):
            self.neighborhood_lookup = RadialNeighborhoodLookup()
        elif isinstance(self.neighborhood_properties, KNNNeighborhoodProperties):
            self.neighborhood_lookup = KNNNeighborhoodLookup()
        else:
            raise NotImplementedError()
        self.neighborhood_lookup.set_source_data(self.data_store.get_lowres_land_sea_data())
        self.neighborhood_lookup.set_neighborhood_properties(self.neighborhood_properties)
        self._neighborhood_graph = None

    def set_downscaler_properties(self, properties):
        self.downscaler_properties = properties
        if self.downscaler is not None:
            try:
                self.downscaler.set_downscaler_properties(properties)
            except AssertionError:
                self._build_downscaler()
        else:
            self._build_downscaler()
        self._downscaler_output = None

    def _build_downscaler(self):
        if isinstance(self.downscaler_properties, LapseRateDownscalerProperties):
            self.downscaler = LapseRateDownscaler(self)
        else:
            raise NotImplementedError()
        self.downscaler.set_downscaler_properties(self.downscaler_properties)
        self._downscaler_output = None

    def _update_domain_data(self):
        if self._domain_data is None:
            self._domain_data = self.data_store.query_domain_data(self.domain_bounds)
            self._neighborhood_graph = None

    def _update_neighborhood_graph(self):
        self._update_domain_data()
        if self._neighborhood_graph is None:
            query_locations = self._domain_data.surface_mesh_lr.locations
            self._neighborhood_graph = self.neighborhood_lookup.query_neighborhood(query_locations)
            self._neighborhood_samples = None

    def _update_neighborhood_samples(self):
        self._update_neighborhood_graph()
        if self._neighborhood_samples is None:
            self._neighborhood_samples = self.data_store.query_sample_data(self._neighborhood_graph)
            self._downscaler_output = None

    def _update_downscaler_output(self):
        self._update_domain_data()
        self._update_neighborhood_samples()
        if self._downscaler_output is None:
            target = self._domain_data.get_lowres_orography()
            self._downscaler_output = self.downscaler.compute_temperatures(target, self._neighborhood_samples)

    def update(self):
        self._update_downscaler_output()
        logging.info('Pipeline update completed')
        return self

    def get_output(self):
        x = np.array([1, 2, 1, 2])
        y = np.array([1, 1, 2, 2])
        coords = Coordinates(lat_lon_system, x, y)
        locs = LocationBatch(coords)
        mesh = TriangleMesh(locs, np.array([[0, 1, 2], [1, 2, 3]]))
        return SurfaceDataset(mesh, np.array([1, 2, 3, 5]) * 4000)

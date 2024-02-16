from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, Optional, List, Any

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from src.interaction.downscaling.data_store import DomainData, GlobalData
from src.interaction.downscaling.domain_selection import DomainSelectionModel, DomainSelectionController
from src.interaction.downscaling.geometry import Coordinates, LocationBatch
from src.model.neighborhood_lookup.neighborhood_graphs import NeighborhoodGraph, UniformNeighborhoodGraph, \
    RadialNeighborhoodGraph
from src.model.visualization.interface import PropertyModel, PropertyModelUpdateError, PropertyModelView, \
    PropertyModelController


class GridConfiguration(Enum):
    O1280 = 'o1280'
    O8000 = 'o8000'


class SurfaceFieldType(Enum):
    T2M = 't2m'
    T2M_INTERPOLATION = 't2m_interpolation'
    T2M_DIFFERENCE = 't2m_difference'
    LAPSE_RATE = 'lapse_rate'
    LAPSE_RATE_INTERPOLATION = 'lapse_rate_interpolation'
    Z = 'z'
    Z_INTERPOLATION = 'z_interpolation'
    Z_DIFFERENCE = 'z_difference'
    LSM = 'lsm'


class VolumeFieldType(Enum):
    Z_QUANTILES = 'z_quantiles'
    T2M_VOLUME = 't2m_volume'


class OutputDataset(object):

    def __init__(self, parent=None, reference=None):
        self.parent = parent
        self.reference = reference
        self.groups = {}
        self.surface_fields = {}
        self.volume_fields = {}

    def get_group(self, key: Any):
        if key not in self.groups:
            raise RuntimeError()
        return self.groups[key]

    def create_group(self, key: Any, reference=None):
        if key in self.groups:
            raise RuntimeError()
        group = OutputDataset(parent=self, reference=reference)
        self.groups[key] = group
        return group

    def add_surface_field(self, field_type: SurfaceFieldType, data: xr.DataArray):
        if field_type in self.surface_fields:
            raise RuntimeError(f'[ERROR] Error assigning surface field: field {field_type.name} already occupied.')
        self.surface_fields[field_type] = data
        return self

    def add_volume_field(self, field_type: VolumeFieldType, data: xr.Dataset):
        if field_type in self.volume_fields:
            raise RuntimeError(f'[ERROR] Error assigning surface field: field {field_type.name} already occupied.')
        assert 'z' in data.data_vars, \
            f'[ERROR] Error in assigning volume field {field_type.name}: parameter z is not defined.'
        self.volume_fields[field_type] = data
        return self


class InterpolationType(Enum):
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    BARYCENTRIC = 'barycentric'


class InterpolationMethod(object):

    def __init__(self, source: DomainData):
        self.source = source

    def set_source(self, source: DomainData):
        self.source = source
        return self

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables: Optional[List[str]] = None):
        raise NotImplementedError()


class NearestNeighborInterpolation(InterpolationMethod):

    def __init__(self, source: DomainData):
        super(NearestNeighborInterpolation, self).__init__(source)
        self._build_search_structure()

    def set_source(self, source: DomainData):
        super().set_source(source)
        self._build_search_structure()
        return self

    @staticmethod
    def _to_search_structure_coords(coords: Coordinates):
        return coords.as_xyz().values

    def _build_search_structure(self):
        self.search_structure = self.source.get_grid_lookup()

    def interpolate(self, target: LocationBatch, data: xr.Dataset, variables=None):
        xyz = self._to_search_structure_coords(target.coords)
        indices = self.search_structure.kneighbors(xyz, return_distance=False)
        data = data if variables is None else data[variables]
        return data.isel(values=indices)


class BarycentricInterpolation(object):
    pass


class DownscalerType(Enum):
    FIXED_LAPSE_RATE = 'fixed_lapse_rate'
    ADAPTIVE_LAPSE_RATE = 'adaptive_lapse_rate'
    NETWORK = 'network'


class DownscalingMethodModel(PropertyModel):

    class Properties(PropertyModel.Properties):
        pass

    def __init__(self):
        super().__init__(None)
        self.source_data: DomainData = None
        self.target_data: DomainData = None
        self.output: OutputDataset = None

    def set_data(self, source_data: DomainData, target_data: DomainData):
        self.source_data = source_data
        self.target_data = target_data
        self.output = None
        return self

    def update(self):
        raise NotImplementedError()

    def supports_update(self, properties):
        return isinstance(properties, self.Properties)


class DownscalingPipelineModel(object):

    def __init__(
            self,
            source_domain: DomainSelectionModel,
            target_domain: DomainSelectionModel,
    ):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.downscaler: DownscalingMethodModel = None

    @property
    def data(self):
        if self.downscaler is None:
            return None
        return self.downscaler.output

    def set_downscaler(self, downscaler: DownscalingMethodModel):
        self.downscaler = downscaler
        self.update_downscaler_data()
        return self

    def update_downscaler_data(self):
        self.downscaler.set_data(self.source_domain.data, self.target_domain.data)
        return self

    def update(self):
        self.downscaler.update()
        return self


class DownscalingMethodView(PropertyModelView):
    pass


class DownscalingMethodController(PropertyModelController):

    @classmethod
    def from_view(cls, view: DownscalingMethodView, pipeline: DownscalingPipelineModel) -> 'DownscalingMethodController':
        raise NotImplementedError()

    def __init__(self, view: DownscalingMethodView, model: DownscalingMethodModel, parent=None, apply_defaults=True):
        super().__init__(view, model, parent, apply_defaults)


class FixedLapseRateDownscaler(DownscalingMethodModel):
    pass


class FixedLapseRateDownscalerView(DownscalingMethodView):
    pass


class FixedLapseRateDownscalerController(DownscalingMethodController):

    @classmethod
    def from_view(cls, view: FixedLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'FixedLapseRateDownscalerController':
        model = FixedLapseRateDownscaler()
        return cls(view, model)


class _InterpolatedDownscaler(DownscalingMethodModel):

    @dataclass
    class Properties:
        interpolation: InterpolationType

    def __init__(self):
        super().__init__()
        self.interpolator = None
        self._interpolators = {
            InterpolationType.NEAREST_NEIGHBOR: NearestNeighborInterpolation,
            InterpolationType.BARYCENTRIC: BarycentricInterpolation,
        }

    def set_data(self, source_data: DomainData, target_data: DomainData):
        super().set_data(source_data, target_data)
        self.interpolator = None

    def set_properties(self, properties: '_InterpolatedDownscaler.Properties') -> '_InterpolatedDownscaler':
        if self.properties == properties:
            return self
        super().set_properties(properties)
        self.interpolator = None
        return self

    def update(self):
        if self.properties is None or self.source_data is None:
            raise RuntimeError('[ERROR] Error updating interpolated downscaler')
        self._update_interpolator()
        return self

    def _update_interpolator(self):
        interpolation_type = self.properties.interpolation
        if (self.interpolator is None) or (self.interpolator.type != interpolation_type):
            cls = self._interpolators.get(interpolation_type)
            self.interpolator = cls(self.source_data)
        else:
            self.interpolator.set_source(self.source_data)
        return self


class NeighborhoodType(Enum):
    RADIAL = 'radial'
    NEAREST_NEIGHBORS = 'nearest_neighbors'


class TreeType(Enum):
    AUTO = 'auto'
    KD_TREE = 'kd_tree'
    BALL_TREE = 'ball_tree'
    BRUTE = 'brute'


class NeighborhoodModel(PropertyModel):

    @dataclass
    class Properties(PropertyModel.Properties):
        neighborhood_type: NeighborhoodType
        neighborhood_size: Union[int, float]
        tree_type: TreeType
        num_jobs: int # 1 for single-process, -1 for all processors
        lsm_threshold: float


    def __init__(self, data_store: GlobalData):
        super().__init__(None)
        self.data_store = data_store
        self.domain: DomainData = None

        self.search_structure = None
        self.data: xr.Dataset = None
        self.graph: NeighborhoodGraph = None

        self._actions = {
            NeighborhoodType.NEAREST_NEIGHBORS: self._query_k_nearest_neighbors,
            NeighborhoodType.RADIAL: self._query_radial_neighbors,
        }

    def update(self):
        if self.properties is None or self.domain is None:
            message = []
            if self.properties is None:
                message.append('properties not set')
            if self.domain is None:
                message.append('domain not set')
            message = ' and '.join(message)
            raise RuntimeError(f'[ERROR] Error in updating neighborhood data: {message}')
        if self.search_structure is None:
            self._build_search_structure()
        self.graph = self.query_neighbor_graph(self.domain.sites)
        self.data = self.data_store.query_link_data(self.graph.links)
        return self

    def tree_update_required(self, old_properties: 'NeighborhoodModel.Properties') -> bool:
        new_properties = self.properties
        if new_properties is None:
            return True
        if new_properties.lsm_threshold != old_properties.lsm_threshold:
            return True
        if new_properties.tree_type != old_properties.tree_type:
            return True
        if new_properties.num_jobs != old_properties.num_jobs:
            return True
        return False

    def set_properties(self, properties: 'NeighborhoodModel.Properties') -> 'NeighborhoodModel':
        old_properties = self.properties
        if old_properties == properties:
            return self
        super().set_properties(properties)
        if self.tree_update_required(old_properties):
            self.search_structure = None
        self.data = None
        self.graph = None
        return self

    def set_domain(self, domain: DomainData):
        self.domain = domain
        self.search_structure = None
        self.data = None
        self.graph = None
        return self

    def _build_search_structure(self):
        properties = self.properties
        self.search_structure = NearestNeighbors(
            algorithm=self.properties.tree_type.value,
            n_jobs=self.properties.num_jobs,
            leaf_size=100,
        )
        data = self.data_store.get_lsm()
        if data is None:
            raise RuntimeError('[ERROR] Error while building neighborhood lookup from properties: LSM unavailable.')
        mask = np.argwhere(data.values >= properties.lsm_threshold)
        data = data.isel(values=mask)
        coords = Coordinates.from_xarray(data).as_geocentric().values
        self.search_structure.fit(coords)

    def query_neighbor_graph(self, sites: LocationBatch) -> NeighborhoodGraph:
        action = self._actions.get(self.properties.neighborhood_type, None)
        if action is None:
            raise RuntimeError()
        return action(sites)

    def query_neighbor_data(self, sites: LocationBatch) -> Tuple[xr.Dataset, NeighborhoodGraph]:
        graph = self.query_neighbor_graph(sites)
        data = self.data_store.query_link_data(graph.links)
        return data, graph

    def _query_k_nearest_neighbors(self, locations: LocationBatch) -> UniformNeighborhoodGraph:
        return UniformNeighborhoodGraph.from_tree_query(
            locations, self.search_structure, self.properties.neighborhood_size)

    def _query_radial_neighbors(self, locations: LocationBatch) -> RadialNeighborhoodGraph:
        return RadialNeighborhoodGraph.from_tree_query(
            locations, self.search_structure, self.properties.neighborhood_size)


class NeighborhoodModelView(PropertyModelView):
    pass


class NeighborhoodModelController(PropertyModelController):

    @classmethod
    def from_view(cls, view: NeighborhoodModelView, data_store: GlobalData) -> 'NeighborhoodModelController':
        model = NeighborhoodModel(data_store)
        return cls(view, model)

    def default_settings(self):
        return NeighborhoodModel.Properties(
            neighborhood_type=NeighborhoodType.RADIAL,
            neighborhood_size=60.,
            tree_type=TreeType.AUTO,
            num_jobs=-1,
            lsm_threshold=0.5
        )


class LapseRateEstimator(PropertyModel):

    @dataclass
    class Properties:
        use_volume: bool
        use_weights: bool
        weight_scale_km: float
        min_num_neighbors: int
        fit_intercept: bool
        default_lapse_rate: float
        neighborhood: NeighborhoodModel.Properties

    def __init__(self, neighborhood_model: NeighborhoodModel):
        super().__init__(None)
        self.neighborhood = neighborhood_model
        self.source_data: DomainData = None
        self.target_data: DomainData = None
        self.output = None

    def set_data(self, source_data: DomainData, target_data: DomainData):
        self.source_data = source_data
        self.target_data = target_data
        self.neighborhood.set_domain(self.source_data)
        self.output = None
        return self

    def set_properties(self, properties: 'LapseRateEstimator.Properties') -> 'LapseRateEstimator':
        if properties == self.properties:
            return self
        super().set_properties(properties)
        self.neighborhood.set_properties(self.properties.neighborhood)
        self.output = None
        return self

    def update(self):
        if self.output is None:
            if self.neighborhood.data is None:
                self.neighborhood.update()
            self._compute_lapse_rates()
        return self

    def synchronize_properties(self):
        self.properties.neighborhood = self.neighborhood.properties
        return self

    def _compute_lapse_rates(self):
        site_data = self.source_data.data
        neighbor_data = self.neighborhood.data
        neighbor_graph = self.neighborhood.graph

        num_links = neighbor_graph.num_links
        site_id_at_link = neighbor_graph.links['location'].values
        distance_at_link = neighbor_graph.links['distance'].values
        t2m_at_site = site_data.t2m.values
        t2m_site_at_link = t2m_at_site[site_id_at_link]
        z_at_site = site_data.z.values
        z_site_at_link = z_at_site[site_id_at_link]
        split_indices = np.unique(np.cumsum(num_links))
        dt_around_site = np.split(neighbor_data.t2m.values - t2m_site_at_link, split_indices)
        dz_around_site = np.split(neighbor_data.z.values - z_site_at_link, split_indices)
        distance_around_site = np.split(distance_at_link, split_indices)
        lapse_rates = np.full_like(t2m_at_site, - 0.0065)
        mask = num_links > 0
        count = int(np.sum(mask))
        lapse_rates[mask] = np.fromiter(
            (
                self._estimate_lapse_rate(dt, dz, d)
                for dt, dz, d in zip(dt_around_site, dz_around_site, distance_around_site)
            ),
            count=count, dtype=float
        )

        coords = self.source_data.sites.coords.as_lat_lon()
        self.output = xr.DataArray(
            data=lapse_rates,
            dims=['values'],
            name=SurfaceFieldType.LAPSE_RATE.value,
            coords={
                'latitude': ('values', coords.y),
                'longitude': ('values', coords.x),
            },
        )

        return self.output

    def _estimate_lapse_rate(self, dt, dz, d):
        props = self.properties
        if len(dt) < props.min_num_neighbors:
            return props.default_lapse_rate
        lm = LinearRegression(fit_intercept=props.fit_intercept)
        if props.use_weights:
            weights = np.exp(-(d / (props.weight_scale_km * 1000.)) ** 2.)
        else:
            weights = None
        lm.fit(dz[:, None], dt, sample_weight=weights)
        return lm.coef_[0]


class LapseRateEstimatorView(PropertyModelView):

    def get_neighborhood_view(self) -> NeighborhoodModelView:
        raise NotImplementedError()


class LapseRateEstimatorController(PropertyModelController):

    @classmethod
    def from_view(cls, view: LapseRateEstimatorView, data_store: GlobalData):
        neighborhood_view = view.get_neighborhood_view()
        neighborhood_controller = NeighborhoodModelController.from_view(neighborhood_view, data_store)
        model = LapseRateEstimator(neighborhood_controller.model)
        return cls(view, model, neighborhood_controller)

    def __init__(
            self,
            view: LapseRateEstimatorView, model: LapseRateEstimator,
            neighborhood_controller: NeighborhoodModelController,
            parent=None, apply_defaults: bool = True
    ):
        super().__init__(view, model, parent, apply_defaults)
        self.neighborhood_controller = neighborhood_controller
        self.neighborhood_controller.model_changed.connect(self._on_neighborhood_changed)

    def _on_neighborhood_changed(self):
        self.model.synchronize_properties()
        self.model.output = None
        self.model_changed.emit()


class AdaptiveLapseRateDownscaler(_InterpolatedDownscaler):

    FIELDS_O1280 = [
        SurfaceFieldType.LAPSE_RATE,
        SurfaceFieldType.T2M,
        SurfaceFieldType.Z,
        SurfaceFieldType.LSM
    ]
    FIELDS_O8000 = [
        SurfaceFieldType.LAPSE_RATE_INTERPOLATION,
        SurfaceFieldType.T2M_INTERPOLATION, SurfaceFieldType.T2M_DIFFERENCE,
        SurfaceFieldType.Z, SurfaceFieldType.Z_INTERPOLATION, SurfaceFieldType.Z_DIFFERENCE,
        SurfaceFieldType.LSM
    ]

    @dataclass
    class Properties(_InterpolatedDownscaler.Properties):
        estimator: LapseRateEstimator.Properties

    def __init__(self, estimator: LapseRateEstimator):
        super().__init__()
        self.estimator = estimator

    def set_data(self, source_data: DomainData, target_data: DomainData):
        super().set_data(source_data, target_data)
        self.estimator.set_data(source_data, target_data)

    def set_properties(self, properties: 'AdaptiveLapseRateDownscaler.Properties') -> 'AdaptiveLapseRateDownscaler':
        if self.properties == properties:
            return self
        super().set_properties(properties)
        self.estimator.set_properties(properties.estimator)
        return self

    def update(self):
        if self.output is None:
            super().update()
            self.estimator.update()
            self._interpolate_estimator_outputs()
        return self

    def _interpolate_estimator_outputs(self):
        source_data = self.source_data.data
        target_data = self.target_data.data
        sites = self.target_data.sites
        interpolated_source = self.interpolator.interpolate(sites, source_data, variables=['t2m', 'z'])
        interpolated_lapse_rate = self.interpolator.interpolate(sites, self.estimator.output)

        dz = target_data.z.values - interpolated_source.z.values
        lapse_rate = interpolated_lapse_rate.values
        dt2m = lapse_rate * dz
        t2m_hr = interpolated_source.t2m.values + dt2m

        data_hr = target_data.assign({
            SurfaceFieldType.T2M.value: ('values', t2m_hr),
            SurfaceFieldType.T2M_INTERPOLATION: interpolated_source['t2m'].rename(SurfaceFieldType.T2M_INTERPOLATION),
            SurfaceFieldType.T2M_DIFFERENCE: ('values', dt2m),
            SurfaceFieldType.LAPSE_RATE.value: ('values', np.full_like(t2m_hr, lapse_rate)),
            SurfaceFieldType.Z_INTERPOLATION.value: (
            'values', interpolated_source['z'].rename(SurfaceFieldType.Z_INTERPOLATION.value)),
            SurfaceFieldType.Z_DIFFERENCE.value: ('values', dz),
        })

        output = OutputDataset()

        output_hr = output.get_group(GridConfiguration.O8000)
        for field_type in self.FIELDS_O8000:
            output_hr.add_surface_field(field_type, data_hr[field_type.value])

        output_lr = output.get_group(GridConfiguration.O1280)
        for field_type in self.FIELDS_O1280:
            output_lr.add_surface_field(field_type, source_data[field_type.value])

        self.output = output

    def synchronize_properties(self):
        self.properties.estimator = self.estimator.properties


class AdaptiveLapseRateDownscalerView(DownscalingMethodView):

    def get_estimator_view(self) -> LapseRateEstimatorView():
        raise NotImplementedError()


class AdaptiveLapseRateDownscalerController(DownscalingMethodController):

    @classmethod
    def from_view(cls, view: AdaptiveLapseRateDownscalerView, pipeline: DownscalingPipelineModel) -> 'AdaptiveLapseRateDownscalerController':
        estimator_view = view.get_estimator_view()
        global_data = pipeline.source_domain.data_store
        estimator_controller = LapseRateEstimatorController.from_view(estimator_view, global_data)
        estimator: LapseRateEstimator = estimator_controller.model
        model = AdaptiveLapseRateDownscaler(estimator)
        return cls(view, model, estimator_controller)

    def __init__(
            self,
            view: AdaptiveLapseRateDownscalerView, model: AdaptiveLapseRateDownscaler,
            estimator_controller: LapseRateEstimatorController,
            parent=None, apply_defaults=True
    ):
        super().__init__(view, model, parent, apply_defaults)
        self.estimator_controller = estimator_controller
        self.estimator_controller.model_changed.connect(self._on_estimator_changed)

    def _on_estimator_changed(self):
        self.model.synchronize_properties()
        self.model.output = None
        self.model_changed.emit()





class NetworkDownscaler(DownscalingMethodModel):
    pass


class NetworkDownscalerView(DownscalingMethodView):
    pass


class NetworkDownscalerController(DownscalingMethodController):
    pass


class DownscalerFactory(object):

    def __init__(self, downscaler_type: DownscalerType):
        self.downscaler_type = downscaler_type
        self.controller_class = {
            DownscalerType.FIXED_LAPSE_RATE: FixedLapseRateDownscalerController,
            DownscalerType.ADAPTIVE_LAPSE_RATE: AdaptiveLapseRateDownscalerController,
            DownscalerType.NETWORK: NetworkDownscalerController,
        }.get(self.downscaler_type)
        self.view_class = {
            DownscalerType.FIXED_LAPSE_RATE: FixedLapseRateDownscalerView,
            DownscalerType.ADAPTIVE_LAPSE_RATE: AdaptiveLapseRateDownscalerView,
            DownscalerType.NETWORK: NetworkDownscalerView,
        }.get(self.downscaler_type)

    def build_from_view(self, downscaler_view: DownscalingMethodView) -> DownscalingMethodController:
        assert isinstance(downscaler_view, self.view_class)
        controller = self.controller_class.from_view(downscaler_view)
        return controller


class DownscalingPipelineView(QWidget):
    downscaler_type_changed = pyqtSignal(DownscalerType)

    def get_settings(self) -> DownscalingMethodModel.Properties:
        raise NotImplementedError()

    def get_current_downscaler_view(self):
        raise NotImplementedError()


class DownscalingPipelineController(QObject):

    def __init__(
            self,
            view: DownscalingPipelineView,
            model: DownscalingPipelineModel,
            downscaling_controller: DownscalingMethodController,
            data_controller: DomainSelectionController,
            parent=None
    ):
        super().__init__(parent)
        self.view = view
        self.view.downscaler_type_changed.connect(self._on_downscaler_type_changed)
        self.model = model
        self.data_controller = data_controller
        self.data_controller.domain_changed.connect(self._on_domain_changed)
        self.downscaling_controller = downscaling_controller

    def _on_downscaler_type_changed(self, downscaler_type: DownscalerType):
        controller = DownscalerFactory(downscaler_type).build_from_view(self.view.get_current_downscaler_view())
        downscaler = controller.model
        self.model.set_downscaler(downscaler)
        self.downscaling_controller = controller
        self.downscaling_controller.model_changed.connect(self._on_model_changed)
        self.model.downscaler.update()

    def _on_domain_changed(self):
        self.model.update_downscaler_data()
        self.model.downscaler.update()

    def _on_model_changed(self):
        self.model.downscaler.update()

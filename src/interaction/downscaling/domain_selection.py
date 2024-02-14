import xarray as xr

from src.interaction.downscaling.data_store import DataStore
from src.interaction.downscaling.geometry import OctahedralGrid, DomainLimits, DomainBoundingBox


class DomainData(object):

    def __init__(self, grid: OctahedralGrid, data_store: DataStore):
        self.bounding_box = None
        self.data_store = data_store
        self.grid = grid
        self.mesh = None
        self.data: xr.Dataset = None

    @property
    def sites(self):
        if self.mesh is None:
            return None
        return self.mesh.nodes

    def set_bounds(self, bounds: DomainLimits):
        if bounds is not None:
            self.bounding_box = DomainBoundingBox(bounds)
        else:
            self.bounding_box = None
        self._update()
        return self

    def _update(self):
        if self.bounding_box is not None:
            self.mesh = self.grid.get_mesh_for_subdomain(self.bounding_box)
            self.site_data = self.data_store.query_site_data(self.mesh.nodes)
        else:
            self.reset()
        return self

    def reset(self):
        return self.set_bounds(None)


DEFAULT_DOMAIN = DomainLimits(43., 47., 6., 12.)


class DomainSettingsView(object):
    domain_limits_changed = None

    def get_settings(self) -> DomainLimits:
        raise NotImplementedError()

    def update_settings(self, settings: DomainLimits):
        raise NotImplementedError()


class DomainController(object):
    domain_changed = None

    def __init__(
            self,
            settings_view: DomainSettingsView,
            domain_lr: DomainData, domain_hr: DomainData
    ):
        self.view = settings_view
        self.model_lr = domain_lr
        self.model_hr = domain_hr
        self.view.domain_limits_changed.connect(self._on_domain_changed)
        self.view.update_settings(DEFAULT_DOMAIN)

    def _on_domain_changed(self):
        self._synchronize_domain_settings()
        self.domain_changed.emit()

    def _synchronize_domain_settings(self):
        bounds = self.view.get_settings()
        self.model_lr.set_bounds(bounds)
        self.model_hr.set_bounds(bounds)
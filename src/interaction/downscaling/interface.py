

class DataStore():

    def query_domain_data(self, domain_bounds):
        raise NotImplementedError()


class DomainSelectionModel():

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.domain_bounds = None
        self.domain_data = None

    def set_bounds(self, bounds):
        self.domain_bounds = bounds
        self._update()
        return self

    def _update(self):
        if self.domain_bounds is not None:
            self.domain_data = self.data_store.query_domain_data(self.domain_bounds)
        return self

    def get_domain_data(self):
        return self.domain_data


class NeighborhoodModel():

    def __init__(self, domain_model: DomainSelectionModel):
        self.domain_model = domain_model
        self.lookup = None
        self.neighborhood_data = None

    def _update(self):
        if self.lookup is None:
            return self
        domain_data = self.domain_model.get_domain_data()



class DownscalerModel():

    def __init__(self, neighborhood: NeighborhoodModel, data_store):
        self.neighborhood = neighborhood
        self.data_store = data_store
        self.downscaler = None



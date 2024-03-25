from src.paper.volume_visualization.station_data_representation import StationDataRepresentation


class StationSiteReferenceVisualization(StationDataRepresentation):

    def _update_mesh_scaling(self):
        self.mesh.points[:, -1] = self.station_data.compute_station_elevation(self.scaling).ravel()

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_sites(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)


class StationOnTerrainReferenceVisualization(StationDataRepresentation):

    def _update_mesh_scaling(self):
        z_site = self.station_data.compute_station_elevation(self.scaling).ravel()
        z_surf = self.station_data.compute_terrain_elevation(self.scaling).ravel()
        n = len(z_site)
        self.mesh.points[:n, -1] = z_site
        self.mesh.points[n:, -1] = z_surf

    def _build_and_show_mesh(self):
        self.mesh = self.station_data.get_station_reference(self.scaling)
        self.slot.show_reference_mesh(self.mesh, self.properties, render=False)

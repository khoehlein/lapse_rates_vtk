import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from sklearn.neighbors import NearestNeighbors

from src.model.downscaling.neighborhood_graphs import RadialNeighborhoodGraph
from src.model.geometry import LocationBatch, Coordinates

path_to_station_data = "C:\\Users\\kevin\\PycharmProjects\\lapse_rates_vtk\\station_locations_nearest.csv"
path_to_lsm_o1280 = "C:\\Users\\kevin\\data\\ECMWF\\LSM_HRES_Sep2022.grib"
path_to_lsm_o8000 = "C:\\Users\\kevin\\data\\ECMWF\\lsm_from_watermask.nc"

station_data = pd.read_csv(path_to_station_data)
data_o1280 = xr.open_dataset(path_to_lsm_o1280, engine='cfgrib').lsm
data_o8000 = xr.open_dataset(path_to_lsm_o8000, engine='netcdf4').lsm
lsm_o8000 = data_o8000.isel(values=station_data['nearest_node_o8000']).values

station_data = station_data.loc[lsm_o8000 < 0.10]
lsm_o1280 = data_o1280.isel(values=station_data['nearest_node_o1280']).values
lsm_o8000 = data_o8000.isel(values=station_data['nearest_node_o8000']).values

tree = NearestNeighbors()
tree.fit(Coordinates.from_xarray(data_o1280).as_xyz().values)

neighbors = RadialNeighborhoodGraph.from_tree_query(LocationBatch(Coordinates.from_xarray(station_data)), tree, 18).links
neighbors['lsm'] = data_o1280.isel(values=neighbors['neighbor'].values).values
max_fraction = neighbors.groupby('location')['lsm'].max().values

fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
ax.scatter(max_fraction, lsm_o8000, alpha=0.1)
ax.set(xlabel='O1280', ylabel='O8000', xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
plt.tight_layout()
plt.show()
plt.close()

cmap = mpl.cm.get_cmap('RdYlBu_r')
bounds = [0, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(2, 1, figsize=(12, 15), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
ax = axs[0]
p = ax.scatter(station_data.longitude.values, station_data.latitude.values, c=max_fraction, alpha=1., cmap=cmap, norm=norm, s=5)
cbar = plt.colorbar(p, ax=ax, spacing='proportional')
cbar.set_label('O1280')
ax.coastlines()
ax.gridlines()
ax.set(ylabel='O1280', facecolor='gray')
ax = axs[1]
ax.scatter(station_data.longitude.values, station_data.latitude.values, c=lsm_o8000, alpha=1., cmap=cmap, norm=norm, s=5)
cbar = plt.colorbar(p, ax=ax, spacing='proportional')
cbar.set_label('O8000')
ax.coastlines()
ax.gridlines()
ax.set(ylabel='O8000', facecolor='gray')
plt.tight_layout()
plt.show()
plt.close()

station_data = station_data.loc[max_fraction < 0.10]

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
ax.scatter(station_data.longitude.values, station_data.latitude.values, alpha=1, s=5, color='red')
ax.set(xlim=(-180, 180), ylim=(-90, 90))
ax.coastlines()
ax.gridlines()
plt.tight_layout()
plt.show()
plt.close()

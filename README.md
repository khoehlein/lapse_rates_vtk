# lapse_rates_vtk
Visualization of near-surface temperatures and lapse rates using VTK

# Installation
```
conda create --name lapse_rates_vtk_new --solver=libmamba -c conda-forge python=3.10 xarray dask h5py netcdf4 zarr tqdm jupyterlab jupyterlab-lsp python-lsp-server eccodes cfgrib cartopy pyqt scikit-learn python-eccodes
pip install -r environment.txt
```

# Software architecture

# Data cleansing
- run `src/observations/compute_point_predictions.py` to export station predictions for plain HRES and corrected HRES
- run `src/observations/compute_station_corrections.py` with various confidence settings to compute RANSAC corrected station predictions based on HRES
- run `src/observations/blacklist/build_mask.py` to export filter outputs and masked observations  





import xarray as xr


t2m = xr.open_dataset("/mnt/ssd4tb/ECMWF/HRES_2m_temp_20211219.grib")
lnsp = xr.open_dataset("/mnt/ssd4tb/ECMWF/19Dec21_06Z_lnsp_O1280.grib")
q = xr.open_dataset("/mnt/ssd4tb/ECMWF/19Dec21_06Z_q_model_levels_O1280.grib")
z = xr.open_dataset("/mnt/ssd4tb/ECMWF/HRES_orog_o1279_2021-2022.grib")
lsm = xr.open_dataset("/mnt/ssd4tb/ECMWF/LSM_HRES_Sep2022.grib")


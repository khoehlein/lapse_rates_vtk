import functools
import os.path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

# from model.constants import THRESHOLD_LSM_LAND
# from model.visualization_model.py import load_temperatures
# from model.storage import load_orography, Orography, load_lsm, LandSeaMask

R_DRY_AIR = 287.0528 # J / (K * kg)
R_WATER_VAPOR = 461.52 # J / (K * kg)
GRAVITATION = 9.81 # m/s^2


@functools.lru_cache(maxsize=1)
def load_level_coefficients(min_level: int = 117, max_level: int = 137, return_xr=False):
    # Coefficients correspond to half levels.
    # Full levels are obtained by averaging subsequent levels as follows:
    #
    #     p_full[n] = (a[n] + a[n - 1]) / 2 + (b[n] + b[n - 1]) / 2 * p_surf
    #
    coeffs = pd.read_csv(os.path.join(os.path.dirname(__file__), 'ecmwf_hybrid_level_coeffs_l137.csv'), index_col=0)
    coeffs = coeffs.loc[int(min_level):int(max_level + 1), ['a', 'b', 'ph', 'pf']]
    if not return_xr:
        return coeffs
    return coeffs.to_xarray().rename_dims(n='level')


def compute_half_level_pressure(surface_pressure: np.ndarray, min_level=117, max_level=137):
    coeffs = load_level_coefficients(min_level=min_level, max_level=max_level, return_xr=True)
    a_half = coeffs['a'].values[:, None]
    b_half = coeffs['b'].values[:, None]
    p_half = a_half + b_half * surface_pressure
    return p_half


def compute_full_level_pressure(surface_pressure: np.ndarray, min_level=118, max_level=137):
    p_half = compute_half_level_pressure(surface_pressure, min_level - 1, max_level)
    p_full = (p_half[1:, :] + p_half[:-1, :]) / 2.
    return p_full


def compute_standard_surface_pressure(
        surface_height: np.ndarray,
        base_pressure: Union[float, np.ndarray] = 101325.,
        base_pressure_height: Union[float, np.ndarray] = 0.,
        lapse_rate: Union[float, np.ndarray] = 0.0065,
        base_temperature: Union[float, np.ndarray] = 288.15,
        base_temperature_height: Union[float, np.ndarray] = 0.
):
    surface_temperature = base_temperature - lapse_rate * (surface_height - base_temperature_height)
    temperature_at_base_pressure_height = base_temperature - lapse_rate * (base_pressure_height - base_temperature_height)
    reduced_temperature = surface_temperature / temperature_at_base_pressure_height
    alpha = GRAVITATION / (R_DRY_AIR * lapse_rate)
    return base_pressure * np.power(reduced_temperature, alpha)


def compute_approximate_level_height(
        level_pressure: np.ndarray,
        base_pressure: Union[float, np.ndarray] = 101325,
        base_pressure_height: Union[float, np.ndarray] = 0.,
        lapse_rate: Union[float, np.ndarray] = 0.0065,
        base_temperature: Union[float, np.ndarray] = 288.15,
        base_temperature_height: Union[float, np.ndarray] = 0.
):
    reduced_pressure = level_pressure / base_pressure
    alpha = (R_DRY_AIR * lapse_rate) / GRAVITATION
    temperature_at_base_pressure_height = base_temperature - lapse_rate * (base_pressure_height - base_temperature_height)
    delta_height = (temperature_at_base_pressure_height / lapse_rate) * (1. - np.power(reduced_pressure, alpha))
    return  base_pressure_height + delta_height


def compute_physical_level_height(
        surface_pressure: np.ndarray,
        surface_height: np.ndarray,
        temperature: np.ndarray,
        specific_humidity: np.ndarray,
):
    p_half = compute_half_level_pressure(surface_pressure)
    t_virtual = temperature * (1. + ((R_WATER_VAPOR / R_DRY_AIR) - 1.) * specific_humidity)
    delta_z = - R_DRY_AIR * t_virtual * np.log(p_half[:-1] / p_half[1:])
    delta_z = np.cumsum(delta_z[::-1], axis=0)[::-1]
    delta_z[:-1] += delta_z[1:]
    delta_z /= 2.
    height = delta_z / GRAVITATION + surface_height
    return height


def _compute_isa76_level_heights(orography: np.ndarray, min_level=118, max_level=137):
    p_surf = compute_standard_surface_pressure(orography)
    p_level = compute_full_level_pressure(p_surf, min_level=min_level, max_level=max_level)
    height = compute_approximate_level_height(p_level)
    return height


def _compute_isa76_t2m_level_height(t2m:np.ndarray, orography:np.ndarray, min_level=118, max_level=137):
    height_2m = orography + 2.
    p_surf = compute_standard_surface_pressure(orography, base_temperature=t2m, base_temperature_height=height_2m)
    p_level = compute_full_level_pressure(p_surf, min_level=min_level, max_level=max_level)
    height = compute_approximate_level_height(p_level, base_temperature=t2m, base_temperature_height=height_2m)
    return height


def _compute_sp_t2m_level_height(p_surf: np.ndarray, t2m: np.ndarray, orography: np.ndarray, min_level=118, max_level=137):
    height_2m = orography + 2.
    p_level = compute_full_level_pressure(p_surf, min_level=min_level, max_level=max_level)
    height = compute_approximate_level_height(
        p_level,
        base_pressure=p_surf, base_pressure_height=orography,
        base_temperature=t2m, base_temperature_height=height_2m
    )
    return height


def silly_interpolation(query, coords, data):
    # silly interpolation due to absence of efficient broadcasting interpolation solution in xarray
    # assuming that coords are ordered DESCENDING
    i_lower = ((coords > query).astype(int).sum(dim='level') - 1).clip(min=0, max=(coords.sizes['level'] - 2))
    i_upper = i_lower + 1
    c_lower = coords.isel(level=i_lower)
    c_upper = coords.isel(level=i_upper)
    a_upper = (query - c_lower) / (c_upper - c_lower)
    # circumvent interpolation to areas below the available 3d data by clamping,
    # admit linear extrapolation to altitude above the available 3d data
    a_upper = a_upper.clip(max=1.)
    v_lower = data.isel(level=i_lower)
    v_upper = data.isel(level=i_upper)
    out = v_lower + a_upper * (v_upper - v_lower)
    return out


def silly_interpolation_np(query, coords, data):
    # silly interpolation due to absence of efficient broadcasting interpolation solution in xarray
    # assuming that coords are ordered DESCENDING
    i_lower = np.clip((np.sum((coords > query).astype(int), axis=0) - 1), 0, (coords.shape[0] - 2))
    i_upper = i_lower + 1
    site_idx = np.arange(coords.shape[1])
    c_lower = coords[i_lower, site_idx]
    c_upper = coords[i_upper, site_idx]
    a_upper = (query - c_lower) / (c_upper - c_lower)
    # circumvent interpolation to areas below the available 3d data by clamping,
    # admit linear extrapolation to altitude above the available 3d data
    a_upper = np.clip(a_upper, a_min=None, a_max=1.)
    v_lower = data[i_lower, site_idx]
    v_upper = data[i_upper, site_idx]
    out = v_lower + a_upper * (v_upper - v_lower)
    return out
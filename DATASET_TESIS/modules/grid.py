"""
Definición y validación de la cuadrícula objetivo del proyecto basada en MUR SST.
"""

import numpy as np
import xarray as xr

def define_target_grid(ds_mur: xr.Dataset) -> xr.Dataset:
    """
    Extrae y valida la cuadrícula objetivo maestra a partir del dataset MUR SST.
    Dimensiones esperadas: lat=86, lon=96 (0.01° de resolución).
    """
    if "lat" not in ds_mur.coords or "lon" not in ds_mur.coords:
        raise ValueError("El dataset de referencia MUR no contiene coordenadas 'lat' y 'lon'.")
        
    lat_coords = ds_mur.lat.values
    lon_coords = ds_mur.lon.values
    
    grid = xr.Dataset(
        coords={
            "lat": ("lat", lat_coords, {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
            "lon": ("lon", lon_coords, {"units": "degrees_east", "standard_name": "longitude", "axis": "X"})
        },
        attrs={
            "description": "Cuadrícula maestra objetivo de downscaling basada en MUR SST v4.1",
            "lat_min": float(lat_coords.min()),
            "lat_max": float(lat_coords.max()),
            "lon_min": float(lon_coords.min()),
            "lon_max": float(lon_coords.max()),
            "lat_size": int(len(lat_coords)),
            "lon_size": int(len(lon_coords))
        }
    )
    return grid

def check_grid_uniformity(grid: xr.Dataset) -> dict:
    """
    Verifica la monotonicidad, uniformidad del espaciado y resolución de la cuadrícula.
    """
    lat = grid.lat.values
    lon = grid.lon.values
    
    lat_diff = np.diff(lat)
    lon_diff = np.diff(lon)
    
    lat_monotonic = bool(np.all(lat_diff > 0))
    lon_monotonic = bool(np.all(lon_diff > 0))
    
    lat_res_mean = float(np.mean(lat_diff))
    lon_res_mean = float(np.mean(lon_diff))
    
    lat_res_std = float(np.std(lat_diff))
    lon_res_std = float(np.std(lon_diff))
    
    lat_uniform = bool(lat_res_std < 1e-6)
    lon_uniform = bool(lon_res_std < 1e-6)
    
    stats = {
        "lat_count": len(lat),
        "lon_count": len(lon),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lat_monotonic": lat_monotonic,
        "lon_monotonic": lon_monotonic,
        "lat_res_mean": lat_res_mean,
        "lon_res_mean": lon_res_mean,
        "lat_uniform": lat_uniform,
        "lon_uniform": lon_uniform,
        "total_cells": int(len(lat) * len(lon))
    }
    return stats

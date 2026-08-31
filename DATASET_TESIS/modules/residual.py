"""
Módulo para interpolación bilineal de OISST (con halo) a la cuadrícula MUR y cálculo de residuales.
R(t, x, y) = SST_MUR(t, x, y) - SST_BIL(t, x, y) sobre ocean_mask_final == 1.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def interpolate_oisst_to_mur(oisst_da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    """
    Realiza la interpolación bilineal de NOAA OISST (con halo espacial) a las coordenadas exactas de MUR.
    
    Parámetros:
      oisst_da: xr.DataArray 2D de OISST con coordenadas (lat, lon) que abarcan el halo.
      target_lat: Coordenadas 1D de latitud MUR (86 puntos).
      target_lon: Coordenadas 1D de longitud MUR (96 puntos).
      
    Retorna:
      xr.DataArray con sst_bil en las coordenadas exactas de MUR [lat=86, lon=96].
    """
    src_lat = oisst_da.lat.values
    src_lon = oisst_da.lon.values
    src_vals = oisst_da.values
    
    # Asegurar que las coordenadas de origen estén en orden estrictamente ascendente para RegularGridInterpolator
    if src_lat[1] < src_lat[0]:
        src_lat = src_lat[::-1]
        src_vals = src_vals[::-1, :]
    if src_lon[1] < src_lon[0]:
        src_lon = src_lon[::-1]
        src_vals = src_vals[:, ::-1]
        
    # Crear interpolador bilineal (lineal en 2D)
    # bounds_error=False, fill_value=np.nan para manejar extrapolación segura si fuera necesario
    interp = RegularGridInterpolator(
        (src_lat, src_lon), 
        src_vals, 
        method="linear", 
        bounds_error=False, 
        fill_value=np.nan
    )
    
    # Crear malla de destino
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    query_points = np.column_stack((tgt_lat_grid.ravel(), tgt_lon_grid.ravel()))
    
    # Evaluar interpolación
    interpolated_vals_1d = interp(query_points)
    sst_bil_2d = interpolated_vals_1d.reshape((len(target_lat), len(target_lon))).astype(np.float32)
    
    # Construir DataArray con coordenadas exactas
    da_bil = xr.DataArray(
        sst_bil_2d,
        coords={
            "lat": ("lat", target_lat, {"units": "degrees_north"}),
            "lon": ("lon", target_lon, {"units": "degrees_east"})
        },
        dims=["lat", "lon"],
        attrs={
            "long_name": "SST OISST interpolada bilinealmente a la cuadrícula MUR",
            "units": "degree_C",
            "interpolation_method": "bilinear"
        }
    )
    return da_bil

def compute_residual(sst_mur: xr.DataArray, sst_bil: xr.DataArray, ocean_mask_final: np.ndarray) -> xr.DataArray:
    """
    Calcula el residual de temperatura superficial del mar:
      R = SST_MUR - SST_BIL
    
    Aplica la máscara final:
      - ocean_mask_final == 1: R
      - ocean_mask_final == 0 (tierra): NaN
    """
    mur_vals = sst_mur.values.astype(np.float32)
    bil_vals = sst_bil.values.astype(np.float32)
    
    residual_vals = mur_vals - bil_vals
    # Enmascarar tierra como NaN
    residual_vals[ocean_mask_final == 0] = np.nan
    
    da_residual = xr.DataArray(
        residual_vals,
        coords=sst_mur.coords,
        dims=sst_mur.dims,
        attrs={
            "long_name": "Residual SST (MUR - OISST_BIL)",
            "units": "degree_C",
            "formula": "R = SST_MUR - SST_BIL"
        }
    )
    return da_residual

def compute_residual_stats(residual_da: xr.DataArray) -> dict:
    """
    Calcula estadísticas descriptivas del residual únicamente sobre celdas válidas (océano).
    Retorna: min, max, mean, median, std, P1, P5, P95, P99.
    """
    vals = residual_da.values
    valid = vals[~np.isnan(vals)]
    
    if len(valid) == 0:
        return {
            "count": 0,
            "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan,
            "p1": np.nan, "p5": np.nan, "p95": np.nan, "p99": np.nan
        }
        
    return {
        "count": int(len(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "p1": float(np.percentile(valid, 1)),
        "p5": float(np.percentile(valid, 5)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99))
    }

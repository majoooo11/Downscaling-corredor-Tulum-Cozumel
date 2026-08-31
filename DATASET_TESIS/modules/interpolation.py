"""
Módulo de interpolación espacial exclusiva OISST -> MUR.
Responsabilidades:
  - Selección de OISST con halo espacial amplio para evitar efectos de borde.
  - Inspección de coordenadas nativas de OISST.
  - Diagnóstico de celdas terrestres NaN en OISST (~0.25°).
  - Estrategia A: Extensión costera en malla OISST de 0.25° con nearest-ocean -> Interpolación bilineal.
  - Estrategia B: Triangulación lineal 2D con fallback nearest fuera del convex hull.
  - Trazabilidad explícita mediante la máscara diagnóstica oisst_coastal_support_mask.
  - Verificación estricta de alineación de coordenadas.
  - Control de valores NaN y bordes de dominio.
"""

from typing import Tuple, Dict, Any
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import cKDTree

def select_oisst_with_halo(
    ds_oisst: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    halo: float = 0.5
) -> Tuple[xr.DataArray, Dict[str, Any]]:
    """
    Selecciona la región espacial de OISST que cubre el dominio objetivo más un halo de seguridad.
    """
    lat_var = "latitude" if "latitude" in ds_oisst.coords else "lat"
    lon_var = "longitude" if "longitude" in ds_oisst.coords else "lon"
    
    q_lat_min = lat_min - halo
    q_lat_max = lat_max + halo
    q_lon_min = lon_min - halo
    q_lon_max = lon_max + halo
    
    all_lons = ds_oisst[lon_var].values
    if np.max(all_lons) > 180 and q_lon_min < 0:
        sel_lon_min = q_lon_min + 360
        sel_lon_max = q_lon_max + 360
    else:
        sel_lon_min = q_lon_min
        sel_lon_max = q_lon_max
        
    sub_ds = ds_oisst.sel({
        lat_var: slice(q_lat_min, q_lat_max),
        lon_var: slice(sel_lon_min, sel_lon_max)
    })
    
    da = sub_ds["sst"].squeeze()
    
    rename_dict = {}
    if lat_var != "lat":
        rename_dict[lat_var] = "lat"
    if lon_var != "lon":
        rename_dict[lon_var] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)
        
    if np.max(da.lon.values) > 180:
        da = da.assign_coords(lon=(da.lon - 360))
        
    if da.lat.values[1] < da.lat.values[0]:
        da = da.reindex(lat=da.lat.values[::-1])
    if da.lon.values[1] < da.lon.values[0]:
        da = da.reindex(lon=da.lon.values[::-1])
        
    lats_selected = da.lat.values
    lons_selected = da.lon.values
    
    stats = {
        "lat_min_halo": float(lats_selected.min()),
        "lat_max_halo": float(lats_selected.max()),
        "lon_min_halo": float(lons_selected.min()),
        "lon_max_halo": float(lons_selected.max()),
        "selected_lats": lats_selected.tolist(),
        "selected_lons": lons_selected.tolist(),
        "shape": da.shape,
        "n_pixels": int(da.size),
        "n_nans": int(np.isnan(da.values).sum()),
        "resolution_lat": float(np.diff(lats_selected).mean()) if len(lats_selected) > 1 else 0.25,
        "resolution_lon": float(np.diff(lons_selected).mean()) if len(lons_selected) > 1 else 0.25
    }
    
    return da, stats

def bilinear_interp_standard(
    da_oisst_halo: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray
) -> xr.DataArray:
    """
    Interpolación bilineal estándar (sin relleno de nodos terrestres).
    Produce NaN cuando algún nodo vecino en OISST es terrestre.
    """
    src_lat = da_oisst_halo.lat.values
    src_lon = da_oisst_halo.lon.values
    src_data = da_oisst_halo.values
    
    interp_func = RegularGridInterpolator(
        (src_lat, src_lon),
        src_data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    query_pts = np.column_stack((tgt_lat_grid.ravel(), tgt_lon_grid.ravel()))
    
    interp_vals_1d = interp_func(query_pts)
    sst_bil_2d = interp_vals_1d.reshape((len(target_lat), len(target_lon))).astype(np.float32)
    
    return xr.DataArray(
        sst_bil_2d,
        coords={
            "lat": ("lat", target_lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", target_lon, {"units": "degrees_east", "standard_name": "longitude"})
        },
        dims=["lat", "lon"],
        attrs={
            "long_name": "SST OISST Bilineal Estándar (sin extensión costera)",
            "units": "degree_C",
            "method": "bilinear_standard"
        }
    )

def interpolate_strategy_a_coastal_support(
    da_oisst_halo: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray
) -> Tuple[xr.DataArray, np.ndarray, xr.DataArray]:
    """
    ESTRATEGIA A: Extensión costera en la malla OISST de 0.25° mediante nearest-ocean
    exclusivamente como soporte matemático para la interpolación bilineal.
    
    Retorna:
      - da_bil_a: xr.DataArray (86x96) con la interpolación bilineal.
      - oisst_extended: np.ndarray con los valores OISST de 0.25° extendidos.
      - support_mask: xr.DataArray booleano (en malla OISST) indicando los nodos rellenados por soporte.
    """
    src_lat = da_oisst_halo.lat.values
    src_lon = da_oisst_halo.lon.values
    src_data = da_oisst_halo.values.copy()
    
    is_valid = ~np.isnan(src_data)
    valid_coords = []
    valid_vals = []
    for i in range(len(src_lat)):
        for j in range(len(src_lon)):
            if is_valid[i, j]:
                valid_coords.append([src_lat[i], src_lon[j]])
                valid_vals.append(src_data[i, j])
                
    if len(valid_coords) == 0:
        raise ValueError("No hay píxeles oceánicos válidos en OISST halo.")
        
    tree = cKDTree(valid_coords)
    support_mask_arr = np.zeros(src_data.shape, dtype=bool)
    
    # Extender hacia celdas terrestres usando nearest ocean
    for i in range(len(src_lat)):
        for j in range(len(src_lon)):
            if np.isnan(src_data[i, j]):
                _, idx = tree.query([src_lat[i], src_lon[j]], k=1)
                src_data[i, j] = valid_vals[idx]
                support_mask_arr[i, j] = True
                
    # Interpolación bilineal regular sobre la cuadrícula completa con soporte
    interp_a = RegularGridInterpolator(
        (src_lat, src_lon),
        src_data,
        method="linear",
        bounds_error=True
    )
    
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    query_pts = np.column_stack((tgt_lat_grid.ravel(), tgt_lon_grid.ravel()))
    
    bil_a_1d = interp_a(query_pts)
    bil_a_2d = bil_a_1d.reshape((len(target_lat), len(target_lon))).astype(np.float32)
    
    da_bil_a = xr.DataArray(
        bil_a_2d,
        coords={
            "lat": ("lat", target_lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", target_lon, {"units": "degrees_east", "standard_name": "longitude"})
        },
        dims=["lat", "lon"],
        attrs={
            "long_name": "NOAA OISST v2.1 SST interpolada bilinealmente (Estrategia A: Soporte Costero)",
            "units": "degree_C",
            "interpolation_method": "bilinear_with_coastal_support",
            "source_resolution": "0.25 degrees",
            "target_resolution": "0.01 degrees"
        }
    )
    
    da_support_mask = xr.DataArray(
        support_mask_arr,
        coords={"lat": src_lat, "lon": src_lon},
        dims=["lat", "lon"],
        attrs={
            "long_name": "Máscara de soporte costero OISST (True = nodo terrestre rellenado para soporte)",
            "description": "Indica nodos OISST 0.25° rellenados exclusivamente como soporte matemático para la interpolación bilineal costera"
        }
    )
    
    return da_bil_a, src_data, da_support_mask

def interpolate_strategy_b_triangulation(
    da_oisst_halo: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray
) -> xr.DataArray:
    """
    ESTRATEGIA B: Interpolación lineal usando exclusivamente puntos oceánicos válidos (triangulación 2D)
    con fallback nearest únicamente para celdas que caigan fuera del convex hull.
    """
    src_lat = da_oisst_halo.lat.values
    src_lon = da_oisst_halo.lon.values
    src_data = da_oisst_halo.values
    
    is_valid = ~np.isnan(src_data)
    valid_coords = []
    valid_vals = []
    for i in range(len(src_lat)):
        for j in range(len(src_lon)):
            if is_valid[i, j]:
                valid_coords.append([src_lat[i], src_lon[j]])
                valid_vals.append(src_data[i, j])
                
    pts_arr = np.array(valid_coords)
    vals_arr = np.array(valid_vals)
    
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    
    grid_linear = griddata(pts_arr, vals_arr, (tgt_lat_grid, tgt_lon_grid), method="linear")
    grid_nearest = griddata(pts_arr, vals_arr, (tgt_lat_grid, tgt_lon_grid), method="nearest")
    
    # Fallback nearest donde linear sea NaN
    grid_b = np.where(np.isnan(grid_linear), grid_nearest, grid_linear).astype(np.float32)
    
    da_bil_b = xr.DataArray(
        grid_b,
        coords={
            "lat": ("lat", target_lat, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", target_lon, {"units": "degrees_east", "standard_name": "longitude"})
        },
        dims=["lat", "lon"],
        attrs={
            "long_name": "NOAA OISST v2.1 SST interpolada (Estrategia B: Triangulación 2D + Fallback Nearest)",
            "units": "degree_C",
            "interpolation_method": "delaunay_linear_with_nearest_fallback",
            "source_resolution": "0.25 degrees",
            "target_resolution": "0.01 degrees"
        }
    )
    return da_bil_b

def verify_coordinates_alignment(
    da_bil: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Verifica que las dimensiones y coordenadas de sst_bil coincidan exactamente con la malla MUR.
    """
    bil_lat = da_bil.lat.values
    bil_lon = da_bil.lon.values
    
    err_lat = float(np.max(np.abs(bil_lat - target_lat)))
    err_lon = float(np.max(np.abs(bil_lon - target_lon)))
    shape_ok = bool(da_bil.shape == (len(target_lat), len(target_lon)))
    coords_ok = bool(err_lat < tolerance and err_lon < tolerance)
    
    return {
        "shape": da_bil.shape,
        "expected_shape": (len(target_lat), len(target_lon)),
        "shape_match": shape_ok,
        "max_abs_lat_diff": err_lat,
        "max_abs_lon_diff": err_lon,
        "coords_match": coords_ok,
        "tolerance": tolerance,
        "status": "OK" if (shape_ok and coords_ok) else "MISMATCH"
    }

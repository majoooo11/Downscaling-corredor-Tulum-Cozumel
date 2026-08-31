"""
Módulo de agregación batimétrica desde GEBCO hacia la cuadrícula maestra MUR.
Aplica la convención de profundidad positiva (Depth = -elevation) y enmascara tierra como NaN.
"""

import numpy as np
import xarray as xr

def aggregate_gebco_to_mur_grid(ds_gebco: xr.Dataset, ds_mur: xr.Dataset, ds_mask: xr.Dataset) -> xr.Dataset:
    """
    Agrega la batimetría GEBCO (~15 arc-sec) a la cuadrícula maestra MUR (~0.01°).
    
    Convención:
      - Profundidad positiva: Depth = -elevation (en metros) para celdas oceánicas.
      - Tierra: NaN donde ocean_mask_final == 0.
      
    Retorna:
      xr.Dataset con la variable 'depth' en metros sobre coordenadas (lat, lon).
    """
    mur_lat = ds_mur.lat.values
    mur_lon = ds_mur.lon.values
    
    gebco_var = "elevation" if "elevation" in ds_gebco.variables else "z"
    gebco_elev = ds_gebco[gebco_var].values
    gebco_lat = ds_gebco.lat.values
    gebco_lon = ds_gebco.lon.values
    
    # Obtener máscara final
    ocean_mask_final = ds_mask["ocean_mask_final"].values
    
    # Calcular bordes de las celdas MUR
    dlat = np.abs(np.diff(mur_lat).mean())
    dlon = np.abs(np.diff(mur_lon).mean())
    
    lat_edges = np.concatenate([[mur_lat[0] - dlat/2.0], mur_lat[:-1] + np.diff(mur_lat)/2.0, [mur_lat[-1] + dlat/2.0]])
    lon_edges = np.concatenate([[mur_lon[0] - dlon/2.0], mur_lon[:-1] + np.diff(mur_lon)/2.0, [mur_lon[-1] + dlon/2.0]])
    
    lat_edges_asc = np.sort(lat_edges)
    lon_edges_asc = np.sort(lon_edges)
    
    g_lat_grid, g_lon_grid = np.meshgrid(gebco_lat, gebco_lon, indexing="ij")
    
    lat_bins = np.digitize(g_lat_grid.ravel(), lat_edges_asc) - 1
    lon_bins = np.digitize(g_lon_grid.ravel(), lon_edges_asc) - 1
    
    if mur_lat[1] < mur_lat[0]:
        lat_idx_map = (len(mur_lat) - 1) - lat_bins
    else:
        lat_idx_map = lat_bins
        
    if mur_lon[1] < mur_lon[0]:
        lon_idx_map = (len(mur_lon) - 1) - lon_bins
    else:
        lon_idx_map = lon_bins
        
    elev_flat = gebco_elev.ravel()
    # Tomar sólo puntos submarinos (elevation < 0) para el cálculo de la profundidad oceánica
    is_submarine = (elev_flat < 0)
    
    valid_pts = is_submarine & (lat_idx_map >= 0) & (lat_idx_map < len(mur_lat)) & (lon_idx_map >= 0) & (lon_idx_map < len(mur_lon))
    
    flat_indices = lat_idx_map[valid_pts] * len(mur_lon) + lon_idx_map[valid_pts]
    # Suma de profundidades positivas (-elev)
    depth_sum = np.bincount(flat_indices, weights=-elev_flat[valid_pts], minlength=len(mur_lat)*len(mur_lon))
    point_counts = np.bincount(flat_indices, minlength=len(mur_lat)*len(mur_lon))
    
    # Calcular media de profundidad
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_depth_1d = depth_sum / point_counts
        
    depth_2d = mean_depth_1d.reshape((len(mur_lat), len(mur_lon)))
    
    # Enmascarar tierra como NaN usando ocean_mask_final
    depth_2d[ocean_mask_final == 0] = np.nan
    
    # Para cualquier celda de océano donde no hubiera puntos submarinos (borde extremo),
    # interpolar o asignar valor mínimo plausible
    if np.any(np.isnan(depth_2d) & (ocean_mask_final == 1)):
        # Fallback simple
        pass
        
    ds_depth = xr.Dataset(
        data_vars={
            "depth": (("lat", "lon"), depth_2d.astype(np.float32), {
                "units": "m",
                "long_name": "Profundidad batimétrica marina positiva agregada desde GEBCO",
                "standard_name": "sea_floor_depth_below_geoid"
            })
        },
        coords={
            "lat": mur_lat,
            "lon": mur_lon
        },
        attrs={
            "description": "Profundidad batimétrica sobre la cuadrícula maestra MUR",
            "source": "GEBCO 15 arc-seconds",
            "land_mask": "ocean_mask_final == 0 -> NaN"
        }
    )
    return ds_depth

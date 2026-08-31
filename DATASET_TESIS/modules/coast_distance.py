"""
Módulo de cálculo de distancia mínima a la costa en proyección métrica (UTM 16N / EPSG:32616).
Enmascara tierra como NaN según ocean_mask_final.
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pyproj import Transformer

def calculate_distance_to_coast(ds_gebco: xr.Dataset, ds_mur: xr.Dataset, ds_mask: xr.Dataset) -> xr.Dataset:
    """
    Calcula la distancia geodésica mínima (en km) desde cada celda oceánica MUR a la costa GEBCO.
    
    Proceso:
      1. Extrae los puntos de la línea de costa a partir de la interfaz tierra-mar en GEBCO (elevation == 0 o contornos de costa).
      2. Proyecta las coordenadas geográficas (WGS84 / EPSG:4326) a proyección métrica UTM 16N (EPSG:32616).
      3. Utiliza un árbol KD (cKDTree: es una clase optimizada en el módulo scipy.spatial de Python que sirve para hacer búsquedas 
         rápidas de vecinos cercanos en un espacio multidimensional.) para calcular la distancia euclidiana en metros a la costa más cercana.
      4. Convierte a kilómetros.
      5. Aplica ocean_mask_final (tierra -> NaN).
      
    Retorna:
      xr.Dataset con 'distance_coast_km' en coordenadas (lat, lon).
    """
    mur_lat = ds_mur.lat.values
    mur_lon = ds_mur.lon.values
    ocean_mask_final = ds_mask["ocean_mask_final"].values
    
    # 1. Extraer interfaz de costa de GEBCO
    gebco_var = "elevation" if "elevation" in ds_gebco.variables else "z"
    gebco_elev = ds_gebco[gebco_var].values
    gebco_lat = ds_gebco.lat.values
    gebco_lon = ds_gebco.lon.values
    
    # Identificar celdas de tierra (elev >= 0) y mar (elev < 0)
    is_land_gebco = (gebco_elev >= 0)
    
    # Encontrar celdas costeras en GEBCO: celdas de tierra que tienen al menos un vecino de mar, o viceversa
    # Usando diferencias 2D rápidas
    land_int = is_land_gebco.astype(int)
    diff_lat = np.abs(np.diff(land_int, axis=0, append=land_int[-1:, :]))
    diff_lon = np.abs(np.diff(land_int, axis=1, append=land_int[:, -1:]))
    is_coast_gebco = (diff_lat > 0) | (diff_lon > 0)
    
    g_lat_grid, g_lon_grid = np.meshgrid(gebco_lat, gebco_lon, indexing="ij")
    coast_lats = g_lat_grid[is_coast_gebco]
    coast_lons = g_lon_grid[is_coast_gebco]
    
    if len(coast_lats) == 0:
        # Fallback: puntos más cercanos a 0
        coast_lats = g_lat_grid[np.abs(gebco_elev) < 10]
        coast_lons = g_lon_grid[np.abs(gebco_elev) < 10]
        
    # 2. Transformador de coordenadas a UTM 16N (EPSG:32616)
    # always_xy=True para que tome (lon, lat) -> (x, y)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    
    # Proyectar puntos de costa
    coast_x, coast_y = transformer.transform(coast_lons, coast_lats)
    coast_points_utm = np.column_stack((coast_x, coast_y))
    
    # Construir cKDTree con puntos de costa
    coast_tree = cKDTree(coast_points_utm)
    
    # 3. Proyectar centros de las celdas MUR
    mur_lat_grid, mur_lon_grid = np.meshgrid(mur_lat, mur_lon, indexing="ij")
    mur_x, mur_y = transformer.transform(mur_lon_grid.ravel(), mur_lat_grid.ravel())
    mur_points_utm = np.column_stack((mur_x, mur_y))
    
    # 4. Consultar distancias euclidianas mínimas
    distances_m, _ = coast_tree.query(mur_points_utm, k=1)
    distances_km_2d = (distances_m / 1000.0).reshape((len(mur_lat), len(mur_lon)))
    
    # 5. Aplicar ocean_mask_final (tierra -> NaN)
    distances_km_2d[ocean_mask_final == 0] = np.nan
    
    ds_dist = xr.Dataset(
        data_vars={
            "distance_coast_km": (("lat", "lon"), distances_km_2d.astype(np.float32), {
                "units": "km",
                "long_name": "Distancia mínima a la línea de costa en kilómetros",
                "projection": "EPSG:32616 (UTM 16N)"
            })
        },
        coords={
            "lat": mur_lat,
            "lon": mur_lon
        },
        attrs={
            "description": "Distancia a la costa calculada en UTM 16N sobre la cuadrícula maestra MUR",
            "source_coastline": "GEBCO 15 arc-seconds",
            "land_mask": "ocean_mask_final == 0 -> NaN"
        }
    )
    return ds_dist

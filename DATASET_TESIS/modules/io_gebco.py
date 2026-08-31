"""
Módulo de entrada/salida e inspección de batimetría GEBCO (~15 arc-sec).
"""

from pathlib import Path
import numpy as np
import xarray as xr

def inspect_gebco_sources(config) -> dict:
    """
    Inspecciona el archivo GEBCO configurado y retorna metadatos clave.
    """
    gebco_path = Path(config.GEBCO_FILE)
    if not gebco_path.exists():
        return {
            "exists": False,
            "variable_name": "N/A",
            "units": "N/A",
            "res_lat": 0.0,
            "lat_min": 0.0,
            "lat_max": 0.0,
            "lon_min": 0.0,
            "lon_max": 0.0
        }
        
    with xr.open_dataset(gebco_path) as ds:
        var_name = None
        for v in ["elevation", "z", "topo"]:
            if v in ds.variables:
                var_name = v
                break
        if var_name is None:
            var_name = list(ds.data_vars.keys())[0]
            
        units = ds[var_name].attrs.get("units", "m")
        lat_vals = ds.lat.values
        lon_vals = ds.lon.values
        
        lat_diff = np.diff(lat_vals)
        res_lat = float(np.abs(lat_diff).mean()) if len(lat_diff) > 0 else 0.00416667
        
        return {
            "exists": True,
            "variable_name": var_name,
            "units": units,
            "res_lat": res_lat,
            "lat_min": float(lat_vals.min()),
            "lat_max": float(lat_vals.max()),
            "lon_min": float(lon_vals.min()),
            "lon_max": float(lon_vals.max()),
            "shape": (len(lat_vals), len(lon_vals))
        }

def load_gebco_dataset(config) -> xr.Dataset:
    """
    Carga el dataset GEBCO desde la ruta configurada.
    """
    gebco_path = Path(config.GEBCO_FILE)
    if not gebco_path.exists():
        raise FileNotFoundError(f"Archivo GEBCO no encontrado en: {gebco_path}")
    ds = xr.open_dataset(gebco_path)
    return ds

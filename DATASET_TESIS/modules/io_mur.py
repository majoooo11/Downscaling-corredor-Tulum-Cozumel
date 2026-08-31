"""
Módulo de entrada/salida y carga de MUR SST v4.1 (0.01°).
Maneja tanto el bloque histórico (2015-2019) como la serie diaria (2019-2025).
"""

from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import xarray as xr

def inspect_mur_sources(config) -> dict:
    """
    Inspecciona las fuentes locales de MUR (histórico y diario).
    """
    hist_dir = Path(config.MUR_HIST_DIR)
    daily_dir = Path(config.MUR_DAILY_DIR)
    
    hist_files = sorted(hist_dir.glob("*.nc")) if hist_dir.exists() else []
    daily_files = sorted(daily_dir.glob("*.nc4")) if daily_dir.exists() else []
    
    hist_range = "N/A"
    if hist_files:
        hist_range = f"{hist_files[0].name} a {hist_files[-1].name}"
        
    daily_range = "N/A"
    dims = {}
    res_lat = 0.01
    if daily_files:
        daily_range = f"{daily_files[0].name[:8]} a {daily_files[-1].name[:8]}"
        try:
            with xr.open_dataset(daily_files[0]) as ds:
                dims = {"lat": ds.sizes.get("lat", config.LAT_GRID_SIZE), "lon": ds.sizes.get("lon", config.LON_GRID_SIZE)}
                if "lat" in ds.coords:
                    lat_diff = np.diff(ds.lat.values)
                    if len(lat_diff) > 0:
                        res_lat = float(np.abs(lat_diff).mean())
        except Exception:
            dims = {"lat": config.LAT_GRID_SIZE, "lon": config.LON_GRID_SIZE}
            
    return {
        "hist_files_found": len(hist_files),
        "hist_range": hist_range,
        "daily_files_found": len(daily_files),
        "daily_range": daily_range,
        "dims": dims,
        "res_lat": res_lat
    }

def load_mur_reference(config) -> xr.Dataset:
    """
    Carga un dataset de referencia para obtener la cuadrícula espacial maestra (lat=86, lon=96).
    """
    hist_dir = Path(config.MUR_HIST_DIR)
    daily_dir = Path(config.MUR_DAILY_DIR)
    
    if hist_dir.exists() and list(hist_dir.glob("*.nc")):
        ref_file = sorted(hist_dir.glob("*.nc"))[0]
        ds = xr.open_dataset(ref_file)
        # Tomar sólo el primer slice temporal si tiene dimensión tiempo
        if "time" in ds.dims:
            ds = ds.isel(time=0)
        return ds
    elif daily_dir.exists() and list(daily_dir.glob("*.nc4")):
        ref_file = sorted(daily_dir.glob("*.nc4"))[0]
        ds = xr.open_dataset(ref_file)
        if "time" in ds.dims:
            ds = ds.squeeze()
        return ds
    else:
        raise FileNotFoundError("No se encontraron archivos MUR de referencia ni en histórico ni en diario.")

def load_mur_dataset(config) -> xr.Dataset:
    """
    Alias para obtener el dataset de referencia espacial de MUR.
    """
    return load_mur_reference(config)

def load_mur_single_day(date_str: str, config) -> xr.DataArray:
    """
    Carga la SST MUR de alta resolución para un día específico (YYYY-MM-DD).
    Convierte analysed_sst de Kelvin a Celsius (SST_C = SST_K - 273.15).
    Retorna un DataArray 2D con coordenadas lat y lon.
    """
    target_date = pd.to_datetime(date_str).date()
    hist_dir = Path(config.MUR_HIST_DIR)
    daily_dir = Path(config.MUR_DAILY_DIR)
    
    # 1. Si la fecha está en el rango histórico (2015-01-01 a 2019-07-22)
    if target_date <= datetime.date(2019, 7, 22):
        # Determinar el archivo anual correspondiente
        year = target_date.year
        if year in [2015, 2016, 2017, 2018]:
            hist_file = hist_dir / f"MUR_{year}_Tulum_Cozumel.nc"
        elif year == 2019:
            hist_file = hist_dir / "MUR_2019_01-01_07-22_Tulum_Cozumel.nc"
        else:
            raise ValueError(f"Año {year} fuera de rango histórico.")
            
        if not hist_file.exists():
            raise FileNotFoundError(f"Archivo MUR histórico no encontrado: {hist_file}")
            
        with xr.open_dataset(hist_file) as ds:
            # Buscar el slice correspondiente a date_str
            times = pd.to_datetime(ds.time.values).tz_localize(None).normalize()
            mask_date = (times == pd.to_datetime(date_str).normalize())
            if not np.any(mask_date):
                raise KeyError(f"Fecha {date_str} no encontrada dentro de {hist_file.name}")
            idx = int(np.where(mask_date)[0][0])
            da_k = ds["analysed_sst"].isel(time=idx)
            # Convertir a Celsius
            da_c = da_k - 273.15
            da_c.attrs["units"] = "degree_C"
            da_c.attrs["long_name"] = "MUR SST en grados Celsius"
            return da_c.load()
            
    # 2. Si la fecha es posterior a 2019-07-22 (archivos diarios .nc4)
    else:
        date_tag = target_date.strftime("%Y%m%d")
        daily_files = list(daily_dir.glob(f"{date_tag}*JPL-L4_GHRSST-SSTfnd-MUR-GLOB*.nc4"))
        if not daily_files:
            # Búsqueda alternativa por prefijo
            daily_files = list(daily_dir.glob(f"{date_tag}*.nc4"))
            
        if not daily_files:
            raise FileNotFoundError(f"Archivo MUR diario no encontrado para la fecha {date_str} en {daily_dir}")
            
        daily_file = daily_files[0]
        with xr.open_dataset(daily_file) as ds:
            if "time" in ds.dims:
                da_k = ds["analysed_sst"].squeeze()
            else:
                da_k = ds["analysed_sst"]
            da_c = da_k - 273.15
            da_c.attrs["units"] = "degree_C"
            da_c.attrs["long_name"] = "MUR SST en grados Celsius"
            return da_c.load()

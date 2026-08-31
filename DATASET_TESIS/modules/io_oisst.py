"""
Módulo de entrada/salida y consulta de NOAA OISST v2.1 (~0.25°).
Soporta ERDDAP OPeNDAP, NOAA PSL THREDDS, NOAA NCEI Direct HTTP, halo espacial y el archivo local para 2025-01-14.
"""

from pathlib import Path
import urllib.request
import io
import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("tesis_pipeline")

def inspect_oisst_sources(config) -> dict:
    """
    Inspecciona las fuentes de datos OISST (archivo local 2025-01-14 y URL ERDDAP).
    """
    local_file = Path(config.OISST_LOCAL_FILE)
    local_found = local_file.exists()
    local_dims = {}
    units = "degree_C"
    
    if local_found:
        try:
            with xr.open_dataset(local_file) as ds:
                local_dims = dict(ds.sizes)
                if "sst" in ds.data_vars:
                    units = ds.sst.attrs.get("units", "degree_C")
        except Exception:
            local_dims = {"info": "Error al leer dimensiones"}
            
    return {
        "local_file_found": local_found,
        "local_file_path": str(local_file),
        "local_dims": local_dims,
        "units": units,
        "erddap_url": config.OISST_ERDDAP_URL
    }

def load_oisst_day(date_str: str, config, halo: bool = True) -> xr.DataArray:
    """
    Carga la SST OISST para una fecha específica (YYYY-MM-DD) con tolerancia a fallas de red
    mediante múltiples proveedores oficiales de NOAA (PSL THREDDS, NCEI Direct HTTP y CoastWatch ERDDAP).
    """
    dt = pd.to_datetime(date_str)
    target_date = dt.strftime("%Y-%m-%d")
    year = dt.year
    yyyymm = dt.strftime("%Y%m")
    yyyymmdd = dt.strftime("%Y%m%d")
    
    if halo:
        lat_min, lat_max = config.OISST_LAT_MIN, config.OISST_LAT_MAX
        lon_min, lon_max = config.OISST_LON_MIN, config.OISST_LON_MAX
    else:
        lat_min, lat_max = config.LAT_MIN, config.LAT_MAX
        lon_min, lon_max = config.LON_MIN, config.LON_MAX

    # 1. Caso especial: Archivo local recuperado para 2025-01-14
    if target_date == "2025-01-14":
        local_file = Path(config.OISST_LOCAL_FILE)
        if local_file.exists():
            with xr.open_dataset(local_file) as ds:
                return _extract_spatial_subset(ds, target_date, lat_min, lat_max, lon_min, lon_max)

    # 2. Intentar NOAA PSL THREDDS (archivos anuales agregados, alta velocidad y disponibilidad)
    psl_url = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    try:
        with xr.open_dataset(psl_url) as ds_psl:
            return _extract_spatial_subset(ds_psl, target_date, lat_min, lat_max, lon_min, lon_max)
    except Exception as e_psl:
        logger.debug(f"PSL THREDDS no disponible para {target_date}: {e_psl}. Intentando NCEI Direct HTTP...")

    # 3. Intentar NOAA NCEI Direct HTTP (archivos netCDF diarios individuales)
    ncei_url = f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
    try:
        req = urllib.request.Request(ncei_url, headers={"User-Agent": "Mozilla/5.0 (Scientific Pipeline)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            with xr.open_dataset(io.BytesIO(data)) as ds_ncei:
                return _extract_spatial_subset(ds_ncei, target_date, lat_min, lat_max, lon_min, lon_max)
    except Exception as e_ncei:
        logger.debug(f"NCEI Direct HTTP no disponible para {target_date}: {e_ncei}. Intentando CoastWatch ERDDAP...")

    # 4. Intentar CoastWatch ERDDAP
    try:
        with xr.open_dataset(config.OISST_ERDDAP_URL, chunks={"time": 1}) as ds_erddap:
            return _extract_spatial_subset(ds_erddap, target_date, lat_min, lat_max, lon_min, lon_max)
    except Exception as e_erddap:
        raise RuntimeError(f"Error crítico al obtener OISST para {target_date} desde todos los proveedores NOAA: {e_erddap}")

def _extract_spatial_subset(
    ds: xr.Dataset,
    target_date: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> xr.DataArray:
    """Extrae y estandariza el DataArray de SST para la región y fecha requeridas."""
    lat_var = "latitude" if "latitude" in ds.coords else "lat"
    lon_var = "longitude" if "longitude" in ds.coords else "lon"
    
    max_lon = float(ds[lon_var].max())
    q_lon_min = lon_min + 360 if (max_lon > 180 and lon_min < 0) else lon_min
    q_lon_max = lon_max + 360 if (max_lon > 180 and lon_max < 0) else lon_max
    
    sel_kwargs = {
        lat_var: slice(lat_min, lat_max),
        lon_var: slice(q_lon_min, q_lon_max)
    }
    if "time" in ds.dims:
        sel_kwargs["time"] = target_date
        
    ds_sub = ds.sel(sel_kwargs)
    da = ds_sub["sst"].squeeze()
    
    rename_dict = {}
    if lat_var != "lat":
        rename_dict[lat_var] = "lat"
    if lon_var != "lon":
        rename_dict[lon_var] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)
        
    if max_lon > 180 and float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon - 360))
        
    # Asegurar orden ascendente de lat y lon
    if da.lat.values[1] < da.lat.values[0]:
        da = da.reindex(lat=da.lat.values[::-1])
    if da.lon.values[1] < da.lon.values[0]:
        da = da.reindex(lon=da.lon.values[::-1])
        
    da = da.astype(np.float32)
    da.attrs["units"] = "degree_C"
    da.attrs["long_name"] = f"NOAA OISST v2.1 SST ({target_date})"
    return da.compute()

def load_oisst_year_batch(year: int, config, halo: bool = True) -> xr.DataArray:
    """
    Carga de forma optimizada el bloque anual completo de NOAA OISST v2.1 para un año dado,
    descargando el recorte espacial (~7x7 píxeles) vía ERDDAP REST y rellenando cualquier
    fecha faltante o corrupta con NCEI Direct HTTP.
    Retorna un xr.DataArray con dimensiones (time, lat, lon) para todos los días del año.
    """
    if halo:
        lat_min, lat_max = config.OISST_LAT_MIN, config.OISST_LAT_MAX
        lon_min, lon_max = config.OISST_LON_MIN, config.OISST_LON_MAX
    else:
        lat_min, lat_max = config.LAT_MIN, config.LAT_MAX
        lon_min, lon_max = config.LON_MIN, config.LON_MAX

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    dates_expected = pd.date_range(start_date, end_date, freq="D")
    
    # 1. Intentar descarga en lote anual vía CoastWatch ERDDAP REST
    erddap_rest_url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?"
        f"sst[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)][(0.0):1:(0.0)][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    
    da_erddap = None
    try:
        req = urllib.request.Request(erddap_rest_url, headers={"User-Agent": "Mozilla/5.0 (Scientific Pipeline)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with xr.open_dataset(io.BytesIO(data)) as ds:
                da = ds["sst"].squeeze()
                rename_dict = {}
                if "latitude" in da.coords: rename_dict["latitude"] = "lat"
                if "longitude" in da.coords: rename_dict["longitude"] = "lon"
                if rename_dict: da = da.rename(rename_dict)
                if float(da.lon.max()) > 180: da = da.assign_coords(lon=(da.lon - 360))
                if da.lat.values[1] < da.lat.values[0]: da = da.reindex(lat=da.lat.values[::-1])
                if da.lon.values[1] < da.lon.values[0]: da = da.reindex(lon=da.lon.values[::-1])
                
                times = pd.to_datetime(da.time.values).tz_localize(None).normalize()
                da = da.assign_coords(time=times)
                da = da.astype(np.float32)
                da_erddap = da.load()
    except Exception as e_erddap:
        logger.warning(f"Descarga batch ERDDAP no disponible para el año {year}: {e_erddap}. Procesando con NCEI...")

    # Si se obtuvo ERDDAP, verificar si faltan días o si hay fechas especiales
    if da_erddap is not None:
        times_found = pd.DatetimeIndex(da_erddap.time.values).normalize()
        missing_in_erddap = dates_expected[~dates_expected.isin(times_found)]
        
        # Crear dataset completo con todas las fechas
        daily_slices = []
        for d in dates_expected:
            d_norm = d.normalize()
            if d_norm in times_found:
                idx_f = int(np.where(times_found == d_norm)[0][0])
                da_day = da_erddap.isel(time=idx_f)
                # Caso especial 2025-01-14 o si el día tiene anomalías
                if d.strftime("%Y-%m-%d") == "2025-01-14" or np.isnan(da_day.values).sum() > 25:
                    da_day = _fetch_single_day_ncei(d.strftime("%Y-%m-%d"), lat_min, lat_max, lon_min, lon_max)
                daily_slices.append(da_day)
            else:
                # Recuperar fecha faltante en ERDDAP vía NCEI Direct HTTP
                d_str = d.strftime("%Y-%m-%d")
                logger.info(f"Recuperando fecha faltante en ERDDAP ({d_str}) desde NCEI Direct HTTP...")
                da_day = _fetch_single_day_ncei(d_str, lat_min, lat_max, lon_min, lon_max)
                daily_slices.append(da_day)
                
        da_full = xr.concat(daily_slices, dim="time")
        da_full = da_full.assign_coords(time=dates_expected)
        da_full.attrs["units"] = "degree_C"
        da_full.attrs["long_name"] = f"NOAA OISST v2.1 SST ({year})"
        return da_full

    # 2. Fallback completo: Carga secuencial día a día usando NCEI Direct HTTP
    daily_list = []
    for d in dates_expected:
        d_str = d.strftime("%Y-%m-%d")
        da_day = _fetch_single_day_ncei(d_str, lat_min, lat_max, lon_min, lon_max)
        daily_list.append(da_day)
        
    da_concat = xr.concat(daily_list, dim="time")
    da_concat = da_concat.assign_coords(time=dates_expected)
    da_concat.attrs["units"] = "degree_C"
    da_concat.attrs["long_name"] = f"NOAA OISST v2.1 SST ({year})"
    return da_concat

def _fetch_single_day_ncei(date_str: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    """Descarga e interpola un solo día desde NCEI Direct HTTP."""
    dt = pd.to_datetime(date_str)
    yyyymm = dt.strftime("%Y%m")
    yyyymmdd = dt.strftime("%Y%m%d")
    ncei_url = f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
    
    req = urllib.request.Request(ncei_url, headers={"User-Agent": "Mozilla/5.0 (Scientific Pipeline)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
        with xr.open_dataset(io.BytesIO(data)) as ds:
            q_lon_min = lon_min + 360 if lon_min < 0 else lon_min
            q_lon_max = lon_max + 360 if lon_max < 0 else lon_max
            sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(q_lon_min, q_lon_max))
            da = sub["sst"].squeeze().load()
            if "latitude" in da.coords: da = da.rename({"latitude": "lat"})
            if "longitude" in da.coords: da = da.rename({"longitude": "lon"})
            if float(da.lon.max()) > 180: da = da.assign_coords(lon=(da.lon - 360))
            if da.lat.values[1] < da.lat.values[0]: da = da.reindex(lat=da.lat.values[::-1])
            if da.lon.values[1] < da.lon.values[0]: da = da.reindex(lon=da.lon.values[::-1])
            da = da.astype(np.float32)
            return da


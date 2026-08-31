#!/usr/bin/env python3
"""
Script de auditoría maestra de datos de tesis.
Realiza control de calidad y verificación para MUR SST, NOAA OISST y GEBCO,
determina el periodo común real y genera el reporte final.
"""

import os
import sys
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# Rutas de entrada y salida
MUR_DIR = Path("/Users/universidaddecolima/Desktop/Lole/MUR_ZARR/MUR-JPL-L4-GLOB-v4.1_4.1-20260814_211604")
OISST_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"
GEBCO_PATH = Path("/Users/universidaddecolima/Desktop/Lole/GEBCO/gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc")
LOG_PATH = Path("/Users/universidaddecolima/Desktop/Lole/auditoria_datos_tesis.log")

# Dominios y periodos objetivo
TARGET_START = "2015-01-01"
TARGET_END = "2025-12-31"
LAT_MIN, LAT_MAX = 19.90, 20.75
LON_MIN, LON_MAX = -87.60, -86.65
EXPECTED_DAYS = 4018  # Entre 2015-01-01 y 2025-12-31 inclusive

def main():
    print("============================================================")
    print("INICIANDO AUDITORÍA MAESTRA DE DATOS PARA TESIS")
    print("============================================================")
    print(f"Fecha local: {datetime.datetime.now().isoformat()}")
    print(f"Directorio de reporte: {LOG_PATH}")
    
    # -------------------------------------------------------------------------
    # 1. AUDITORÍA MUR SST v4.1
    # -------------------------------------------------------------------------
    print("\n--- Auditando MUR SST v4.1 ---")
    mur_status = "INCOMPLETO"
    mur_first_date = "N/A"
    mur_last_date = "N/A"
    mur_days_available = 0
    mur_days_missing = EXPECTED_DAYS
    mur_duplicates = 0
    mur_corrupted = 0
    mur_zero_size = 0
    mur_time_missing = 0
    mur_units = "N/A"
    mur_res = "N/A"
    mur_dims = "N/A"
    mur_lat_min, mur_lat_max = np.nan, np.nan
    mur_lon_min, mur_lon_max = np.nan, np.nan
    
    mur_dates_in_target = pd.DatetimeIndex([])
    
    if not MUR_DIR.exists():
        print(f"Error: La carpeta de MUR {MUR_DIR} no existe.")
    else:
        nc_files = sorted(MUR_DIR.glob("*.nc4"))
        print(f"Archivos MUR encontrados: {len(nc_files)}")
        
        fechas_mur = []
        fechas_duplicadas = []
        
        for file in nc_files:
            file_size = file.stat().st_size
            if file_size == 0:
                mur_zero_size += 1
                mur_corrupted += 1
                continue
                
            try:
                # Open dataset using xarray to read time variable
                with xr.open_dataset(file) as ds:
                    if 'time' not in ds.variables:
                        mur_time_missing += 1
                        mur_corrupted += 1
                        continue
                        
                    # Extract date
                    date_val = pd.to_datetime(ds.time.values[0]).date()
                    
                    # Capture spatial metadata from the first valid file
                    if len(fechas_mur) == 0:
                        mur_dims = f"lat: {ds.sizes.get('lat', 'N/A')}, lon: {ds.sizes.get('lon', 'N/A')}"
                        if 'analysed_sst' in ds.data_vars:
                            mur_units = ds.analysed_sst.attrs.get('units', 'No especificado')
                        if 'lat' in ds.coords and 'lon' in ds.coords:
                            mur_lat_min = float(ds.lat.min().values)
                            mur_lat_max = float(ds.lat.max().values)
                            mur_lon_min = float(ds.lon.min().values)
                            mur_lon_max = float(ds.lon.max().values)
                            
                            # Estimate resolution
                            lat_diff = np.diff(ds.lat.values)
                            if len(lat_diff) > 0:
                                mur_res = f"{lat_diff.mean():.4f}°"
                            else:
                                mur_res = "0.0100°"
                                
                    if date_val in fechas_mur:
                        fechas_duplicadas.append(date_val)
                    else:
                        fechas_mur.append(date_val)
                        
            except Exception as e:
                print(f"Error al abrir {file.name}: {e}")
                mur_corrupted += 1
                
        if fechas_mur:
            fechas_mur_idx = pd.DatetimeIndex(sorted(fechas_mur))
            mur_first_date = fechas_mur_idx.min().strftime('%Y-%m-%d')
            mur_last_date = fechas_mur_idx.max().strftime('%Y-%m-%d')
            
            # Filter for target period 2015-01-01 to 2025-12-31
            target_range = pd.date_range(start=TARGET_START, end=TARGET_END, freq='D')
            mur_dates_in_target = fechas_mur_idx[fechas_mur_idx.isin(target_range)]
            mur_days_available = len(mur_dates_in_target)
            
            # Calculate missing dates in target range
            missing_dates = target_range.difference(fechas_mur_idx)
            mur_days_missing = len(missing_dates)
            mur_duplicates = len(fechas_duplicadas)
            
            if mur_days_missing == 0 and mur_corrupted == 0 and mur_duplicates == 0:
                mur_status = "OK"
            else:
                mur_status = "INCOMPLETO"
                
    print(f"MUR SST v4.1 Status: {mur_status}")
    print(f"MUR Rango: {mur_first_date} a {mur_last_date} ({len(fechas_mur)} días en total)")
    print(f"MUR Días en 2015-2025: {mur_days_available}/{EXPECTED_DAYS}")
    print(f"MUR Días faltantes en 2015-2025: {mur_days_missing}")
    
    # -------------------------------------------------------------------------
    # 2. AUDITORÍA NOAA OISST v2.1 (ERDDAP OPeNDAP)
    # -------------------------------------------------------------------------
    print("\n--- Auditando NOAA OISST v2.1 ---")
    oisst_status = "INCOMPLETO"
    oisst_first_date = "N/A"
    oisst_last_date = "N/A"
    oisst_days_available = 0
    oisst_days_missing = EXPECTED_DAYS
    oisst_duplicates = 0
    oisst_units = "N/A"
    oisst_res = "N/A"
    oisst_domain_coverage = "REVISAR"
    
    oisst_dates_in_target = pd.DatetimeIndex([])
    
    try:
        print(f"Conectando a NOAA ERDDAP: {OISST_ERDDAP_URL}")
        ds_oisst = xr.open_dataset(OISST_ERDDAP_URL, chunks={'time': 100})
        print("Conexión establecida exitosamente.")
        
        # Verify coordinates
        if 'time' in ds_oisst.coords:
            times_oisst = pd.to_datetime(ds_oisst.time.values).tz_localize(None).normalize()
            oisst_first_date = times_oisst.min().strftime('%Y-%m-%d')
            oisst_last_date = times_oisst.max().strftime('%Y-%m-%d')
            
            # Filter for target range
            target_range = pd.date_range(start=TARGET_START, end=TARGET_END, freq='D')
            oisst_dates_in_target = times_oisst[times_oisst.isin(target_range)]
            oisst_days_available = len(oisst_dates_in_target)
            
            missing_dates_oisst = target_range.difference(times_oisst)
            oisst_days_missing = len(missing_dates_oisst)
            oisst_duplicates = int(times_oisst.duplicated().sum())
            
            # Check sst variable
            if 'sst' in ds_oisst.data_vars:
                oisst_units = ds_oisst.sst.attrs.get('units', 'degree_C')
                
                # Check resolution
                lat_spacing = np.diff(ds_oisst.latitude.values)
                lon_spacing = np.diff(ds_oisst.longitude.values)
                if len(lat_spacing) > 0 and len(lon_spacing) > 0:
                    oisst_res = f"{abs(lat_spacing.mean()):.2f}° x {abs(lon_spacing.mean()):.2f}°"
                else:
                    oisst_res = "0.25°"
                
                # Check Tulum-Cozumel coverage
                # Slice test date 2015-01-01
                test_slice = ds_oisst.sst.sel(
                    time="2015-01-01",
                    latitude=slice(LAT_MIN, LAT_MAX),
                    longitude=slice(LON_MIN, LON_MAX)
                ).compute()
                
                total_px = test_slice.size
                valid_px = int(test_slice.notnull().sum())
                
                if total_px > 0 and valid_px > 0:
                    oisst_domain_coverage = f"OK ({valid_px}/{total_px} píxeles válidos en 2015-01-01)"
                else:
                    oisst_domain_coverage = "REVISAR (Sin píxeles válidos)"
                    
            if oisst_days_missing == 0 and oisst_duplicates == 0:
                oisst_status = "OK"
            else:
                oisst_status = "INCOMPLETO"
                
        ds_oisst.close()
    except Exception as e:
        print(f"Error al verificar NOAA OISST en ERDDAP: {e}")
        oisst_status = "INCOMPLETO"
        
    print(f"OISST v2.1 Status: {oisst_status}")
    print(f"OISST Rango ERDDAP: {oisst_first_date} a {oisst_last_date}")
    print(f"OISST Días en 2015-2025: {oisst_days_available}/{EXPECTED_DAYS}")
    print(f"OISST Días faltantes en 2015-2025: {oisst_days_missing}")
    
    # -------------------------------------------------------------------------
    # 3. AUDITORÍA GEBCO
    # -------------------------------------------------------------------------
    print("\n--- Auditando GEBCO ---")
    gebco_status = "INCOMPLETO"
    gebco_var = "N/A"
    gebco_units = "N/A"
    gebco_res = "N/A"
    gebco_lat_min, gebco_lat_max = np.nan, np.nan
    gebco_lon_min, gebco_lon_max = np.nan, np.nan
    gebco_nans = 0
    gebco_infs = 0
    gebco_coverage = "INCOMPLETA"
    
    if not GEBCO_PATH.exists():
        print(f"Error: El archivo de GEBCO {GEBCO_PATH} no existe.")
    else:
        try:
            with xr.open_dataset(GEBCO_PATH) as ds_gebco:
                # Find elevation variable
                for name in ['elevation', 'z', 'topo']:
                    if name in ds_gebco.variables:
                        gebco_var = name
                        break
                
                if gebco_var != "N/A":
                    gebco_units = ds_gebco[gebco_var].attrs.get('units', 'm')
                    data = ds_gebco[gebco_var].values
                    
                    gebco_nans = int(np.isnan(data).sum())
                    gebco_infs = int(np.isinf(data).sum())
                    
                    if 'lat' in ds_gebco.coords and 'lon' in ds_gebco.coords:
                        gebco_lat_min = float(ds_gebco.lat.min().values)
                        gebco_lat_max = float(ds_gebco.lat.max().values)
                        gebco_lon_min = float(ds_gebco.lon.min().values)
                        gebco_lon_max = float(ds_gebco.lon.max().values)
                        
                        lat_diff = np.diff(ds_gebco.lat.values)
                        if len(lat_diff) > 0:
                            res_arcsec = lat_diff.mean() * 3600.0
                            gebco_res = f"{res_arcsec:.2f} arc-seconds (~{lat_diff.mean():.6f}°)"
                        else:
                            gebco_res = "15 arc-seconds"
                            
                        # Verify spatial coverage
                        covers_lat = (gebco_lat_min <= LAT_MIN) and (gebco_lat_max >= LAT_MAX)
                        covers_lon = (gebco_lon_min <= LON_MIN) and (gebco_lon_max >= LON_MAX)
                        
                        if covers_lat and covers_lon:
                            gebco_coverage = "COMPLETA"
                            gebco_status = "OK"
                        else:
                            gebco_coverage = "INCOMPLETA"
                            # Detailed missing coverage message
                            missing_details = []
                            if gebco_lat_min > LAT_MIN:
                                missing_details.append(f"Falta latitud sur (requiere {LAT_MIN}°, tiene {gebco_lat_min:.6f}°)")
                            if gebco_lat_max < LAT_MAX:
                                missing_details.append(f"Falta latitud norte (requiere {LAT_MAX}°, tiene {gebco_lat_max:.6f}°)")
                            if gebco_lon_min > LON_MIN:
                                missing_details.append(f"Falta longitud oeste (requiere {LON_MIN}°, tiene {gebco_lon_min:.6f}°)")
                            if gebco_lon_max < LON_MAX:
                                missing_details.append(f"Falta longitud este (requiere {LON_MAX}°, tiene {gebco_lon_max:.6f}°)")
                            gebco_coverage = f"INCOMPLETA ({', '.join(missing_details)})"
                            gebco_status = "INCOMPLETO"
                            
        except Exception as e:
            print(f"Error al abrir GEBCO: {e}")
            gebco_status = "INCOMPLETO"
            
    print(f"GEBCO Status: {gebco_status}")
    print(f"GEBCO Cobertura: {gebco_coverage}")
    print(f"GEBCO Rango Lat: {gebco_lat_min}° a {gebco_lat_max}°")
    print(f"GEBCO Rango Lon: {gebco_lon_min}° a {gebco_lon_max}°")
    
    # -------------------------------------------------------------------------
    # 4. DETERMINAR PERIODO COMÚN REAL
    # -------------------------------------------------------------------------
    print("\n--- Determinando Periodo Común ---")
    common_start = "N/A"
    common_end = "N/A"
    common_days = 0
    common_missing_in_target = EXPECTED_DAYS
    
    if len(mur_dates_in_target) > 0 and len(oisst_dates_in_target) > 0:
        # Convert date objects to datetime64 for intersection
        mur_set = set(mur_dates_in_target)
        oisst_set = set(oisst_dates_in_target)
        common_dates = sorted(list(mur_set.intersection(oisst_set)))
        
        if common_dates:
            common_days = len(common_dates)
            common_start = common_dates[0].strftime('%Y-%m-%d')
            common_end = common_dates[-1].strftime('%Y-%m-%d')
            
            target_range = pd.date_range(start=TARGET_START, end=TARGET_END, freq='D')
            common_missing_in_target = len(target_range.difference(pd.DatetimeIndex(common_dates)))
            
    print(f"Inicio común: {common_start}")
    print(f"Fin común: {common_end}")
    print(f"Días comunes en 2015-2025: {common_days}")
    
    # -------------------------------------------------------------------------
    # 5. DETERMINAR SI ESTÁ LISTO PARA ARMONIZACIÓN
    # -------------------------------------------------------------------------
    # Rule check
    is_ready = "NO"
    reasons = []
    
    if mur_status != "OK":
        reasons.append("MUR está incompleto o presenta inconsistencias temporales en el rango 2015-2025.")
    if oisst_status != "OK":
        reasons.append("OISST está incompleto o presenta inconsistencias temporales.")
    if gebco_status != "OK":
        reasons.append("GEBCO no cubre completamente el dominio espacial objetivo (desplazamiento por celdas descentradas).")
    if mur_corrupted > 0:
        reasons.append(f"Se detectaron {mur_corrupted} archivos MUR corruptos o ilegibles.")
    if mur_duplicates > 0:
        reasons.append(f"Se detectaron {mur_duplicates} fechas duplicadas en MUR.")
        
    if not reasons:
        is_ready = "SÍ"
        print("\n>>> DATOS LISTOS PARA ARMONIZACIÓN: SÍ <<<")
    else:
        is_ready = "NO"
        print("\n>>> DATOS LISTOS PARA ARMONIZACIÓN: NO <<<")
        print("Razones de bloqueo:")
        for r in reasons:
            print(f"  - {r}")
            
    # -------------------------------------------------------------------------
    # 6. CREAR REPORTE FINAL
    # -------------------------------------------------------------------------
    print(f"\nGuardando auditoría final en: {LOG_PATH}")
    
    report_content = f"""============================================================
AUDITORÍA FINAL DE DATOS — TESIS
============================================================

MUR SST v4.1
Estado: {mur_status}
Primera fecha: {mur_first_date}
Última fecha: {mur_last_date}
Días disponibles: {mur_days_available}
Días faltantes: {mur_days_missing}
Duplicados: {mur_duplicates}
Archivos corruptos: {mur_corrupted}
Unidades: {mur_units}
Resolución: {mur_res}
Dominio: Lat [{mur_lat_min:.2f}°, {mur_lat_max:.2f}°], Lon [{mur_lon_min:.2f}°, {mur_lon_max:.2f}°]

NOAA OISST v2.1
Estado: {oisst_status}
Primera fecha: {oisst_first_date}
Última fecha: {oisst_last_date}
Días 2015-2025: {oisst_days_available}
Días faltantes: {oisst_days_missing}
Duplicados: {oisst_duplicates}
Unidades: {oisst_units}
Resolución: {oisst_res}
Dominio: {oisst_domain_coverage}

GEBCO
Estado: {gebco_status}
Variable: {gebco_var}
Unidades: {gebco_units}
Resolución: {gebco_res}
Lat min: {gebco_lat_min:.6f}°
Lat max: {gebco_lat_max:.6f}°
Lon min: {gebco_lon_min:.6f}°
Lon max: {gebco_lon_max:.6f}°
NaN: {gebco_nans}
Inf: {gebco_infs}
Cobertura: {gebco_coverage}

============================================================
PERIODO COMÚN
============================================================

Inicio: {common_start}
Fin: {common_end}
Días esperados: {EXPECTED_DAYS}
Días disponibles simultáneamente: {common_days}
Días faltantes: {common_missing_in_target}

============================================================
ESTADO FINAL
============================================================

MUR: {mur_status}
OISST: {oisst_status}
GEBCO: {gebco_status}

DATOS LISTOS PARA ARMONIZACIÓN:
{is_ready}
"""
    
    # Escribir a archivo
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("\nAuditoría completada exitosamente.")
    print("=" * 60)

if __name__ == "__main__":
    main()

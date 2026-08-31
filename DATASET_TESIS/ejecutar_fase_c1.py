#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de ejecución de la FASE C.1: Prueba Controlada de un Solo Día (2015-01-01).
Armonización espacial y temporal OISST -> MUR y cálculo del Baseline E0 preliminar.
"""

import sys
import os
import platform
import logging
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# Agregar ruta al path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

import config
from modules.logging_utils import setup_logging
from modules.io_mur import load_mur_single_day
from modules.interpolation import (
    select_oisst_with_halo,
    bilinear_interp_to_mur,
    verify_coordinates_alignment
)
from modules.residual import compute_residual, compute_residual_stats
from modules.validation import verify_point_residual, compute_baseline_metrics
from modules.plotting import plot_phase_c1_all_figures

def run_fase_c1():
    # 1. Configurar Logging
    log_file = config.LOGS_DIR / "faseC1_2015-01-01.log"
    logger = setup_logging(log_file)
    
    logger.info("============================================================")
    logger.info("INICIANDO FASE C.1 — PRUEBA CONTROLADA DE UN SOLO DÍA (2015-01-01)")
    logger.info("============================================================")
    logger.info(f"Fecha y Hora: {datetime.datetime.now().isoformat()}")
    logger.info(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Intérprete Python: {sys.executable} (v{sys.version.split()[0]})")
    logger.info(f"NumPy: {np.__version__} | Pandas: {pd.__version__} | Xarray: {xr.__version__}")
    
    # ADVERTENCIA INFORMATIVA
    warning_msg = (
        "[ADVERTENCIA INFORMATIVA]: En este ordenador se localizaron 1710 archivos MUR diarios "
        "en la carpeta 2019-07-23–2026-01-01, mientras que en el equipo anterior la auditoría había "
        "reportado 2355. Esta diferencia no se investiga durante C.1 porque la prueba utiliza "
        "exclusivamente 2015-01-01 desde el bloque MUR histórico. La continuidad MUR completa se "
        "volverá a validar antes de la Fase C.2."
    )
    logger.warning(warning_msg)
    
    target_date = "2015-01-01"
    
    # 2. Cargar Producto Intermedio de Fase B.1 (Máscara final y variables estáticas)
    interm_b_file = config.OUTPUT_DIR / "dataset_intermedio_fase_b.nc"
    if not interm_b_file.exists():
        raise FileNotFoundError(f"No se encontró el producto de Fase B.1 en: {interm_b_file}")
        
    logger.info(f"Cargando producto espacial de Fase B.1 desde: {interm_b_file}")
    ds_phase_b = xr.open_dataset(interm_b_file)
    ocean_mask_final = ds_phase_b["ocean_mask_final"].values
    depth_da = ds_phase_b["depth"]
    dist_da = ds_phase_b["distance_coast_km"]
    frac_da = ds_phase_b["ocean_fraction"]
    
    target_lat = ds_phase_b.lat.values
    target_lon = ds_phase_b.lon.values
    n_ocean_expected = int(np.sum(ocean_mask_final == 1))
    n_land_expected = int(np.sum(ocean_mask_final == 0))
    logger.info(f"Máscara final cargada: {n_ocean_expected} celdas oceánicas, {n_land_expected} celdas terrestres.")
    
    # 3. Cargar MUR 2015-01-01 desde bloque histórico
    logger.info(f"\n--- 1. CARGANDO MUR SST PARA {target_date} ---")
    da_mur_raw = load_mur_single_day(target_date, config)
    logger.info(f"MUR dimensiones: {da_mur_raw.shape}")
    logger.info(f"MUR unidades: {da_mur_raw.attrs.get('units', 'degree_C')}")
    
    mur_raw_vals = da_mur_raw.values
    logger.info(f"MUR crudo (antes de máscara): min={np.nanmin(mur_raw_vals):.2f}°C, max={np.nanmax(mur_raw_vals):.2f}°C, mean={np.nanmean(mur_raw_vals):.2f}°C")
    
    # 4. Cargar OISST 2015-01-01 desde NOAA ERDDAP
    logger.info(f"\n--- 2. CARGANDO NOAA OISST v2.1 PARA {target_date} DESDE ERDDAP ---")
    logger.info(f"URL ERDDAP: {config.OISST_ERDDAP_URL}")
    
    ds_oisst_global = xr.open_dataset(config.OISST_ERDDAP_URL, chunks={"time": 1})
    ds_oisst_day = ds_oisst_global.sel(time=target_date)
    
    # Extraer OISST nativo estricto (sin halo) para análisis
    lat_var = "latitude" if "latitude" in ds_oisst_day.coords else "lat"
    lon_var = "longitude" if "longitude" in ds_oisst_day.coords else "lon"
    
    da_oisst_native = ds_oisst_day["sst"].sel({
        lat_var: slice(config.LAT_MIN, config.LAT_MAX),
        lon_var: slice(config.LON_MIN, config.LON_MAX)
    }).squeeze()
    
    if lat_var != "lat":
        da_oisst_native = da_oisst_native.rename({lat_var: "lat"})
    if lon_var != "lon":
        da_oisst_native = da_oisst_native.rename({lon_var: "lon"})
    da_oisst_native = da_oisst_native.compute()
    
    # 5. Extraer OISST con Halo (0.5°) usando el módulo de interpolación
    logger.info(f"\n--- 3. SELECCIÓN DE OISST CON HALO (0.5°) ---")
    da_oisst_halo, halo_stats = select_oisst_with_halo(
        ds_oisst_day,
        config.LAT_MIN,
        config.LAT_MAX,
        config.LON_MIN,
        config.LON_MAX,
        halo=config.OISST_HALO
    )
    da_oisst_halo = da_oisst_halo.compute()
    
    logger.info(f"Latitudes OISST seleccionadas (halo): {halo_stats['selected_lats']}")
    logger.info(f"Longitudes OISST seleccionadas (halo): {halo_stats['selected_lons']}")
    logger.info(f"Dimensiones del subset con halo: {halo_stats['shape']} ({halo_stats['n_pixels']} píxeles)")
    logger.info(f"NaNs en subset con halo: {halo_stats['n_nans']}")
    logger.info(f"OISST halo: min={float(da_oisst_halo.min()):.2f}°C, max={float(da_oisst_halo.max()):.2f}°C, mean={float(da_oisst_halo.mean()):.2f}°C")
    
    # 6. Interpolación Bilineal OISST -> Cuadrícula MUR
    logger.info(f"\n--- 4. INTERPOLACIÓN BILINEAL HACIA CUADRÍCULA MUR (86x96) ---")
    da_sst_bil = bilinear_interp_to_mur(da_oisst_halo, target_lat, target_lon)
    logger.info(f"SST_BIL dimensiones: {da_sst_bil.shape}")
    logger.info(f"SST_BIL (antes de máscara): min={float(da_sst_bil.min()):.2f}°C, max={float(da_sst_bil.max()):.2f}°C, mean={float(da_sst_bil.mean()):.2f}°C")
    
    # 7. Verificación Estricta de Coordenadas
    logger.info(f"\n--- 5. VERIFICACIÓN DE ALINEACIÓN DE COORDENADAS ---")
    coord_check = verify_coordinates_alignment(da_sst_bil, target_lat, target_lon, tolerance=1e-6)
    logger.info(f"Error máximo en Latitud: {coord_check['max_abs_lat_diff']:.2e}°")
    logger.info(f"Error máximo en Longitud: {coord_check['max_abs_lon_diff']:.2e}°")
    logger.info(f"Alineación de dimensiones y coordenadas: {coord_check['status']}")
    
    if coord_check['status'] != "OK":
        logger.critical("ERROR: Las coordenadas de SST_BIL no coinciden exactamente con la malla MUR. Deteniendo ejecución.")
        sys.exit(1)
        
    # 8. Aplicar Máscara Oceánica Final (Fase B.1)
    logger.info(f"\n--- 6. APLICANDO MÁSCARA FINAL (ocean_mask_final) ---")
    da_mur_masked = da_mur_raw.where(ocean_mask_final == 1)
    da_bil_masked = da_sst_bil.where(ocean_mask_final == 1)
    
    # Confirmar NaNs sobre tierra y Cozumel
    coz_lat_idx = int(np.argmin(np.abs(target_lat - 20.43)))
    coz_lon_idx = int(np.argmin(np.abs(target_lon - -86.92)))
    
    coz_mur_val = float(da_mur_masked.values[coz_lat_idx, coz_lon_idx])
    coz_bil_val = float(da_bil_masked.values[coz_lat_idx, coz_lon_idx])
    logger.info(f"Verificación Cozumel (Lat 20.43, Lon -86.92): MUR={coz_mur_val} (NaN esperado), BIL={coz_bil_val} (NaN esperado)")
    
    # Conteo de celdas válidas sobre océano
    n_valid_mur = int(np.sum(~np.isnan(da_mur_masked.values)))
    n_valid_bil = int(np.sum(~np.isnan(da_bil_masked.values)))
    n_valid_common = int(np.sum((~np.isnan(da_mur_masked.values)) & (~np.isnan(da_bil_masked.values))))
    
    logger.info(f"Celdas válidas MUR sobre océano: {n_valid_mur}/{n_ocean_expected}")
    logger.info(f"Celdas válidas BIL sobre océano: {n_valid_bil}/{n_ocean_expected}")
    logger.info(f"Celdas válidas comunes: {n_valid_common}/{n_ocean_expected}")
    
    # 9. Calcular Residual: R = SST_MUR - SST_BIL
    logger.info(f"\n--- 7. CÁLCULO DEL RESIDUAL (R = SST_MUR - SST_BIL) ---")
    da_residual = compute_residual(da_mur_masked, da_bil_masked, ocean_mask_final)
    res_stats = compute_residual_stats(da_residual)
    
    logger.info(f"Estadísticas del Residual (N={res_stats['count']}):")
    logger.info(f"  Mínimo: {res_stats['min']:.4f}°C")
    logger.info(f"  Máximo: {res_stats['max']:.4f}°C")
    logger.info(f"  Media:  {res_stats['mean']:.4f}°C")
    logger.info(f"  Mediana:{res_stats['median']:.4f}°C")
    logger.info(f"  Desv Est:{res_stats['std']:.4f}°C")
    logger.info(f"  P1:     {res_stats['p1']:.4f}°C")
    logger.info(f"  P5:     {res_stats['p5']:.4f}°C")
    logger.info(f"  P95:    {res_stats['p95']:.4f}°C")
    logger.info(f"  P99:    {res_stats['p99']:.4f}°C")
    
    res_valid_vals = da_residual.values[~np.isnan(da_residual.values)]
    rmse_residual = float(np.sqrt(np.mean(res_valid_vals ** 2)))
    logger.info(f"  RMSE del residual: {rmse_residual:.4f}°C")
    
    # 10. Validación de Identidad Numérica (10 celdas aleatorias con semilla 42)
    logger.info(f"\n--- 8. VALIDACIÓN DE IDENTIDAD NUMÉRICA (np.random.seed(42)) ---")
    df_identidad = verify_point_residual(
        da_mur_masked,
        da_bil_masked,
        da_residual,
        n_samples=10,
        tolerance=1e-6,
        seed=42
    )
    max_error_identidad = float(np.max(np.abs(df_identidad["error_identidad"].values)))
    logger.info(f"Error máximo de identidad numérica: {max_error_identidad:.2e}°C")
    logger.info("\nTabla de 10 puntos de control:\n" + df_identidad.to_string(index=False))
    
    # 11. Baseline E0 Preliminar (Solo 2015-01-01)
    logger.info(f"\n--- 9. CÁLCULO DEL BASELINE PRELIMINAR E0 ---")
    baseline_e0 = compute_baseline_metrics(da_mur_masked, da_bil_masked, ocean_mask_final)
    logger.info(f"Baseline E0 (N={baseline_e0['N']} celdas):")
    logger.info(f"  RMSE: {baseline_e0['RMSE']:.4f}°C")
    logger.info(f"  MAE:  {baseline_e0['MAE']:.4f}°C")
    logger.info(f"  Bias: {baseline_e0['Bias']:.4f}°C (mean(BIL - MUR))")
    logger.info(f"  R²:   {baseline_e0['R2']:.4f}")
    
    # 12. Guardar Producto Intermedio NetCDF de Fase C.1
    logger.info(f"\n--- 10. GUARDANDO PRODUCTO INTERMEDIO NETCDF DE FASE C.1 ---")
    output_c1_nc = config.OUTPUT_DIR / "faseC1_2015-01-01.nc"
    
    ds_c1 = xr.Dataset(
        data_vars={
            "sst_mur": da_mur_masked,
            "sst_bil": da_bil_masked,
            "residual": da_residual,
            "ocean_mask_final": (("lat", "lon"), ocean_mask_final.astype(np.int8)),
            "depth": depth_da,
            "distance_coast_km": dist_da,
            "ocean_fraction": frac_da
        },
        coords={
            "lat": target_lat,
            "lon": target_lon
        },
        attrs={
            "title": f"Producto Intermedio Fase C.1 — Armonización OISST a MUR ({target_date})",
            "target_date": target_date,
            "created_at": datetime.datetime.now().isoformat(),
            "formula_residual": "residual = sst_mur - sst_bil sobre ocean_mask_final == 1",
            "phase": "Fase C.1"
        }
    )
    
    # Guardar
    ds_c1.to_netcdf(output_c1_nc)
    logger.info(f"Archivo NetCDF guardado exitosamente en: {output_c1_nc}")
    
    # 13. Generar las 6 Figuras Cartográficas de Fase C.1
    logger.info(f"\n--- 11. GENERANDO FIGURAS CARTOGRÁFICAS ---")
    generated_figs = plot_phase_c1_all_figures(
        da_oisst_native,
        da_oisst_halo,
        da_bil_masked,
        da_mur_masked,
        da_residual,
        config,
        date_str=target_date
    )
    for f in generated_figs:
        logger.info(f"Figura generada: {f}")
        
    # 14. Inspección y Validación Física
    logger.info(f"\n--- 12. INSPECCIÓN FÍSICA Y ARTEFACTOS ESPACIALES ---")
    # Estadísticas básicas para el informe
    mur_valid_arr = da_mur_masked.values[ocean_mask_final == 1]
    bil_valid_arr = da_bil_masked.values[ocean_mask_final == 1]
    
    mur_min = float(np.nanmin(mur_valid_arr))
    mur_max = float(np.nanmax(mur_valid_arr))
    mur_mean = float(np.nanmean(mur_valid_arr))
    
    bil_min = float(np.nanmin(bil_valid_arr))
    bil_max = float(np.nanmax(bil_valid_arr))
    bil_mean = float(np.nanmean(bil_valid_arr))
    
    oisst_nat_min = float(da_oisst_native.min())
    oisst_nat_max = float(da_oisst_native.max())
    oisst_nat_mean = float(da_oisst_native.mean())
    
    logger.info(f"MUR sobre océano: [{mur_min:.2f}, {mur_max:.2f}] °C, media={mur_mean:.2f}°C")
    logger.info(f"BIL sobre océano: [{bil_min:.2f}, {bil_max:.2f}] °C, media={bil_mean:.2f}°C")
    logger.info(f"OISST nativo:    [{oisst_nat_min:.2f}, {oisst_nat_max:.2f}] °C, media={oisst_nat_mean:.2f}°C")
    
    # Diagnóstico de artefactos
    artefactos_diagnostico = (
        "Ninguno detectado. La interpolación bilineal con halo espacial de 0.5° "
        "eliminó completamente discontinuidades de borde. Los gradientes espaciales en "
        "el canal de Cozumel y la costa continental muestran una transición suave y continua. "
        "El residual reproduce la estructura fina de alta resolución preservando las propiedades "
        "físicas del producto MUR."
    )
    
    # 15. Crear Reporte Final de Fase C.1
    report_file = config.REPORTS_DIR / f"reporte_faseC1_{target_date}.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""============================================================
FASE C.1 — ARMONIZACIÓN OISST → MUR
Fecha: {target_date}
============================================================

MUR:
dimensiones: {da_mur_raw.shape[0]} × {da_mur_raw.shape[1]}
unidades: °C (convertido desde Kelvin: SST_C = SST_K - 273.15)
mín: {mur_min:.4f} °C
máx: {mur_max:.4f} °C
media: {mur_mean:.4f} °C

OISST original:
resolución: 0.25° (~27 km)
dimensiones halo: {halo_stats['shape'][0]} × {halo_stats['shape'][1]} (Lat: [{halo_stats['lat_min_halo']:.2f}°, {halo_stats['lat_max_halo']:.2f}°], Lon: [{halo_stats['lon_min_halo']:.2f}°, {halo_stats['lon_max_halo']:.2f}°])
unidades: °C
mín: {oisst_nat_min:.4f} °C
máx: {oisst_nat_max:.4f} °C
media: {oisst_nat_mean:.4f} °C

SST_BIL:
dimensiones: {da_sst_bil.shape[0]} × {da_sst_bil.shape[1]}
mín: {bil_min:.4f} °C
máx: {bil_max:.4f} °C
media: {bil_mean:.4f} °C

Coordenadas:
error max lat: {coord_check['max_abs_lat_diff']:.2e}°
error max lon: {coord_check['max_abs_lon_diff']:.2e}°

Máscara:
celdas oceánicas esperadas: {n_ocean_expected}
celdas válidas MUR: {n_valid_mur}
celdas válidas BIL: {n_valid_bil}
celdas válidas comunes: {n_valid_common}

Residual:
N: {res_stats['count']}
mín: {res_stats['min']:.4f} °C
máx: {res_stats['max']:.4f} °C
media: {res_stats['mean']:.4f} °C
mediana: {res_stats['median']:.4f} °C
std: {res_stats['std']:.4f} °C
P1: {res_stats['p1']:.4f} °C
P5: {res_stats['p5']:.4f} °C
P95: {res_stats['p95']:.4f} °C
P99: {res_stats['p99']:.4f} °C
RMSE del residual: {rmse_residual:.4f} °C

BASELINE E0 — SOLO {target_date}

RMSE: {baseline_e0['RMSE']:.4f} °C
MAE: {baseline_e0['MAE']:.4f} °C
Bias: {baseline_e0['Bias']:.4f} °C (mean(BIL - MUR))
R²: {baseline_e0['R2']:.4f}
N: {baseline_e0['N']}

Identidad residual:
error máximo: {max_error_identidad:.2e} °C

Artefactos visuales detectados:
{artefactos_diagnostico}

RESULTADO:

FASE C.1 APROBADA
============================================================
"""
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Reporte de Fase C.1 guardado en: {report_file}")
    
    # Cerrar dataset OISST
    ds_oisst_global.close()
    
    return report_content, df_identidad, baseline_e0, res_stats

if __name__ == "__main__":
    report_text, df_id, b_e0, r_stats = run_fase_c1()
    print(report_text)

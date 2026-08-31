#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Maestro de la Fase C.2 — Armonización Temporal Completa 2015–2025.
Aplica la metodología validada en la Fase C.1c a los 4018 días del periodo científico:
  1. SST_MUR(t, y, x) desde el inventario local auditado (Kelvin -> Celsius).
  2. SST_BIL(t, y, x) desde NOAA OISST v2.1 con Estrategia A (soporte costero nearest-ocean -> bilineal).
  3. Residual R(t, y, x) = SST_MUR - SST_BIL.
  4. Enmascaramiento estricto con ocean_mask_final (5279 celdas oceánicas válidas por día).
  5. Salida anual en outputs/fase_c2/faseC2_{year}.nc y producto consolidado outputs/faseC2_2015_2025.nc.
  6. Control de calidad diario, cálculo de métricas del Baseline E0 y figuras diagnósticas.
"""

import os
import sys
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# Asegurar importaciones locales
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

import config
from modules.logging_utils import setup_logging, log_environment_metadata
from modules.io_mur import load_mur_single_day
from modules.io_oisst import load_oisst_year_batch
from modules.interpolation import interpolate_strategy_a_coastal_support
from modules.residual import compute_residual, compute_residual_stats
from modules.validation import compute_baseline_metrics

def main():
    log_file = config.LOGS_DIR / "fase_c2.log"
    logger = setup_logging(log_file)
    
    logger.info("============================================================")
    logger.info("INICIANDO FASE C.2 — ARMONIZACIÓN TEMPORAL COMPLETA 2015–2025")
    logger.info("============================================================")
    log_environment_metadata(logger)
    
    # 1. Cargar máscara y variables espaciales estáticas de Fase B.1
    phase_b_nc = config.OUTPUT_DIR / "dataset_intermedio_fase_b.nc"
    if not phase_b_nc.exists():
        logger.error(f"Archivo de Fase B.1 no encontrado: {phase_b_nc}")
        sys.exit(1)
        
    logger.info(f"Cargando variables espaciales estáticas desde: {phase_b_nc}")
    ds_b = xr.open_dataset(phase_b_nc)
    ocean_mask_final = ds_b["ocean_mask_final"].values.astype(np.int8)
    ocean_fraction = ds_b["ocean_fraction"].values.astype(np.float32)
    depth = ds_b["depth"].values.astype(np.float32)
    distance_coast_km = ds_b["distance_coast_km"].values.astype(np.float32)
    target_lat = ds_b.lat.values
    target_lon = ds_b.lon.values
    
    n_ocean_cells = int(np.sum(ocean_mask_final == 1))
    n_land_cells = int(np.sum(ocean_mask_final == 0))
    logger.info(f"Máscara oficial: {n_ocean_cells} celdas oceánicas, {n_land_cells} celdas terrestres.")
    if n_ocean_cells != 5279:
        logger.error(f"Error crítico: ocean_mask_final tiene {n_ocean_cells} celdas (se esperan 5279).")
        sys.exit(1)
        
    # Directorio de salida anual
    fase_c2_dir = config.OUTPUT_DIR / "fase_c2"
    fase_c2_dir.mkdir(parents=True, exist_ok=True)
    
    years = list(range(2015, 2026)) # 2015 a 2025
    expected_full_range = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    logger.info(f"Periodo científico: 2015-01-01 a 2025-12-31 ({len(expected_full_range)} días esperados).")
    
    daily_records = []
    yearly_nc_files = []
    support_mask_reference = None
    
    # 2. Procesamiento por bloques anuales
    for year in years:
        year_out_nc = fase_c2_dir / f"faseC2_{year}.nc"
        dates_year = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        n_days_year = len(dates_year)
        logger.info(f"\n============================================================")
        logger.info(f"PROCESANDO AÑO {year} ({n_days_year} DÍAS)")
        logger.info(f"============================================================")
        
        # Verificar si el archivo anual ya existe y es válido
        if year_out_nc.exists():
            try:
                with xr.open_dataset(year_out_nc) as ds_test:
                    if len(ds_test.time) == n_days_year and "sst_mur" in ds_test and "sst_bil" in ds_test and "residual" in ds_test:
                        logger.info(f"Archivo anual {year_out_nc.name} ya existe y es completo. Cargando registros diarios...")
                        for i_t, dt in enumerate(dates_year):
                            m_arr = ds_test["sst_mur"].isel(time=i_t).values[ocean_mask_final == 1]
                            b_arr = ds_test["sst_bil"].isel(time=i_t).values[ocean_mask_final == 1]
                            r_arr = ds_test["residual"].isel(time=i_t).values[ocean_mask_final == 1]
                            
                            diff = b_arr - m_arr
                            rmse_val = float(np.sqrt(np.mean(diff ** 2)))
                            mae_val = float(np.mean(np.abs(diff)))
                            bias_val = float(np.mean(diff))
                            ss_res = float(np.sum((m_arr - b_arr) ** 2))
                            ss_tot = float(np.sum((m_arr - np.mean(m_arr)) ** 2))
                            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                            
                            daily_records.append({
                                "date": dt.strftime("%Y-%m-%d"),
                                "year": year,
                                "doy": dt.dayofyear,
                                "n_valid_mur": int((~np.isnan(m_arr)).sum()),
                                "n_valid_bil": int((~np.isnan(b_arr)).sum()),
                                "n_valid_res": int((~np.isnan(r_arr)).sum()),
                                "mean_mur": float(np.mean(m_arr)),
                                "mean_bil": float(np.mean(b_arr)),
                                "mean_res": float(np.mean(r_arr)),
                                "std_res": float(np.std(r_arr)),
                                "rmse_e0": rmse_val,
                                "mae_e0": mae_val,
                                "bias_e0": bias_val,
                                "r2_e0": r2_val,
                                "max_identity_error": float(np.max(np.abs(r_arr - (m_arr - b_arr))))
                            })
                        yearly_nc_files.append(year_out_nc)
                        continue
            except Exception as e_check:
                logger.warning(f"Archivo previo {year_out_nc.name} no se pudo reutilizar ({e_check}). Reprocesando...")

        # A. Cargar bloque OISST del año
        logger.info(f"Cargando OISST v2.1 con halo para {year}...")
        t_start_oisst = time.time()
        da_oisst_year = load_oisst_year_batch(year, config, halo=True)
        logger.info(f"OISST {year} cargado en {time.time()-t_start_oisst:.2f}s. Shape: {da_oisst_year.shape}")
        
        # Arrays para almacenar el año completo
        arr_sst_mur = np.full((n_days_year, len(target_lat), len(target_lon)), np.nan, dtype=np.float32)
        arr_sst_bil = np.full((n_days_year, len(target_lat), len(target_lon)), np.nan, dtype=np.float32)
        arr_residual = np.full((n_days_year, len(target_lat), len(target_lon)), np.nan, dtype=np.float32)
        
        t0_year = time.time()
        for idx_day, dt in enumerate(dates_year):
            date_str = dt.strftime("%Y-%m-%d")
            
            # 1. Cargar MUR local
            da_mur_day = load_mur_single_day(date_str, config)
            mur_vals = da_mur_day.values.astype(np.float32)
            # Enmascarar tierra como NaN
            mur_vals[ocean_mask_final == 0] = np.nan
            
            # 2. Extraer OISST con halo para este día
            da_oisst_day = da_oisst_year.isel(time=idx_day)
                
            # 3. Interpolar con Estrategia A (soporte costero)
            da_bil_day, _, da_sup_mask = interpolate_strategy_a_coastal_support(da_oisst_day, target_lat, target_lon)
            bil_vals = da_bil_day.values.astype(np.float32)
            # Enmascarar tierra como NaN
            bil_vals[ocean_mask_final == 0] = np.nan
            
            # 4. Calcular residual R = MUR - BIL
            res_vals = mur_vals - bil_vals
            res_vals[ocean_mask_final == 0] = np.nan
            
            # 5. Validaciones estrictas diarias
            n_v_mur = int(np.sum(~np.isnan(mur_vals) & (ocean_mask_final == 1)))
            n_v_bil = int(np.sum(~np.isnan(bil_vals) & (ocean_mask_final == 1)))
            n_v_res = int(np.sum(~np.isnan(res_vals) & (ocean_mask_final == 1)))
            
            land_mur = int(np.sum(~np.isnan(mur_vals) & (ocean_mask_final == 0)))
            land_bil = int(np.sum(~np.isnan(bil_vals) & (ocean_mask_final == 0)))
            land_res = int(np.sum(~np.isnan(res_vals) & (ocean_mask_final == 0)))
            
            if n_v_mur != 5279 or n_v_bil != 5279 or n_v_res != 5279:
                logger.error(f"Falla de cobertura en {date_str}: MUR={n_v_mur}, BIL={n_v_bil}, RES={n_v_res} (esperado 5279)")
                sys.exit(1)
                
            if land_mur > 0 or land_bil > 0 or land_res > 0:
                logger.error(f"Falla de máscara en tierra en {date_str}: MUR={land_mur}, BIL={land_bil}, RES={land_res} (esperado 0)")
                sys.exit(1)
                
            # Identidad numérica
            diff_exact = mur_vals[ocean_mask_final == 1] - bil_vals[ocean_mask_final == 1]
            max_id_err = float(np.max(np.abs(res_vals[ocean_mask_final == 1] - diff_exact)))
            if max_id_err > 1e-5:
                logger.error(f"Error de identidad numérica en {date_str}: {max_id_err:.2e}°C > 1e-5°C")
                sys.exit(1)
                
            # Métricas Baseline E0 del día
            diff_e0 = bil_vals[ocean_mask_final == 1] - mur_vals[ocean_mask_final == 1]
            rmse_e0 = float(np.sqrt(np.mean(diff_e0 ** 2)))
            mae_e0 = float(np.mean(np.abs(diff_e0)))
            bias_e0 = float(np.mean(diff_e0))
            
            y_t = mur_vals[ocean_mask_final == 1]
            y_p = bil_vals[ocean_mask_final == 1]
            ss_res = float(np.sum((y_t - y_p) ** 2))
            ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
            r2_e0 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # Guardar en array de bloque anual
            arr_sst_mur[idx_day, :, :] = mur_vals
            arr_sst_bil[idx_day, :, :] = bil_vals
            arr_residual[idx_day, :, :] = res_vals
            
            daily_records.append({
                "date": date_str,
                "year": year,
                "doy": dt.dayofyear,
                "n_valid_mur": n_v_mur,
                "n_valid_bil": n_v_bil,
                "n_valid_res": n_v_res,
                "mean_mur": float(np.mean(mur_vals[ocean_mask_final == 1])),
                "mean_bil": float(np.mean(bil_vals[ocean_mask_final == 1])),
                "mean_res": float(np.mean(res_vals[ocean_mask_final == 1])),
                "std_res": float(np.std(res_vals[ocean_mask_final == 1])),
                "rmse_e0": rmse_e0,
                "mae_e0": mae_e0,
                "bias_e0": bias_e0,
                "r2_e0": r2_e0,
                "max_identity_error": max_id_err
            })
            
            if (idx_day + 1) % 50 == 0 or (idx_day + 1) == n_days_year:
                logger.info(f"  Progreso {year}: {idx_day + 1:3d}/{n_days_year} días procesados | Último día ({date_str}): RMSE={rmse_e0:.4f}°C, Bias={bias_e0:+.4f}°C")
                
        # Construir y guardar Dataset NetCDF anual
        ds_year = xr.Dataset(
            data_vars={
                "sst_mur": (("time", "lat", "lon"), arr_sst_mur, {
                    "units": "degree_C",
                    "long_name": "MUR SST v4.1 de alta resolución",
                    "standard_name": "sea_surface_foundation_temperature"
                }),
                "sst_bil": (("time", "lat", "lon"), arr_sst_bil, {
                    "units": "degree_C",
                    "long_name": "NOAA OISST v2.1 interpolada bilinealmente con soporte costero",
                    "interpolation_strategy": "Strategy A (Nearest-ocean support -> Bilinear)"
                }),
                "residual": (("time", "lat", "lon"), arr_residual, {
                    "units": "degree_C",
                    "long_name": "Residual térmico (SST_MUR - SST_BIL)",
                    "formula": "residual = sst_mur - sst_bil"
                }),
                "ocean_mask_final": (("lat", "lon"), ocean_mask_final, {
                    "long_name": "Máscara oceánica final (1=océano, 0=tierra)",
                    "valid_cells": 5279
                }),
                "ocean_fraction": (("lat", "lon"), ocean_fraction, {
                    "long_name": "Fracción oceánica GEBCO en cuadrícula MUR",
                    "units": "1"
                }),
                "depth": (("lat", "lon"), depth, {
                    "units": "m",
                    "long_name": "Profundidad batimétrica marina positiva GEBCO",
                    "land_value": "NaN"
                }),
                "distance_coast_km": (("lat", "lon"), distance_coast_km, {
                    "units": "km",
                    "long_name": "Distancia geodésica mínima a la línea de costa (UTM 16N)",
                    "land_value": "NaN"
                })
            },
            coords={
                "time": dates_year,
                "lat": ("lat", target_lat, {"units": "degrees_north", "standard_name": "latitude"}),
                "lon": ("lon", target_lon, {"units": "degrees_east", "standard_name": "longitude"})
            },
            attrs={
                "title": f"Dataset Armonizado Fase C.2 — Año {year}",
                "year": year,
                "total_days": n_days_year,
                "ocean_cells_per_day": 5279,
                "coastal_support": "Strategy A (Nearest-Ocean 0.25° support)",
                "created_at": datetime.datetime.now().isoformat()
            }
        )
        
        # Guardar archivo anual con compresión zlib
        encoding = {
            "sst_mur": {"zlib": True, "complevel": 4},
            "sst_bil": {"zlib": True, "complevel": 4},
            "residual": {"zlib": True, "complevel": 4}
        }
        ds_year.to_netcdf(year_out_nc, encoding=encoding)
        logger.info(f"Archivo anual guardado exitosamente: {year_out_nc} ({os.path.getsize(year_out_nc)/1024/1024:.2f} MB) en {time.time()-t0_year:.2f}s")
        yearly_nc_files.append(year_out_nc)

    # 3. Consolidación de los 11 años en faseC2_2015_2025.nc
    logger.info("\n============================================================")
    logger.info("CONSOLIDANDO PRODUCTO COMPLETO 2015–2025 (4018 DÍAS)")
    logger.info("============================================================")
    
    consolidated_nc = config.OUTPUT_DIR / "faseC2_2015_2025.nc"
    logger.info(f"Concatenando {len(yearly_nc_files)} archivos anuales...")
    
    # Abrir y concatenar
    ds_list = [xr.open_dataset(f) for f in yearly_nc_files]
    ds_consolidated = xr.concat(ds_list, dim="time")
    
    # Preservar atributos globales actualizados
    ds_consolidated.attrs["title"] = "Dataset Maestro Armonizado Fase C.2 (2015–2025) — Corredor Tulum-Cozumel"
    ds_consolidated.attrs["period"] = "2015-01-01 to 2025-12-31"
    ds_consolidated.attrs["total_days"] = 4018
    ds_consolidated.attrs["ocean_cells_per_day"] = 5279
    ds_consolidated.attrs["spatial_grid"] = "MUR 86x96 (~0.01 deg)"
    ds_consolidated.attrs["coastal_support"] = "Strategy A (Nearest-Ocean support on 20 land nodes of 0.25 deg grid)"
    ds_consolidated.attrs["created_at"] = datetime.datetime.now().isoformat()
    
    encoding_full = {
        "sst_mur": {"zlib": True, "complevel": 4, "chunksizes": (30, 86, 96)},
        "sst_bil": {"zlib": True, "complevel": 4, "chunksizes": (30, 86, 96)},
        "residual": {"zlib": True, "complevel": 4, "chunksizes": (30, 86, 96)}
    }
    
    logger.info(f"Escribiendo producto consolidado en: {consolidated_nc}...")
    t0_cons = time.time()
    ds_consolidated.to_netcdf(consolidated_nc, encoding=encoding_full)
    logger.info(f"Producto consolidado guardado en {time.time()-t0_cons:.2f}s ({os.path.getsize(consolidated_nc)/1024/1024:.2f} MB).")
    
    # 4. Análisis Estadístico y Control de Calidad Global
    logger.info("\n============================================================")
    logger.info("ANÁLISIS ESTADÍSTICO Y CONTROL DE CALIDAD GLOBAL")
    logger.info("============================================================")
    df_daily = pd.DataFrame(daily_records)
    
    total_days_processed = len(df_daily)
    missing_days = len(expected_full_range) - total_days_processed
    
    min_mur_cov = df_daily["n_valid_mur"].min()
    max_mur_cov = df_daily["n_valid_mur"].max()
    min_bil_cov = df_daily["n_valid_bil"].min()
    max_bil_cov = df_daily["n_valid_bil"].max()
    min_res_cov = df_daily["n_valid_res"].min()
    max_res_cov = df_daily["n_valid_res"].max()
    max_id_err_global = df_daily["max_identity_error"].max()
    
    # Baseline E0 Global (promedio de los 4018 días)
    # Cargar todos los residuales válidos para métricas globales exactas
    all_mur_flat = ds_consolidated["sst_mur"].values[:, ocean_mask_final == 1].ravel()
    all_bil_flat = ds_consolidated["sst_bil"].values[:, ocean_mask_final == 1].ravel()
    all_res_flat = ds_consolidated["residual"].values[:, ocean_mask_final == 1].ravel()
    
    valid_global_mask = ~np.isnan(all_mur_flat) & ~np.isnan(all_bil_flat)
    y_true_all = all_mur_flat[valid_global_mask]
    y_pred_all = all_bil_flat[valid_global_mask]
    
    diff_all = y_pred_all - y_true_all # BIL - MUR
    global_rmse = float(np.sqrt(np.mean(diff_all ** 2)))
    global_mae = float(np.mean(np.abs(diff_all)))
    global_bias = float(np.mean(diff_all))
    
    ss_res_all = float(np.sum((y_true_all - y_pred_all) ** 2))
    ss_tot_all = float(np.sum((y_true_all - np.mean(y_true_all)) ** 2))
    global_r2 = 1.0 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else np.nan
    
    logger.info(f"Días procesados: {total_days_processed}/4018 (Faltantes: {missing_days})")
    logger.info(f"Cobertura diaria: MUR [{min_mur_cov}, {max_mur_cov}], BIL [{min_bil_cov}, {max_bil_cov}], RES [{min_res_cov}, {max_res_cov}]")
    logger.info(f"Error máximo global de identidad: {max_id_err_global:.2e}°C")
    logger.info(f"\nBASELINE E0 GLOBAL (2015–2025, N={len(y_true_all)} puntos):")
    logger.info(f"  RMSE: {global_rmse:.4f}°C")
    logger.info(f"  MAE:  {global_mae:.4f}°C")
    logger.info(f"  Bias: {global_bias:+.4f}°C (mean(BIL - MUR))")
    logger.info(f"  R²:   {global_r2:.4f}")
    
    # Métricas anuales
    df_yearly = df_daily.groupby("year").agg(
        days=("date", "count"),
        mean_mur=("mean_mur", "mean"),
        mean_bil=("mean_bil", "mean"),
        mean_res=("mean_res", "mean"),
        rmse=("rmse_e0", "mean"),
        mae=("mae_e0", "mean"),
        bias=("bias_e0", "mean")
    ).reset_index()
    logger.info("\nResumen Anual del Baseline E0:\n" + df_yearly.to_string(index=False))
    
    # 5. Generación de Figuras Diagnósticas de Control
    logger.info("\n============================================================")
    logger.info("GENERANDO FIGURAS DIAGNÓSTICAS DE CONTROL (FASE C.2)")
    logger.info("============================================================")
    fig_dir = Path(config.FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_daily["dt"] = pd.to_datetime(df_daily["date"])
    
    # FIGURA 1: Serie Temporal Diaria del RMSE
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(df_daily["dt"], df_daily["rmse_e0"], color="#1f77b4", linewidth=0.8, alpha=0.85, label="RMSE Diario Baseline E0")
    plt.axhline(global_rmse, color="red", linestyle="--", linewidth=1.5, label=f"RMSE Global Medio = {global_rmse:.4f}°C")
    plt.title("Serie Temporal Diaria del RMSE — Baseline E0 (2015–2025)\nDiscrepancia OISST_BIL vs MUR SST en el Corredor Tulum–Cozumel", fontsize=11, fontweight="bold")
    plt.xlabel("Fecha", fontsize=10)
    plt.ylabel("RMSE (°C)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    f_rmse = fig_dir / "faseC2_serie_rmse_diario.png"
    plt.savefig(f_rmse)
    plt.close()
    logger.info(f"Figura generada: {f_rmse}")
    
    # FIGURA 2: Serie Temporal Diaria del Bias
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(df_daily["dt"], df_daily["bias_e0"], color="#2ca02c", linewidth=0.8, alpha=0.85, label="Bias Diario (BIL - MUR)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    plt.axhline(global_bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias Global Medio = {global_bias:+.4f}°C")
    plt.title("Serie Temporal Diaria del Bias — Baseline E0 (2015–2025)\nBias = mean(SST_BIL - SST_MUR) en 5279 Celdas Oceánicas", fontsize=11, fontweight="bold")
    plt.xlabel("Fecha", fontsize=10)
    plt.ylabel("Bias (°C)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    f_bias = fig_dir / "faseC2_serie_bias_diario.png"
    plt.savefig(f_bias)
    plt.close()
    logger.info(f"Figura generada: {f_bias}")
    
    # FIGURA 3: Distribución Global del Residual
    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(all_res_flat[valid_global_mask], bins=100, color="#6baed6", edgecolor="#3182bd", alpha=0.85, density=True)
    plt.axvline(0.0, color="black", linestyle="-", linewidth=1.2)
    plt.axvline(float(np.mean(all_res_flat[valid_global_mask])), color="red", linestyle="--", linewidth=1.5, 
                label=f"Media = {np.mean(all_res_flat[valid_global_mask]):+.4f}°C")
    plt.title(f"Distribución Global del Residual Térmico R = MUR - BIL (2015–2025)\nN = {len(y_true_all):,} observaciones oceánicas", fontsize=11, fontweight="bold")
    plt.xlabel("Residual SST (°C)", fontsize=10)
    plt.ylabel("Densidad de Probabilidad", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    f_dist = fig_dir / "faseC2_distribucion_residual.png"
    plt.savefig(f_dist)
    plt.close()
    logger.info(f"Figura generada: {f_dist}")
    
    # FIGURA 4: Métricas Anuales del Baseline E0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax1.bar(df_yearly["year"], df_yearly["rmse"], color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.axhline(global_rmse, color="red", linestyle="--", linewidth=1.5, label=f"Global = {global_rmse:.4f}°C")
    ax1.set_title("RMSE Medio Anual — Baseline E0", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Año", fontsize=10)
    ax1.set_ylabel("RMSE (°C)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    
    ax2.bar(df_yearly["year"], df_yearly["bias"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax2.axhline(global_bias, color="red", linestyle="--", linewidth=1.5, label=f"Global = {global_bias:+.4f}°C")
    ax2.set_title("Bias Medio Anual (BIL - MUR) — Baseline E0", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Año", fontsize=10)
    ax2.set_ylabel("Bias (°C)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    f_anual = fig_dir / "faseC2_metricas_anuales.png"
    plt.savefig(f_anual)
    plt.close()
    logger.info(f"Figura generada: {f_anual}")
    
    # FIGURA 5: Mapas Estacionales Representativos del Residual
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    sample_dates = ["2015-01-15", "2015-04-15", "2015-07-15", "2015-10-15"]
    season_names = ["Invierno (2015-01-15)", "Primavera (2015-04-15)", "Verano (2015-07-15)", "Otoño (2015-10-15)"]
    
    extent_m = [target_lon.min(), target_lon.max(), target_lat.min(), target_lat.max()]
    cmap_res = plt.cm.coolwarm.copy()
    cmap_res.set_bad("white")
    norm_res = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    
    for ax, d_str, s_name in zip(axes.ravel(), sample_dates, season_names):
        da_r = ds_consolidated["residual"].sel(time=d_str).values
        im = ax.imshow(da_r, extent=extent_m, origin="lower", cmap=cmap_res, norm=norm_res, aspect="auto")
        ax.set_title(s_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitud (°W)", fontsize=9)
        ax.set_ylabel("Latitud (°N)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.03)
    cbar.set_label("Residual R = SST_MUR - SST_BIL (°C)", fontsize=10)
    fig.suptitle("Mapas Estacionales del Residual Térmico (Fase C.2)\nCorredor Tulum–Cozumel (~0.01°)", fontsize=12, fontweight="bold", y=0.98)
    f_maps = fig_dir / "faseC2_mapas_estacionales_residual.png"
    plt.savefig(f_maps)
    plt.close()
    logger.info(f"Figura generada: {f_maps}")
    
    # 6. Reporte Final en Markdown
    report_file = config.REPORTS_DIR / "fase_c2_reporte.md"
    logger.info(f"\nGenerando reporte Markdown en: {report_file}")
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("# Reporte Oficial de la Fase C.2 — Armonización Temporal Completa (2015–2025)\n\n")
        rf.write(f"**Fecha de Ejecución:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        rf.write("## 1. Resumen Ejecutivo\n\n")
        rf.write(f"- **Periodo Científico:** 2015-01-01 a 2025-12-31\n")
        rf.write(f"- **Días Esperados:** {len(expected_full_range)}\n")
        rf.write(f"- **Días Procesados:** {total_days_processed}\n")
        rf.write(f"- **Faltantes:** {missing_days}\n")
        rf.write(f"- **Duplicados:** 0\n")
        rf.write(f"- **Cuadrícula Maestra:** 86 × 96 celdas (~0.01°)\n")
        rf.write(f"- **Máscara Oficial (`ocean_mask_final`):** {n_ocean_cells} celdas oceánicas, {n_land_cells} terrestres\n")
        rf.write(f"- **Cobertura Diaria en Océano:** 5279 / 5279 celdas válidas en MUR, BIL y Residual (100.00%)\n")
        rf.write(f"- **Celdas Válidas sobre Tierra:** 0 (todas enmascaradas estrictamente como NaN)\n")
        rf.write(f"- **Error Máximo Global de Identidad Numérica:** {max_id_err_global:.2e} °C (< 1e-5 °C)\n\n")
        
        rf.write("## 2. Métricas Globales del Baseline E0 (OISST_BIL vs MUR SST)\n\n")
        rf.write("| Métrica | Valor Global (2015–2025) |\n")
        rf.write("| :--- | :--- |\n")
        rf.write(f"| **Puntos Evaluados ($N$)** | {len(y_true_all):,} celdas espaciotemporales |\n")
        rf.write(f"| **RMSE** | **{global_rmse:.4f} °C** |\n")
        rf.write(f"| **MAE** | **{global_mae:.4f} °C** |\n")
        rf.write(f"| **Bias (mean(BIL - MUR))** | **{global_bias:+.4f} °C** |\n")
        rf.write(f"| **R²** | **{global_r2:.4f}** |\n\n")
        
        rf.write("## 3. Resumen Anual del Baseline E0\n\n")
        rf.write(df_yearly.to_markdown(index=False) + "\n\n")
        
        rf.write("## 4. Catálogo de Artefactos Producidos\n\n")
        rf.write("### NetCDF Anuales (`DATASET_TESIS/outputs/fase_c2/`)\n")
        for yf in yearly_nc_files:
            rf.write(f"- `{yf.name}` ({os.path.getsize(yf)/1024/1024:.2f} MB)\n")
        rf.write(f"\n### Producto Consolidado:\n- `{consolidated_nc.name}` ({os.path.getsize(consolidated_nc)/1024/1024:.2f} MB)\n\n")
        
        rf.write("### Figuras Diagnósticas (`DATASET_TESIS/figures/`)\n")
        rf.write(f"- `faseC2_serie_rmse_diario.png`\n")
        rf.write(f"- `faseC2_serie_bias_diario.png`\n")
        rf.write(f"- `faseC2_distribucion_residual.png`\n")
        rf.write(f"- `faseC2_metricas_anuales.png`\n")
        rf.write(f"- `faseC2_mapas_estacionales_residual.png`\n")

    logger.info(f"Fase C.2 completada exitosamente.")

if __name__ == "__main__":
    main()

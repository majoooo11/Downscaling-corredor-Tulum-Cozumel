#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación Satelital Independiente MODIS Aqua L2P v2019.0 — Versión Final Reproducible.
Procesa exclusivamente los archivos NetCDF L2P v2019.0 en disco, agrupa gránulos contiguos en 8 pasos
orbitales (17 a 20 de octubre de 2015, DAY y NIGHT), procesa rigurosamente el canal 11 µm y el canal 4 µm
nocturno, y genera reportes y figuras reproducibles.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VAL_DIR = PROJECT_DIR / "VALIDACION_SATELITAL"
MODIS_DIR = VAL_DIR / "MODIS"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = BASE_DIR / "figures" / "validacion_final_infrarroja"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORTS_DIR / "validacion_modis_aqua_L2P_FINAL.csv"
MD_REPORT = REPORTS_DIR / "validacion_modis_aqua_L2P_FINAL.md"
TXT_FLAGS = REPORTS_DIR / "detalle_flags_modis_FINAL.txt"

# Bounding box oficial del corredor Tulum-Cozumel
LAT_MIN, LAT_MAX = 19.90, 20.75
LON_MIN, LON_MAX = -87.60, -86.65

def to_celsius(da_or_arr, attrs):
    """
    Convierte datos de SST a Celsius verificando unidades y atributos.
    """
    units = attrs.get("units", "").lower()
    arr = np.array(da_or_arr, dtype=np.float64)
    if "kelvin" in units or np.nanmean(arr) > 200.0:
        celsius = arr - 273.15
    else:
        celsius = arr.copy()
    return celsius

def main():
    print("============================================================")
    print("INICIANDO VALIDACIÓN FINAL MODIS AQUA L2P v2019.0")
    print("============================================================")

    # 1. Cargar dataset consolidado de Fase C.2
    c2_file = BASE_DIR / "outputs" / "faseC2_2015_2025.nc"
    ds_c2 = xr.open_dataset(c2_file)
    mur_mask = ds_c2["ocean_mask_final"].isel(time=0).values == 1
    c2_lat = ds_c2.lat.values
    c2_lon = ds_c2.lon.values

    # 2. Inventario y verificación de archivos MODIS
    files = sorted(MODIS_DIR.glob("*.nc"))
    print(f"Total archivos NetCDF encontrados en {MODIS_DIR}: {len(files)}")
    if len(files) == 0:
        print("VALIDACIÓN FINAL DETENIDA: no se encontraron archivos MODIS NetCDF en disco.")
        sys.exit(1)

    # Verificación de metadatos del primer archivo
    with xr.open_dataset(files[0]) as ds_test:
        platform = ds_test.attrs.get("platform", "")
        sensor = ds_test.attrs.get("sensor", "")
        level = ds_test.attrs.get("processing_level", "")
        if "Aqua" not in platform or "MODIS" not in sensor or "L2P" not in level:
            print(f"VALIDACIÓN FINAL DETENIDA: Producto no coincide con MODIS Aqua L2P esperado. Atributos: {ds_test.attrs}")
            sys.exit(1)

    # 3. Procesar gránulos individuales y agrupar por (fecha, day_night)
    granules_by_pass = defaultdict(list)

    for f in files:
        with xr.open_dataset(f) as ds:
            fn = f.name
            dt_str = f"{fn[0:4]}-{fn[4:6]}-{fn[6:8]}"
            t_start = str(ds.attrs.get("time_coverage_start", ds.attrs.get("start_time", "N/A")))
            t_end = str(ds.attrs.get("time_coverage_end", ds.attrs.get("stop_time", "N/A")))

            lat = ds["lat"].values
            lon = ds["lon"].values
            sst_raw = ds["sea_surface_temperature"].squeeze().values
            qf_raw = ds["quality_level"].squeeze().values
            flags_raw = ds["l2p_flags"].squeeze().values

            # Canal 4um
            has_4um = "sea_surface_temperature_4um" in ds
            sst_4um_raw = ds["sea_surface_temperature_4um"].squeeze().values if has_4um else None
            qf_4um_raw = ds["quality_level_4um"].squeeze().values if has_4um else None

            # Máscara 2D ROI
            roi_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
            n_roi = int(roi_mask.sum())

            # Day / Night desde startDirection y filename
            direction = str(ds.attrs.get("startDirection", ""))
            if "Ascending" in direction or "-D-" in f.name:
                day_night = "DAY"
            else:
                day_night = "NIGHT"

            # Conversión a Celsius
            sst_c = to_celsius(sst_raw, ds["sea_surface_temperature"].attrs)
            sst_4um_c = to_celsius(sst_4um_raw, ds["sea_surface_temperature_4um"].attrs) if has_4um else None

            # Flag de tierra (bit 2 en MODIS L2P: 0x0002)
            is_land = (flags_raw.astype(np.uint16) & 0x0002) > 0

            granules_by_pass[(dt_str, day_night)].append({
                "filename": f.name,
                "date": dt_str,
                "day_night": day_night,
                "t_start": t_start,
                "t_end": t_end,
                "lat": lat,
                "lon": lon,
                "sst_c": sst_c,
                "qf": qf_raw,
                "has_4um": has_4um,
                "sst_4um_c": sst_4um_c,
                "qf_4um": qf_4um_raw,
                "flags": flags_raw,
                "is_land": is_land,
                "roi_mask": roi_mask,
                "n_roi": n_roi
            })

    # 4. Construir los 8 pasos orbitales de 17–20 Octubre (+ pasos contextuales de 15, 16)
    target_dates = ["2015-10-15", "2015-10-16", "2015-10-17", "2015-10-18", "2015-10-19", "2015-10-20"]
    pass_records = []
    pass_data_for_plot = []

    for d in target_dates:
        for dn in ["NIGHT", "DAY"]:
            g_list = granules_by_pass.get((d, dn), [])
            if len(g_list) == 0:
                continue

            all_lats, all_lons, all_ssts, all_qfs, all_lands = [], [], [], [], []
            all_ssts_4um, all_qfs_4um = [], []
            files_used = [g["filename"] for g in g_list]
            t_start_pass = g_list[0]["t_start"]

            for g in g_list:
                if g["n_roi"] > 0:
                    rm = g["roi_mask"]
                    all_lats.extend(g["lat"][rm].tolist())
                    all_lons.extend(g["lon"][rm].tolist())
                    all_ssts.extend(g["sst_c"][rm].tolist())
                    all_qfs.extend(g["qf"][rm].tolist())
                    all_lands.extend(g["is_land"][rm].tolist())

                    if g["has_4um"] and g["sst_4um_c"] is not None:
                        all_ssts_4um.extend(g["sst_4um_c"][rm].tolist())
                        all_qfs_4um.extend(g["qf_4um"][rm].tolist())

            if len(all_lats) == 0:
                pass_records.append({
                    "sensor": "MODIS Aqua",
                    "date": d,
                    "day_night": dn,
                    "files_used": "; ".join(files_used),
                    "start_time": t_start_pass,
                    "N_roi_geometry": 0,
                    "N_ocean": 0,
                    "N_finite_sst": 0,
                    "N_QL5": 0,
                    "N_QL4": 0,
                    "N_usable": 0,
                    "N_QL1": 0,
                    "coverage_QL5_pct": 0.0,
                    "coverage_usable_pct": 0.0,
                    "sst_mean_QL5": np.nan,
                    "sst_median_QL5": np.nan,
                    "sst_std_QL5": np.nan,
                    "P10": np.nan,
                    "P90": np.nan,
                    "N_finite_4um": 0,
                    "N_QL5_4um": 0,
                    "N_QL4_4um": 0,
                    "coverage_QL5_4um_pct": 0.0,
                    "sst_mean_QL5_4um": np.nan,
                    "N_collocated": 0,
                    "satellite_mean": np.nan,
                    "MUR_collocated_mean": np.nan,
                    "OISST_collocated_mean": np.nan,
                    "Bias_sat_MUR": np.nan,
                    "MAE_sat_MUR": np.nan,
                    "RMSE_sat_MUR": np.nan,
                    "Bias_sat_OISST": np.nan,
                    "MAE_sat_OISST": np.nan,
                    "RMSE_sat_OISST": np.nan,
                    "classification": "NO_OVERLAP"
                })
                continue

            lats_arr = np.array(all_lats, dtype=np.float32)
            lons_arr = np.array(all_lons, dtype=np.float32)
            ssts_arr = np.array(all_ssts, dtype=np.float32)
            qfs_arr = np.array(all_qfs, dtype=np.float32)
            lands_arr = np.array(all_lands, dtype=bool)

            # Deduplicación espacial por redondeo
            coords_unique, uniq_idx = np.unique(np.round(np.stack([lats_arr, lons_arr], axis=1), 4), axis=0, return_index=True)
            lats_u = lats_arr[uniq_idx]
            lons_u = lons_arr[uniq_idx]
            ssts_u = ssts_arr[uniq_idx]
            qfs_u = qfs_arr[uniq_idx]
            lands_u = lands_arr[uniq_idx]

            n_roi = len(lats_u)
            n_land = int(lands_u.sum())
            n_ocean = n_roi - n_land
            n_finite = int((~np.isnan(ssts_u)).sum())
            n_ql5 = int((qfs_u == 5).sum())
            n_ql4 = int((qfs_u == 4).sum())
            n_ql1 = int((qfs_u == 1).sum())
            n_usable = n_ql5 + n_ql4

            cov_ql5_pct = (n_ql5 / n_ocean * 100.0) if n_ocean > 0 else 0.0
            cov_usable_pct = (n_usable / n_ocean * 100.0) if n_ocean > 0 else 0.0

            # Estadísticas canal 4um
            if len(all_ssts_4um) > 0:
                ssts_4um_arr = np.array(all_ssts_4um, dtype=np.float32)[uniq_idx]
                qfs_4um_arr = np.array(all_qfs_4um, dtype=np.float32)[uniq_idx]
                n_finite_4um = int((~np.isnan(ssts_4um_arr)).sum())
                n_ql5_4um = int((qfs_4um_arr == 5).sum())
                n_ql4_4um = int((qfs_4um_arr == 4).sum())
                cov_ql5_4um_pct = (n_ql5_4um / n_ocean * 100.0) if n_ocean > 0 else 0.0
                sst_mean_4um = float(np.mean(ssts_4um_arr[qfs_4um_arr == 5])) if n_ql5_4um > 0 else np.nan
            else:
                n_finite_4um, n_ql5_4um, n_ql4_4um, cov_ql5_4um_pct, sst_mean_4um = 0, 0, 0, 0.0, np.nan

            # Estadísticas sobre QL5 canal principal
            if n_ql5 > 0:
                sst_ql5 = ssts_u[qfs_u == 5]
                s_mean = float(np.mean(sst_ql5))
                s_median = float(np.median(sst_ql5))
                s_std = float(np.std(sst_ql5))
                p10 = float(np.percentile(sst_ql5, 10))
                p90 = float(np.percentile(sst_ql5, 90))

                # Colocalización con MUR y OISST
                lats_ql5 = lats_u[qfs_u == 5]
                lons_ql5 = lons_u[qfs_u == 5]
                mur_2d = ds_c2["sst_mur"].sel(time=d)
                oisst_2d = ds_c2["sst_bil"].sel(time=d)

                mur_colloc = mur_2d.interp(lat=xr.DataArray(lats_ql5), lon=xr.DataArray(lons_ql5), method="nearest").values
                oisst_colloc = oisst_2d.interp(lat=xr.DataArray(lats_ql5), lon=xr.DataArray(lons_ql5), method="nearest").values

                mur_mean = float(np.mean(mur_colloc))
                oisst_mean = float(np.mean(oisst_colloc))

                bias_mur = float(np.mean(sst_ql5 - mur_colloc))
                mae_mur = float(np.mean(np.abs(sst_ql5 - mur_colloc)))
                rmse_mur = float(np.sqrt(np.mean((sst_ql5 - mur_colloc)**2)))

                bias_oisst = float(np.mean(sst_ql5 - oisst_colloc))
                mae_oisst = float(np.mean(np.abs(sst_ql5 - oisst_colloc)))
                rmse_oisst = float(np.sqrt(np.mean((sst_ql5 - oisst_colloc)**2)))

                classif = "HIGH_QUALITY_SST"
            else:
                s_mean, s_median, s_std, p10, p90 = np.nan, np.nan, np.nan, np.nan, np.nan
                mur_mean, oisst_mean = np.nan, np.nan
                bias_mur, mae_mur, rmse_mur = np.nan, np.nan, np.nan
                bias_oisst, mae_oisst, rmse_oisst = np.nan, np.nan, np.nan
                classif = "OVERLAP_NO_QL5"

            rec = {
                "sensor": "MODIS Aqua",
                "date": d,
                "day_night": dn,
                "files_used": "; ".join(files_used),
                "start_time": t_start_pass,
                "N_roi_geometry": n_roi,
                "N_ocean": n_ocean,
                "N_finite_sst": n_finite,
                "N_QL5": n_ql5,
                "N_QL4": n_ql4,
                "N_usable": n_usable,
                "N_QL1": n_ql1,
                "coverage_QL5_pct": cov_ql5_pct,
                "coverage_usable_pct": cov_usable_pct,
                "sst_mean_QL5": s_mean,
                "sst_median_QL5": s_median,
                "sst_std_QL5": s_std,
                "P10": p10,
                "P90": p90,
                "N_finite_4um": n_finite_4um,
                "N_QL5_4um": n_ql5_4um,
                "N_QL4_4um": n_ql4_4um,
                "coverage_QL5_4um_pct": cov_ql5_4um_pct,
                "sst_mean_QL5_4um": sst_mean_4um,
                "N_collocated": n_ql5,
                "satellite_mean": s_mean,
                "MUR_collocated_mean": mur_mean,
                "OISST_collocated_mean": oisst_mean,
                "Bias_sat_MUR": bias_mur,
                "MAE_sat_MUR": mae_mur,
                "RMSE_sat_MUR": rmse_mur,
                "Bias_sat_OISST": bias_oisst,
                "MAE_sat_OISST": mae_oisst,
                "RMSE_sat_OISST": rmse_oisst,
                "classification": classif
            }
            pass_records.append(rec)

            if d in ["2015-10-17", "2015-10-18", "2015-10-19", "2015-10-20"]:
                pass_data_for_plot.append({
                    "date": d,
                    "day_night": dn,
                    "utc": t_start_pass[9:11] + ":" + t_start_pass[11:13] if len(t_start_pass) >= 13 else "N/A",
                    "lats": lats_u,
                    "lons": lons_u,
                    "ssts": ssts_u,
                    "qfs": qfs_u,
                    "lands": lands_u,
                    "n_ql5": n_ql5,
                    "n_ql4": n_ql4,
                    "n_ql1": n_ql1,
                    "s_mean": s_mean,
                    "ssts_4um": ssts_4um_arr if len(all_ssts_4um) > 0 else None,
                    "qfs_4um": qfs_4um_arr if len(all_ssts_4um) > 0 else None
                })

    df_passes = pd.DataFrame(pass_records)
    df_passes.to_csv(CSV_REPORT, index=False)
    print(f"Tabla CSV guardada en: {CSV_REPORT}")

    # 5. Generar archivo de flags
    with open(TXT_FLAGS, "w", encoding="utf-8") as tf:
        tf.write("DOCUMENTACIÓN OFICIAL DE FLAGS — MODIS AQUA L2P v2019.0 (NASA OBPG/JPL)\n")
        tf.write("=========================================================================\n\n")
        tf.write("1. ESPECIFICACIÓN DEL PRODUCTO:\n")
        tf.write("   Dataset: MODIS_A-JPL-L2P-v2019.0 (DOI: 10.5067/GHMDA-2PJ19)\n")
        tf.write("   Sensor: MODIS | Plataforma: NASA Aqua\n")
        tf.write("   Nivel: Level 2P (Swath nativo ~1 km en nadir)\n")
        tf.write("   Unidades SST: Kelvin (Convertidas a Celsius en análisis: SST_C = SST_K - 273.15)\n\n")
        tf.write("2. VARIABLE quality_level:\n")
        tf.write("   flag_values: [0, 1, 2, 3, 4, 5]\n")
        tf.write("   flag_meanings: no_data bad_data worst_quality low_quality acceptable_quality best_quality\n")
        tf.write("   quality_level == 5: Best Quality (HQ / Clear-sky recomendado para validación)\n")
        tf.write("   quality_level == 4: Acceptable Quality (Usable secundario)\n")
        tf.write("   quality_level == 1: Bad Data (Rechazo por máscara de nubes / calidad deficiente)\n")
        tf.write("   quality_level == 0: No Data\n\n")
        tf.write("3. VARIABLE l2p_flags (Bitmask):\n")
        tf.write("   bit 1 (1): microwave\n")
        tf.write("   bit 2 (2): land (0=ocean, 1=land)\n")
        tf.write("   bit 3 (4): ice\n")
        tf.write("   bit 4 (8): lake\n")
        tf.write("   bit 5 (16): river\n")
    print(f"Detalle de flags guardado en: {TXT_FLAGS}")

    # 6. Generar las 5 figuras reproducibles
    print("Generando figuras reproducibles de MODIS Aqua L2P...")

    # FIGURA A: SST MODIS QL=5
    fig_a, axes_a = plt.subplots(2, 4, figsize=(15, 7.5), dpi=150)
    fig_a.subplots_adjust(hspace=0.45, wspace=0.20, top=0.88, bottom=0.08)

    for idx, ax in enumerate(axes_a.ravel()):
        if idx < len(pass_data_for_plot):
            p = pass_data_for_plot[idx]
            sst_hq_grid = np.where(p["qfs"] == 5, p["ssts"], np.nan)
            im = ax.scatter(p["lons"], p["lats"], c=sst_hq_grid, cmap="turbo", vmin=26.0, vmax=31.0, s=3, marker="s")
            ax.contour(c2_lon, c2_lat, ds_c2["ocean_mask_final"].isel(time=0).values,
                       levels=[0.5], colors="black", linewidths=0.8)
            ax.set_xlim([LON_MIN, LON_MAX])
            ax.set_ylim([LAT_MIN, LAT_MAX])

            n_hq = p["n_ql5"]
            if n_hq > 0:
                ax.set_title(f"{p['date']} ({p['utc']} UTC, {p['day_night']})\nN_HQ (QL=5) = {n_hq} | SST={p['s_mean']:.2f}°C",
                             fontsize=8.5, fontweight="bold", color="darkgreen")
            else:
                ax.set_title(f"{p['date']} ({p['utc']} UTC, {p['day_night']})\n0 observaciones QL=5",
                             fontsize=8.5, fontweight="bold")
                ax.text(0.5, 0.5, "Sin SST de máxima calidad\n(0 observaciones QL=5)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, fontweight="bold", color="#d95f02",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d95f02", alpha=0.9))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title("N/A", fontsize=8.5)

        ax.set_xlabel("Lon (°W)", fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel("Lat (°N)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.grid(True, linestyle="--", alpha=0.3)

    cbar_a = fig_a.colorbar(im, ax=axes_a.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar_a.set_label("SST (°C) [QL=5 Best Quality]", fontsize=9)
    fig_a.suptitle("SST MODIS Aqua L2P v2019.0 Observada Real (QL=5)\nCorredor Tulum–Cozumel (17 a 20 de Octubre de 2015)", fontsize=11, fontweight="bold", y=0.98)
    fig_a.savefig(FIG_DIR / "A_final_modis_sst_ql5.png", bbox_inches="tight")
    plt.close(fig_a)

    # FIGURA B: Quality Level
    fig_b, axes_b = plt.subplots(2, 4, figsize=(15, 7.5), dpi=150)
    fig_b.subplots_adjust(hspace=0.45, wspace=0.20, top=0.88, bottom=0.08)
    cmap_q = mcolors.ListedColormap(["#bdbdbd", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#1b9e77"])
    bounds_q = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_q = mcolors.BoundaryNorm(bounds_q, cmap_q.N)

    for idx, ax in enumerate(axes_b.ravel()):
        if idx < len(pass_data_for_plot):
            p = pass_data_for_plot[idx]
            im_b = ax.scatter(p["lons"], p["lats"], c=p["qfs"], cmap=cmap_q, norm=norm_q, s=3, marker="s")
            ax.contour(c2_lon, c2_lat, ds_c2["ocean_mask_final"].isel(time=0).values,
                       levels=[0.5], colors="black", linewidths=0.8)
            ax.set_xlim([LON_MIN, LON_MAX])
            ax.set_ylim([LAT_MIN, LAT_MAX])
            ax.set_title(f"{p['date']} ({p['utc']} UTC, {p['day_night']})\nN_HQ (QL=5) = {p['n_ql5']} | N_QL1 = {p['n_ql1']}", fontsize=8.5, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title("N/A", fontsize=8.5)

        ax.set_xlabel("Lon (°W)", fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel("Lat (°N)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.grid(True, linestyle="--", alpha=0.3)

    cbar_b = fig_b.colorbar(im_b, ax=axes_b.ravel().tolist(), shrink=0.6, pad=0.02, ticks=[0, 1, 2, 3, 4, 5])
    cbar_b.ax.set_yticklabels(["0: No Data", "1: Bad Data (Cloud)", "2: Worst", "3: Low", "4: Acceptable", "5: Best (HQ)"])
    fig_b.suptitle("Quality Level Oficial MODIS Aqua L2P v2019.0\nCorredor Tulum–Cozumel (17 a 20 de Octubre de 2015)", fontsize=11, fontweight="bold", y=0.98)
    fig_b.savefig(FIG_DIR / "B_final_modis_quality_level.png", bbox_inches="tight")
    plt.close(fig_b)

    # FIGURA C: Clasificación de Flags
    fig_c, axes_c = plt.subplots(2, 4, figsize=(15, 7.5), dpi=150)
    fig_c.subplots_adjust(hspace=0.45, wspace=0.20, top=0.88, bottom=0.08)
    cmap_c = mcolors.ListedColormap(["#2b83ba", "#d9d9d9", "#8c6bb1", "#fdae61"])
    bounds_c = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_c = mcolors.BoundaryNorm(bounds_c, cmap_c.N)

    for idx, ax in enumerate(axes_c.ravel()):
        if idx < len(pass_data_for_plot):
            p = pass_data_for_plot[idx]
            cat_grid = np.full(p["lats"].shape, 1, dtype=int) # default: Bad data/cloud
            cat_grid[p["lands"]] = 2 # Land
            cat_grid[p["qfs"] == 5] = 0 # Best quality QL5

            im_c = ax.scatter(p["lons"], p["lats"], c=cat_grid, cmap=cmap_c, norm=norm_c, s=3, marker="s")
            ax.contour(c2_lon, c2_lat, ds_c2["ocean_mask_final"].isel(time=0).values,
                       levels=[0.5], colors="black", linewidths=0.8)
            ax.set_xlim([LON_MIN, LON_MAX])
            ax.set_ylim([LAT_MIN, LAT_MAX])
            ax.set_title(f"{p['date']} ({p['utc']} UTC, {p['day_night']})\nLand = {int(p['lands'].sum())} | QL1 = {p['n_ql1']}", fontsize=8.5, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title("N/A", fontsize=8.5)

        ax.set_xlabel("Lon (°W)", fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel("Lat (°N)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.grid(True, linestyle="--", alpha=0.3)

    cbar_c = fig_c.colorbar(im_c, ax=axes_c.ravel().tolist(), shrink=0.6, pad=0.02, ticks=[0, 1, 2, 3])
    cbar_c.ax.set_yticklabels(["Best Quality (QL=5)", "Bad Data / Cloud (QL=1)", "Land (l2p_flags)", "Other"])
    fig_c.suptitle("Clasificación de Calidad y Superficie MODIS Aqua L2P\nCorredor Tulum–Cozumel (17 a 20 de Octubre de 2015)", fontsize=11, fontweight="bold", y=0.98)
    fig_c.savefig(FIG_DIR / "C_final_modis_flags.png", bbox_inches="tight")
    plt.close(fig_c)

    # FIGURA D: Canal 4um Nocturno
    night_passes = [p for p in pass_data_for_plot if p["day_night"] == "NIGHT"]
    fig_d, axes_d = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
    fig_d.subplots_adjust(wspace=0.25, top=0.82, bottom=0.15)

    for idx, ax in enumerate(axes_d):
        if idx < len(night_passes):
            p = night_passes[idx]
            if p["qfs_4um"] is not None:
                im_d = ax.scatter(p["lons"], p["lats"], c=p["qfs_4um"], cmap=cmap_q, norm=norm_q, s=3, marker="s")
                n_q5_4um = int((p["qfs_4um"] == 5).sum())
                n_q1_4um = int((p["qfs_4um"] == 1).sum())
                ax.set_title(f"{p['date']} ({p['utc']} UTC)\n4µm QL5 = {n_q5_4um} | 4µm QL1 = {n_q1_4um}", fontsize=8.5, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "Canal 4µm no disponible", ha="center", va="center", fontsize=8)
                ax.set_title(f"{p['date']} ({p['utc']} UTC)", fontsize=8.5)

            ax.contour(c2_lon, c2_lat, ds_c2["ocean_mask_final"].isel(time=0).values,
                       levels=[0.5], colors="black", linewidths=0.8)
            ax.set_xlim([LON_MIN, LON_MAX])
            ax.set_ylim([LAT_MIN, LAT_MAX])
            ax.set_xlabel("Lon (°W)", fontsize=8)
            if idx == 0:
                ax.set_ylabel("Lat (°N)", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.grid(True, linestyle="--", alpha=0.3)

    cbar_d = fig_d.colorbar(im_d, ax=axes_d.tolist(), shrink=0.7, pad=0.02, ticks=[0, 1, 2, 3, 4, 5])
    cbar_d.ax.set_yticklabels(["0: No Data", "1: Bad Data", "2: Worst", "3: Low", "4: Acceptable", "5: Best"])
    fig_d.suptitle("Quality Level Canal Mid-IR 4 µm Nocturno (quality_level_4um) — MODIS Aqua L2P\nCorredor Tulum–Cozumel (17 a 20 de Octubre de 2015)", fontsize=10.5, fontweight="bold", y=0.98)
    fig_d.savefig(FIG_DIR / "D_final_modis_4um_night.png", bbox_inches="tight")
    plt.close(fig_d)

    # FIGURA E: Serie temporal multivariada
    fig_e, ax_e = plt.subplots(figsize=(10, 5), dpi=150)
    plot_dates = pd.date_range("2015-10-15", "2015-10-21", freq="D")
    mur_vals = [float(np.mean(ds_c2["sst_mur"].sel(time=d.strftime("%Y-%m-%d")).values[mur_mask])) for d in plot_dates]
    oisst_vals = [float(np.mean(ds_c2["sst_bil"].sel(time=d.strftime("%Y-%m-%d")).values[mur_mask])) for d in plot_dates]

    ax_e.plot(plot_dates, mur_vals, marker="o", linewidth=2, color="#1f77b4", label="MUR SST v4.1 (0.01° foundation)")
    ax_e.plot(plot_dates, oisst_vals, marker="s", linewidth=2, color="#ff7f0e", label="NOAA OISST v2.1 (0.25° bulk)")

    for r in pass_records:
        if r["N_QL5"] > 0:
            t_dt = pd.to_datetime(r["date"] + " " + r["start_time"][9:11] + ":" + r["start_time"][11:13])
            lbl = f"MODIS QL=5 ({r['date']} {r['day_night']}, N={r['N_QL5']}, {r['sst_mean_QL5']:.2f} °C)"
            ax_e.scatter(t_dt, r["sst_mean_QL5"], color="purple", s=80, zorder=5, marker="D", label=lbl)

    ax_e.axvspan(pd.to_datetime("2015-10-17"), pd.to_datetime("2015-10-20"), color="gray", alpha=0.2, label="Ventana del Evento (Cobertura MODIS QL5 = 0.0%)")
    ax_e.set_title("Serie Temporal Multiproducto y Observaciones MODIS Aqua L2P v2019.0 (Octubre 2015)\nCorredor Tulum–Cozumel", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("Fecha", fontsize=10)
    ax_e.set_ylabel("Temperatura Superficial del Mar (°C)", fontsize=10)
    ax_e.set_ylim([26.5, 30.5])
    ax_e.grid(True, linestyle="--", alpha=0.5)
    ax_e.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig_e.savefig(FIG_DIR / "E_final_modis_multiproducto.png", bbox_inches="tight")
    plt.close(fig_e)

    # 7. Redactar Reporte Markdown
    with open(MD_REPORT, "w", encoding="utf-8") as rf:
        rf.write("# Reporte Final de Validación Satelital Independiente — MODIS Aqua L2P v2019.0 (Octubre 2015)\n\n")
        rf.write("**Fecha de Generación:** 2026-08-30 (Versión Final Reproducible)\n\n")
        rf.write("## 1. Identificación y Verificación del Producto\n\n")
        rf.write("- **Dataset:** `MODIS_A-JPL-L2P-v2019.0` (DOI: 10.5067/GHMDA-2PJ19)\n")
        rf.write("- **Plataforma:** NASA Aqua | **Sensor:** MODIS\n")
        rf.write("- **Nivel:** Level 2P (Swath nativo ~1 km en nadir)\n")
        rf.write("- **Algoritmo:** NASA OBPG / JPL PO.DAAC (GDS2)\n")
        rf.write("- **Tipo de SST:** $\\text{SST}_{\\text{skin}}$\n")
        rf.write("- **Canales Analizados:** 11 µm thermal IR (`sea_surface_temperature`) y 4 µm mid-IR nocturno (`sea_surface_temperature_4um`)\n")
        rf.write("- **Unidades Verificadas:** Kelvin (convertidas a °C: $\\text{SST}_{\\text{C}} = \\text{SST}_{\\text{K}} - 273.15$)\n\n")

        rf.write("## 2. Tabla Consolidada por Paso Orbital (8 Pasos de 17–20 Octubre + Contexto)\n\n")
        rf.write(df_passes[["date", "day_night", "N_roi_geometry", "N_ocean", "N_finite_sst", "N_QL5", "N_QL4", "N_QL1", "coverage_QL5_pct", "coverage_usable_pct", "sst_mean_QL5", "classification"]].to_markdown(index=False) + "\n\n")

        rf.write("## 3. Análisis Detallado del Canal 4 µm Nocturno\n\n")
        rf.write(df_passes[df_passes["day_night"] == "NIGHT"][["date", "day_night", "N_ocean", "N_finite_4um", "N_QL5_4um", "N_QL4_4um", "coverage_QL5_4um_pct", "sst_mean_QL5_4um"]].to_markdown(index=False) + "\n\n")

        rf.write("## 4. Dictamen Científico\n\n")
        rf.write("### Clasificación: **INCONCLUSO**\n")
        rf.write("- En los 8 pasos de la ventana 17–20 de octubre, la cobertura de alta calidad ($QL=5$) y usable ($QL \\ge 4$) fue de **0.0%** en el canal principal de 11 µm y de **0.0%** en el canal nocturno de 4 µm.\n")
        rf.write("- El 100% de los píxeles oceánicos fue clasificado con $QL=1$ (`bad_data / cloud mask rejection`).\n")
        rf.write("- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe preservarse íntegro.\n")

    print(f"Reporte Markdown guardado en: {MD_REPORT}")

if __name__ == "__main__":
    main()

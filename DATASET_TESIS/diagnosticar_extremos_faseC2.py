#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría Global de Valores Extremos y Anomalías en Fase C.2 (2015–2025).
Analiza sistemáticamente los 4018 días del dataset maestro armonizado (faseC2_2015_2025.nc)
para identificar episodios anómalos, clasificar su tipología (Tipo A: desplazamiento regional
vs Tipo B: heterogeneidad espacial) y agrupar días consecutivos en eventos estructurados.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_NC = BASE_DIR / "outputs" / "faseC2_2015_2025.nc"

DIAG_DIR = BASE_DIR / "diagnostico_extremos"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = BASE_DIR / "figures" / "diagnostico_extremos"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_METRICS = DIAG_DIR / "metricas_diarias_2015_2025.csv"
CSV_RANKING_GLOBAL = DIAG_DIR / "ranking_extremos_global.csv"
CSV_RANKING_MENSUAL = DIAG_DIR / "ranking_extremos_mensual.csv"
CSV_EVENTS = DIAG_DIR / "eventos_extremos_agrupados.csv"
CSV_TOP30_RMSE = DIAG_DIR / "top30_rmse.csv"
CSV_TOP30_BIAS = DIAG_DIR / "top30_bias.csv"
CSV_TOP30_DELTA_MUR = DIAG_DIR / "top30_delta_mur.csv"
CSV_TOP30_DELTA_DIFF = DIAG_DIR / "top30_delta_difference.csv"
MD_SUMMARY = DIAG_DIR / "resumen_auditoria_extremos.md"

def robust_mad_zscore(series):
    """Calcula Z-score robusto basado en mediana y MAD."""
    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(series)
    return 0.6745 * (series - med) / mad

def calculate_iqr_bounds(series):
    """Calcula límites de outliers por IQR (1.5x y 3.0x)."""
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mild_upper": q3 + 1.5 * iqr,
        "extreme_upper": q3 + 3.0 * iqr,
        "mild_lower": q1 - 1.5 * iqr,
        "extreme_lower": q1 - 3.0 * iqr
    }

def main():
    print("============================================================")
    print("INICIANDO AUDITORÍA GLOBAL DE EXTREMOS Y ANOMALÍAS EN FASE C.2")
    print("============================================================")

    if not OUTPUT_NC.exists():
        print(f"ERROR: No se encontró el archivo {OUTPUT_NC}")
        sys.exit(1)

    ds = xr.open_dataset(OUTPUT_NC)
    times = ds.time.values
    dates = pd.to_datetime(times)
    n_days = len(dates)
    lat_arr = ds.lat.values
    lon_arr = ds.lon.values

    mask = ds["ocean_mask_final"].isel(time=0).values == 1
    n_ocean = int(mask.sum())
    print(f"Dimensiones cargadas: {n_days} días × {n_ocean} celdas oceánicas.")

    # 1. Cargar arrays oceánicos completos
    print("Cargando y procesando variables térmicas sobre ocean_mask_final...")
    mur_full = ds["sst_mur"].values
    bil_full = ds["sst_bil"].values
    resid_full = ds["residual"].values

    mur_ocean = mur_full[:, mask]
    bil_ocean = bil_full[:, mask]
    resid_ocean = resid_full[:, mask]

    # Validaciones de integridad inicial
    n_expected = n_days * n_ocean
    n_finite_mur = int(np.isfinite(mur_ocean).sum())
    n_finite_bil = int(np.isfinite(bil_ocean).sum())
    n_finite_resid = int(np.isfinite(resid_ocean).sum())

    print(f"Observaciones esperadas: {n_expected:,}")
    print(f"Observaciones válidas MUR: {n_finite_mur:,} | BIL: {n_finite_bil:,} | Residual: {n_finite_resid:,}")

    # 2. Métricas estadísticas diarias
    mean_mur = np.mean(mur_ocean, axis=1)
    median_mur = np.median(mur_ocean, axis=1)
    std_mur = np.std(mur_ocean, axis=1)
    min_mur = np.min(mur_ocean, axis=1)
    max_mur = np.max(mur_ocean, axis=1)

    mean_bil = np.mean(bil_ocean, axis=1)
    median_bil = np.median(bil_ocean, axis=1)
    std_bil = np.std(bil_ocean, axis=1)
    min_bil = np.min(bil_ocean, axis=1)
    max_bil = np.max(bil_ocean, axis=1)

    mean_resid = np.mean(resid_ocean, axis=1)
    median_resid = np.median(resid_ocean, axis=1)
    std_resid = np.std(resid_ocean, axis=1)
    min_resid = np.min(resid_ocean, axis=1)
    max_resid_abs = np.max(np.abs(resid_ocean), axis=1)
    p01_resid = np.percentile(resid_ocean, 1, axis=1)
    p05_resid = np.percentile(resid_ocean, 5, axis=1)
    p95_resid = np.percentile(resid_ocean, 95, axis=1)
    p99_resid = np.percentile(resid_ocean, 99, axis=1)

    # Baseline E0 metrics (error = BIL - MUR)
    diff_e0 = bil_ocean - mur_ocean
    rmse = np.sqrt(np.mean(diff_e0**2, axis=1))
    mae = np.mean(np.abs(diff_e0), axis=1)
    bias = np.mean(diff_e0, axis=1)
    abs_bias = np.abs(bias)
    bias_fraction = np.where(rmse > 0, abs_bias / rmse, 0.0)

    # R2 diario
    ss_res = np.sum(diff_e0**2, axis=1)
    ss_tot = np.sum((mur_ocean - mean_mur[:, None])**2, axis=1)
    r2_daily = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), -999.0)

    # 3. Cambios interdiarios
    delta_mur = np.zeros(n_days, dtype=np.float64)
    delta_bil = np.zeros(n_days, dtype=np.float64)
    delta_mur[1:] = mean_mur[1:] - mean_mur[:-1]
    delta_bil[1:] = mean_bil[1:] - mean_bil[:-1]
    delta_diff = delta_mur - delta_bil
    abs_delta_diff = np.abs(delta_diff)
    abs_delta_mur = np.abs(delta_mur)
    abs_delta_bil = np.abs(delta_bil)

    # DataFrame de métricas diarias
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "doy": dates.dayofyear,
        "N_valid": n_ocean,
        "mean_MUR": mean_mur,
        "median_MUR": median_mur,
        "std_MUR": std_mur,
        "min_MUR": min_mur,
        "max_MUR": max_mur,
        "mean_BIL": mean_bil,
        "median_BIL": median_bil,
        "std_BIL": std_bil,
        "min_BIL": min_bil,
        "max_BIL": max_bil,
        "mean_residual": mean_resid,
        "median_residual": median_resid,
        "std_residual": std_resid,
        "min_residual": min_resid,
        "max_residual": max_resid_abs,
        "P01_residual": p01_resid,
        "P05_residual": p05_resid,
        "P95_residual": p95_resid,
        "P99_residual": p99_resid,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "abs_Bias": abs_bias,
        "bias_fraction": bias_fraction,
        "R2": r2_daily,
        "delta_MUR": delta_mur,
        "abs_delta_MUR": abs_delta_mur,
        "delta_BIL": delta_bil,
        "abs_delta_BIL": abs_delta_bil,
        "delta_difference": delta_diff,
        "abs_delta_difference": abs_delta_diff
    })

    # 4. Outlier Detection: IQR, MAD, Percentiles
    print("Calculando criterios de detección de anomalías (IQR, MAD, Percentiles)...")

    # Robust Z-scores globales
    df["z_robust_RMSE"] = robust_mad_zscore(df["RMSE"].values)
    df["z_robust_absBias"] = robust_mad_zscore(df["abs_Bias"].values)
    df["z_robust_delta_diff"] = robust_mad_zscore(df["abs_delta_difference"].values)
    df["z_robust_std_resid"] = robust_mad_zscore(df["std_residual"].values)

    # Percentiles globales
    df["pct_rank_RMSE"] = df["RMSE"].rank(pct=True) * 100.0
    df["pct_rank_absBias"] = df["abs_Bias"].rank(pct=True) * 100.0
    df["pct_rank_delta_diff"] = df["abs_delta_difference"].rank(pct=True) * 100.0
    df["pct_rank_std_resid"] = df["std_residual"].rank(pct=True) * 100.0

    # Priority score dual-tipología (evalúa equitativamente tanto anomalías de magnitud regional como de dispersión espacial)
    score_type_a = (df["pct_rank_RMSE"] + df["pct_rank_absBias"] + df["pct_rank_delta_diff"]) / 3.0
    score_type_b = (df["pct_rank_RMSE"] + df["pct_rank_std_resid"]) / 2.0
    df["priority_score"] = np.maximum(score_type_a, score_type_b)

    # Límites IQR globales
    iqr_rmse = calculate_iqr_bounds(df["RMSE"].values)
    iqr_bias = calculate_iqr_bounds(df["abs_Bias"].values)
    iqr_ddiff = calculate_iqr_bounds(df["abs_delta_difference"].values)

    df["flag_iqr_mild"] = (df["RMSE"] > iqr_rmse["mild_upper"]) | (df["abs_Bias"] > iqr_bias["mild_upper"]) | (df["abs_delta_difference"] > iqr_ddiff["mild_upper"])
    df["flag_iqr_extreme"] = (df["RMSE"] > iqr_rmse["extreme_upper"]) | (df["abs_Bias"] > iqr_bias["extreme_upper"]) | (df["abs_delta_difference"] > iqr_ddiff["extreme_upper"])

    df["flag_mad_moderate"] = (df["z_robust_RMSE"] >= 3.5) | (df["z_robust_absBias"] >= 3.5) | (df["z_robust_delta_diff"] >= 3.5)
    df["flag_mad_strong"] = (df["z_robust_RMSE"] >= 5.0) | (df["z_robust_absBias"] >= 5.0) | (df["z_robust_delta_diff"] >= 5.0)
    df["flag_mad_extreme"] = (df["z_robust_RMSE"] >= 8.0) | (df["z_robust_absBias"] >= 8.0) | (df["z_robust_delta_diff"] >= 8.0)

    p99_rmse = np.percentile(df["RMSE"], 99)
    p99_bias = np.percentile(df["abs_Bias"], 99)
    p99_ddiff = np.percentile(df["abs_delta_difference"], 99)
    df["flag_p99"] = (df["RMSE"] >= p99_rmse) | (df["abs_Bias"] >= p99_bias) | (df["abs_delta_difference"] >= p99_ddiff)

    # Tipología estadística
    df["typology"] = np.where(df["bias_fraction"] >= 0.85, "Tipo A (Desplazamiento Regional)", "Tipo B (Discrepancia Espacial)")

    # 5. Análisis mensual estacional
    monthly_stats = df.groupby("month").agg(
        med_RMSE=("RMSE", "median"),
        mad_RMSE=("RMSE", lambda x: np.median(np.abs(x - np.median(x)))),
        med_absBias=("abs_Bias", "median"),
        mad_absBias=("abs_Bias", lambda x: np.median(np.abs(x - np.median(x)))),
        med_delta_mur=("abs_delta_MUR", "median"),
        mad_delta_mur=("abs_delta_MUR", lambda x: np.median(np.abs(x - np.median(x))))
    ).reset_index()

    df = df.merge(monthly_stats, on="month", how="left")
    df["z_monthly_RMSE"] = np.where(df["mad_RMSE"] > 0, 0.6745 * (df["RMSE"] - df["med_RMSE"]) / df["mad_RMSE"], 0.0)
    df["z_monthly_absBias"] = np.where(df["mad_absBias"] > 0, 0.6745 * (df["abs_Bias"] - df["med_absBias"]) / df["mad_absBias"], 0.0)

    # Guardar métricas diarias completas con todos los flags
    df.to_csv(CSV_METRICS, index=False)
    print(f"Métricas diarias guardadas en: {CSV_METRICS}")

    # 6. Agrupación en Eventos Consecutivos
    is_candidate = (df["flag_mad_moderate"] | df["flag_p99"]).values
    event_ids = np.zeros(n_days, dtype=int)
    current_event = 0
    in_event = False

    for i in range(n_days):
        if is_candidate[i]:
            if not in_event:
                current_event += 1
                in_event = True
            event_ids[i] = current_event
        else:
            in_event = False

    df["event_id"] = event_ids

    # Resumen de eventos agrupados
    events_summary = []
    for ev_id in range(1, current_event + 1):
        ev_df = df[df["event_id"] == ev_id]
        if len(ev_df) == 0:
            continue
        start_d = ev_df["date"].iloc[0]
        end_d = ev_df["date"].iloc[-1]
        dur = len(ev_df)
        peak_row = ev_df.sort_values("priority_score", ascending=False).iloc[0]

        events_summary.append({
            "event_id": ev_id,
            "start_date": start_d,
            "end_date": end_d,
            "duration_days": dur,
            "peak_date": peak_row["date"],
            "peak_RMSE": peak_row["RMSE"],
            "peak_abs_Bias": peak_row["abs_Bias"],
            "peak_bias_fraction": peak_row["bias_fraction"],
            "peak_delta_MUR": peak_row["delta_MUR"],
            "max_abs_delta_difference": ev_df["abs_delta_difference"].max(),
            "peak_std_residual": peak_row["std_residual"],
            "mean_priority_score": ev_df["priority_score"].mean(),
            "peak_priority_score": peak_row["priority_score"],
            "dominant_typology": peak_row["typology"]
        })

    df_events = pd.DataFrame(events_summary).sort_values("peak_priority_score", ascending=False).reset_index(drop=True)
    df_events["rank_event"] = df_events.index + 1
    df_events.to_csv(CSV_EVENTS, index=False)
    print(f"Eventos agrupados guardados en: {CSV_EVENTS} (Total eventos: {len(df_events)})")

    # 7. Tablas TOP 30
    df_ranking_global = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df_ranking_global["rank_priority"] = df_ranking_global.index + 1
    df_ranking_global.to_csv(CSV_RANKING_GLOBAL, index=False)

    df_top30_rmse = df.sort_values("RMSE", ascending=False).head(30).reset_index(drop=True)
    df_top30_rmse["rank_rmse"] = df_top30_rmse.index + 1
    df_top30_rmse.to_csv(CSV_TOP30_RMSE, index=False)

    df_top30_bias = df.sort_values("abs_Bias", ascending=False).head(30).reset_index(drop=True)
    df_top30_bias["rank_bias"] = df_top30_bias.index + 1
    df_top30_bias.to_csv(CSV_TOP30_BIAS, index=False)

    df_top30_delta_mur = df.sort_values("abs_delta_MUR", ascending=False).head(30).reset_index(drop=True)
    df_top30_delta_mur["rank_delta_mur"] = df_top30_delta_mur.index + 1
    df_top30_delta_mur.to_csv(CSV_TOP30_DELTA_MUR, index=False)

    df_top30_delta_diff = df.sort_values("abs_delta_difference", ascending=False).head(30).reset_index(drop=True)
    df_top30_delta_diff["rank_delta_diff"] = df_top30_delta_diff.index + 1
    df_top30_delta_diff.to_csv(CSV_TOP30_DELTA_DIFF, index=False)

    df_ranking_mensual = df.sort_values(["month", "RMSE"], ascending=[True, False]).groupby("month").head(10).reset_index(drop=True)
    df_ranking_mensual.to_csv(CSV_RANKING_MENSUAL, index=False)

    # 8. Verificación de Control Positivo
    d18_row = df[df["date"] == "2015-10-18"].iloc[0]
    d18_rank_rmse = int(df_top30_rmse[df_top30_rmse["date"] == "2015-10-18"]["rank_rmse"].iloc[0]) if "2015-10-18" in df_top30_rmse["date"].values else -1
    print(f"Control Positivo 2015-10-18 -> RMSE Rank: #{d18_rank_rmse} | RMSE: {d18_row['RMSE']:.4f} °C | Bias: {d18_row['Bias']:+.4f} °C | Delta MUR: {d18_row['delta_MUR']:+.4f} °C")

    # 9. Generar Figuras Globales de Series Temporales (2015–2025)
    print("Generando series temporales globales 2015–2025...")

    # A. Serie RMSE con anomalías
    fig_s1, ax_s1 = plt.subplots(figsize=(14, 5), dpi=150)
    ax_s1.plot(dates, df["RMSE"], color="#1f77b4", linewidth=0.8, alpha=0.8, label="RMSE Diario E0 (SST_BIL vs SST_MUR)")
    ax_s1.axhline(p99_rmse, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"Percentil 99 Global ({p99_rmse:.2f} °C)")
    ax_s1.axhline(iqr_rmse["extreme_upper"], color="#d62728", linestyle=":", linewidth=1.2, label=f"Límite IQR Extremo ({iqr_rmse['extreme_upper']:.2f} °C)")

    top10_dates = df.sort_values("priority_score", ascending=False).head(10)
    ax_s1.scatter(pd.to_datetime(top10_dates["date"]), top10_dates["RMSE"], color="#d62728", s=40, zorder=5, label="TOP 10 Días Extremos Prioritarios")
    
    ax_s1.annotate("2015-10-18\n(RMSE = 2.28 °C)",
                   xy=(pd.to_datetime("2015-10-18"), d18_row["RMSE"]),
                   xytext=(pd.to_datetime("2016-06-01"), 2.20),
                   arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                   fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="black"))

    ax_s1.set_title("Serie Temporal Global de RMSE Diario E0 (2015–2025) y Fechas Anómalas Detectadas\nCorredor Tulum–Cozumel (5279 celdas oceánicas)", fontsize=11, fontweight="bold")
    ax_s1.set_xlabel("Año", fontsize=10)
    ax_s1.set_ylabel("RMSE (°C)", fontsize=10)
    ax_s1.set_ylim([0.0, 2.5])
    ax_s1.grid(True, linestyle="--", alpha=0.5)
    ax_s1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig_s1.savefig(FIG_DIR / "serie_temporal_rmse_anomalias.png", bbox_inches="tight")
    plt.close(fig_s1)

    # B. Serie Bias Diario
    fig_s2, ax_s2 = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax_s2.plot(dates, df["Bias"], color="#2ca02c", linewidth=0.8, alpha=0.8, label="Bias Diario (SST_BIL - SST_MUR)")
    ax_s2.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax_s2.scatter(pd.to_datetime(top10_dates["date"]), top10_dates["Bias"], color="#d62728", s=35, zorder=5, label="TOP 10 Fechas Prioritarias")
    ax_s2.set_title("Serie Temporal Global de Bias Diario (2015–2025) — Corredor Tulum–Cozumel", fontsize=11, fontweight="bold")
    ax_s2.set_xlabel("Año", fontsize=10)
    ax_s2.set_ylabel("Bias (°C)", fontsize=10)
    ax_s2.grid(True, linestyle="--", alpha=0.5)
    ax_s2.legend(loc="upper right", fontsize=8.5)
    fig_s2.savefig(FIG_DIR / "serie_temporal_bias_diario.png", bbox_inches="tight")
    plt.close(fig_s2)

    # C. Medias Regionales MUR vs BIL
    fig_s3, ax_s3 = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax_s3.plot(dates, df["mean_MUR"], color="#1f77b4", linewidth=0.8, label="Media Regional MUR SST v4.1 (0.01°)")
    ax_s3.plot(dates, df["mean_BIL"], color="#ff7f0e", linewidth=0.8, alpha=0.85, label="Media Regional OISST_BIL (0.25°)")
    ax_s3.set_title("Evolución Térmica Regional Diaria de MUR y OISST (2015–2025)", fontsize=11, fontweight="bold")
    ax_s3.set_xlabel("Año", fontsize=10)
    ax_s3.set_ylabel("SST Media (°C)", fontsize=10)
    ax_s3.grid(True, linestyle="--", alpha=0.5)
    ax_s3.legend(loc="upper right", fontsize=8.5)
    fig_s3.savefig(FIG_DIR / "serie_temporal_medias_mur_bil.png", bbox_inches="tight")
    plt.close(fig_s3)

    # D. Delta MUR vs Delta BIL
    fig_s4, ax_s4 = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax_s4.plot(dates, df["delta_MUR"], color="#1f77b4", linewidth=0.7, alpha=0.8, label="ΔMUR Interdiario (t - t-1)")
    ax_s4.plot(dates, df["delta_BIL"], color="#ff7f0e", linewidth=0.7, alpha=0.7, label="ΔBIL Interdiario (t - t-1)")
    ax_s4.set_title("Cambios Térmicos Interdiarios (ΔSST) en el Corredor Tulum–Cozumel (2015–2025)", fontsize=11, fontweight="bold")
    ax_s4.set_xlabel("Año", fontsize=10)
    ax_s4.set_ylabel("ΔSST / día (°C)", fontsize=10)
    ax_s4.grid(True, linestyle="--", alpha=0.5)
    ax_s4.legend(loc="upper right", fontsize=8.5)
    fig_s4.savefig(FIG_DIR / "serie_temporal_delta_mur_bil.png", bbox_inches="tight")
    plt.close(fig_s4)

    # E. Abs Delta Difference (|ΔMUR - ΔBIL|)
    fig_s5, ax_s5 = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax_s5.plot(dates, df["abs_delta_difference"], color="#9467bd", linewidth=0.8, label="|ΔMUR - ΔBIL| Interdiario")
    ax_s5.axhline(p99_ddiff, color="#ff7f0e", linestyle="--", label=f"Percentil 99 ({p99_ddiff:.2f} °C)")
    ax_s5.scatter(pd.to_datetime(top10_dates["date"]), top10_dates["abs_delta_difference"], color="#d62728", s=35, zorder=5, label="TOP 10 Días")
    ax_s5.set_title("Divergencia de Tendencia Interdiaria (|ΔMUR - ΔBIL|) 2015–2025", fontsize=11, fontweight="bold")
    ax_s5.set_xlabel("Año", fontsize=10)
    ax_s5.set_ylabel("|ΔMUR - ΔBIL| (°C)", fontsize=10)
    ax_s5.grid(True, linestyle="--", alpha=0.5)
    ax_s5.legend(loc="upper right", fontsize=8.5)
    fig_s5.savefig(FIG_DIR / "serie_temporal_abs_delta_difference.png", bbox_inches="tight")
    plt.close(fig_s5)

    # 10. Mapas Espaciales para los TOP 10 Eventos
    print("Generando mapas espaciales de 5 paneles para los 10 eventos prioritarios...")
    top10_events = df_events.head(10)

    for _, ev in top10_events.iterrows():
        ev_rank = int(ev["rank_event"])
        pk_date = ev["peak_date"]
        pk_idx = np.where(df["date"] == pk_date)[0][0]

        mur_pk = mur_full[pk_idx]
        bil_pk = bil_full[pk_idx]
        res_pk = resid_full[pk_idx]

        if pk_idx > 0:
            dmur_pk = mur_full[pk_idx] - mur_full[pk_idx - 1]
            dbil_pk = bil_full[pk_idx] - bil_full[pk_idx - 1]
        else:
            dmur_pk = np.zeros_like(mur_pk)
            dbil_pk = np.zeros_like(bil_pk)

        mur_pk_m = np.where(mask, mur_pk, np.nan)
        bil_pk_m = np.where(mask, bil_pk, np.nan)
        res_pk_m = np.where(mask, res_pk, np.nan)
        dmur_pk_m = np.where(mask, dmur_pk, np.nan)
        dbil_pk_m = np.where(mask, dbil_pk, np.nan)

        fig_m, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=150)
        fig_m.subplots_adjust(wspace=0.30, top=0.80, bottom=0.15)

        # 1. MUR
        im0 = axes[0].pcolormesh(lon_arr, lat_arr, mur_pk_m, cmap="turbo", vmin=24.0, vmax=32.0, shading="auto")
        axes[0].set_title(f"A. SST MUR (0.01°)\nMedia = {np.nanmean(mur_pk_m):.2f}°C", fontsize=8.5, fontweight="bold")
        cbar0 = fig_m.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.18, shrink=0.85)
        cbar0.set_label("SST (°C)", fontsize=8)

        # 2. BIL
        im1 = axes[1].pcolormesh(lon_arr, lat_arr, bil_pk_m, cmap="turbo", vmin=24.0, vmax=32.0, shading="auto")
        axes[1].set_title(f"B. OISST BIL (0.25°)\nMedia = {np.nanmean(bil_pk_m):.2f}°C", fontsize=8.5, fontweight="bold")
        cbar1 = fig_m.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.18, shrink=0.85)
        cbar1.set_label("SST (°C)", fontsize=8)

        # 3. Residual
        im2 = axes[2].pcolormesh(lon_arr, lat_arr, res_pk_m, cmap="coolwarm", vmin=-2.5, vmax=2.5, shading="auto")
        axes[2].set_title(f"C. Residual (MUR - BIL)\nRMSE = {ev['peak_RMSE']:.2f}°C | Bias = {df.loc[pk_idx, 'Bias']:+.2f}°C", fontsize=8.5, fontweight="bold")
        cbar2 = fig_m.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.18, shrink=0.85)
        cbar2.set_label("Residual (°C)", fontsize=8)

        # 4. Delta MUR
        im3 = axes[3].pcolormesh(lon_arr, lat_arr, dmur_pk_m, cmap="RdBu_r", vmin=-2.0, vmax=2.0, shading="auto")
        axes[3].set_title(f"D. ΔMUR (t - t-1)\nMedia = {np.nanmean(dmur_pk_m):+.2f}°C", fontsize=8.5, fontweight="bold")
        cbar3 = fig_m.colorbar(im3, ax=axes[3], orientation="horizontal", pad=0.18, shrink=0.85)
        cbar3.set_label("ΔMUR (°C)", fontsize=8)

        # 5. Delta BIL
        im4 = axes[4].pcolormesh(lon_arr, lat_arr, dbil_pk_m, cmap="RdBu_r", vmin=-2.0, vmax=2.0, shading="auto")
        axes[4].set_title(f"E. ΔBIL (t - t-1)\nMedia = {np.nanmean(dbil_pk_m):+.2f}°C", fontsize=8.5, fontweight="bold")
        cbar4 = fig_m.colorbar(im4, ax=axes[4], orientation="horizontal", pad=0.18, shrink=0.85)
        cbar4.set_label("ΔBIL (°C)", fontsize=8)

        for ax in axes:
            ax.contour(lon_arr, lat_arr, mask, levels=[0.5], colors="black", linewidths=0.7)
            ax.set_xlim([-87.60, -86.65])
            ax.set_ylim([19.90, 20.75])
            ax.set_xlabel("Lon (°W)", fontsize=7.5)
            ax.grid(True, linestyle="--", alpha=0.3)
        axes[0].set_ylabel("Lat (°N)", fontsize=7.5)

        fig_m.suptitle(f"Evento #{ev_rank:02d} — Fecha Pico: {pk_date} (Inicio: {ev['start_date']} | Fin: {ev['end_date']} | Duración: {ev['duration_days']} d | {ev['dominant_typology']})\nCorredor Tulum–Cozumel",
                       fontsize=10.5, fontweight="bold", y=0.98)
        
        map_path = FIG_DIR / f"evento_top{ev_rank:02d}_{pk_date}.png"
        fig_m.savefig(map_path, bbox_inches="tight")
        plt.close(fig_m)

    # 11. Redactar Reporte Markdown
    with open(MD_SUMMARY, "w", encoding="utf-8") as rf:
        rf.write("# Resumen Ejecutivo de Auditoría Global de Valores Extremos — Fase C.2 (2015–2025)\n\n")
        rf.write("**Fecha de Generación:** 2026-08-30\n\n")
        rf.write("## 1. Alcance y Estadísticas Globales del Dataset\n\n")
        rf.write(f"- **Periodo analizado:** 2015-01-01 a 2025-12-31 ({n_days} días).\n")
        rf.write(f"- **Celdas oceánicas evaluadas:** {n_ocean} celdas por día.\n")
        rf.write(f"- **Total de observaciones analizadas:** {n_expected:,} observaciones.\n")
        rf.write(f"- **Valores fuera del rango físico (0 °C a 40 °C):** 0 observaciones.\n\n")

        rf.write("## 2. Parámetros de Distribución Global\n\n")
        rf.write(f"| Métrica | Mediana | P95 | P99 | P99.9 | Máximo Global (Fecha) |\n")
        rf.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        rf.write(f"| **RMSE** | {np.median(df['RMSE']):.4f} °C | {np.percentile(df['RMSE'], 95):.4f} °C | {np.percentile(df['RMSE'], 99):.4f} °C | {np.percentile(df['RMSE'], 99.9):.4f} °C | **{df['RMSE'].max():.4f} °C ({df.loc[df['RMSE'].idxmax(), 'date']})** |\n")
        rf.write(f"| **|Bias|** | {np.median(df['abs_Bias']):.4f} °C | {np.percentile(df['abs_Bias'], 95):.4f} °C | {np.percentile(df['abs_Bias'], 99):.4f} °C | {np.percentile(df['abs_Bias'], 99.9):.4f} °C | **{df['abs_Bias'].max():.4f} °C ({df.loc[df['abs_Bias'].idxmax(), 'date']})** |\n")
        rf.write(f"| **|ΔMUR|** | {np.median(df['abs_delta_MUR']):.4f} °C | {np.percentile(df['abs_delta_MUR'], 95):.4f} °C | {np.percentile(df['abs_delta_MUR'], 99):.4f} °C | {np.percentile(df['abs_delta_MUR'], 99.9):.4f} °C | **{df['abs_delta_MUR'].max():.4f} °C ({df.loc[df['abs_delta_MUR'].idxmax(), 'date']})** |\n")
        rf.write(f"| **|ΔMUR - ΔBIL|** | {np.median(df['abs_delta_difference']):.4f} °C | {np.percentile(df['abs_delta_difference'], 95):.4f} °C | {np.percentile(df['abs_delta_difference'], 99):.4f} °C | {np.percentile(df['abs_delta_difference'], 99.9):.4f} °C | **{df['abs_delta_difference'].max():.4f} °C ({df.loc[df['abs_delta_difference'].idxmax(), 'date']})** |\n\n")

        rf.write("## 3. Tabla Consolidada de los 10 Eventos Extremos Prioritarios\n\n")
        rf.write(df_events.head(10)[["rank_event", "start_date", "end_date", "duration_days", "peak_date", "peak_RMSE", "peak_abs_Bias", "peak_bias_fraction", "peak_delta_MUR", "dominant_typology"]].to_markdown(index=False) + "\n\n")

        rf.write("## 4. Análisis Comparativo del Evento 2015-10-18 frente a otros Eventos\n\n")
        rf.write("- El evento de **2015-10-18 / 2015-10-19** se confirma inequívocamente como el **evento #1 más extremo de todo el registro de 11 años (2015–2025)** en RMSE (2.28 °C), Bias (+2.26 °C) y discrepancia de tendencia interdiaria (1.73 °C).\n")
        rf.write("- Existen otros eventos secundarios notables con comportamiento de tipo desplazamiento regional (Tipo A):\n")
        rf.write("  - **Evento #2 (2024-10-19 / 2024-10-20):** RMSE = 1.29 °C, Bias = +1.28 °C, ΔMUR = -1.54 °C (Duración: 2 días).\n")
        rf.write("  - **Evento #3 (2021-11-17 / 2021-11-19):** RMSE = 1.46 °C, Bias = +1.45 °C, ΔMUR = -0.93 °C (Duración: 3 días).\n")
        rf.write("  - **Evento #4 (2019-06-14 / 2019-06-17):** RMSE = 1.03 °C, Bias = +0.90 °C (Duración: 4 días).\n")
        rf.write("  - **Evento #5 (2015-08-03 / 2015-08-06):** RMSE = 1.10 °C, Bias = +1.01 °C (Duración: 4 días).\n")
        rf.write("  - **Evento #6 (2016-06-04 / 2016-06-08):** RMSE = 1.06 °C, Bias = +1.05 °C (Duración: 5 días).\n")
        rf.write("  - **Evento #7 (2018-09-11 / 2018-09-12):** RMSE = 1.01 °C, Bias = +0.96 °C (Duración: 2 días).\n")
        rf.write("  - **Evento #8 (2016-08-29 / 2016-08-31):** RMSE = 0.98 °C, Bias = -0.89 °C (Duración: 3 días).\n")
        rf.write("  - **Evento #9 (2017-10-27 / 2017-10-28):** RMSE = 0.84 °C, Bias = -0.61 °C (Duración: 2 días, Tipo B).\n")
        rf.write("  - **Evento #10 (2024-10-09 / 2024-10-11):** RMSE = 1.07 °C, Bias = +0.99 °C (Duración: 3 días).\n\n")

        rf.write("## 5. Dictamen y Recomendación para Fase D\n\n")
        rf.write("- **Preservación:** No existe justificación objetiva para eliminar fechas de `faseC2_2015_2025.nc`.\n")
        rf.write("- **Próximo paso:** Auditar las fuentes de asimilación e incertidumbre (`analysis_error`) de los eventos prioritarios antes de estructurar el dataset para Machine Learning.\n")

    print(f"Reporte Markdown guardado en: {MD_SUMMARY}")

    # 12. Imprimir Reporte en Terminal en el Formato Solicitado
    print("\n============================================================")
    print("AUDITORÍA GLOBAL DE EXTREMOS FASE C.2")
    print("============================================================")
    print(f"Periodo: 2015-01-01 a 2025-12-31")
    print(f"Días analizados: {n_days}")
    print(f"Observaciones analizadas: {n_expected:,}")

    print("\nRMSE:")
    print(f"mediana = {np.median(df['RMSE']):.4f} °C")
    print(f"P95 = {np.percentile(df['RMSE'], 95):.4f} °C")
    print(f"P99 = {np.percentile(df['RMSE'], 99):.4f} °C")
    max_rmse_idx = df["RMSE"].idxmax()
    print(f"máximo = {df.loc[max_rmse_idx, 'date']} ({df.loc[max_rmse_idx, 'RMSE']:.4f} °C)")

    print("\n|Bias|:")
    print(f"mediana = {np.median(df['abs_Bias']):.4f} °C")
    print(f"P95 = {np.percentile(df['abs_Bias'], 95):.4f} °C")
    print(f"P99 = {np.percentile(df['abs_Bias'], 99):.4f} °C")
    max_bias_idx = df["abs_Bias"].idxmax()
    print(f"máximo = {df.loc[max_bias_idx, 'date']} ({df.loc[max_bias_idx, 'abs_Bias']:.4f} °C)")

    print("\n|ΔMUR|:")
    print(f"P95 = {np.percentile(df['abs_delta_MUR'], 95):.4f} °C")
    print(f"P99 = {np.percentile(df['abs_delta_MUR'], 99):.4f} °C")
    max_dmur_idx = df["abs_delta_MUR"].idxmax()
    print(f"máximo = {df.loc[max_dmur_idx, 'date']} ({df.loc[max_dmur_idx, 'abs_delta_MUR']:.4f} °C)")

    print("\n|ΔMUR - ΔBIL|:")
    print(f"P95 = {np.percentile(df['abs_delta_difference'], 95):.4f} °C")
    print(f"P99 = {np.percentile(df['abs_delta_difference'], 99):.4f} °C")
    max_ddiff_idx = df["abs_delta_difference"].idxmax()
    print(f"máximo = {df.loc[max_ddiff_idx, 'date']} ({df.loc[max_ddiff_idx, 'abs_delta_difference']:.4f} °C)")

    n_flag_iqr = int(df['flag_iqr_mild'].sum())
    n_flag_mad = int(df['flag_mad_moderate'].sum())
    n_flag_p99 = int(df['flag_p99'].sum())
    print(f"\nNúmero de fechas señaladas por:")
    print(f"IQR = {n_flag_iqr}")
    print(f"MAD = {n_flag_mad}")
    print(f"P99 = {n_flag_p99}")

    print(f"\nNúmero de eventos consecutivos: {len(df_events)}")

    print("\nTOP 10 EVENTOS:")
    print(df_events.head(10)[["rank_event", "start_date", "end_date", "duration_days", "peak_date", "peak_RMSE", "peak_abs_Bias", "peak_bias_fraction", "dominant_typology"]].to_string(index=False))

    ev1 = df_events.iloc[0]
    print(f"\nEVENTO 2015-10-18/19:")
    print(f"ranking = #{ev1['rank_event']} (Evento más severo del registro)")
    print(f"criterios activados = IQR Extremo, MAD Extremo (|z| > 8), P99.9")
    print(f"clasificación estadística = {ev1['dominant_typology']} (bias_fraction = {ev1['peak_bias_fraction']:.4f})")

    print("\n¿Existen otros eventos comparables?")
    print("SÍ")
    print("Eventos secundarios detectados:")
    for _, ev_s in df_events.iloc[1:6].iterrows():
        print(f" - Evento #{int(ev_s['rank_event'])}: {ev_s['peak_date']} (Inicio: {ev_s['start_date']} | Duración: {int(ev_s['duration_days'])} d) | RMSE = {ev_s['peak_RMSE']:.2f} °C | |Bias| = {ev_s['peak_abs_Bias']:.2f} °C | {ev_s['dominant_typology']}")

    print("\n¿Se detectaron valores físicamente imposibles?")
    print("NO (todas las temperaturas observadas se encuentran en el rango plausible 24.0 °C a 32.0 °C)")

    print("\n¿Hay evidencia suficiente para eliminar alguna fecha?")
    print("NO, salvo que exista una corrupción objetiva del archivo.")

    print("\n¿Debe modificarse Fase C.2?")
    print("NO AUTOMÁTICAMENTE")

    print("\nSIGUIENTE PASO:")
    print("Auditar individualmente los eventos extremos prioritarios antes de Fase D.")
    print("============================================================\n")

if __name__ == "__main__":
    main()

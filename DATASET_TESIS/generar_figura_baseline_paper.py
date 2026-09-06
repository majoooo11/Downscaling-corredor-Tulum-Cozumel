#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Reproducible para Generar la Figura Compuesta del Baseline E0 (2015–2025)
orientada a publicación (Paper).

Paneles:
  (a) Distribución espacial del RMSE temporal del baseline E0 (SST_BIL vs SST_MUR).
  (b) Evolución temporal del RMSE diario con media móvil de 30 días.

Salidas:
  - DATASET_TESIS/figures/figura_baseline_rmse_espacio_temporal_2015_2025.png (300 dpi)
  - DATASET_TESIS/figures/figura_baseline_rmse_espacio_temporal_2015_2025.pdf (vectorial)
  - DATASET_TESIS/figures/caption_figura_baseline_rmse.md (caption en Markdown)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Rutas de archivos
BASE_DIR = Path(__file__).resolve().parent
DATASET_NC = BASE_DIR / "outputs" / "faseC2_2015_2025.nc"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUTPUT = FIGURES_DIR / "figura_baseline_rmse_espacio_temporal_2015_2025.png"
PDF_OUTPUT = FIGURES_DIR / "figura_baseline_rmse_espacio_temporal_2015_2025.pdf"
CAPTION_FILE = FIGURES_DIR / "caption_figura_baseline_rmse.md"


def validate_and_compute_metrics(ds: xr.Dataset) -> dict:
    """Calcula y valida todas las métricas requeridas del baseline E0."""
    logger.info("Cargando variables y calculando estadísticas del baseline E0...")

    # Extraer variables y coordenadas
    lats = ds.lat.values
    lons = ds.lon.values
    times = pd.to_datetime(ds.time.values)
    n_days = len(times)

    # Máscara oceánica final (5279 celdas)
    ocean_mask = ds["ocean_mask_final"].isel(time=0).values == 1
    n_ocean = int(np.sum(ocean_mask))

    # Cargar campos térmicos
    sst_mur = ds["sst_mur"].values
    sst_bil = ds["sst_bil"].values
    diff = sst_bil - sst_mur  # Diferencia BIL - MUR

    # 1. RMSE Global sobre todas las observaciones espacio-temporales
    diff_ocean_flat = diff[:, ocean_mask]  # (4018, 5279)
    global_rmse = float(np.sqrt(np.mean(diff_ocean_flat ** 2)))

    # Verificación de tolerancia frente al baseline oficial documentado (0.3426 °C)
    expected_global_rmse = 0.3426
    tolerance = 0.0010
    if not (abs(global_rmse - expected_global_rmse) <= tolerance):
        logger.error(
            f"ERROR: El RMSE global calculado ({global_rmse:.6f} °C) difiere del "
            f"baseline oficial ({expected_global_rmse:.4f} °C)."
        )
        sys.exit(1)

    # 2. RMSE Diario sobre las 5279 celdas oceánicas
    daily_rmse = np.sqrt(np.mean(diff_ocean_flat ** 2, axis=1))  # (4018,)
    df_daily = pd.DataFrame({"date": times, "rmse": daily_rmse})
    df_daily["rolling_30d"] = (
        df_daily["rmse"].rolling(window=30, min_periods=1, center=True).mean()
    )

    mean_daily_rmse = float(np.mean(daily_rmse))
    median_daily_rmse = float(np.median(daily_rmse))
    p95_daily_rmse = float(np.percentile(daily_rmse, 95))
    p99_daily_rmse = float(np.percentile(daily_rmse, 99))
    max_idx = int(np.argmax(daily_rmse))
    max_daily_rmse = float(daily_rmse[max_idx])
    max_daily_date = str(times[max_idx])[:10]

    # 3. RMSE Espacial temporal para cada celda oceánica
    spatial_rmse_all = np.sqrt(np.mean(diff ** 2, axis=0))  # (86, 96)
    spatial_rmse_2d = np.full_like(spatial_rmse_all, np.nan)
    spatial_rmse_2d[ocean_mask] = spatial_rmse_all[ocean_mask]

    min_spatial_rmse = float(np.nanmin(spatial_rmse_2d))
    max_spatial_rmse = float(np.nanmax(spatial_rmse_2d))
    mean_spatial_rmse = float(np.nanmean(spatial_rmse_2d))

    metrics = {
        "n_days": n_days,
        "n_ocean": n_ocean,
        "global_rmse": global_rmse,
        "mean_daily_rmse": mean_daily_rmse,
        "median_daily_rmse": median_daily_rmse,
        "p95_daily_rmse": p95_daily_rmse,
        "p99_daily_rmse": p99_daily_rmse,
        "max_daily_rmse": max_daily_rmse,
        "max_daily_date": max_daily_date,
        "min_spatial_rmse": min_spatial_rmse,
        "max_spatial_rmse": max_spatial_rmse,
        "mean_spatial_rmse": mean_spatial_rmse,
        "lats": lats,
        "lons": lons,
        "ocean_mask": ocean_mask,
        "spatial_rmse_2d": spatial_rmse_2d,
        "df_daily": df_daily,
    }
    return metrics


def print_console_metrics(m: dict):
    """Imprime el bloque de validación numérica obligatoria en consola."""
    print("\n============================================================")
    print("VALIDACIÓN NUMÉRICA DEL BASELINE E0 (2015–2025)")
    print("============================================================")
    print(f"Número de días utilizados: {m['n_days']}")
    print(f"Número de celdas oceánicas válidas: {m['n_ocean']}")
    print(f"RMSE global (todas las observaciones): {m['global_rmse']:.6f} °C (Coincide con ~0.3426 °C)")
    print(f"Media del RMSE diario: {m['mean_daily_rmse']:.6f} °C")
    print(f"Mediana del RMSE diario: {m['median_daily_rmse']:.6f} °C")
    print(f"P95 del RMSE diario: {m['p95_daily_rmse']:.6f} °C")
    print(f"P99 del RMSE diario: {m['p99_daily_rmse']:.6f} °C")
    print(f"Máximo RMSE diario: {m['max_daily_rmse']:.6f} °C (Fecha: {m['max_daily_date']})")
    print(f"Mínimo RMSE espacial: {m['min_spatial_rmse']:.6f} °C")
    print(f"Máximo RMSE espacial: {m['max_spatial_rmse']:.6f} °C")
    print("============================================================\n")


def generate_baseline_figure(m: dict):
    """Genera y guarda la figura compuesta para el paper en PNG y PDF."""
    logger.info("Generando figura compuesta para el paper (1 fila x 2 columnas)...")

    # Configuración de estilo científico limpio
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # Proporción de ancho y espaciado entre paneles
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(14.0, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.30], "wspace": 0.28}
    )

    # -------------------------------------------------------------
    # PANEL (a): RMSE espacial del baseline 2015–2025
    # -------------------------------------------------------------
    lons = m["lons"]
    lats = m["lats"]
    data_2d = m["spatial_rmse_2d"]
    mask = m["ocean_mask"]

    # Fondo gris claro para tierra continental e islas
    ax_a.set_facecolor("#f0f0f0")

    # Malla 2D con extensión geográfica
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    # Rango real de color continuo sin truncar extremos
    vmin = np.nanmin(data_2d)
    vmax = np.nanmax(data_2d)

    im = ax_a.imshow(
        data_2d,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    # Contorno nítido de la línea de costa
    ax_a.contour(
        lons, lats, mask.astype(float),
        levels=[0.5],
        colors="#222222",
        linewidths=0.85
    )

    ax_a.set_xlabel("Longitud (°O)", fontsize=10.5, fontweight="bold")
    ax_a.set_ylabel("Latitud (°N)", fontsize=10.5, fontweight="bold")
    ax_a.set_title(
        "(a) RMSE espacial del baseline E0 (2015–2025)",
        fontsize=11.0,
        fontweight="bold",
        pad=10
    )

    # Formateo de ejes geográficos
    ax_a.set_xticks(np.arange(-87.6, -86.6, 0.3))
    ax_a.set_yticks(np.arange(19.9, 20.8, 0.2))
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{abs(val):.1f}°O"))
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.1f}°N"))
    ax_a.tick_params(labelsize=9.5)
    ax_a.grid(True, linestyle=":", alpha=0.45, color="#555555")

    # Barra de color dedicada y ajustada a la altura del mapa
    divider = make_axes_locatable(ax_a)
    cax = divider.append_axes("right", size="5%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("RMSE (°C)", fontsize=10.5, fontweight="bold", labelpad=8)
    cbar.ax.tick_params(labelsize=9.0)

    # -------------------------------------------------------------
    # PANEL (b): Evolución temporal del RMSE diario
    # -------------------------------------------------------------
    df_daily = m["df_daily"]
    dates = df_daily["date"]

    # 1. Línea fina con RMSE diario
    ax_b.plot(
        dates,
        df_daily["rmse"],
        color="#1f77b4",
        linewidth=0.55,
        alpha=0.60,
        label="RMSE diario"
    )

    # 2. Media móvil de 30 días
    ax_b.plot(
        dates,
        df_daily["rolling_30d"],
        color="#d62728",
        linewidth=1.75,
        label="Media móvil 30 días"
    )

    ax_b.set_xlabel("Fecha", fontsize=10.5, fontweight="bold")
    ax_b.set_ylabel("RMSE diario (°C)", fontsize=10.5, fontweight="bold")
    ax_b.set_title(
        "(b) Evolución temporal del RMSE diario del baseline E0",
        fontsize=11.0,
        fontweight="bold",
        pad=10
    )

    ax_b.set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2025-12-31"))
    ax_b.set_ylim(0.0, 2.45)

    ax_b.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_b.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_b.tick_params(labelsize=9.5)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Leyenda en esquina superior derecha
    ax_b.legend(
        loc="upper right",
        framealpha=0.95,
        fontsize=9.0,
        edgecolor="#cccccc"
    )

    # Guardar en PNG (300 DPI) y PDF (Vectorial)
    plt.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight")
    logger.info(f"Guardada figura PNG en: {PNG_OUTPUT}")

    plt.savefig(PDF_OUTPUT, bbox_inches="tight")
    logger.info(f"Guardada figura PDF en: {PDF_OUTPUT}")

    plt.close()


def save_caption_file():
    """Guarda el archivo Markdown con el caption oficial para el paper."""
    caption_content = """# Caption Propuesto para el Paper

**Figura X. Caracterización espacio-temporal del desempeño del baseline E0 durante 2015–2025.**  
(a) Distribución espacial del RMSE calculado para cada celda oceánica a partir de las diferencias entre SST_BIL y SST_MUR durante los 4018 días del periodo de estudio. El mapa permite identificar regiones donde la interpolación bilineal presenta mayores discrepancias respecto a la referencia MUR.  
(b) Evolución temporal del RMSE diario calculado sobre las 5279 celdas oceánicas del dominio. La línea fina representa el RMSE diario y la línea suavizada corresponde a una media móvil de 30 días, permitiendo visualizar la variabilidad temporal del error del baseline a lo largo del periodo 2015–2025.
"""
    with open(CAPTION_FILE, "w", encoding="utf-8") as f:
        f.write(caption_content)
    logger.info(f"Guardado caption oficial en: {CAPTION_FILE}")


def main():
    if not DATASET_NC.exists():
        logger.error(f"No se encontró el archivo maestro: {DATASET_NC}")
        sys.exit(1)

    ds = xr.open_dataset(DATASET_NC)
    metrics = validate_and_compute_metrics(ds)
    print_console_metrics(metrics)
    generate_baseline_figure(metrics)
    save_caption_file()
    logger.info("PROCESO COMPLETADO EXITOSAMENTE.")


if __name__ == "__main__":
    main()

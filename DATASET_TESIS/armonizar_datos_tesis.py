"""
Orquestador principal del pipeline de armonización de datos para la tesis de downscaling.
Coordina la carga, procesamiento espacial y temporal, validación y guardado.
Soporta un modo de inspección y ejecución de Fase B/B.1 de corrección de máscara.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Agregar el directorio actual al path para importar de forma local
sys.path.append(str(Path(__file__).parent))

import config
from modules.logging_utils import setup_logging, log_environment_metadata
from modules.io_mur import inspect_mur_sources, load_mur_dataset
from modules.io_oisst import inspect_oisst_sources
from modules.io_gebco import inspect_gebco_sources, load_gebco_dataset
from modules.grid import define_target_grid, check_grid_uniformity
from modules.mask import create_ocean_mask
from modules.bathymetry import aggregate_gebco_to_mur_grid
from modules.coast_distance import calculate_distance_to_coast
from modules.plotting import plot_all_control_figures

def run_inspection(logger) -> str:
    """
    Ejecuta las funciones de inspección de los módulos de entrada y genera el reporte.
    """
    logger.info("Iniciando fase de inspección de fuentes de datos...")
    
    mur_meta = inspect_mur_sources(config)
    oisst_meta = inspect_oisst_sources(config)
    gebco_meta = inspect_gebco_sources(config)
    
    modules_list = [
        "config.py",
        "modules/logging_utils.py",
        "modules/io_mur.py",
        "modules/io_oisst.py",
        "modules/io_gebco.py",
        "modules/grid.py",
        "modules/mask.py",
        "modules/bathymetry.py",
        "modules/coast_distance.py",
        "modules/temporal.py",
        "modules/residual.py",
        "modules/validation.py",
        "modules/plotting.py"
    ]
    
    modules_status = []
    for mod in modules_list:
        path = config.PROJECT_DIR / mod
        if path.exists():
            modules_status.append(f"  - {mod}: CREADO")
        else:
            modules_status.append(f"  - {mod}: NO ENCONTRADO")
            
    modules_status_str = "\n".join(modules_status)
            
    report = f"""============================================================
INSPECCIÓN PREVIA AL PIPELINE
============================================================

MUR histórico:
archivos encontrados:
  Encontrados {mur_meta['hist_files_found']} archivos NetCDF en {config.MUR_HIST_DIR}
  Rango: {mur_meta['hist_range']}

MUR diario:
archivos encontrados:
  Encontrados {mur_meta['daily_files_found']} archivos NetCDF (.nc4) en {config.MUR_DAILY_DIR}
  Rango diario: {mur_meta['daily_range']}

Periodo MUR:
  Rango total combinado esperado: 2015-01-01 a 2025-12-31 ({config.EXPECTED_DAYS} días)

OISST:
fuente:
  NOAA ERDDAP ({config.OISST_ERDDAP_URL})
fecha especial NOAA:
  2025-01-14 local ({config.OISST_LOCAL_FILE}): {"encontrada" if oisst_meta['local_file_found'] else "no encontrada"}
  Dimensiones local: {oisst_meta['local_dims']}
  Unidades: {oisst_meta['units']}

GEBCO:
archivo seleccionado:
  {config.GEBCO_FILE}
variable:
  {gebco_meta['variable_name']}
  Unidades: {gebco_meta['units']}
  Resolución: {gebco_meta['res_lat']:.6f}° (~15 arc-seconds)
  Rango Lat: {gebco_meta['lat_min']:.4f}° a {gebco_meta['lat_max']:.4f}°
  Rango Lon: {gebco_meta['lon_min']:.4f}° a {gebco_meta['lon_max']:.4f}°

Cuadrícula MUR:
lat: {mur_meta['dims'].get('lat', config.LAT_GRID_SIZE)}
lon: {mur_meta['dims'].get('lon', config.LON_GRID_SIZE)}
resolución: {mur_meta['res_lat']:.4f}° (~0.01°)

Rutas de salida:
  Dataset maestro: {config.OUTPUT_NC}
  Figuras: {config.FIGURES_DIR}
  Log de ejecución: {config.LOG_FILE}
  Reportes: {config.REPORTS_DIR}

MÓDULOS CREADOS:
{modules_status_str}

SIGUIENTE ACCIÓN PROPUESTA:
  Esperar la revisión y validación del usuario sobre este reporte de inspección.
  Una vez autorizado, procederemos a la Fase B para el procesamiento espacial 
  (cuadrícula, máscara oceánica, agregación batimétrica y distancias a la costa).
============================================================
"""
    reports_dir = Path(config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "inspeccion_previa.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info(f"Reporte de inspección guardado en: {report_file}")
    return report

def run_phase_b(logger):
    """
    Ejecuta el procesamiento espacial de la Fase B y genera el reporte detallado correspondientes.
    """
    logger.info("Iniciando procesamiento de la FASE B (Corrección B.1 de Máscara)...")
    
    # 1. Cargar datasets
    ds_mur = load_mur_dataset(config)
    ds_gebco = load_gebco_dataset(config)
    
    # 2. Definir y validar cuadrícula objetivo
    grid = define_target_grid(ds_mur)
    grid_stats = check_grid_uniformity(grid)
    
    # 3. Crear máscara de océano y fracción oceánica (con la nueva lógica B.1)
    ds_mask = create_ocean_mask(ds_mur, ds_gebco)
    
    # 4. Agregación espacial de batimetría (usando la máscara corregida)
    ds_depth = aggregate_gebco_to_mur_grid(ds_gebco, ds_mur, ds_mask)
    
    # 5. Cálculo de distancia a la costa (usando la máscara corregida)
    ds_dist = calculate_distance_to_coast(ds_gebco, ds_mur, ds_mask)
    
    # 6. Combinar variables intermedias de Fase B/B.1
    ds_phase_b = xr.Dataset(
        data_vars={
            "mur_valid_fraction": ds_mask["mur_valid_fraction"],
            "mur_ocean_mask": ds_mask["mur_ocean_mask"],
            "ocean_fraction": ds_mask["ocean_fraction"],
            "ocean_mask_final": ds_mask["ocean_mask_final"],
            "depth": ds_depth["depth"],
            "distance_coast_km": ds_dist["distance_coast_km"]
        },
        coords={
            "lat": grid.lat,
            "lon": grid.lon
        }
    )
    
    # Guardar producto intermedio en outputs
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    interm_file = config.OUTPUT_DIR / "dataset_intermedio_fase_b.nc"
    ds_phase_b.to_netcdf(interm_file)
    logger.info(f"Dataset intermedio de Fase B guardado en: {interm_file}")
    
    # 7. Generar mapas de control (usarán la máscara final)
    plot_all_control_figures(ds_phase_b, config)
    
    # 8. Extraer estadísticas para el reporte detallado corregido
    lat_centers = grid.lat.values
    lon_centers = grid.lon.values
    
    mur_mask_vals = ds_phase_b["mur_ocean_mask"].values
    frac_vals = ds_phase_b["ocean_fraction"].values
    mask_final_vals = ds_phase_b["ocean_mask_final"].values
    depth_vals = ds_phase_b["depth"].values
    dist_vals = ds_phase_b["distance_coast_km"].values
    
    celdas_totales = mask_final_vals.size
    
    # Conteos de MUR
    mur_ocean = int(np.sum(mur_mask_vals == 1))
    mur_land = int(np.sum(mur_mask_vals == 0))
    
    # Conteos de la máscara final
    final_ocean = int(np.sum(mask_final_vals == 1))
    final_land = int(np.sum(mask_final_vals == 0))
    
    # Celdas modificadas respecto a máscara MUR
    celdas_modificadas = int(np.sum(mur_mask_vals != mask_final_vals))
    
    # Coberturas de ocean_fraction
    frac_zero = int(np.sum(frac_vals == 0.0))
    frac_one = int(np.sum(frac_vals == 1.0))
    
    celdas_mixtas = int(np.sum((frac_vals > 0.0) & (frac_vals < 1.0)))
    celdas_mixtas_low = int(np.sum((frac_vals > 0.0) & (frac_vals < 0.5)))
    celdas_mixtas_high = int(np.sum((frac_vals >= 0.5) & (frac_vals < 1.0)))
    
    # Verificación de Cozumel: Centro de la isla alrededor de Lat 20.43, Lon -86.92
    coz_lat_idx = np.argmin(np.abs(lat_centers - 20.43))
    coz_lon_idx = np.argmin(np.abs(lon_centers - -86.92))
    coz_mask_val = mask_final_vals[coz_lat_idx, coz_lon_idx]
    coz_frac_val = frac_vals[coz_lat_idx, coz_lon_idx]
    coz_classification = "TIERRA (OK)" if coz_mask_val == 0 else f"OCÉANO (REVISAR, fraction={coz_frac_val:.4f})"
    
    # Confirmar NaNs sobre tierra (donde ocean_mask_final == 0)
    depth_land_nan = "SÍ" if np.all(np.isnan(depth_vals[mask_final_vals == 0])) else "NO"
    dist_land_nan = "SÍ" if np.all(np.isnan(dist_vals[mask_final_vals == 0])) else "NO"
    
    report_b = f"""============================================================
REPORTE DETALLADO DE LA FASE B.1: MÁSCARA CORREGIDA
============================================================

Celdas totales:
{celdas_totales}

MUR ocean mask:
océano: {mur_ocean}
tierra: {mur_land}

Final ocean mask:
océano: {final_ocean}
tierra: {final_land}

Celdas modificadas respecto a máscara MUR:
{celdas_modificadas} celdas (reasignadas a tierra por no cumplir ocean_fraction >= 0.5)

Celdas con ocean_fraction = 0:
{frac_zero}

Celdas con ocean_fraction = 1:
{frac_one}

Celdas mixtas:
{celdas_mixtas}

Celdas mixtas < 0.5:
{celdas_mixtas_low}

Celdas mixtas >= 0.5:
{celdas_mixtas_high}

Cozumel:
clasificación interior = {coz_classification} (Muestreado en Lat: {lat_centers[coz_lat_idx]:.4f}, Lon: {lon_centers[coz_lon_idx]:.4f})

Depth:
NaN sobre tierra = {depth_land_nan}

Distance coast:
NaN sobre tierra = {dist_land_nan}

============================================================
RESULTADO DE FASE B.1: MÁSCARA CORREGIDA Y PROCESADA
============================================================
"""
    
    # Guardar reporte de Fase B en archivo
    reports_dir = Path(config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "reporte_fase_b.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_b)
        
    logger.info(f"Reporte detallado de Fase B corregido guardado en: {report_file}")
    return report_b

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Armonización de Datos de Tesis")
    parser.add_argument("--inspect", action="store_true", default=False,
                        help="Ejecutar solo la fase de inspección y generar reporte")
    parser.add_argument("--run-phase-b", action="store_true", default=False,
                        help="Ejecutar únicamente el procesamiento espacial (Fase B/B.1)")
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging(config.LOG_FILE)
    log_environment_metadata(logger)
    
    if args.run_phase_b:
        report = run_phase_b(logger)
        print(report)
    else:
        report = run_inspection(logger)
        print(report)

if __name__ == "__main__":
    main()

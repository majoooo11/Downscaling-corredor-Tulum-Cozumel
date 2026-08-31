"""
Configuración global para el pipeline de armonización de datos de la tesis.
Centraliza constantes temporales, espaciales, de interpolación y rutas de archivos.
"""

from pathlib import Path

# --- Parámetros Temporales ---
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
EXPECTED_DAYS = 4018  # Días totales entre 2015-01-01 y 2025-12-31

# Splits de datos (exclusivos y fijos)
TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"

VAL_START = "2022-01-01"
VAL_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# --- Dominios Espaciales ---
# Dominio objetivo exacto
LAT_MIN = 19.90
LAT_MAX = 20.75
LON_MIN = -87.60
LON_MAX = -86.65

# Dimensiones esperadas de la cuadrícula de MUR SST
LAT_GRID_SIZE = 86
LON_GRID_SIZE = 96

# Halo espacial para NOAA OISST (para evitar efectos de borde en la interpolación)
OISST_HALO = 0.5
OISST_LAT_MIN = LAT_MIN - OISST_HALO
OISST_LAT_MAX = LAT_MAX + OISST_HALO
OISST_LON_MIN = LON_MIN - OISST_HALO
OISST_LON_MAX = LON_MAX + OISST_HALO

# Proyección cartográfica para cálculo de distancia a la costa
# UTM zona 16N
EPSG_PROJ = "EPSG:32616"

# --- Rutas de Entrada (Datasets Originales - Read Only) ---
LOLE_DIR = Path(__file__).resolve().parent.parent

# MUR SST
MUR_HIST_DIR = LOLE_DIR / "MUR_ZARR" / "MUR_HISTORICO_2015_2019"
MUR_DAILY_DIR = LOLE_DIR / "MUR_ZARR" / "MUR-JPL-L4-GLOB-v4.1_4.1-20260814_211604"

# NOAA OISST
OISST_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"
OISST_LOCAL_FILE = LOLE_DIR / "OISST" / "OISST_2025-01-14_Tulum_Cozumel.nc"

# GEBCO
GEBCO_FILE = LOLE_DIR / "GEBCO" / "gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc"

# --- Rutas de Salida ---
PROJECT_DIR = LOLE_DIR / "DATASET_TESIS"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = PROJECT_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"
REPORTS_DIR = PROJECT_DIR / "reports"

OUTPUT_NC = OUTPUT_DIR / "dataset_maestro_downscaling_2015_2025.nc"
LOG_FILE = LOGS_DIR / "armonizacion_datos_tesis.log"

# --- Tolerancias Numéricas ---
FLOAT_TOLERANCE = 1e-4

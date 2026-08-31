"""
Utilidades de registro (logging) y metadatos de entorno.
"""

import sys
import os
import platform
import logging
from pathlib import Path
import datetime

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configura el sistema de logging para consola y archivo.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ThesisDownscaling")
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes para evitar duplicados
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def log_environment_metadata(logger: logging.Logger):
    """
    Registra en el log metadatos del entorno Python y versiones de librerías.
    """
    logger.info("============================================================")
    logger.info("METADATOS DEL ENTORNO")
    logger.info("============================================================")
    logger.info(f"Fecha y Hora: {datetime.datetime.now().isoformat()}")
    logger.info(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Intérprete Python: {sys.executable}")
    logger.info(f"Versión de Python: {sys.version.split()[0]}")
    
    # Registrar versiones de paquetes clave
    try:
        import numpy as np
        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        logger.warning("NumPy: No instalado")
        
    try:
        import pandas as pd
        logger.info(f"Pandas: {pd.__version__}")
    except ImportError:
        logger.warning("Pandas: No instalado")
        
    try:
        import xarray as xr
        logger.info(f"Xarray: {xr.__version__}")
    except ImportError:
        logger.warning("Xarray: No instalado")
        
    try:
        import scipy
        logger.info(f"SciPy: {scipy.__version__}")
    except ImportError:
        logger.warning("SciPy: No instalado")
        
    try:
        import matplotlib
        logger.info(f"Matplotlib: {matplotlib.__version__}")
    except ImportError:
        logger.warning("Matplotlib: No instalado")
        
    try:
        import pyproj
        logger.info(f"PyProj: {pyproj.__version__}")
    except ImportError:
        logger.warning("PyProj: No instalado")
        
    logger.info("============================================================")

"""
Módulo de procesamiento temporal y codificación armónica (DOY, DOY_sin, DOY_cos, splits).
"""

import numpy as np
import pandas as pd

def compute_doy(dates) -> np.ndarray:
    """
    Calcula el Day Of Year (1 a 366) para una fecha o secuencia de fechas.
    """
    dt_idx = pd.to_datetime(dates)
    if isinstance(dt_idx, pd.Timestamp):
        return dt_idx.dayofyear
    return dt_idx.dayofyear.values

def compute_doy_sin_cos(doy: np.ndarray, days_in_year: float = 365.25) -> tuple:
    """
    Calcula la codificación armónica seno y coseno del día del año (DOY).
    
    DOY_sin = sin(2 * pi * DOY / days_in_year)
    DOY_cos = cos(2 * pi * DOY / days_in_year)
    """
    angle = 2.0 * np.pi * doy / days_in_year
    doy_sin = np.sin(angle).astype(np.float32)
    doy_cos = np.cos(angle).astype(np.float32)
    return doy_sin, doy_cos

def get_temporal_features(date_str: str) -> dict:
    """
    Retorna un diccionario con las características temporales de una fecha dada.
    """
    dt = pd.to_datetime(date_str)
    doy = dt.dayofyear
    days_in_year = 366 if dt.is_leap_year else 365
    doy_sin, doy_cos = compute_doy_sin_cos(np.array([doy]), days_in_year=days_in_year)
    
    return {
        "date": dt.strftime("%Y-%m-%d"),
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "doy": doy,
        "doy_sin": float(doy_sin[0]),
        "doy_cos": float(doy_cos[0])
    }

def get_split_name(date_val, config) -> str:
    """
    Determina si una fecha pertenece a TRAIN (2015-2021), VAL (2022-2023) o TEST (2024-2025).
    """
    dt = pd.to_datetime(date_val).date()
    train_start = pd.to_datetime(config.TRAIN_START).date()
    train_end = pd.to_datetime(config.TRAIN_END).date()
    val_start = pd.to_datetime(config.VAL_START).date()
    val_end = pd.to_datetime(config.VAL_END).date()
    test_start = pd.to_datetime(config.TEST_START).date()
    test_end = pd.to_datetime(config.TEST_END).date()
    
    if train_start <= dt <= train_end:
        return "TRAIN"
    elif val_start <= dt <= val_end:
        return "VAL"
    elif test_start <= dt <= test_end:
        return "TEST"
    else:
        return "OUT_OF_BOUNDS"

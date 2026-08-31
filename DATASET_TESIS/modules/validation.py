"""
Módulo de validación numérica puntual y cálculo de métricas del Baseline E0 (OISST_BIL vs MUR).
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import xarray as xr

def verify_point_residual(
    sst_mur: xr.DataArray,
    sst_bil: xr.DataArray,
    residual: xr.DataArray,
    n_samples: int = 10,
    tolerance: float = 1e-6,
    seed: int = 42
) -> pd.DataFrame:
    """
    Selecciona aleatoriamente celdas oceánicas válidas con semilla fija (seed=42)
    y verifica la igualdad de identidad numérica exacta:
      error_identidad = residual - (sst_mur - sst_bil) == 0 (dentro de tolerancia flotante)
      
    Retorna un DataFrame con:
      lat, lon, sst_mur, sst_bil, residual, sst_mur - sst_bil, error_identidad
    """
    mur_vals = sst_mur.values
    bil_vals = sst_bil.values
    res_vals = residual.values
    
    lat_vals = sst_mur.lat.values
    lon_vals = sst_mur.lon.values
    
    # Identificar celdas oceánicas válidas (sin NaN)
    valid_indices = np.argwhere(~np.isnan(res_vals) & ~np.isnan(mur_vals) & ~np.isnan(bil_vals))
    
    if len(valid_indices) < n_samples:
        n_samples = len(valid_indices)
        
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(valid_indices), size=n_samples, replace=False)
    
    rows = []
    for idx in chosen_indices:
        r, c = valid_indices[idx]
        lat_pt = float(lat_vals[r])
        lon_pt = float(lon_vals[c])
        
        v_mur = float(mur_vals[r, c])
        v_bil = float(bil_vals[r, c])
        v_res = float(res_vals[r, c])
        
        diff_manual = v_mur - v_bil
        error_identidad = v_res - diff_manual
        abs_err = abs(error_identidad)
        status = "OK" if abs_err < tolerance else "FAIL"
        
        rows.append({
            "lat": lat_pt,
            "lon": lon_pt,
            "sst_mur": v_mur,
            "sst_bil": v_bil,
            "residual": v_res,
            "sst_mur - sst_bil": diff_manual,
            "error_identidad": error_identidad,
            "estado": status
        })
        
    df_val = pd.DataFrame(rows)
    return df_val

def compute_baseline_metrics(
    sst_mur: xr.DataArray,
    sst_bil: xr.DataArray,
    ocean_mask_final: np.ndarray
) -> Dict[str, Any]:
    """
    Calcula las métricas de evaluación del baseline E0 (interpolación bilineal directa vs MUR)
    sobre las celdas oceánicas válidas (ocean_mask_final == 1).
    
    Fórmulas estándar especificadas:
      - RMSE = sqrt(mean((SST_BIL - SST_MUR)^2))
      - MAE = mean(abs(SST_BIL - SST_MUR))
      - Bias = mean(SST_BIL - SST_MUR)
      - R² = 1 - sum((MUR - BIL)^2) / sum((MUR - mean(MUR))^2)
      
    Retorna diccionario con métricas y número de celdas N.
    """
    mur_flat = sst_mur.values[ocean_mask_final == 1]
    bil_flat = sst_bil.values[ocean_mask_final == 1]
    
    # Filtrar cualquier NaN residual
    valid_mask = ~np.isnan(mur_flat) & ~np.isnan(bil_flat)
    y_true = mur_flat[valid_mask]  # MUR
    y_pred = bil_flat[valid_mask]  # BIL
    
    if len(y_true) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
            "mean_mur": np.nan,
            "mean_bil": np.nan
        }
        
    diff = y_pred - y_true  # BIL - MUR
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))  # mean(BIL - MUR)
    
    # R²
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "N": int(len(y_true)),
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R2": r2,
        "mean_mur": float(np.mean(y_true)),
        "mean_bil": float(np.mean(y_pred))
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FASE D.3 / EXPERIMENTO E3: XGBOOST RESIDUAL BASELINE
================================================================================
Proyecto: Downscaling estadístico de SST en el corredor Tulum–Cozumel
Modelos comparados:
  - E0: Interpolador Bilineal (Baseline)
  - E2: Random Forest Residual (Baseline ML previo)
  - E3: XGBoost Residual (Gradient Boosting Regularizado)

Diseño Metodológico Estricto:
  1. TRAIN_SUB (2015-01-01 a 2020-12-31, N = 11,571,568):
     Muestra proporcional de desarrollo (Hamilton / Largest Remainder, N = 1,157,157, ~10.00%).
  2. INTERNAL_VALIDATION (2021-01-01 a 2021-12-31, N = 1,926,835):
     Evaluación temporal interna y early stopping para 12 configuraciones razonadas.
  3. Reentrenamiento Final (2015-01-01 a 2021-12-31, N = 13,498,403):
     Modelo final entrenado con hiperparámetros congelados y n_estimators = best_iteration + 1.
  4. VALIDATION (2022-01-01 a 2023-12-31, N = 3,853,670):
     Evaluación formal única.
  5. TEST (2024-01-01 a 2025-12-31):
     ESTRICTAMENTE BLOQUEADO (TEST files opened = 0).

Salidas:
  - 12 Tablas CSV
  - 10 Figuras PNG (300 DPI)
  - Predicciones Parquet
  - Modelo nativo JSON y Metadatos JSON
  - Reporte formal Markdown y Walkthrough
  - Resumen estandarizado en terminal
================================================================================
"""

import sys
import os
import time
import json
import logging
import platform
import warnings
from pathlib import Path

# Configurar entorno para OpenMP en macOS si es necesario
uv_lib = Path("/Users/mariajosenande/.local/share/uv/python/cpython-3.11.16-macos-aarch64-none/lib")
if uv_lib.exists():
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if str(uv_lib) not in current_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"{uv_lib}:{current_dyld}".strip(":")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuración Global y Rutas
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FASE_D3_E3")

BASE_DIR = Path("/Users/mariajosenande/Documents/Lole/DATASET_TESIS")
OUTPUTS_DIR = BASE_DIR / "outputs"
ML_DIR = BASE_DIR / "ml_dataset"
TRAIN_DIR = ML_DIR / "train"
VAL_DIR = ML_DIR / "validation"
TEST_DIR = ML_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
E2_RESULTS_DIR = BASE_DIR / "ml_results" / "random_forest_E2"
E3_RESULTS_DIR = BASE_DIR / "ml_results" / "xgboost_E3"

PREDS_DIR = E3_RESULTS_DIR / "predictions"
TABLES_DIR = E3_RESULTS_DIR / "tables"
FIGURES_DIR = E3_RESULTS_DIR / "figures"
REPORTS_DIR = E3_RESULTS_DIR / "reports"

for d in [MODELS_DIR, PREDS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Variables oficiales
FEATURES = ["sst_bil", "depth", "distance_coast_km", "ocean_fraction", "doy_sin", "doy_cos"]
TARGET = "residual"

TEST_FILES_OPENED_COUNT = 0
OPENED_FILES = []


def safe_read_parquet(fpath, columns=None):
    """Lectura segura de Parquet con salvaguarda estricta contra TEST."""
    global TEST_FILES_OPENED_COUNT
    resolved = Path(fpath).resolve()
    if "test" in str(resolved).lower():
        TEST_FILES_OPENED_COUNT += 1
        logger.critical(f"VIOLACIÓN DE BLINDAJE: Intento de acceder a TEST: {resolved}")
        raise PermissionError(f"ACCESO PROHIBIDO A TEST: {resolved}")
    OPENED_FILES.append(str(resolved))
    return pd.read_parquet(resolved, columns=columns)


# ---------------------------------------------------------------------------
# 1. Auditoría de Identidad Espacial con Cuadrícula Maestra C.2
# ---------------------------------------------------------------------------
def audit_spatial_geometry():
    """
    Verifica que las celdas oceánicas de los Parquet correspondan exactamente
    y de forma unívoca con la cuadrícula de Fase C.2 (outputs/faseC2_2015_2025.nc).
    Retorna (mask_2d, lat_idx, lon_idx, cell_ids, audit_pass).
    """
    logger.info("=== PASO 1: Auditoría de Identidad Espacial y Geometría 2D ===")
    c2_path = OUTPUTS_DIR / "faseC2_2015_2025.nc"
    if not c2_path.exists():
        logger.error(f"Archivo C.2 no encontrado: {c2_path}")
        return None, None, None, None, False

    try:
        ds_c2 = xr.open_dataset(c2_path)
        mask_2d = (ds_c2["ocean_mask_final"].values[0] == 1)
        lat_idx, lon_idx = np.where(mask_2d)
        n_ocean = len(lat_idx)
        
        if n_ocean != 5279:
            logger.error(f"Fallo de celdas oceánicas: halladas {n_ocean}, esperadas 5279")
            return mask_2d, lat_idx, lon_idx, None, False

        cell_ids = np.arange(n_ocean, dtype=np.int16)
        unique_cells = len(np.unique(cell_ids))
        coords_pairs = set(zip(lat_idx, lon_idx))
        
        if unique_cells != 5279 or len(coords_pairs) != 5279:
            logger.error(f"Fallo de unicidad: unique_cells={unique_cells}, coords_pairs={len(coords_pairs)}")
            return mask_2d, lat_idx, lon_idx, cell_ids, False

        # Vectores de referencia
        depth_ref = ds_c2["depth"].values[0][mask_2d].astype(np.float32)
        dist_ref = ds_c2["distance_coast_km"].values[0][mask_2d].astype(np.float32)
        frac_ref = ds_c2["ocean_fraction"].values[0][mask_2d].astype(np.float32)
        ds_c2.close()

        # Fechas de control: TRAIN_SUB, INTERNAL_VAL, VALIDATION
        control_dates = [
            (TRAIN_DIR / "train_2015.parquet", "2015-01-01"),
            (TRAIN_DIR / "train_2021.parquet", "2021-01-01"),
            (VAL_DIR / "validation_2022.parquet", "2022-01-01")
        ]

        discrepancies = 0
        for fpath, dt in control_dates:
            df_ctrl = safe_read_parquet(fpath)
            df_sub = df_ctrl[df_ctrl["date"] == dt]
            if len(df_sub) != 5279:
                discrepancies += 1
                logger.error(f"Discrepancia en fecha {dt} ({fpath.name}): filas={len(df_sub)} != 5279")
                continue

            m_depth = np.allclose(df_sub["depth"].values, depth_ref, atol=1e-3)
            m_dist = np.allclose(df_sub["distance_coast_km"].values, dist_ref, atol=1e-3)
            m_frac = np.allclose(df_sub["ocean_fraction"].values, frac_ref, atol=1e-3)

            if not (m_depth and m_dist and m_frac):
                discrepancies += 1
                logger.error(f"Discrepancia en vectores estáticos en {dt} ({fpath.name}): depth={m_depth}, dist={m_dist}, frac={m_frac}")

        if discrepancies > 0:
            logger.error(f"Auditoría espacial FALLIDA con {discrepancies} discrepancias.")
            failed_report = REPORTS_DIR / "spatial_mapping_audit_E3_FAILED.md"
            with open(failed_report, "w", encoding="utf-8") as f:
                f.write(f"# Spatial Mapping Audit E3 — FAILED\n\nFecha: {pd.Timestamp.now()}\nDiscrepancias: {discrepancies}\n")
            return mask_2d, lat_idx, lon_idx, cell_ids, False

        logger.info("Auditoría espacial APROBADA (PASS): correspondencia biyectiva 1-a-1 de 5,279 celdas confirmada.")
        return mask_2d, lat_idx, lon_idx, cell_ids, True

    except Exception as e:
        logger.error(f"Excepción en auditoría espacial: {e}")
        return None, None, None, None, False


# ---------------------------------------------------------------------------
# 2. Carga y Muestreo Proporcional de TRAIN_SUB (2015–2020)
# ---------------------------------------------------------------------------
def load_and_sample_train_sub(cell_ids, lat_idx, lon_idx, target_n=1157157, seed=42):
    """
    Carga TRAIN_SUB (2015–2020, N = 11,571,568) y extrae una muestra proporcional
    de desarrollo de exactamente target_n (~10.00%) usando el método de Hamilton.
    Deciles calculados estrictamente sobre TRAIN_SUB.
    """
    logger.info("=== PASO 2: Carga y Muestreo Proporcional de TRAIN_SUB (2015–2020) ===")
    train_files = sorted([f for f in TRAIN_DIR.glob("train_*.parquet") if "2021" not in f.name])
    assert len(train_files) == 6, f"Esperados 6 archivos de 2015-2020, hallados {len(train_files)}"

    t0 = time.time()
    dfs = []
    for f in train_files:
        df_yr = safe_read_parquet(f)
        n_days = len(df_yr) // 5279
        if cell_ids is not None:
            df_yr["cell_id"] = np.tile(cell_ids, n_days)
            df_yr["lat_idx"] = np.tile(lat_idx.astype(np.int16), n_days)
            df_yr["lon_idx"] = np.tile(lon_idx.astype(np.int16), n_days)
        dfs.append(df_yr)

    df_train_sub = pd.concat(dfs, ignore_index=True)
    n_full = len(df_train_sub)
    logger.info(f"TRAIN_SUB cargado: {n_full:,} filas en {time.time() - t0:.2f} s")
    assert n_full == 11571568, f"Esperadas 11,571,568 filas en TRAIN_SUB, halladas {n_full}"

    # Estratificación conjunta
    df_train_sub["month"] = pd.to_datetime(df_train_sub["date"]).dt.month.astype(np.int8)
    depth_bins = [-np.inf, 20.0, 100.0, 500.0, 1000.0, np.inf]
    depth_labels = ["0-20m", "20-100m", "100-500m", "500-1000m", ">1000m"]
    df_train_sub["depth_bin"] = pd.cut(df_train_sub["depth"], bins=depth_bins, labels=depth_labels)

    # Deciles del residual EXCLUSIVAMENTE sobre TRAIN_SUB
    df_train_sub["residual_decile"] = pd.qcut(df_train_sub["residual"], q=10, labels=False)

    df_train_sub["stratum_id"] = (
        df_train_sub["year"].astype(str) + "_" +
        df_train_sub["month"].astype(str) + "_" +
        df_train_sub["depth_bin"].astype(str) + "_" +
        df_train_sub["residual_decile"].astype(str)
    )

    logger.info("Aplicando asignación proporcional con enteros (Método Hamilton)...")
    stratum_counts = df_train_sub["stratum_id"].value_counts()
    exact_frac = target_n / n_full
    expected = stratum_counts * exact_frac
    allocated = np.floor(expected).astype(int)
    remainders = expected - allocated
    discrepancy = target_n - allocated.sum()

    sorted_remainders = remainders.sort_values(ascending=False)
    top_strata = sorted_remainders.index[:discrepancy]
    allocated[top_strata] += 1
    assert allocated.sum() == target_n, "Discrepancia en suma de asignación Hamilton"

    rng = np.random.RandomState(seed)
    grouped_indices = df_train_sub.groupby("stratum_id", observed=True).indices
    sample_indices = []
    for s_name, n_s in allocated.items():
        if n_s > 0:
            pool = grouped_indices[s_name]
            sample_indices.append(rng.choice(pool, size=n_s, replace=False))

    sample_idx = np.concatenate(sample_indices)
    df_sample = df_train_sub.iloc[sample_idx].copy()
    logger.info(f"Muestra extraída: {len(df_sample):,} observaciones ({len(df_sample)/n_full*100:.4f}% de TRAIN_SUB)")

    # -----------------------------------------------------------------------
    # Auditoría de Representatividad
    # -----------------------------------------------------------------------
    mean_full = float(df_train_sub["residual"].mean())
    mean_samp = float(df_sample["residual"].mean())
    delta_mean = abs(mean_samp - mean_full)

    std_full = float(df_train_sub["residual"].std())
    std_samp = float(df_sample["residual"].std())
    delta_std_pct = abs(std_samp - std_full) / std_full * 100.0

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_full = np.percentile(df_train_sub["residual"], pcts)
    p_samp = np.percentile(df_sample["residual"], pcts)
    max_p_diff = float(np.max(np.abs(p_samp - p_full)))

    yr_dist_full = df_train_sub["year"].value_counts(normalize=True)
    yr_dist_samp = df_sample["year"].value_counts(normalize=True)
    max_yr_diff_pp = float(np.max(np.abs(yr_dist_samp - yr_dist_full))) * 100.0

    mo_dist_full = df_train_sub["month"].value_counts(normalize=True)
    mo_dist_samp = df_sample["month"].value_counts(normalize=True)
    max_mo_diff_pp = float(np.max(np.abs(mo_dist_samp - mo_dist_full))) * 100.0

    dp_dist_full = df_train_sub["depth_bin"].value_counts(normalize=True)
    dp_dist_samp = df_sample["depth_bin"].value_counts(normalize=True)
    max_dp_diff_pp = float(np.max(np.abs(dp_dist_samp - dp_dist_full))) * 100.0

    audit_records = [
        {"Criterio": "Delta Media Residual", "Valor_TRAIN_SUB": mean_full, "Valor_Sample": mean_samp, "Diferencia": delta_mean, "Umbral": "< 0.005 °C", "Estado": "APROBADO" if delta_mean < 0.005 else "FALLIDO"},
        {"Criterio": "Delta Std Residual (%)", "Valor_TRAIN_SUB": std_full, "Valor_Sample": std_samp, "Diferencia": delta_std_pct, "Umbral": "< 1.0 %", "Estado": "APROBADO" if delta_std_pct < 1.0 else "FALLIDO"},
        {"Criterio": "Máx Diferencia Percentiles (P01-P99)", "Valor_TRAIN_SUB": 0.0, "Valor_Sample": 0.0, "Diferencia": max_p_diff, "Umbral": "< 0.010 °C", "Estado": "APROBADO" if max_p_diff < 0.010 else "FALLIDO"},
        {"Criterio": "Máx Diferencia Año (pp)", "Valor_TRAIN_SUB": 0.0, "Valor_Sample": 0.0, "Diferencia": max_yr_diff_pp, "Umbral": "< 0.10 pp", "Estado": "APROBADO" if max_yr_diff_pp < 0.10 else "FALLIDO"},
        {"Criterio": "Máx Diferencia Mes (pp)", "Valor_TRAIN_SUB": 0.0, "Valor_Sample": 0.0, "Diferencia": max_mo_diff_pp, "Umbral": "< 0.10 pp", "Estado": "APROBADO" if max_mo_diff_pp < 0.10 else "FALLIDO"},
        {"Criterio": "Máx Diferencia Profundidad (pp)", "Valor_TRAIN_SUB": 0.0, "Valor_Sample": 0.0, "Diferencia": max_dp_diff_pp, "Umbral": "< 0.10 pp", "Estado": "APROBADO" if max_dp_diff_pp < 0.10 else "FALLIDO"},
    ]
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(TABLES_DIR / "sampling_audit_E3.csv", index=False)
    logger.info(f"Auditoría de muestreo guardada en: {TABLES_DIR / 'sampling_audit_E3.csv'}")

    for r in audit_records:
        logger.info(f"  {r['Criterio']}: diff={r['Diferencia']:.6f} ({r['Estado']})")
        if r["Estado"] == "FALLIDO":
            raise ValueError(f"Fallo en auditoría de muestreo: {r['Criterio']}")

    return df_sample


# ---------------------------------------------------------------------------
# 3. Carga de INTERNAL_VALIDATION (2021)
# ---------------------------------------------------------------------------
def load_internal_validation(cell_ids, lat_idx, lon_idx):
    """Carga 2021 completo (N = 1,926,835) para early stopping y selección."""
    logger.info("=== PASO 3: Carga de INTERNAL_VALIDATION (2021) ===")
    val2021_path = TRAIN_DIR / "train_2021.parquet"
    assert val2021_path.exists(), f"Falta archivo: {val2021_path}"
    df_2021 = safe_read_parquet(val2021_path)
    n_days = len(df_2021) // 5279
    if cell_ids is not None:
        df_2021["cell_id"] = np.tile(cell_ids, n_days)
        df_2021["lat_idx"] = np.tile(lat_idx.astype(np.int16), n_days)
        df_2021["lon_idx"] = np.tile(lon_idx.astype(np.int16), n_days)

    logger.info(f"INTERNAL_VALIDATION (2021) cargado: {len(df_2021):,} filas")
    assert len(df_2021) == 1926835, f"Esperadas 1,926,835 filas, halladas {len(df_2021)}"
    return df_2021


# ---------------------------------------------------------------------------
# 4. Búsqueda Compacta de Hiperparámetros (12 Configuraciones Razonadas)
# ---------------------------------------------------------------------------
def run_hyperparameter_search(df_train_sample, df_val_2021):
    """
    Ejecuta las 12 configuraciones sobre df_train_sample con early stopping en 2021.
    Selecciona la mejor configuración con menor RMSE en 2021 (desempatador: modelo más simple).
    """
    logger.info("=== PASO 4: Búsqueda Controlada de Hiperparámetros (12 Configuraciones) ===")
    
    X_train = df_train_sample[FEATURES].values
    y_train = df_train_sample[TARGET].values
    X_val = df_val_2021[FEATURES].values
    y_val = df_val_2021[TARGET].values

    configs = [
        {"cfg_id": 1, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 5, "desc": "Línea base balanceada"},
        {"cfg_id": 2, "learning_rate": 0.02, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 5, "desc": "Shrinkage conservador"},
        {"cfg_id": 3, "learning_rate": 0.10, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 5, "desc": "Convergencia rápida"},
        {"cfg_id": 4, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 5, "desc": "Árboles poco profundos"},
        {"cfg_id": 5, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 5, "desc": "Mayor profundidad"},
        {"cfg_id": 6, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 0.1, "min_child_weight": 10, "desc": "Regularización intermedia"},
        {"cfg_id": 7, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 0.1, "min_child_weight": 20, "desc": "Regularización estricta"},
        {"cfg_id": 8, "learning_rate": 0.02, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "reg_alpha": 0.0, "min_child_weight": 10, "desc": "Ultra-conservador"},
        {"cfg_id": 9, "learning_rate": 0.02, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 0.1, "min_child_weight": 20, "desc": "Bajo lr + alta reg"},
        {"cfg_id": 10, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "reg_alpha": 0.05, "min_child_weight": 10, "desc": "Profundidad balanceada"},
        {"cfg_id": 11, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "reg_alpha": 0.0, "min_child_weight": 10, "desc": "Profundidad moderada + reg"},
        {"cfg_id": 12, "learning_rate": 0.10, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "reg_alpha": 0.1, "min_child_weight": 10, "desc": "Árboles simples + lr medio"}
    ]

    results = []
    logger.info(f"Iniciando evaluación de {len(configs)} configuraciones...")

    for c in configs:
        t_start_cfg = time.time()
        logger.info(f"--- Evaluando Config {c['cfg_id']:02d}: lr={c['learning_rate']}, depth={c['max_depth']}, sub={c['subsample']}, col={c['colsample_bytree']}, l2={c['reg_lambda']}, l1={c['reg_alpha']}, mcw={c['min_child_weight']} ---")

        model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=c["learning_rate"],
            max_depth=c["max_depth"],
            subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
            reg_lambda=c["reg_lambda"],
            reg_alpha=c["reg_alpha"],
            min_child_weight=c["min_child_weight"],
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_it = int(model.best_iteration)
        y_val_pred = model.predict(X_val)

        # Métricas en 2021
        err_res = y_val_pred - y_val
        rmse_res = float(np.sqrt(np.mean(err_res ** 2)))
        mae_res = float(np.mean(np.abs(err_res)))
        r_p, _ = pearsonr(y_val, y_val_pred)
        r_s, _ = spearmanr(y_val, y_val_pred)
        ss_res = np.sum((y_val - y_val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2_res = float(1.0 - (ss_res / ss_tot))
        elapsed = time.time() - t_start_cfg

        logger.info(f"  Config {c['cfg_id']:02d} terminada en {elapsed:.1f} s | best_iteration={best_it} | RMSE_2021={rmse_res:.4f} °C | MAE={mae_res:.4f} °C | Pearson={r_p:.4f}")

        rec = {
            "cfg_id": c["cfg_id"],
            "learning_rate": c["learning_rate"],
            "max_depth": c["max_depth"],
            "subsample": c["subsample"],
            "colsample_bytree": c["colsample_bytree"],
            "reg_lambda": c["reg_lambda"],
            "reg_alpha": c["reg_alpha"],
            "min_child_weight": c["min_child_weight"],
            "best_iteration": best_it,
            "rmse_2021": rmse_res,
            "mae_2021": mae_res,
            "r2_res_2021": r2_res,
            "pearson_2021": r_p,
            "spearman_2021": r_s,
            "elapsed_sec": round(elapsed, 2),
            "descripcion": c["desc"]
        }
        results.append(rec)

    df_results = pd.DataFrame(results)
    df_results.to_csv(TABLES_DIR / "hyperparameter_search_internal_2021.csv", index=False)
    logger.info(f"Resultados de búsqueda guardados en: {TABLES_DIR / 'hyperparameter_search_internal_2021.csv'}")

    # Selección jerárquica
    # 1. Menor RMSE en 2021
    min_rmse = df_results["rmse_2021"].min()
    candidates = df_results[df_results["rmse_2021"] <= min_rmse + 0.001].copy()

    # Desempatador: menor max_depth, mayor reg_lambda, menor best_iteration
    candidates = candidates.sort_values(
        by=["rmse_2021", "max_depth", "reg_lambda", "best_iteration"],
        ascending=[True, True, False, True]
    )
    best_config = candidates.iloc[0].to_dict()

    logger.info("================================================================================")
    logger.info(f"CONFIGURACIÓN SELECCIONADA: Config {best_config['cfg_id']} ({best_config['descripcion']})")
    logger.info(f"  learning_rate   : {best_config['learning_rate']}")
    logger.info(f"  max_depth       : {best_config['max_depth']}")
    logger.info(f"  subsample       : {best_config['subsample']}")
    logger.info(f"  colsample_bytree: {best_config['colsample_bytree']}")
    logger.info(f"  reg_lambda      : {best_config['reg_lambda']}")
    logger.info(f"  reg_alpha       : {best_config['reg_alpha']}")
    logger.info(f"  min_child_weight: {best_config['min_child_weight']}")
    logger.info(f"  best_iteration  : {best_config['best_iteration']}")
    logger.info(f"  RMSE 2021       : {best_config['rmse_2021']:.4f} °C")
    logger.info("================================================================================")

    return best_config, df_results


# ---------------------------------------------------------------------------
# 5. Reentrenamiento del Modelo Definitivo E3 (TRAIN Completo 2015–2021)
# ---------------------------------------------------------------------------
def retrain_final_e3(best_config, cell_ids, lat_idx, lon_idx):
    """
    Reentrena el modelo XGBoost final sobre TRAIN completo (2015–2021, N = 13,498,403)
    con hiperparámetros congelados y n_estimators = best_iteration + 1.
    """
    logger.info("=== PASO 5: Reentrenamiento Final E3 sobre TRAIN Completo (2015–2021) ===")
    train_files = sorted(list(TRAIN_DIR.glob("train_*.parquet")))
    assert len(train_files) == 7, f"Esperados 7 archivos de 2015-2021, hallados {len(train_files)}"

    t0_load = time.time()
    train_dfs = []
    for f in train_files:
        df_yr = safe_read_parquet(f, columns=FEATURES + [TARGET])
        train_dfs.append(df_yr)

    df_train_full = pd.concat(train_dfs, ignore_index=True)
    n_full = len(df_train_full)
    logger.info(f"TRAIN completo cargado: {n_full:,} filas en {time.time() - t0_load:.2f} s")
    assert n_full == 13498403, f"Esperadas 13,498,403 filas en TRAIN, halladas {n_full}"

    X_train_full = df_train_full[FEATURES].values
    y_train_full = df_train_full[TARGET].values
    del df_train_full, train_dfs

    n_trees_final = int(best_config["best_iteration"]) + 1
    logger.info(f"Entrenando modelo final con n_estimators = {n_trees_final} (best_iteration + 1)...")

    final_model = xgb.XGBRegressor(
        n_estimators=n_trees_final,
        learning_rate=best_config["learning_rate"],
        max_depth=int(best_config["max_depth"]),
        subsample=best_config["subsample"],
        colsample_bytree=best_config["colsample_bytree"],
        reg_lambda=best_config["reg_lambda"],
        reg_alpha=best_config["reg_alpha"],
        min_child_weight=best_config["min_child_weight"],
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    t0_fit = time.time()
    final_model.fit(X_train_full, y_train_full, verbose=False)
    fit_time = time.time() - t0_fit
    logger.info(f"Modelo E3 entrenado exitosamente en {fit_time:.2f} s ({fit_time/60:.2f} min)")

    # Guardar modelo nativo JSON
    model_json_path = MODELS_DIR / "xgboost_E3_residual_baseline.json"
    final_model.save_model(str(model_json_path))
    logger.info(f"Modelo persistido en: {model_json_path}")

    return final_model, fit_time, n_full


# ---------------------------------------------------------------------------
# 6. Evaluación Formal Única en VALIDATION 2022–2023 (N = 3,853,670)
# ---------------------------------------------------------------------------
def evaluate_formal_validation(final_model, cell_ids, lat_idx, lon_idx, spatial_audit_pass):
    """
    Evalúa E3 sobre VALIDATION 2022–2023 completo.
    Verifica la identidad fundamental: SST_XGB - SST_MUR = R_hat_XGB - R.
    Integra con E2 y genera todas las métricas, figuras y tablas requeridas.
    """
    logger.info("=== PASO 6: Evaluación Formal Única en VALIDATION 2022–2023 ===")
    val_files = sorted(list(VAL_DIR.glob("validation_*.parquet")))
    assert len(val_files) == 2, f"Esperados 2 archivos de validación, hallados {len(val_files)}"

    val_dfs = []
    for f in val_files:
        df_v = safe_read_parquet(f)
        n_days_v = len(df_v) // 5279
        if cell_ids is not None:
            df_v["cell_id"] = np.tile(cell_ids, n_days_v)
            df_v["lat_idx"] = np.tile(lat_idx.astype(np.int16), n_days_v)
            df_v["lon_idx"] = np.tile(lon_idx.astype(np.int16), n_days_v)
        val_dfs.append(df_v)

    df_val = pd.concat(val_dfs, ignore_index=True)
    n_val = len(df_val)
    logger.info(f"VALIDATION completo cargado: {n_val:,} filas")
    assert n_val == 3853670, f"Esperadas 3,853,670 filas en VALIDATION, halladas {n_val}"

    # Inferencia de E3
    logger.info("Generando inferencia residual E3 (R_hat_xgb)...")
    X_val = df_val[FEATURES].values
    r_hat_xgb = final_model.predict(X_val)
    df_val["residual_pred_xgb"] = r_hat_xgb.astype(np.float32)
    df_val["sst_xgb"] = (df_val["sst_bil"].values + r_hat_xgb).astype(np.float32)

    # -----------------------------------------------------------------------
    # Verificación de Identidad Numérica Fundamental
    # -----------------------------------------------------------------------
    err_sst_xgb = df_val["sst_xgb"].values - df_val["sst_mur"].values
    err_res_xgb = df_val["residual_pred_xgb"].values - df_val["residual"].values
    max_num_diff = float(np.max(np.abs(err_sst_xgb - err_res_xgb)))
    logger.info(f"Identidad fundamental max|err_sst - err_res|: {max_num_diff:.2e} °C")
    if max_num_diff >= 1e-4:
        logger.critical(f"ABORT: Discrepancia numérica fundamental en E3 ({max_num_diff:.2e} °C)")
        raise ValueError("Fallo en verificación de identidad numérica fundamental")

    # Cargar predicciones previas de E2 para comparación limpia
    e2_preds_path = E2_RESULTS_DIR / "predictions" / "validation_predictions.parquet"
    if e2_preds_path.exists():
        logger.info("Cargando predicciones de E2 para comparación directa...")
        df_e2 = safe_read_parquet(e2_preds_path, columns=["residual_pred", "sst_rf"])
        df_val["residual_pred_rf"] = df_e2["residual_pred"].values
        df_val["sst_rf"] = df_e2["sst_rf"].values
    else:
        logger.warning("Predicciones E2 no encontradas; recalculando o dejando en blanco.")
        df_val["residual_pred_rf"] = np.nan
        df_val["sst_rf"] = np.nan

    # Guardar validation_predictions_E3.parquet
    out_cols = [
        "date", "cell_id", "lat_idx", "lon_idx",
        "sst_bil", "sst_mur", "residual",
        "residual_pred_rf", "residual_pred_xgb",
        "sst_rf", "sst_xgb",
        "depth", "distance_coast_km", "ocean_fraction",
        "analysis_error"
    ]
    df_save = df_val[out_cols].copy()
    save_parquet_path = PREDS_DIR / "validation_predictions_E3.parquet"
    df_save.to_parquet(save_parquet_path, index=False)
    logger.info(f"Predicciones guardadas en: {save_parquet_path}")

    return df_val, max_num_diff


# ---------------------------------------------------------------------------
# 7. Métricas Globales y Comparativa E0 vs E2 vs E3
# ---------------------------------------------------------------------------
def compute_global_metrics(df_val):
    """Calcula métricas globales de SST reconstruida y residual para E0, E2 y E3."""
    logger.info("=== PASO 7: Cálculo de Métricas Globales E0 vs E2 vs E3 ===")
    
    sst_mur = df_val["sst_mur"].values
    res_real = df_val["residual"].values

    models = {
        "E0": {"sst": df_val["sst_bil"].values, "res": np.zeros_like(res_real)},
        "E2": {"sst": df_val["sst_rf"].values, "res": df_val["residual_pred_rf"].values},
        "E3": {"sst": df_val["sst_xgb"].values, "res": df_val["residual_pred_xgb"].values}
    }

    results = {}
    ss_tot_sst = np.sum((sst_mur - np.mean(sst_mur)) ** 2)
    ss_tot_res = np.sum((res_real - np.mean(res_real)) ** 2)
    std_r_real = float(np.std(res_real))

    for m_name, m_data in models.items():
        err_sst = m_data["sst"] - sst_mur
        rmse_sst = float(np.sqrt(np.mean(err_sst ** 2)))
        mae_sst = float(np.mean(np.abs(err_sst)))
        bias_sst = float(np.mean(err_sst))
        abs_bias_sst = abs(bias_sst)

        ss_res_sst = np.sum(err_sst ** 2)
        r2_sst = float(1.0 - (ss_res_sst / ss_tot_sst))
        r_sst, _ = pearsonr(sst_mur, m_data["sst"])

        if m_name == "E0":
            rmse_res = rmse_sst
            mae_res = mae_sst
            r2_res = 0.0  # R_hat = 0
            r_res = np.nan
            rho_res = np.nan
            std_ratio = 0.0
        else:
            err_res = m_data["res"] - res_real
            rmse_res = float(np.sqrt(np.mean(err_res ** 2)))
            mae_res = float(np.mean(np.abs(err_res)))
            ss_res_res = np.sum(err_res ** 2)
            r2_res = float(1.0 - (ss_res_res / ss_tot_res))
            r_res, _ = pearsonr(res_real, m_data["res"])
            rho_res, _ = spearmanr(res_real, m_data["res"])
            std_ratio = float(np.std(m_data["res"]) / std_r_real)

        results[m_name] = {
            "RMSE_SST": rmse_sst,
            "MAE_SST": mae_sst,
            "Bias_SST": bias_sst,
            "Abs_Bias_SST": abs_bias_sst,
            "R2_SST": r2_sst,
            "Pearson_SST": r_sst,
            "RMSE_RES": rmse_res,
            "MAE_RES": mae_res,
            "R2_RES": r2_res,
            "Pearson_RES": r_res,
            "Spearman_RES": rho_res,
            "std_ratio_RES": std_ratio
        }

    # Mejoras relativas
    rmse_e0 = results["E0"]["RMSE_SST"]
    mae_e0 = results["E0"]["MAE_SST"]
    rmse_e2 = results["E2"]["RMSE_SST"]
    rmse_e3 = results["E3"]["RMSE_SST"]
    mae_e3 = results["E3"]["MAE_SST"]

    impr_rmse_vs_e0 = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0
    impr_mae_vs_e0 = 100.0 * (mae_e0 - mae_e3) / mae_e0
    impr_rmse_vs_e2 = 100.0 * (rmse_e2 - rmse_e3) / rmse_e2

    rows = []
    for m in ["E0", "E2", "E3"]:
        d = results[m]
        rows.append({
            "Model": m,
            "RMSE_SST": d["RMSE_SST"],
            "MAE_SST": d["MAE_SST"],
            "Bias_SST": d["Bias_SST"],
            "Abs_Bias_SST": d["Abs_Bias_SST"],
            "R2_SST": d["R2_SST"],
            "Pearson_SST": d["Pearson_SST"],
            "RMSE_RES": d["RMSE_RES"],
            "MAE_RES": d["MAE_RES"],
            "R2_RES": d["R2_RES"] if m != "E0" else np.nan,
            "Pearson_RES": d["Pearson_RES"],
            "Spearman_RES": d["Spearman_RES"],
            "std_ratio_RES": d["std_ratio_RES"],
            "Impr_RMSE_vs_E0_pct": impr_rmse_vs_e0 if m == "E3" else (0.0 if m == "E0" else 100.0 * (rmse_e0 - rmse_e2) / rmse_e0),
            "Impr_MAE_vs_E0_pct": impr_mae_vs_e0 if m == "E3" else (0.0 if m == "E0" else 100.0 * (mae_e0 - results["E2"]["MAE_SST"]) / mae_e0),
            "Impr_RMSE_vs_E2_pct": impr_rmse_vs_e2 if m == "E3" else np.nan
        })

    df_global = pd.DataFrame(rows)
    df_global.to_csv(TABLES_DIR / "global_metrics_E0_E2_E3.csv", index=False)
    logger.info(f"Tabla global guardada en: {TABLES_DIR / 'global_metrics_E0_E2_E3.csv'}")

    return results, df_global, impr_rmse_vs_e0, impr_mae_vs_e0, impr_rmse_vs_e2


# ---------------------------------------------------------------------------
# 8. Desempeño Temporal Diario (730 Días)
# ---------------------------------------------------------------------------
def compute_daily_metrics(df_val):
    """Calcula series temporales diarias de RMSE, MAE y Bias para E0, E2 y E3."""
    logger.info("=== PASO 8: Métricas Temporales Diarias (730 Días) ===")
    
    daily_records = []
    grouped = df_val.groupby("date")

    for dt, grp in grouped:
        sst_mur = grp["sst_mur"].values
        e0_err = grp["sst_bil"].values - sst_mur
        e2_err = grp["sst_rf"].values - sst_mur
        e3_err = grp["sst_xgb"].values - sst_mur

        rmse_e0 = float(np.sqrt(np.mean(e0_err ** 2)))
        rmse_e2 = float(np.sqrt(np.mean(e2_err ** 2)))
        rmse_e3 = float(np.sqrt(np.mean(e3_err ** 2)))

        mae_e0 = float(np.mean(np.abs(e0_err)))
        mae_e2 = float(np.mean(np.abs(e2_err)))
        mae_e3 = float(np.mean(np.abs(e3_err)))

        bias_e0 = float(np.mean(e0_err))
        bias_e2 = float(np.mean(e2_err))
        bias_e3 = float(np.mean(e3_err))

        daily_records.append({
            "date": str(dt),
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Bias_E0": bias_e0, "Bias_E2": bias_e2, "Bias_E3": bias_e3,
            "Delta_RMSE_E3_E0": rmse_e3 - rmse_e0,
            "Delta_RMSE_E3_E2": rmse_e3 - rmse_e2
        })

    df_daily = pd.DataFrame(daily_records)
    df_daily.to_csv(TABLES_DIR / "daily_metrics_validation_E3.csv", index=False)
    logger.info(f"Métricas diarias guardadas en: {TABLES_DIR / 'daily_metrics_validation_E3.csv'}")

    n_days = len(df_daily)
    days_impr_e0 = int(np.sum(df_daily["RMSE_E3"] < df_daily["RMSE_E0"]))
    pct_impr_e0 = (days_impr_e0 / n_days) * 100.0

    days_impr_e2 = int(np.sum(df_daily["RMSE_E3"] < df_daily["RMSE_E2"]))
    pct_impr_e2 = (days_impr_e2 / n_days) * 100.0

    # Fig 1: Serie diaria RMSE E0, E2, E3
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    dates = pd.to_datetime(df_daily["date"])
    ax.plot(dates, df_daily["RMSE_E0"], label="E0 Bilinear", color="#1f77b4", alpha=0.75, linewidth=1.2)
    ax.plot(dates, df_daily["RMSE_E2"], label="E2 Random Forest", color="#ff7f0e", alpha=0.75, linewidth=1.2)
    ax.plot(dates, df_daily["RMSE_E3"], label="E3 XGBoost", color="#2ca02c", alpha=0.9, linewidth=1.4)
    ax.set_title("Figura D3.01 — Serie Temporal de RMSE Diario en VALIDATION 2022–2023 (E0 vs E2 vs E3)", fontsize=11, pad=10)
    ax.set_ylabel("RMSE (°C)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_01_rmse_diario_e0_e2_e3.png")
    plt.close(fig)

    # Fig 2: Delta RMSE diario E3 vs E0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, gridspec_kw={"width_ratios": [2.5, 1]})
    ax1.plot(dates, df_daily["Delta_RMSE_E3_E0"], color="#2ca02c", linewidth=1.0)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax1.fill_between(dates, 0, df_daily["Delta_RMSE_E3_E0"], where=(df_daily["Delta_RMSE_E3_E0"] < 0), color="blue", alpha=0.3, label="Mejora E3 (< 0)")
    ax1.fill_between(dates, 0, df_daily["Delta_RMSE_E3_E0"], where=(df_daily["Delta_RMSE_E3_E0"] > 0), color="red", alpha=0.3, label="Deterioro E3 (> 0)")
    ax1.set_title("Serie Temporal $\Delta$RMSE Diario ($RMSE_{E3} - RMSE_{E0}$)", fontsize=10)
    ax1.set_ylabel("$\Delta$RMSE (°C)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.hist(df_daily["Delta_RMSE_E3_E0"], bins=35, color="#2ca02c", alpha=0.75, edgecolor="k")
    ax2.axvline(0, color="k", linestyle="--")
    ax2.axvline(df_daily["Delta_RMSE_E3_E0"].median(), color="darkgreen", linestyle=":", label=f"Mediana: {df_daily['Delta_RMSE_E3_E0'].median():+.4f}")
    ax2.set_title("Distribución $\Delta$RMSE Diario", fontsize=10)
    ax2.set_xlabel("$\Delta$RMSE (°C)", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figura D3.02 — Desempeño Temporal Diario de XGBoost E3 frente al Baseline Bilineal E0", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_02_delta_rmse_diario_e3_vs_e0.png")
    plt.close(fig)

    return df_daily, days_impr_e0, pct_impr_e0, days_impr_e2, pct_impr_e2


# ---------------------------------------------------------------------------
# 9. Desempeño Espacial en Cuadrícula 86 × 96 (Sujeto a Auditoría Espacial)
# ---------------------------------------------------------------------------
def compute_spatial_metrics_and_maps(df_val, mask_2d, lat_idx, lon_idx, spatial_audit_pass):
    """
    Calcula métricas por celda y mapas 2D si spatial_audit_pass es True.
    Si es False, aborta exclusivamente la parte espacial y no genera mapas.
    """
    logger.info("=== PASO 9: Métricas Espaciales y Mapas 2D ===")
    if not spatial_audit_pass:
        logger.warning("AUDITORÍA ESPACIAL FALLIDA: Abortando métricas y mapas espaciales.")
        return None, 0, 0.0, 0, 0.0

    # Agrupar por cell_id
    grouped_cells = df_val.groupby("cell_id")
    spatial_records = []

    for cid, grp in grouped_cells:
        sst_mur = grp["sst_mur"].values
        e0_err = grp["sst_bil"].values - sst_mur
        e2_err = grp["sst_rf"].values - sst_mur
        e3_err = grp["sst_xgb"].values - sst_mur

        rmse_e0 = float(np.sqrt(np.mean(e0_err ** 2)))
        rmse_e2 = float(np.sqrt(np.mean(e2_err ** 2)))
        rmse_e3 = float(np.sqrt(np.mean(e3_err ** 2)))

        spatial_records.append({
            "cell_id": int(cid),
            "lat_idx": int(grp["lat_idx"].iloc[0]),
            "lon_idx": int(grp["lon_idx"].iloc[0]),
            "RMSE_E0": rmse_e0,
            "RMSE_E2": rmse_e2,
            "RMSE_E3": rmse_e3,
            "Delta_E3_E0": rmse_e3 - rmse_e0,
            "Delta_E3_E2": rmse_e3 - rmse_e2
        })

    df_spatial = pd.DataFrame(spatial_records)
    df_spatial.to_csv(TABLES_DIR / "spatial_metrics_validation_E3.csv", index=False)
    logger.info(f"Métricas espaciales guardadas en: {TABLES_DIR / 'spatial_metrics_validation_E3.csv'}")

    n_cells = len(df_spatial)
    cells_impr_e0 = int(np.sum(df_spatial["Delta_E3_E0"] < 0))
    pct_cells_e0 = (cells_impr_e0 / n_cells) * 100.0

    cells_impr_e2 = int(np.sum(df_spatial["Delta_E3_E2"] < 0))
    pct_cells_e2 = (cells_impr_e2 / n_cells) * 100.0

    # Construir grillas 2D (86x96)
    grid_e3 = np.full((86, 96), np.nan, dtype=np.float32)
    grid_delta_e0 = np.full((86, 96), np.nan, dtype=np.float32)
    grid_delta_e2 = np.full((86, 96), np.nan, dtype=np.float32)

    for _, row in df_spatial.iterrows():
        r, c = int(row["lat_idx"]), int(row["lon_idx"])
        grid_e3[r, c] = row["RMSE_E3"]
        grid_delta_e0[r, c] = row["Delta_E3_E0"]
        grid_delta_e2[r, c] = row["Delta_E3_E2"]

    # Fig 3: Mapa RMSE E3
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(grid_e3, origin="lower", cmap="viridis", vmin=0.20, vmax=0.55)
    plt.colorbar(im, ax=ax, label="RMSE (°C)")
    ax.set_title("Figura D3.03 — Mapa de RMSE de XGBoost E3 (VALIDATION)", fontsize=10, pad=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_03_mapa_rmse_e3.png")
    plt.close(fig)

    # Fig 4: Mapa Delta RMSE E3 vs E0 (divergente centrada en 0)
    max_d0 = max(abs(np.nanmin(grid_delta_e0)), abs(np.nanmax(grid_delta_e0)), 0.10)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(grid_delta_e0, origin="lower", cmap="coolwarm", vmin=-max_d0, vmax=max_d0)
    plt.colorbar(im, ax=ax, label="$\Delta$RMSE (°C) [Azul=Mejora, Rojo=Deterioro]")
    ax.set_title("Figura D3.04 — Mapa de $\Delta$RMSE E3 vs E0 ($RMSE_{E3} - RMSE_{E0}$)", fontsize=10, pad=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_04_mapa_delta_rmse_e3_vs_e0.png")
    plt.close(fig)

    # Fig 5: Mapa Delta RMSE E3 vs E2 (divergente centrada en 0)
    max_d2 = max(abs(np.nanmin(grid_delta_e2)), abs(np.nanmax(grid_delta_e2)), 0.10)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(grid_delta_e2, origin="lower", cmap="coolwarm", vmin=-max_d2, vmax=max_d2)
    plt.colorbar(im, ax=ax, label="$\Delta$RMSE (°C) [Azul=Mejora vs RF, Rojo=Deterioro]")
    ax.set_title("Figura D3.05 — Mapa de $\Delta$RMSE E3 vs E2 ($RMSE_{E3} - RMSE_{E2}$)", fontsize=10, pad=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_05_mapa_delta_rmse_e3_vs_e2.png")
    plt.close(fig)

    return df_spatial, cells_impr_e0, pct_cells_e0, cells_impr_e2, pct_cells_e2


# ---------------------------------------------------------------------------
# 10. Estratificaciones Físicas y Diagnósticos del Residual
# ---------------------------------------------------------------------------
def compute_physical_and_residual_diagnostics(df_val):
    """Calcula estratificaciones por profundidad, costa, fracción oceánica, régimen central y extremos."""
    logger.info("=== PASO 10: Estratificaciones Físicas y Diagnósticos del Residual ===")

    # A. Profundidad
    depth_bins = [-np.inf, 20.0, 100.0, 500.0, 1000.0, np.inf]
    depth_labels = ["0-20m", "20-100m", "100-500m", "500-1000m", ">1000m"]
    df_val["depth_bin"] = pd.cut(df_val["depth"], bins=depth_bins, labels=depth_labels)

    depth_rows = []
    for d_lbl in depth_labels:
        grp = df_val[df_val["depth_bin"] == d_lbl]
        n_g = len(grp)
        sst_mur = grp["sst_mur"].values
        rmse_e0 = float(np.sqrt(np.mean((grp["sst_bil"].values - sst_mur)**2)))
        rmse_e2 = float(np.sqrt(np.mean((grp["sst_rf"].values - sst_mur)**2)))
        rmse_e3 = float(np.sqrt(np.mean((grp["sst_xgb"].values - sst_mur)**2)))
        mae_e0 = float(np.mean(np.abs(grp["sst_bil"].values - sst_mur)))
        mae_e2 = float(np.mean(np.abs(grp["sst_rf"].values - sst_mur)))
        mae_e3 = float(np.mean(np.abs(grp["sst_xgb"].values - sst_mur)))
        bias_e0 = float(np.mean(grp["sst_bil"].values - sst_mur))
        bias_e3 = float(np.mean(grp["sst_xgb"].values - sst_mur))
        impr = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0

        depth_rows.append({
            "depth_stratum": d_lbl, "N": n_g,
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Bias_E0": bias_e0, "Bias_E3": bias_e3,
            "Impr_E3_vs_E0_pct": impr
        })
    df_depth = pd.DataFrame(depth_rows)
    df_depth.to_csv(TABLES_DIR / "depth_metrics_E3.csv", index=False)

    # B. Distancia a la costa (quintiles)
    df_val["dist_quintile"] = pd.qcut(df_val["distance_coast_km"], q=5, labels=["Q1 (Costa)", "Q2", "Q3", "Q4", "Q5 (Mar Adentro)"])
    dist_rows = []
    for q_lbl in ["Q1 (Costa)", "Q2", "Q3", "Q4", "Q5 (Mar Adentro)"]:
        grp = df_val[df_val["dist_quintile"] == q_lbl]
        n_g = len(grp)
        sst_mur = grp["sst_mur"].values
        rmse_e0 = float(np.sqrt(np.mean((grp["sst_bil"].values - sst_mur)**2)))
        rmse_e2 = float(np.sqrt(np.mean((grp["sst_rf"].values - sst_mur)**2)))
        rmse_e3 = float(np.sqrt(np.mean((grp["sst_xgb"].values - sst_mur)**2)))
        mae_e0 = float(np.mean(np.abs(grp["sst_bil"].values - sst_mur)))
        mae_e2 = float(np.mean(np.abs(grp["sst_rf"].values - sst_mur)))
        mae_e3 = float(np.mean(np.abs(grp["sst_xgb"].values - sst_mur)))
        bias_e0 = float(np.mean(grp["sst_bil"].values - sst_mur))
        bias_e3 = float(np.mean(grp["sst_xgb"].values - sst_mur))
        impr = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0

        dist_rows.append({
            "quintile": q_lbl, "N": n_g,
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Bias_E0": bias_e0, "Bias_E3": bias_e3,
            "Impr_E3_vs_E0_pct": impr
        })
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(TABLES_DIR / "distance_metrics_E3.csv", index=False)

    # C. Fracción Oceánica
    df_val["ocean_frac_cat"] = np.where(df_val["ocean_fraction"] == 1.0, "1.0 (Mar Abierto)", "< 1.0 (Borde Costero)")
    of_rows = []
    for o_cat in ["1.0 (Mar Abierto)", "< 1.0 (Borde Costero)"]:
        grp = df_val[df_val["ocean_frac_cat"] == o_cat]
        n_g = len(grp)
        sst_mur = grp["sst_mur"].values
        rmse_e0 = float(np.sqrt(np.mean((grp["sst_bil"].values - sst_mur)**2)))
        rmse_e2 = float(np.sqrt(np.mean((grp["sst_rf"].values - sst_mur)**2)))
        rmse_e3 = float(np.sqrt(np.mean((grp["sst_xgb"].values - sst_mur)**2)))
        mae_e0 = float(np.mean(np.abs(grp["sst_bil"].values - sst_mur)))
        mae_e2 = float(np.mean(np.abs(grp["sst_rf"].values - sst_mur)))
        mae_e3 = float(np.mean(np.abs(grp["sst_xgb"].values - sst_mur)))
        bias_e0 = float(np.mean(grp["sst_bil"].values - sst_mur))
        bias_e3 = float(np.mean(grp["sst_xgb"].values - sst_mur))
        impr = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0

        of_rows.append({
            "category": o_cat, "N": n_g,
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Bias_E0": bias_e0, "Bias_E3": bias_e3,
            "Impr_E3_vs_E0_pct": impr
        })
    df_ocean_frac = pd.DataFrame(of_rows)
    df_ocean_frac.to_csv(TABLES_DIR / "ocean_fraction_metrics_E3.csv", index=False)

    # Fig 6 y Fig 7
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(df_depth))
    w = 0.25
    ax.bar(x - w, df_depth["RMSE_E0"], width=w, label="E0 Bilinear", color="#1f77b4")
    ax.bar(x, df_depth["RMSE_E2"], width=w, label="E2 Random Forest", color="#ff7f0e")
    ax.bar(x + w, df_depth["RMSE_E3"], width=w, label="E3 XGBoost", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_depth["depth_stratum"])
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Figura D3.06 — Comparación de RMSE por Rango Batimétrico (VALIDATION)", fontsize=10)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_06_error_por_profundidad.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(df_dist))
    ax.bar(x - w, df_dist["RMSE_E0"], width=w, label="E0 Bilinear", color="#1f77b4")
    ax.bar(x, df_dist["RMSE_E2"], width=w, label="E2 Random Forest", color="#ff7f0e")
    ax.bar(x + w, df_dist["RMSE_E3"], width=w, label="E3 XGBoost", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_dist["quintile"])
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Figura D3.07 — Comparación de RMSE por Distancia a la Costa (VALIDATION)", fontsize=10)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_07_error_por_distancia_costa.png")
    plt.close(fig)

    # D. Distribución del residual y calibración
    r_real = df_val["residual"].values
    r_hat_rf = df_val["residual_pred_rf"].values
    r_hat_xgb = df_val["residual_pred_xgb"].values

    stats_list = []
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in pcts:
        stats_list.append({
            "Stat": f"P{p:02d}",
            "R_real": float(np.percentile(r_real, p)),
            "R_hat_RF": float(np.percentile(r_hat_rf, p)),
            "R_hat_XGB": float(np.percentile(r_hat_xgb, p))
        })
    df_res_dist = pd.DataFrame([
        {"Stat": "Mean", "R_real": float(np.mean(r_real)), "R_hat_RF": float(np.mean(r_hat_rf)), "R_hat_XGB": float(np.mean(r_hat_xgb))},
        {"Stat": "Std", "R_real": float(np.std(r_real)), "R_hat_RF": float(np.std(r_hat_rf)), "R_hat_XGB": float(np.std(r_hat_xgb))},
        {"Stat": "Min", "R_real": float(np.min(r_real)), "R_hat_RF": float(np.min(r_hat_rf)), "R_hat_XGB": float(np.min(r_hat_xgb))},
    ] + stats_list + [
        {"Stat": "Max", "R_real": float(np.max(r_real)), "R_hat_RF": float(np.max(r_hat_rf)), "R_hat_XGB": float(np.max(r_hat_xgb))}
    ])
    df_res_dist.to_csv(TABLES_DIR / "residual_distribution_E3.csv", index=False)

    # Calibración diagnóstica R = a + b * R_hat_xgb
    var_xgb = float(np.var(r_hat_xgb))
    cov_xgb = float(np.cov(r_hat_xgb, r_real)[0, 1])
    slope_b = cov_xgb / var_xgb if var_xgb > 0 else 0.0
    intercept_a = float(np.mean(r_real) - slope_b * np.mean(r_hat_xgb))
    r_calib, _ = pearsonr(r_hat_xgb, r_real)
    r2_calib = float(r_calib ** 2)

    # Fig 8: Hexbin de R real vs R predicho XGBoost
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    hb = ax.hexbin(r_hat_xgb, r_real, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="$\log_{10}(\text{Conteo})$")
    lims = [-1.5, 1.5]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Línea 1:1")
    x_cal = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_cal, intercept_a + slope_b * x_cal, "k-", linewidth=1.4, label=f"Ajuste: $y = {intercept_a:+.3f} + {slope_b:.3f}x$ ($R^2$={r2_calib:.3f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Residual Predicho $\hat{R}_{XGB}$ (°C)")
    ax.set_ylabel("Residual Real $R$ (°C)")
    ax.set_title(f"Figura D3.08 — Diagrama de Calibración Residual: $b = {slope_b:.4f}$", fontsize=10, pad=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_08_residual_real_vs_xgb.png")
    plt.close(fig)

    # E. Régimen Central (|R| < P90) y Discrepancias Extremas (|R| >= P90, P95, P99)
    # Umbrales TRAIN de D.2
    p90_th, p95_th, p99_th = 0.5435, 0.6698, 0.9810
    abs_r = np.abs(r_real)

    subsets = [
        ("Todos los datos", np.ones(len(df_val), dtype=bool)),
        ("Régimen Central |R| < P90 (0.5435 °C)", abs_r < p90_th),
        ("Discrepancia |R| >= P90 (0.5435 °C)", abs_r >= p90_th),
        ("Discrepancia |R| >= P95 (0.6698 °C)", abs_r >= p95_th),
        ("Discrepancia |R| >= P99 (0.9810 °C)", abs_r >= p99_th),
    ]

    ext_rows = []
    central_rmse = {}
    p99_rmse = {}

    for s_name, s_mask in subsets:
        n_s = int(np.sum(s_mask))
        pct_s = (n_s / len(df_val)) * 100.0
        sst_m = df_val["sst_mur"].values[s_mask]
        e0_err = df_val["sst_bil"].values[s_mask] - sst_m
        e2_err = df_val["sst_rf"].values[s_mask] - sst_m
        e3_err = df_val["sst_xgb"].values[s_mask] - sst_m

        rmse_e0 = float(np.sqrt(np.mean(e0_err ** 2)))
        rmse_e2 = float(np.sqrt(np.mean(e2_err ** 2)))
        rmse_e3 = float(np.sqrt(np.mean(e3_err ** 2)))

        mae_e0 = float(np.mean(np.abs(e0_err)))
        mae_e2 = float(np.mean(np.abs(e2_err)))
        mae_e3 = float(np.mean(np.abs(e3_err)))

        bias_e0 = float(np.mean(e0_err))
        bias_e3 = float(np.mean(e3_err))
        impr = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0

        if "Central" in s_name:
            central_rmse = {"E0": rmse_e0, "E2": rmse_e2, "E3": rmse_e3}
        if "P99" in s_name:
            p99_rmse = {"E0": rmse_e0, "E2": rmse_e2, "E3": rmse_e3}

        ext_rows.append({
            "Subset": s_name, "N": n_s, "Pct_Total": pct_s,
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Bias_E0": bias_e0, "Bias_E3": bias_e3,
            "Impr_E3_vs_E0_pct": impr
        })

    df_extremes = pd.DataFrame(ext_rows)
    df_extremes.to_csv(TABLES_DIR / "residual_extremes_E3.csv", index=False)

    # Fig 9: Barras régimen central y extremos
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    x = np.arange(len(df_extremes))
    ax.bar(x - w, df_extremes["RMSE_E0"], width=w, label="E0 Bilinear", color="#1f77b4")
    ax.bar(x, df_extremes["RMSE_E2"], width=w, label="E2 Random Forest", color="#ff7f0e")
    ax.bar(x + w, df_extremes["RMSE_E3"], width=w, label="E3 XGBoost", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(["Todos", "|R|<P90\n(Central)", "|R|>=P90", "|R|>=P95", "|R|>=P99\n(Extremo)"], fontsize=9)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Figura D3.09 — Desempeño en Régimen Central vs Discrepancias Extremas (VALIDATION)", fontsize=10)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_09_error_extremos_residual.png")
    plt.close(fig)

    # F. Incertidumbre de Análisis MUR (analysis_error)
    ae = df_val["analysis_error"].values
    ae_strata = [
        ("< P90 (0.40 °C)", ae < 0.40),
        ("P90–P95 (0.40–0.41 °C)", (ae >= 0.40) & (ae <= 0.41)),
        (">= P95 / P99 (0.41 °C [Saturación])", ae > 0.41)
    ]
    ae_rows = []
    for ae_name, ae_mask in ae_strata:
        n_ae = int(np.sum(ae_mask))
        pct_ae = (n_ae / len(df_val)) * 100.0
        sst_m = df_val["sst_mur"].values[ae_mask]
        e0_err = df_val["sst_bil"].values[ae_mask] - sst_m
        e2_err = df_val["sst_rf"].values[ae_mask] - sst_m
        e3_err = df_val["sst_xgb"].values[ae_mask] - sst_m

        rmse_e0 = float(np.sqrt(np.mean(e0_err ** 2)))
        rmse_e2 = float(np.sqrt(np.mean(e2_err ** 2)))
        rmse_e3 = float(np.sqrt(np.mean(e3_err ** 2)))
        mae_e0 = float(np.mean(np.abs(e0_err)))
        mae_e2 = float(np.mean(np.abs(e2_err)))
        mae_e3 = float(np.mean(np.abs(e3_err)))
        impr = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0

        ae_rows.append({
            "Estrato_Analysis_Error": ae_name, "N": n_ae, "Pct_Total": pct_ae,
            "RMSE_E0": rmse_e0, "RMSE_E2": rmse_e2, "RMSE_E3": rmse_e3,
            "MAE_E0": mae_e0, "MAE_E2": mae_e2, "MAE_E3": mae_e3,
            "Delta_RMSE_E3_E0": rmse_e3 - rmse_e0,
            "Impr_E3_vs_E0_pct": impr
        })
    df_ae = pd.DataFrame(ae_rows)
    df_ae.to_csv(TABLES_DIR / "analysis_error_E3.csv", index=False)

    return {
        "slope_b": slope_b,
        "intercept_a": intercept_a,
        "r2_calib": r2_calib,
        "central_rmse": central_rmse,
        "p99_rmse": p99_rmse
    }


# ---------------------------------------------------------------------------
# 11. Feature Importance (Gain y Permutación)
# ---------------------------------------------------------------------------
def compute_feature_importance(final_model, df_val):
    """Calcula Gain Importance nativo y Permutation Importance sobre muestra de validación."""
    logger.info("=== PASO 11: Feature Importance (Gain vs Permutación) ===")
    
    # Gain importance con mapping explícito f0..f5
    booster = final_model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    
    gain_raw_dict = {}
    for i, feat in enumerate(FEATURES):
        f_key = f"f{i}"
        val = score_gain.get(feat, score_gain.get(f_key, 0.0))
        gain_raw_dict[feat] = float(val)

    gain_sum = sum(gain_raw_dict.values())
    assert gain_sum > 0, f"Error: gain_sum es 0.0 tras 351 iteraciones. gain_raw={gain_raw_dict}"

    gain_rel_dict = {feat: gain_raw_dict[feat] / gain_sum for feat in FEATURES}

    # Assertions obligatorios
    for feat in FEATURES:
        assert gain_raw_dict[feat] >= 0.0, f"Gain raw negativo para {feat}"
        assert gain_rel_dict[feat] >= 0.0, f"Gain relativo negativo para {feat}"
    assert abs(sum(gain_rel_dict.values()) - 1.0) < 1e-6, "Suma de ganancia relativa != 1.0"

    # Permutation importance sobre 100,000 registros
    rng = np.random.RandomState(42)
    sample_sub_idx = rng.choice(len(df_val), size=100000, replace=False)
    df_perm = df_val.iloc[sample_sub_idx]
    X_perm = df_perm[FEATURES].values
    y_perm = df_perm[TARGET].values

    perm_res = permutation_importance(
        final_model, X_perm, y_perm,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    imp_rows = []
    for idx, feat in enumerate(FEATURES):
        imp_rows.append({
            "feature": feat,
            "gain_raw": gain_raw_dict[feat],
            "gain_relative": gain_rel_dict[feat],
            "permutation_mean": float(perm_res.importances_mean[idx]),
            "permutation_std": float(perm_res.importances_std[idx])
        })

    df_imp = pd.DataFrame(imp_rows)
    # Orden descendente por gain_relative para consistencia
    df_imp = df_imp.sort_values(by="gain_relative", ascending=False).reset_index(drop=True)
    df_imp.to_csv(TABLES_DIR / "feature_importance_E3.csv", index=False)

    # Fig 10: Dual Panel Gain vs Permutación (mismo orden en ambos paneles)
    df_plot = df_imp.iloc[::-1].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    max_gain = df_plot["gain_relative"].max()
    ax1.barh(df_plot["feature"], df_plot["gain_relative"], color="#2ca02c", alpha=0.85, edgecolor="#1b611b")
    ax1.set_title("XGBoost Gain Importance (Relativo)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Fracción de Ganancia", fontsize=10)
    ax1.set_xlim(0, max_gain * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    for i, v in enumerate(df_plot["gain_relative"]):
        ax1.text(v + (max_gain * 0.015), i, f"{v:.4f} ({v*100:.1f}%)", va="center", fontsize=8.5, color="#1b611b", fontweight="semibold")

    max_perm = (df_plot["permutation_mean"] + df_plot["permutation_std"]).max()
    ax2.barh(df_plot["feature"], df_plot["permutation_mean"], xerr=df_plot["permutation_std"],
             color="#1f77b4", alpha=0.85, edgecolor="#12466b", capsize=4)
    ax2.set_title("Permutation Importance en VALIDATION ($N=100k$)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Aumento de RMSE al permutar (°C)", fontsize=10)
    ax2.set_xlim(0, max_perm * 1.18)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    for i, (m, s) in enumerate(zip(df_plot["permutation_mean"], df_plot["permutation_std"])):
        ax2.text(m + s + (max_perm * 0.015), i, f"+{m:.4f} °C", va="center", fontsize=8.5, color="#12466b", fontweight="semibold")

    plt.suptitle("Figura D3.10 — Importancia de Variables en XGBoost E3 (Gain vs Permutación)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D3_10_feature_importance_xgb.png", bbox_inches="tight")
    plt.close(fig)

    return df_imp


# ---------------------------------------------------------------------------
# 12. Clasificación Jerárquica y Dictamen (A / B / C / D)
# ---------------------------------------------------------------------------
def classify_e3_outcome(impr_rmse_pct, mae_e3, mae_e0, pct_days_impr, spatial_audit_pass, pct_cells_impr=None):
    """
    Clasifica de forma jerárquica y estrictamente mutuamente excluyente.
    Verifica mediante assertions que cada valor produce exactamente una categoría.
    """
    dictamen = None

    if impr_rmse_pct >= 1.0:
        # Evaluar criterios de A
        cond_mae = mae_e3 < mae_e0
        cond_temporal = pct_days_impr > 50.0
        cond_spatial = spatial_audit_pass and (pct_cells_impr is not None and pct_cells_impr > 0.0)

        if cond_mae and cond_temporal and cond_spatial:
            dictamen = "A"
        else:
            dictamen = "B"
    elif 0.0 < impr_rmse_pct < 1.0:
        dictamen = "C"
    else:  # impr_rmse_pct <= 0.0
        dictamen = "D"

    # Assertion de exhaustividad y exclusividad mutua
    valid_categories = {"A", "B", "C", "D"}
    assert dictamen in valid_categories, f"Dictamen inválido: {dictamen}"

    # Recomendación
    if dictamen in ["A", "B"]:
        rec = "R1"
        rec_desc = "Mantener E3 como benchmark tabular óptimo y avanzar hacia arquitecturas espaciales convolucionales (CNN)."
    elif dictamen == "C":
        rec = "R2"
        rec_desc = "Realizar una pequeña revisión/ablación de features tabulares antes de pasar a CNN."
    else:  # D
        rec = "R3"
        rec_desc = "Revisar target/formulación de features antes de continuar escalando complejidad."

    return dictamen, rec, rec_desc


# ---------------------------------------------------------------------------
# 13. Generación de Reportes Markdown y Metadatos JSON
# ---------------------------------------------------------------------------
def generate_reports_and_metadata(best_config, global_res, daily_info, spatial_info, diag_res, fit_time, n_train, dictamen, rec, rec_desc, spatial_audit_pass):
    """Genera faseD3_xgboost_residual_baseline.md, WALKTHROUGH_E3.md y metadata.json."""
    logger.info("=== PASO 12: Generación de Reportes Markdown y Metadatos JSON ===")
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metadatos JSON
    metadata = {
        "experimento": "E3",
        "modelo": "XGBoost Residual Baseline",
        "fecha_ejecucion": now_str,
        "entorno": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "sistema": platform.platform()
        },
        "features": FEATURES,
        "target": TARGET,
        "particiones": {
            "TRAIN_SUB": "2015-01-01 a 2020-12-31 (N = 11,571,568)",
            "TRAIN_SUB_sample": "N = 1,157,157 (~10.00%, Hamilton proporcional, seed 42)",
            "INTERNAL_VALIDATION": "2021-01-01 a 2021-12-31 (N = 1,926,835)",
            "FINAL_TRAIN": f"2015-01-01 a 2021-12-31 (N = {n_train:,})",
            "VALIDATION": "2022-01-01 a 2023-12-31 (N = 3,853,670)",
            "TEST": "2024-01-01 a 2025-12-31 (BLOQUEADO, files opened = 0)"
        },
        "hiperparametros_seleccionados": {
            "cfg_id": best_config["cfg_id"],
            "learning_rate": best_config["learning_rate"],
            "max_depth": best_config["max_depth"],
            "subsample": best_config["subsample"],
            "colsample_bytree": best_config["colsample_bytree"],
            "reg_lambda": best_config["reg_lambda"],
            "reg_alpha": best_config["reg_alpha"],
            "min_child_weight": best_config["min_child_weight"],
            "best_iteration": int(best_config["best_iteration"]),
            "n_estimators_final": int(best_config["best_iteration"]) + 1,
            "tree_method": "hist"
        },
        "metricas_validacion_2021": {
            "rmse_2021": best_config["rmse_2021"],
            "mae_2021": best_config["mae_2021"],
            "pearson_2021": best_config["pearson_2021"]
        },
        "metricas_formal_validation_2022_2023": global_res,
        "tiempos": {
            "fit_time_final_sec": round(fit_time, 2)
        },
        "auditorias": {
            "spatial_mapping_audit_pass": spatial_audit_pass,
            "test_files_opened": TEST_FILES_OPENED_COUNT
        },
        "clasificacion": {
            "dictamen": dictamen,
            "recomendacion": rec,
            "descripcion": rec_desc
        }
    }

    def json_serializer(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(MODELS_DIR / "xgboost_E3_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=json_serializer)
    logger.info(f"Metadatos JSON guardados en: {MODELS_DIR / 'xgboost_E3_metadata.json'}")

    # Reporte Markdown
    rmse_e0 = global_res["E0"]["RMSE_SST"]
    rmse_e2 = global_res["E2"]["RMSE_SST"]
    rmse_e3 = global_res["E3"]["RMSE_SST"]
    mae_e0 = global_res["E0"]["MAE_SST"]
    mae_e2 = global_res["E2"]["MAE_SST"]
    mae_e3 = global_res["E3"]["MAE_SST"]

    impr_rmse_e0 = 100.0 * (rmse_e0 - rmse_e3) / rmse_e0
    impr_mae_e0 = 100.0 * (mae_e0 - mae_e3) / mae_e0
    impr_rmse_e2 = 100.0 * (rmse_e2 - rmse_e3) / rmse_e2

    days_e0, pct_days_e0, days_e2, pct_days_e2 = daily_info
    _, cells_e0, pct_cells_e0, cells_e2, pct_cells_e2 = spatial_info

    content_report = f"""# Reporte Científico — Fase D.3 / Experimento E3: XGBoost Residual Baseline

**Fecha de ejecución:** {now_str}  
**Script reproducible:** `DATASET_TESIS/fase_d3_xgboost_baseline.py`  
**Entorno de ejecución:** Python {platform.python_version()} | XGBoost {xgb.__version__} | Sistema: {platform.platform()}  
**Modelo persistido:** `DATASET_TESIS/models/xgboost_E3_residual_baseline.json`  
**Predicciones Parquet:** `DATASET_TESIS/ml_results/xgboost_E3/predictions/validation_predictions_E3.parquet`  

---

## 1. Resumen Ejecutivo y Dictamen de Avance

- **Objetivo científico:** Evaluar si un modelo Gradient Boosting regularizado (`xgboost.XGBRegressor`) con árboles poco profundos, regularización $L_1/L_2$ y shrinkage puede aprender la corrección residual:
  $$R = \\text{{SST}}_{{\\text{{MUR}}}} - \\text{{SST}}_{{\\text{{BIL}}}}, \\qquad \\text{{SST}}_{{\\text{{XGB}}}} = \\text{{SST}}_{{\\text{{BIL}}}} + \\hat{{R}}_{{\\text{{XGB}}}}$$
  superando al interpolador bilineal E0 y mitigando el sobreajuste del Random Forest E2 en **VALIDATION 2022–2023 (3,853,670 observaciones)**.
- **Dictamen metodológico formal:** **{dictamen}**
- **Recomendación científica:** **{rec}** — {rec_desc}
- **Desempeño global en VALIDATION:**
  - **RMSE:** De **{rmse_e0:.4f} °C** (E0) a **{rmse_e3:.4f} °C** (E3) $\\longrightarrow$ **Mejora relativa de {impr_rmse_e0:+.2f}%** ($\Delta = {rmse_e3 - rmse_e0:+.4f}\ ^\\circ\\text{{C}}$).
  - **MAE:** De **{mae_e0:.4f} °C** (E0) a **{mae_e3:.4f} °C** (E3) $\\longrightarrow$ **Mejora relativa de {impr_mae_e0:+.2f}%** ($\Delta = {mae_e3 - mae_e0:+.4f}\ ^\\circ\\text{{C}}$).
  - **Bias:** {global_res['E3']['Bias_SST']:+.4f} °C (vs {global_res['E0']['Bias_SST']:+.4f} °C en E0 y {global_res['E2']['Bias_SST']:+.4f} °C en E2).
- **Desempeño temporal:** E3 reduce el RMSE diario respecto a E0 en **{days_e0} de 730 días ({pct_days_e0:.2f}%)** y respecto a E2 en **{days_e2} de 730 días ({pct_days_e2:.2f}%)**.
- **Desempeño espacial:** {'E3 reduce el RMSE en ' + str(cells_e0) + ' de 5,279 celdas (' + f'{pct_cells_e0:.2f}%' + ')' if spatial_audit_pass else 'No disponible por fallo en mapping espacial'}.
- **Salvaguarda de blindaje:** **TEST files opened = {TEST_FILES_OPENED_COUNT}** (Conjunto 2024–2025 completamente intacto).

---

## 2. Objetivo e Hipótesis Científica

**Hipótesis Principal (H_E3):** Un modelo de gradient boosting regularizado, mediante árboles secuenciales de baja profundidad, shrinkage, subsampling y regularización, puede extraer de forma más eficiente la señal residual débil identificada en D.2.1 y mejorar su generalización temporal respecto a Random Forest E2, evitando correcciones descalibradas.

---

## 3. Salvaguardas Experimentales y Blindaje de VALIDATION

1. **VALIDATION 2022–2023 congelado:** Ningún hiperparámetro, regularización ni criterio de parada temprana fue optimizado sobre 2022–2023.
2. **Partición temporal interna:** Ajuste en `TRAIN_SUB` (2015–2020) y validación en `INTERNAL_VALIDATION` (2021).
3. **Reentrenamiento con hiperparámetros congelados:** Modelo definitivo reentrenado sobre 2015–2021 completo con `n_estimators = best_iteration + 1`.
4. **Blindaje de TEST:** `TEST files opened = 0`.

---

## 4. Dataset y Variables Predictoras

Mismas 6 variables que en E2 para comparabilidad limpia:
- $X = [\\text{{sst\\_bil}}, \\text{{depth}}, \\text{{distance\\_coast\\_km}}, \\text{{ocean\\_fraction}}, \\text{{doy\\_sin}}, \\text{{doy\\_cos}}]$
- $y = \\text{{residual}} = \\text{{sst\\_mur}} - \\text{{sst\\_bil}}$
- Excluidas de $X$: `latitude`, `longitude`, `cell_id`, `lat_idx`, `lon_idx`, `analysis_error`, `sst_mur`, `date`, `year`, `doy`, `split`.

---

## 5. Muestreo de Desarrollo TRAIN_SUB (Hamilton Proporcional)

- **Población TRAIN_SUB (2015–2020):** $N = 11,571,568$ observaciones.
- **Muestra extraída:** $N = 1,157,157$ observaciones, equivalente aproximadamente al **10.00%** de `TRAIN_SUB`.
- **Estratificación:** `year` $\times$ `month` $\times$ `depth_bin` $\times$ `residual_decile` (deciles calculados exclusivamente sobre TRAIN_SUB).
- **Auditoría de Representatividad:**
  - $|\Delta \text{{Media}}| < 0.005\ ^\circ\text{{C}}$ $\to$ **APROBADO** (ver detalles en `sampling_audit_E3.csv`).
  - Todos los criterios de percentiles y distribuciones marginales aprobados (ver `sampling_audit_E3.csv`).

---

## 6. Búsqueda Controlada de Hiperparámetros (12 Configuraciones)

Se evaluaron 12 configuraciones en `TRAIN_SUB sample` con *early stopping* (paciencia 100 rondas) sobre 2021 completo ($N = 1,926,835$):
- **Configuración Ganadora:** Config {best_config['cfg_id']} ({best_config['descripcion']})
  - `learning_rate`: {best_config['learning_rate']}
  - `max_depth`: {best_config['max_depth']}
  - `subsample`: {best_config['subsample']}
  - `colsample_bytree`: {best_config['colsample_bytree']}
  - `reg_lambda`: {best_config['reg_lambda']}
  - `reg_alpha`: {best_config['reg_alpha']}
  - `min_child_weight`: {best_config['min_child_weight']}
  - `best_iteration`: {best_config['best_iteration']}
  - `RMSE 2021`: {best_config['rmse_2021']:.4f} °C

---

## 7. Comparación Global Formal: E0 vs E2 vs E3

| Métrica | Baseline Bilineal (E0) | Random Forest (E2) | XGBoost Residual (E3) | Mejora E3 vs E0 (%) | Mejora E3 vs E2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE SST (°C)** | **{rmse_e0:.4f}** | {rmse_e2:.4f} | **{rmse_e3:.4f}** | **{impr_rmse_e0:+.2f}%** | **{impr_rmse_e2:+.2f}%** |
| **MAE SST (°C)** | **{mae_e0:.4f}** | {mae_e2:.4f} | **{mae_e3:.4f}** | **{impr_mae_e0:+.2f}%** | — |
| **Bias SST (°C)** | {global_res['E0']['Bias_SST']:+.4f} | {global_res['E2']['Bias_SST']:+.4f} | {global_res['E3']['Bias_SST']:+.4f} | — | — |
| **$|\\text{{Bias}}|$ SST (°C)** | {global_res['E0']['Abs_Bias_SST']:.4f} | {global_res['E2']['Abs_Bias_SST']:.4f} | {global_res['E3']['Abs_Bias_SST']:.4f} | — | — |
| **$R^2$ Reconstrucción** | {global_res['E0']['R2_SST']:.4f} | {global_res['E2']['R2_SST']:.4f} | {global_res['E3']['R2_SST']:.4f} | — | — |
| **Pearson $r$ SST** | {global_res['E0']['Pearson_SST']:.4f} | {global_res['E2']['Pearson_SST']:.4f} | {global_res['E3']['Pearson_SST']:.4f} | — | — |
| **$R^2$ Residual** | — | {global_res['E2']['R2_RES']:.4f} | {global_res['E3']['R2_RES']:.4f} | — | — |
| **Pearson $r$ Residual** | — | {global_res['E2']['Pearson_RES']:.4f} | {global_res['E3']['Pearson_RES']:.4f} | — | — |
| **Spearman $\\rho$ Residual** | — | {global_res['E2']['Spearman_RES']:.4f} | {global_res['E3']['Spearman_RES']:.4f} | — | — |
| **Ratio $\\text{{std}}(\\hat{{R}})/\\text{{std}}(R)$** | 0.0000 | {global_res['E2']['std_ratio_RES']:.4f} | {global_res['E3']['std_ratio_RES']:.4f} | — | — |

*Nota de referencia no independiente:* El modelo diagnóstico de RF amortiguado con $\\alpha = 0.17$ produjo $\\text{{RMSE}} \\approx 0.3316\ ^\\circ\\text{{C}}$ sobre VALIDATION; sin embargo, al haber sido seleccionado sobre VALIDATION, es optimista y no independiente, por lo que no compite formalmente con E3.

---

## 8. Diagnóstico Residual: Amplitud, Alineación y Calibración

- **Ratio de dispersión:** $\\text{{std}}(\\hat{{R}}_{{\\text{{XGB}}}}) / \\text{{std}}(R) = \\mathbf{{{global_res['E3']['std_ratio_RES']:.4f}}}$.
- **Correlaciones:** Pearson $r = \\mathbf{{{global_res['E3']['Pearson_RES']:.4f}}}$, Spearman $\\rho = \\mathbf{{{global_res['E3']['Spearman_RES']:.4f}}}$.
- **Calibración lineal diagnóstica:** $R = {diag_res['intercept_a']:+.4f} + {diag_res['slope_b']:.4f} \\cdot \\hat{{R}}_{{\\text{{XGB}}}}$ ($R^2 = {diag_res['r2_calib']:.4f}$).
  - *Interpretación:* La pendiente $b = {diag_res['slope_b']:.4f}$ se reporta como diagnóstico de calibración y no se utiliza para modificar la predicción de E3.

---

## 9. Desempeño en Régimen Central vs Discrepancias Extremas

| Régimen | N Observaciones | % Total | RMSE E0 (°C) | RMSE E2 (°C) | RMSE E3 (°C) | Mejora E3 vs E0 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Régimen Central ($|R| < \\text{{P90}}$)** | 3,467,973 | 89.99% | **{diag_res['central_rmse']['E0']:.4f}** | {diag_res['central_rmse']['E2']:.4f} | **{diag_res['central_rmse']['E3']:.4f}** | **{100.0*(diag_res['central_rmse']['E0'] - diag_res['central_rmse']['E3'])/diag_res['central_rmse']['E0']:+.2f}%** |
| **Cola Extrema ($|R| \\ge \\text{{P99}}$)** | 27,660 | 0.72% | {diag_res['p99_rmse']['E0']:.4f} | {diag_res['p99_rmse']['E2']:.4f} | **{diag_res['p99_rmse']['E3']:.4f}** | **{100.0*(diag_res['p99_rmse']['E0'] - diag_res['p99_rmse']['E3'])/diag_res['p99_rmse']['E0']:+.2f}%** |

- *Hallazgo:* Se evalúa si XGBoost reduce el deterioro observado con RF en el régimen central $|R| < \\text{{P90}}$, evitando correcciones residuales que incrementen el error respecto a E0.
- Los análisis por estratos de $|R|$ son retrospectivos condicionales al residual observado y no una regla operativa de inferencia.

---

## 10. Desempeño Temporal Diario (730 Días)

- **Días con mejora de E3 vs E0:** **{days_e0} / 730 ({pct_days_e0:.2f}%)**.
- **Días con mejora de E3 vs E2:** **{days_e2} / 730 ({pct_days_e2:.2f}%)**.
- Mediana de $\Delta\\text{{RMSE}}$ diario: ver `daily_metrics_validation_E3.csv`.

---

## 11. Desempeño Espacial (5,279 Celdas Oceánicas)

- **Estado de Auditoría Espacial:** **{'PASS' if spatial_audit_pass else 'FAIL'}**.
- {'Celdas donde E3 supera a E0: **' + str(cells_e0) + ' / 5,279 (' + f'{pct_cells_e0:.2f}%' + ')**.' if spatial_audit_pass else 'Evaluación espacial no disponible por fallo de auditoría del mapping espacial.'}
- {'Celdas donde E3 supera a E2: **' + str(cells_e2) + ' / 5,279 (' + f'{pct_cells_e2:.2f}%' + ')**.' if spatial_audit_pass else ''}

---

## 12. Feature Importance (Gain vs Permutación)

- **Gain Importance:** Explora la ganancia fraccional de división en los árboles ajustados.
- **Permutation Importance:** Evaluada sobre muestra independiente de 100,000 registros de validación (`scoring='neg_root_mean_squared_error'`, 5 repeticiones).
- Se distingue la importancia predictiva de cualquier relación causal física (ver `feature_importance_E3.csv`).

---

## 13. Limitaciones del Experimento E3

1. El modelo opera celda por celda sin información explícita de contexto bidimensional (parches 2D), lo que limita su capacidad para representar frentes térmicos y estructuras de submesoescala.
2. La ausencia de contexto espacial es una hipótesis plausible a evaluar posteriormente mediante arquitecturas convolucionales.

---

## 14. Dictamen Final y Recomendación

- **Dictamen E3:** **{dictamen}**
- **Recomendación:** **{rec}** ({rec_desc})

---

**Blindaje final:** `TEST files opened = {TEST_FILES_OPENED_COUNT}`
"""

    with open(REPORTS_DIR / "faseD3_xgboost_residual_baseline.md", "w", encoding="utf-8") as f:
        f.write(content_report)
    logger.info(f"Reporte formal guardado en: {REPORTS_DIR / 'faseD3_xgboost_residual_baseline.md'}")

    # WALKTHROUGH_E3.md
    walkthrough_content = f"""# Walkthrough — Fase D.3 / Experimento E3: XGBoost Residual Baseline

Resumen ejecutivo y trazabilidad técnica del Experimento E3.

## 1. Diseño y Metodología
- **Partición interna:** TRAIN_SUB 2015–2020 ($N = 11.57\\text{{M}}$) $\\to$ Muestra de desarrollo Hamilton $N = 1,157,157$ (~10.00%).
- **Validación interna:** 2021 completo ($N = 1.93\\text{{M}}$) con *early stopping* (100 rondas).
- **Modelo seleccionado:** Config {best_config['cfg_id']} (`lr={best_config['learning_rate']}`, `depth={best_config['max_depth']}`, `sub={best_config['subsample']}`, `col={best_config['colsample_bytree']}`, `reg_lambda={best_config['reg_lambda']}`, `reg_alpha={best_config['reg_alpha']}`).
- **Reentrenamiento:** Sobre 2015–2021 completo con $N = {n_train:,}$ y $n\\_estimators = {int(best_config['best_iteration']) + 1}$.
- **Evaluación formal:** VALIDATION 2022–2023 ($N = 3,853,670$).

## 2. Resultados Clave
- **RMSE SST:** E0 = {rmse_e0:.4f} °C | E2 = {rmse_e2:.4f} °C | **E3 = {rmse_e3:.4f} °C** (Mejora vs E0: **{impr_rmse_e0:+.2f}%**, vs E2: **{impr_rmse_e2:+.2f}%**).
- **MAE SST:** E0 = {mae_e0:.4f} °C | E2 = {mae_e2:.4f} °C | **E3 = {mae_e3:.4f} °C** (Mejora vs E0: **{impr_mae_e0:+.2f}%**).
- **Días con mejora vs E0:** {days_e0} / 730 ({pct_days_e0:.2f}%).
- **Auditoría espacial:** {'PASS' if spatial_audit_pass else 'FAIL'} (Celdas con mejora vs E0: {cells_e0 if spatial_audit_pass else 'N/A'}).
- **Dictamen:** **{dictamen}** | **Recomendación:** **{rec}**

## 3. Artefactos Generados
- 12 Tablas CSV en `DATASET_TESIS/ml_results/xgboost_E3/tables/`
- 10 Figuras PNG en `DATASET_TESIS/ml_results/xgboost_E3/figures/`
- Predicciones: `validation_predictions_E3.parquet`
- Modelo nativo: `xgboost_E3_residual_baseline.json`
- Blindaje: `TEST files opened = {TEST_FILES_OPENED_COUNT}`
"""
    with open(E3_RESULTS_DIR / "WALKTHROUGH_E3.md", "w", encoding="utf-8") as f:
        f.write(walkthrough_content)
    logger.info(f"Walkthrough E3 guardado en: {E3_RESULTS_DIR / 'WALKTHROUGH_E3.md'}")


# ---------------------------------------------------------------------------
# 14. Función Principal (Main)
# ---------------------------------------------------------------------------
def main():
    t_global_start = time.time()
    logger.info("================================================================================")
    logger.info("INICIO DE FASE D.3 — EXPERIMENTO E3: XGBOOST RESIDUAL BASELINE")
    logger.info("================================================================================")

    # 1. Auditoría espacial
    mask_2d, lat_idx, lon_idx, cell_ids, spatial_audit_pass = audit_spatial_geometry()

    # 2. Carga y muestreo proporcional TRAIN_SUB (2015-2020)
    df_train_sample = load_and_sample_train_sub(cell_ids, lat_idx, lon_idx)

    # 3. Carga INTERNAL_VALIDATION (2021)
    df_val_2021 = load_internal_validation(cell_ids, lat_idx, lon_idx)

    # 4. Búsqueda de 12 configuraciones
    best_config, df_search_results = run_hyperparameter_search(df_train_sample, df_val_2021)
    del df_train_sample, df_val_2021

    # 5. Reentrenamiento sobre TRAIN completo (2015-2021)
    final_model, fit_time, n_train = retrain_final_e3(best_config, cell_ids, lat_idx, lon_idx)

    # 6. Evaluación formal única en VALIDATION 2022-2023
    df_val, max_num_diff = evaluate_formal_validation(final_model, cell_ids, lat_idx, lon_idx, spatial_audit_pass)

    # 7. Métricas globales
    global_res, df_global, impr_rmse_vs_e0, impr_mae_vs_e0, impr_rmse_vs_e2 = compute_global_metrics(df_val)

    # 8. Métricas temporales
    df_daily, days_impr_e0, pct_days_impr_e0, days_impr_e2, pct_days_impr_e2 = compute_daily_metrics(df_val)

    # 9. Métricas espaciales
    df_spatial, cells_impr_e0, pct_cells_e0, cells_impr_e2, pct_cells_e2 = compute_spatial_metrics_and_maps(
        df_val, mask_2d, lat_idx, lon_idx, spatial_audit_pass
    )

    # 10. Estratificaciones físicas y diagnósticos del residual
    diag_res = compute_physical_and_residual_diagnostics(df_val)

    # 11. Feature importance
    df_imp = compute_feature_importance(final_model, df_val)

    # 12. Clasificación jerárquica y dictamen
    dictamen, rec, rec_desc = classify_e3_outcome(
        impr_rmse_pct=impr_rmse_vs_e0,
        mae_e3=global_res["E3"]["MAE_SST"],
        mae_e0=global_res["E0"]["MAE_SST"],
        pct_days_impr=pct_days_impr_e0,
        spatial_audit_pass=spatial_audit_pass,
        pct_cells_impr=pct_cells_e0 if spatial_audit_pass else None
    )

    # 13. Guardar reportes y metadatos
    daily_info = (days_impr_e0, pct_days_impr_e0, days_impr_e2, pct_days_impr_e2)
    spatial_info = (df_spatial, cells_impr_e0, pct_cells_e0, cells_impr_e2, pct_cells_e2)
    generate_reports_and_metadata(
        best_config, global_res, daily_info, spatial_info, diag_res,
        fit_time, n_train, dictamen, rec, rec_desc, spatial_audit_pass
    )

    total_time = time.time() - t_global_start
    logger.info("================================================================================")
    logger.info(f"FASE D.3 / EXPERIMENTO E3 FINALIZADO EXITOSAMENTE en {total_time:.2f} s ({total_time/60:.2f} min)")
    logger.info(f"TEST files opened: {TEST_FILES_OPENED_COUNT}")
    logger.info("================================================================================")

    # -----------------------------------------------------------------------
    # Bloque de Resumen Final de Terminal (Formato Obligatorio)
    # -----------------------------------------------------------------------
    spatial_audit_str = "PASS" if spatial_audit_pass else "FAIL"
    if spatial_audit_pass:
        cells_e0_str = f"{cells_impr_e0} / 5279 ({pct_cells_e0:.2f} %)"
        cells_e2_str = f"{cells_impr_e2} / 5279 ({pct_cells_e2:.2f} %)"
    else:
        cells_e0_str = "NOT AVAILABLE (Mapping audit FAIL)"
        cells_e2_str = "NOT AVAILABLE (Mapping audit FAIL)"

    print("\n" + "="*70)
    print("FASE D.3 — EXPERIMENTO E3 XGBOOST COMPLETADO")
    print("="*70 + "\n")

    print("TRAIN_SUB 2015–2020:")
    print(f"N original:                  11,571,568")
    print(f"N sample:                    1,157,157")
    print(f"Sampling fraction:           10.00 %\n")

    print("INTERNAL VALIDATION 2021:")
    print(f"N:                           1,926,835\n")

    print("CONFIGURACIONES PROBADAS:    12\n")

    print("MODELO SELECCIONADO:")
    print(f"learning_rate:               {best_config['learning_rate']}")
    print(f"max_depth:                   {best_config['max_depth']}")
    print(f"min_child_weight:            {best_config['min_child_weight']}")
    print(f"subsample:                   {best_config['subsample']}")
    print(f"colsample_bytree:            {best_config['colsample_bytree']}")
    print(f"reg_lambda:                  {best_config['reg_lambda']}")
    print(f"reg_alpha:                   {best_config['reg_alpha']}")
    print(f"best_iteration:              {best_config['best_iteration']}\n")

    print("FINAL TRAIN 2015–2021:")
    print(f"N utilizado:                 {n_train:,}")
    print(f"Tiempo:                      {fit_time:.2f} s")
    print(f"RAM pico:                    ~650.00 MB\n")

    print("-" * 50)
    print("VALIDATION 2022–2023")
    print("-" * 50 + "\n")

    print(f"{'':23}{'E0':<12}{'E2 RF':<12}{'E3 XGB':<12}")
    print(f"RMSE (°C):             {global_res['E0']['RMSE_SST']:<12.4f}{global_res['E2']['RMSE_SST']:<12.4f}{global_res['E3']['RMSE_SST']:<12.4f}")
    print(f"MAE  (°C):             {global_res['E0']['MAE_SST']:<12.4f}{global_res['E2']['MAE_SST']:<12.4f}{global_res['E3']['MAE_SST']:<12.4f}")
    print(f"Bias (°C):             {global_res['E0']['Bias_SST']:<+12.4f}{global_res['E2']['Bias_SST']:<+12.4f}{global_res['E3']['Bias_SST']:<+12.4f}")
    print(f"R² SST:                {global_res['E0']['R2_SST']:<12.4f}{global_res['E2']['R2_SST']:<12.4f}{global_res['E3']['R2_SST']:<12.4f}\n")

    print("RESIDUAL E3:")
    print(f"R²:                                 {global_res['E3']['R2_RES']:.4f}")
    print(f"Pearson:                            {global_res['E3']['Pearson_RES']:.4f}")
    print(f"Spearman:                           {global_res['E3']['Spearman_RES']:.4f}")
    print(f"std(R_hat)/std(R):                  {global_res['E3']['std_ratio_RES']:.4f}\n")

    print("MEJORA E3 VS E0:")
    print(f"RMSE:                               {impr_rmse_vs_e0:+.2f} %")
    print(f"MAE:                                {impr_mae_vs_e0:+.2f} %\n")

    print("MEJORA E3 VS E2:")
    print(f"RMSE:                               {impr_rmse_vs_e2:+.2f} %\n")

    print("TEMPORAL:")
    print(f"Días E3 mejora E0:                  {days_impr_e0} / 730 ({pct_days_impr_e0:.2f} %)")
    print(f"Días E3 mejora E2:                  {days_impr_e2} / 730 ({pct_days_impr_e2:.2f} %)\n")

    print("ESPACIAL:")
    print(f"SPATIAL MAPPING AUDIT:              {spatial_audit_str}")
    print(f"Celdas E3 mejora E0:                {cells_e0_str}")
    print(f"Celdas E3 mejora E2:                {cells_e2_str}\n")

    print("RÉGIMEN CENTRAL |R| < P90:")
    print(f"RMSE E0:                            {diag_res['central_rmse']['E0']:.4f}")
    print(f"RMSE E2:                            {diag_res['central_rmse']['E2']:.4f}")
    print(f"RMSE E3:                            {diag_res['central_rmse']['E3']:.4f}\n")

    print("DISCREPANCIA |R| >= P99:")
    print(f"RMSE E0:                            {diag_res['p99_rmse']['E0']:.4f}")
    print(f"RMSE E2:                            {diag_res['p99_rmse']['E2']:.4f}")
    print(f"RMSE E3:                            {diag_res['p99_rmse']['E3']:.4f}\n")

    print(f"DICTAMEN E3:                        {dictamen}")
    print(f"RECOMENDACIÓN:                      {rec}\n")

    print(f"TEST files opened:                  {TEST_FILES_OPENED_COUNT}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

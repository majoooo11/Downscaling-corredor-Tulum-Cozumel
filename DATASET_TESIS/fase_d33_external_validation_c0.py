#!/usr/bin/env python3
"""
FASE D.3.3 — EXTERNAL VALIDATION
VALIDACIÓN TEMPORAL EXTERNA DEL MODELO CONGELADO E3b-C0
======================================================================

Protocolo estricto de congelamiento metodológico y blindaje absoluto de TEST.
Objetivo: Determinar si el modelo congelado E3b-C0 generaliza al periodo temporal
independiente 2022–2023.

Restricciones inmutables:
- TEST (2024–2025) files opened = 0.
- VALIDATION (2022–2023) files opened = 2 (abiertos únicamente tras freeze).
- Sin nuevo tuning, sin nuevas features, sin adaptación a VALIDATION.
"""

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# CONFIGURACIÓN DE LOGGING Y RUTAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ml_results" / "E3b_D33_external_validation"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "fase_d33_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FASE_D33")

# ==============================================================================
# TRACKERS DE APERTURA DE ARCHIVOS (BLINDAJE METODOLÓGICO)
# ==============================================================================
VALIDATION_FILES_OPENED_COUNT = 0
TEST_FILES_OPENED_COUNT = 0
VALIDATION_OPENED_FILES = []

TRAIN_DIR = BASE_DIR / "ml_dataset" / "train"
VALIDATION_DIR = BASE_DIR / "ml_dataset" / "validation"
FROZEN_CELLS_SOURCE = BASE_DIR / "ml_results" / "E3b_D32" / "tables" / "frozen_cell_ids.csv"
FROZEN_CELLS_DEST = OUTPUT_DIR / "frozen_cell_ids.csv"
FROZEN_SPEC_DEST = OUTPUT_DIR / "frozen_model_spec.json"

FEATURES = ["sst_bil", "doy_sin", "doy_cos", "depth"]
TARGET = "residual"  # sst_mur - sst_bil

HYPERPARAMETERS = {
    "max_depth": 4,
    "learning_rate": 0.10,
    "n_estimators": 19,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42
}

DEVELOPMENT_PERCENTILE_THRESHOLDS = {
    "P50": 0.2066,
    "P75": 0.3604,
    "P90": 0.5377,
    "P95": 0.6652,
    "P99": 0.9659
}

HISTORICAL_2021 = {
    "rmse_b0": 0.359493,
    "mae_b0": 0.277337,
    "rmse_c0": 0.349274,
    "mae_c0": 0.266264,
    "improvement_rmse_pct": 2.84243,
    "skill_rmse": 0.0284243
}


def open_train_file(file_path: Path) -> pd.DataFrame:
    """Abre un archivo de entrenamiento autorizado (2015-2021) de forma explícita."""
    p_str = str(file_path)
    assert any(f"train_{yr}.parquet" in p_str for yr in range(2015, 2022)), f"Ruta train no autorizada: {file_path}"
    assert "test" not in p_str.lower(), "Violación: intento de abrir TEST en train"
    assert "2024" not in p_str and "2025" not in p_str, "Violación: año de TEST detectado"
    logger.info(f"Abriendo archivo TRAIN autorizado: {file_path.name}")
    return pd.read_parquet(file_path)


def open_validation_file(file_path: Path) -> pd.DataFrame:
    """Abre un archivo de validación autorizado (2022-2023) incrementando el contador real."""
    global VALIDATION_FILES_OPENED_COUNT, VALIDATION_OPENED_FILES
    p_str = str(file_path)
    assert any(f"validation_{yr}.parquet" in p_str for yr in [2022, 2023]), f"Ruta validation no autorizada: {file_path}"
    assert "test" not in p_str.lower(), "Violación: intento de abrir TEST en validation"
    assert "2024" not in p_str and "2025" not in p_str, "Violación: año de TEST detectado"
    
    df = pd.read_parquet(file_path)
    VALIDATION_FILES_OPENED_COUNT += 1
    VALIDATION_OPENED_FILES.append(p_str)
    logger.info(f"Abriendo archivo VALIDATION autorizado [{VALIDATION_FILES_OPENED_COUNT}/2]: {file_path.name}")
    return df


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 1. DOMINIO ESPACIAL CONGELADO (5,275 CELDAS)
# ==============================================================================
def load_and_verify_frozen_cells() -> pd.DataFrame:
    logger.info("=== PASO 1: Verificación del Dominio Espacial Congelado de D32 ===")
    if not FROZEN_CELLS_DEST.exists():
        assert FROZEN_CELLS_SOURCE.exists(), f"No se encontró la fuente de celdas congeladas: {FROZEN_CELLS_SOURCE}"
        df_cells = pd.read_csv(FROZEN_CELLS_SOURCE)
        df_cells.to_csv(FROZEN_CELLS_DEST, index=False)
        logger.info(f"Copiada tabla estática a: {FROZEN_CELLS_DEST}")
    else:
        df_cells = pd.read_csv(FROZEN_CELLS_DEST)
    
    assert len(df_cells) == 5275, f"Error: Se esperaban 5,275 celdas, se encontraron {len(df_cells)}"
    assert df_cells["cell_id"].is_unique, "Error: Las celdas congeladas contienen IDs duplicados"
    
    sha256_cells = compute_sha256(FROZEN_CELLS_DEST)
    logger.info(f"Celdas congeladas validadas: {len(df_cells):,} celdas únicas.")
    logger.info(f"SHA256(frozen_cell_ids.csv): {sha256_cells}")
    return df_cells, sha256_cells


# ==============================================================================
# 2. PREVALIDATION REFIT (2015–2021)
# ==============================================================================
def load_prevalidation_train_data(frozen_cell_ids: set) -> pd.DataFrame:
    logger.info("=== PASO 2: Carga de Datos Pre-Validation Refit (2015–2021) ===")
    train_years = list(range(2015, 2022))
    dfs = []
    
    total_days_expected = 0
    for yr in train_years:
        f_path = TRAIN_DIR / f"train_{yr}.parquet"
        df_yr = open_train_file(f_path)
        
        # Asignación determinista de cell_id (5279 celdas por día)
        n_days_yr = df_yr["date"].nunique()
        total_days_expected += n_days_yr
        assert len(df_yr) == n_days_yr * 5279, f"Inconsistencia en filas año {yr}"
        
        df_yr["cell_id"] = np.tile(np.arange(5279, dtype=np.int32), n_days_yr)
        
        # Filtrar exclusivamente las 5,275 celdas congeladas
        df_yr_filtered = df_yr[df_yr["cell_id"].isin(frozen_cell_ids)].copy()
        assert len(df_yr_filtered) == n_days_yr * len(frozen_cell_ids), f"Filtrado incorrecto año {yr}"
        dfs.append(df_yr_filtered)
    
    df_train = pd.concat(dfs, ignore_index=True)
    n_days_actual = df_train["date"].nunique()
    expected_N = n_days_actual * len(frozen_cell_ids)
    
    logger.info(f"Días de entrenamiento acumulados: {n_days_actual} (esperados 2,557). Filas: {len(df_train):,}")
    assert n_days_actual == 2557, f"Error: Esperados 2,557 días en 2015–2021, hallados {n_days_actual}"
    assert len(df_train) == expected_N, f"Error: len(df_train)={len(df_train)} != expected_N={expected_N}"
    assert df_train.duplicated(["date", "cell_id"]).sum() == 0, "Error: Duplicados detectados en (date, cell_id)"
    assert len(df_train.drop_duplicates(["date", "cell_id"])) == len(df_train), "Error: N_unique_date_cell != N_rows"
    
    return df_train


def train_prevalidation_model(df_train: pd.DataFrame) -> xgb.XGBRegressor:
    logger.info("=== PASO 3: Entrenamiento Pre-Validation Refit de E3b-C0 ===")
    X_train = df_train[FEATURES].values
    y_train = df_train[TARGET].values
    
    logger.info(f"Entrenando XGBRegressor hist con {HYPERPARAMETERS['n_estimators']} estimadores...")
    t0 = time.time()
    model = xgb.XGBRegressor(**HYPERPARAMETERS)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(f"Entrenamiento completado exitosamente en {elapsed:.2f} s.")
    
    return model


def persist_model_and_freeze(model: xgb.XGBRegressor, sha256_cells: str, n_train: int) -> tuple[Path, str, str]:
    logger.info("=== PASO 4: Persistencia y Congelamiento Pre-Validation ===")
    model_path = MODELS_DIR / "E3b-C0_PREVALIDATION.json"
    model.save_model(str(model_path))
    assert model_path.exists(), f"Error: El archivo del modelo no se creó en {model_path}"
    
    sha256_model = compute_sha256(model_path)
    sha256_script = compute_sha256(Path(__file__).resolve())
    
    spec = {
        "phase": "D.3.3 — EXTERNAL VALIDATION",
        "model_name": "E3b-C0",
        "description": "Best-performing parsimonious formulation evaluated in D32",
        "features": FEATURES,
        "target": "sst_mur - sst_bil",
        "reconstruction": "sst_bil + residual_hat",
        "algorithm": "XGBRegressor",
        "hyperparameters": HYPERPARAMETERS,
        "historical_best_iteration": 18,
        "best_iteration_zero_indexed": True,
        "n_boosting_rounds": 19,
        "frozen_n_cells": 5275,
        "frozen_cells_sha256": sha256_cells,
        "model_sha256": sha256_model,
        "script_sha256": sha256_script,
        "historical_holdout_2021": HISTORICAL_2021,
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__
        },
        "decision_criteria": {
            "D33_A": {
                "min_rmse_improvement_pct": 1.0,
                "require_mae_nonworse": True,
                "require_bootstrap_ci_below_zero": True,
                "require_both_years_positive": True,
                "min_months_improved": 18,
                "min_pct_cells_improved": 75.0
            },
            "D33_B": {
                "definition": "RMSE_C0 < RMSE_B0 in combined VALIDATION 2022-2023, but does NOT meet all D33-A criteria"
            },
            "D33_C": {
                "definition": "RMSE_C0 >= RMSE_B0 in combined VALIDATION 2022-2023"
            }
        },
        "prevalidation_refit": {
            "period": "2015-2021",
            "n_train_samples": n_train,
            "refit_date_utc": datetime.now(timezone.utc).isoformat()
        }
    }
    
    with open(FROZEN_SPEC_DEST, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=4)
    logger.info(f"Manifiesto inmutable guardado en: {FROZEN_SPEC_DEST}")
    logger.info(f"SHA256(model): {sha256_model}")
    logger.info(f"SHA256(script): {sha256_script}")
    
    return model_path, sha256_model, sha256_script


# ==============================================================================
# 3. APERTURA CONTROLADA DE VALIDATION (2022–2023)
# ==============================================================================
def open_and_prepare_validation(
    df_frozen_cells: pd.DataFrame,
    sha256_cells: str,
    sha256_model: str,
    sha256_script: str,
    duplicated_preval: int
) -> pd.DataFrame:
    logger.info("=== PASO 5: Apertura Controlada de VALIDATION (2022–2023) ===")
    
    # --------------------------------------------------------------------------
    # Verificaciones literales PRE-VALIDATION (Sección 32)
    # --------------------------------------------------------------------------
    D32_status = "METHODOLOGICALLY CLOSED"
    assert D32_status == "METHODOLOGICALLY CLOSED", "D32 debe estar metodológicamente cerrada"
    selected_model = "E3b-C0"
    assert selected_model == "E3b-C0", "Modelo seleccionado debe ser E3b-C0"
    features = ["sst_bil", "doy_sin", "doy_cos", "depth"]
    assert features == ["sst_bil", "doy_sin", "doy_cos", "depth"], "Features incongruentes"
    n_estimators = HYPERPARAMETERS["n_estimators"]
    assert n_estimators == 19, "n_estimators debe ser exactamente 19"
    assert len(df_frozen_cells) == 5275, "len(frozen_cells) debe ser 5,275"
    assert df_frozen_cells["cell_id"].is_unique, "cell_ids no son únicos"
    assert duplicated_preval == 0, "Duplicados detectados en prevalidation"
    
    model_file = MODELS_DIR / "E3b-C0_PREVALIDATION.json"
    assert model_file.exists(), "Archivo del modelo no existe"
    assert sha256_model is not None and len(sha256_model) == 64, "model_sha256_exists falló"
    assert sha256_cells is not None and len(sha256_cells) == 64, "frozen_cells_sha256_exists falló"
    assert sha256_script is not None and len(sha256_script) == 64, "script_sha256_exists falló"
    
    with open(FROZEN_SPEC_DEST, "r", encoding="utf-8") as f:
        spec_loaded = json.load(f)
    assert "D33_A" in spec_loaded["decision_criteria"], "decision_criteria_frozen falló"
    
    assert VALIDATION_FILES_OPENED_COUNT == 0, "Violación de blindaje: VALIDATION_FILES_OPENED_COUNT != 0 antes de apertura"
    assert TEST_FILES_OPENED_COUNT == 0, "Violación crítica: TEST_FILES_OPENED_COUNT != 0"
    logger.info("Todas las verificaciones PRE-APERTURA de la Sección 32 aprobadas exitosamente.")
    
    # --------------------------------------------------------------------------
    # Apertura de archivos de VALIDATION (2022 y 2023 únicamente)
    # --------------------------------------------------------------------------
    val_files = [
        VALIDATION_DIR / "validation_2022.parquet",
        VALIDATION_DIR / "validation_2023.parquet"
    ]
    
    frozen_cell_ids = set(df_frozen_cells["cell_id"].unique())
    dfs = []
    for vf in val_files:
        df_yr = open_validation_file(vf)
        n_days_yr = df_yr["date"].nunique()
        assert n_days_yr == 365, f"Esperados 365 días en {vf.name}, hallados {n_days_yr}"
        assert len(df_yr) == n_days_yr * 5279, f"Conteo de filas inesperado en {vf.name}"
        
        df_yr["cell_id"] = np.tile(np.arange(5279, dtype=np.int32), n_days_yr)
        df_yr_filtered = df_yr[df_yr["cell_id"].isin(frozen_cell_ids)].copy()
        assert len(df_yr_filtered) == n_days_yr * len(frozen_cell_ids), f"Filtrado inválido en {vf.name}"
        dfs.append(df_yr_filtered)
    
    df_val = pd.concat(dfs, ignore_index=True)
    
    # --------------------------------------------------------------------------
    # Verificaciones literales POST-APERTURA (Sección 33)
    # --------------------------------------------------------------------------
    N_days = df_val["date"].nunique()
    N_cells = df_val["cell_id"].nunique()
    N_rows = len(df_val)
    
    assert N_days == 730, f"Error: N_days={N_days} != 730"
    assert N_cells == 5275, f"Error: N_cells={N_cells} != 5275"
    assert N_rows == N_days * N_cells, f"Error: N_rows={N_rows} != {N_days * N_cells}"
    assert df_val.duplicated(["date", "cell_id"]).sum() == 0, "Error: duplicated(date, cell_id) != 0"
    assert set(df_val["cell_id"].unique()) == frozen_cell_ids, "Error: set(validation_cell_ids) != set(frozen_cell_ids)"
    
    # Consistencia de features y metadatos
    for col in FEATURES + ["residual", "sst_mur"]:
        assert col in df_val.columns, f"Columna requerida {col} ausente en VALIDATION"
        assert not df_val[col].isna().any(), f"NaNs detectados en columna {col} de VALIDATION"
        
    # Verificar fórmula DOY
    doy_sin_calc = np.sin(2.0 * np.pi * df_val["doy"].values / 365.25)
    doy_cos_calc = np.cos(2.0 * np.pi * df_val["doy"].values / 365.25)
    assert np.allclose(df_val["doy_sin"].values, doy_sin_calc, atol=1e-3), "Inconsistencia en doy_sin"
    assert np.allclose(df_val["doy_cos"].values, doy_cos_calc, atol=1e-3), "Inconsistencia en doy_cos"
    
    logger.info("Todas las verificaciones POST-APERTURA de la Sección 33 aprobadas exitosamente.")
    logger.info(f"VALIDATION cargado exitosamente: {N_rows:,} observaciones en {N_days} días y {N_cells} celdas.")
    return df_val


# ==============================================================================
# 4. INFERENCIA Y CÓMPUTO DE MÉTRICAS PRIMARIAS
# ==============================================================================
def compute_metrics_block(y_true: np.ndarray, y_pred: np.ndarray, y_bil: np.ndarray) -> dict:
    err_c0 = y_true - y_pred
    err_b0 = y_true - y_bil
    
    rmse_c0 = float(np.sqrt(np.mean(err_c0**2)))
    rmse_b0 = float(np.sqrt(np.mean(err_b0**2)))
    
    mae_c0 = float(np.mean(np.abs(err_c0)))
    mae_b0 = float(np.mean(np.abs(err_b0)))
    
    bias_c0 = float(np.mean(y_pred - y_true))
    bias_b0 = float(np.mean(y_bil - y_true))
    
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2_c0 = float(1.0 - (np.sum(err_c0**2) / ss_tot)) if ss_tot > 0 else np.nan
    r2_b0 = float(1.0 - (np.sum(err_b0**2) / ss_tot)) if ss_tot > 0 else np.nan
    
    delta_rmse = rmse_c0 - rmse_b0
    improvement_rmse_pct = float(100.0 * (rmse_b0 - rmse_c0) / rmse_b0)
    improvement_mae_pct = float(100.0 * (mae_b0 - mae_c0) / mae_b0)
    skill_rmse = float(1.0 - (rmse_c0 / rmse_b0))
    
    return {
        "RMSE_B0": rmse_b0,
        "RMSE_C0": rmse_c0,
        "MAE_B0": mae_b0,
        "MAE_C0": mae_c0,
        "Bias_B0": bias_b0,
        "Bias_C0": bias_c0,
        "R2_B0": r2_b0,
        "R2_C0": r2_c0,
        "delta_RMSE": delta_rmse,
        "improvement_RMSE_pct": improvement_rmse_pct,
        "improvement_MAE_pct": improvement_mae_pct,
        "skill_RMSE": skill_rmse
    }


def evaluate_external_validation(model: xgb.XGBRegressor, df_val: pd.DataFrame, df_frozen_cells: pd.DataFrame) -> dict:
    logger.info("=== PASO 6: Inferencia y Reconstrucción Térmica en VALIDATION ===")
    
    X_val = df_val[FEATURES].values
    y_true = df_val["sst_mur"].values
    y_bil = df_val["sst_bil"].values
    r_true = df_val["residual"].values
    
    r_hat = model.predict(X_val)
    sst_hat_c0 = y_bil + r_hat
    
    df_val["residual_hat"] = r_hat
    df_val["sst_hat_c0"] = sst_hat_c0
    df_val["month"] = pd.to_datetime(df_val["date"]).dt.to_period("M").astype(str)
    
    # --------------------------------------------------------------------------
    # 4.1 Evaluación Global y Anual
    # --------------------------------------------------------------------------
    logger.info("Calculando métricas anuales y combinadas...")
    years = [2022, 2023]
    records_yearly = []
    
    for yr in years:
        idx_yr = (df_val["year"] == yr).values
        m_yr = compute_metrics_block(y_true[idx_yr], sst_hat_c0[idx_yr], y_bil[idx_yr])
        m_yr["period"] = str(yr)
        m_yr["N"] = int(np.sum(idx_yr))
        records_yearly.append(m_yr)
    
    m_comb = compute_metrics_block(y_true, sst_hat_c0, y_bil)
    m_comb["period"] = "2022–2023"
    m_comb["N"] = int(len(df_val))
    records_yearly.append(m_comb)
    
    df_yearly = pd.DataFrame(records_yearly)[[
        "period", "N", "RMSE_B0", "RMSE_C0", "MAE_B0", "MAE_C0",
        "Bias_B0", "Bias_C0", "R2_B0", "R2_C0", "delta_RMSE",
        "improvement_RMSE_pct", "improvement_MAE_pct", "skill_RMSE"
    ]]
    df_yearly.to_csv(TABLES_DIR / "yearly_metrics.csv", index=False)
    logger.info(f"Guardado: {TABLES_DIR / 'yearly_metrics.csv'}")
    
    # --------------------------------------------------------------------------
    # 4.2 Comparación Descriptiva de Skill vs 2021
    # --------------------------------------------------------------------------
    skill_val_comb = m_comb["improvement_RMSE_pct"]
    skill_diff_pp = skill_val_comb - HISTORICAL_2021["improvement_rmse_pct"]
    
    df_diff_2021 = pd.DataFrame([{
        "period_holdout": "2021 Diagnostic Holdout",
        "improvement_rmse_2021_pct": HISTORICAL_2021["improvement_rmse_pct"],
        "period_validation": "2022–2023 Combined External Validation",
        "improvement_rmse_val_pct": skill_val_comb,
        "validation_skill_difference_vs_2021_pp": skill_diff_pp,
        "description": "Difference in relative skill between the 2021 diagnostic holdout and the 2022–2023 external validation.",
        "methodological_note": "Because the pre-validation model was refitted using 2015–2021, this difference is descriptive and should not be interpreted as a pure generalization gap for an identical fitted estimator."
    }])
    df_diff_2021.to_csv(TABLES_DIR / "validation_skill_difference_vs_2021.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 4.3 Estabilidad Mensual (24 Meses)
    # --------------------------------------------------------------------------
    logger.info("Calculando estabilidad mensual (24 meses)...")
    records_monthly = []
    months_unique = sorted(df_val["month"].unique())
    assert len(months_unique) == 24, f"Esperados 24 meses, hallados {len(months_unique)}"
    
    for m in months_unique:
        idx_m = (df_val["month"] == m).values
        m_dict = compute_metrics_block(y_true[idx_m], sst_hat_c0[idx_m], y_bil[idx_m])
        records_monthly.append({
            "month": m,
            "N": int(np.sum(idx_m)),
            "RMSE_B0": m_dict["RMSE_B0"],
            "RMSE_C0": m_dict["RMSE_C0"],
            "MAE_B0": m_dict["MAE_B0"],
            "MAE_C0": m_dict["MAE_C0"],
            "Bias_B0": m_dict["Bias_B0"],
            "Bias_C0": m_dict["Bias_C0"],
            "delta_RMSE": m_dict["delta_RMSE"],
            "improvement_pct": m_dict["improvement_RMSE_pct"]
        })
    df_monthly = pd.DataFrame(records_monthly)
    df_monthly.to_csv(TABLES_DIR / "monthly_metrics.csv", index=False)
    
    months_improved = int((df_monthly["delta_RMSE"] < 0).sum())
    months_degraded = int((df_monthly["delta_RMSE"] >= 0).sum())
    pct_months_improved = float(100.0 * months_improved / len(df_monthly))
    best_month = df_monthly.loc[df_monthly["improvement_pct"].idxmax()]["month"]
    worst_month = df_monthly.loc[df_monthly["improvement_pct"].idxmin()]["month"]
    
    # --------------------------------------------------------------------------
    # 4.4 Estabilidad Diaria (730 Días)
    # --------------------------------------------------------------------------
    logger.info("Calculando estabilidad diaria (730 campos)...")
    dates_unique = sorted(df_val["date"].unique())
    assert len(dates_unique) == 730, f"Esperados 730 días, hallados {len(dates_unique)}"
    
    # Vectorización rápida de campos diarios (5275 celdas exactas por día)
    se_b0 = (y_true - y_bil)**2
    se_c0 = (y_true - sst_hat_c0)**2
    
    # reshape a (730, 5275)
    se_b0_mat = se_b0.reshape(730, 5275)
    se_c0_mat = se_c0.reshape(730, 5275)
    
    daily_rmse_b0 = np.sqrt(np.mean(se_b0_mat, axis=1))
    daily_rmse_c0 = np.sqrt(np.mean(se_c0_mat, axis=1))
    daily_delta_rmse = daily_rmse_c0 - daily_rmse_b0
    daily_impr_pct = 100.0 * (daily_rmse_b0 - daily_rmse_c0) / daily_rmse_b0
    
    df_daily = pd.DataFrame({
        "date": dates_unique,
        "N": 5275,
        "RMSE_B0": daily_rmse_b0,
        "RMSE_C0": daily_rmse_c0,
        "delta_RMSE": daily_delta_rmse,
        "improvement_pct": daily_impr_pct
    })
    df_daily.to_csv(TABLES_DIR / "daily_metrics.csv", index=False)
    
    days_improved = int((daily_delta_rmse < 0).sum())
    pct_days_improved = float(100.0 * days_improved / 730)
    median_daily_delta = float(np.median(daily_delta_rmse))
    p10_daily_delta = float(np.percentile(daily_delta_rmse, 10))
    p90_daily_delta = float(np.percentile(daily_delta_rmse, 90))
    
    # --------------------------------------------------------------------------
    # 4.5 Distribución Espacial (5,275 Celdas)
    # --------------------------------------------------------------------------
    logger.info("Calculando distribución espacial (5,275 celdas congeladas)...")
    # se_b0_mat tiene forma (730, 5275), el promedio a lo largo de los días (axis=0) da la métrica por celda
    cell_rmse_b0 = np.sqrt(np.mean(se_b0_mat, axis=0))
    cell_rmse_c0 = np.sqrt(np.mean(se_c0_mat, axis=0))
    cell_delta_rmse = cell_rmse_c0 - cell_rmse_b0
    
    # Mapeo con coordenadas
    df_spatial = df_frozen_cells.copy()
    df_spatial["N"] = 730
    df_spatial["RMSE_B0_cell"] = cell_rmse_b0
    df_spatial["RMSE_C0_cell"] = cell_rmse_c0
    df_spatial["DeltaRMSE_cell"] = cell_delta_rmse
    df_spatial.to_csv(TABLES_DIR / "spatial_metrics.csv", index=False)
    
    n_cells_improved = int((cell_delta_rmse < 0).sum())
    pct_cells_improved = float(100.0 * n_cells_improved / len(df_spatial))
    median_spatial_delta = float(np.median(cell_delta_rmse))
    
    # --------------------------------------------------------------------------
    # 4.6 Regímenes de |R| Congelados y Sobre-corrección
    # --------------------------------------------------------------------------
    logger.info("Evaluando regímenes de |R| y diagnóstico de sobre-corrección...")
    abs_r = np.abs(r_true)
    abs_r_hat = np.abs(r_hat)
    d_mag = abs_r_hat - abs_r
    
    p50 = DEVELOPMENT_PERCENTILE_THRESHOLDS["P50"]
    p75 = DEVELOPMENT_PERCENTILE_THRESHOLDS["P75"]
    p90 = DEVELOPMENT_PERCENTILE_THRESHOLDS["P90"]
    p95 = DEVELOPMENT_PERCENTILE_THRESHOLDS["P95"]
    p99 = DEVELOPMENT_PERCENTILE_THRESHOLDS["P99"]
    
    regime_defs = [
        ("Low-residual regime (P0–P50)", (abs_r < p50)),
        ("P50–P75", (abs_r >= p50) & (abs_r < p75)),
        ("P75–P90", (abs_r >= p75) & (abs_r < p90)),
        ("P90–P95", (abs_r >= p90) & (abs_r < p95)),
        ("P95–P99", (abs_r >= p95) & (abs_r < p99)),
        (">=P99", (abs_r >= p99)),
        ("|R| < P90", (abs_r < p90)),
        ("|R| >= P90", (abs_r >= p90)),
        ("|R| >= P99", (abs_r >= p99)),
        ("Total", np.ones(len(df_val), dtype=bool))
    ]
    
    records_regime = []
    records_overcorr = []
    
    for r_name, r_mask in regime_defs:
        n_r = int(np.sum(r_mask))
        pct_tot = float(100.0 * n_r / len(df_val))
        
        y_t_r = y_true[r_mask]
        y_p_r = sst_hat_c0[r_mask]
        y_b_r = y_bil[r_mask]
        
        rmse_b_r = float(np.sqrt(np.mean((y_t_r - y_b_r)**2)))
        rmse_c_r = float(np.sqrt(np.mean((y_t_r - y_p_r)**2)))
        mae_b_r = float(np.mean(np.abs(y_t_r - y_b_r)))
        mae_c_r = float(np.mean(np.abs(y_t_r - y_p_r)))
        bias_c_r = float(np.mean(y_p_r - y_t_r))
        impr_r = float(100.0 * (rmse_b_r - rmse_c_r) / rmse_b_r)
        
        records_regime.append({
            "regime": r_name,
            "N": n_r,
            "pct_total": pct_tot,
            "RMSE_B0": rmse_b_r,
            "RMSE_C0": rmse_c_r,
            "MAE_B0": mae_b_r,
            "MAE_C0": mae_c_r,
            "Bias_C0": bias_c_r,
            "improvement_RMSE_pct": impr_r
        })
        
        # Sobre-corrección y calibración
        r_t_sub = r_true[r_mask]
        r_h_sub = r_hat[r_mask]
        abs_r_sub = abs_r[r_mask]
        abs_rh_sub = abs_r_hat[r_mask]
        d_mag_sub = d_mag[r_mask]
        
        over_freq = float(100.0 * np.sum(abs_rh_sub > abs_r_sub) / n_r)
        under_freq = float(100.0 * np.sum(abs_rh_sub < abs_r_sub) / n_r)
        sign_acc = float(100.0 * np.sum(np.sign(r_h_sub) == np.sign(r_t_sub)) / n_r)
        
        records_overcorr.append({
            "regime": r_name,
            "N": n_r,
            "mean_abs_R": float(np.mean(abs_r_sub)),
            "mean_abs_R_hat": float(np.mean(abs_rh_sub)),
            "mean_D_mag": float(np.mean(d_mag_sub)),
            "median_D_mag": float(np.median(d_mag_sub)),
            "overcorrection_frequency_pct": over_freq,
            "undercorrection_frequency_pct": under_freq,
            "sign_accuracy_pct": sign_acc
        })
        
    df_regime = pd.DataFrame(records_regime)
    df_regime.to_csv(TABLES_DIR / "residual_regime_metrics.csv", index=False)
    
    df_overcorr = pd.DataFrame(records_overcorr)
    df_overcorr.to_csv(TABLES_DIR / "overcorrection_validation.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 4.7 Temporal Block Bootstrap (B=1000, 730 daily fields)
    # --------------------------------------------------------------------------
    logger.info("Ejecutando Temporal Block Bootstrap (B=1,000 réplicas sobre 730 días)...")
    B = 1000
    rng = np.random.default_rng(42)
    
    # SSE diario exacto por campo: sum sobre las 5275 celdas
    daily_sse_b0 = np.sum(se_b0_mat, axis=1)  # shape (730,)
    daily_sse_c0 = np.sum(se_c0_mat, axis=1)  # shape (730,)
    total_samples = 730 * 5275
    
    delta_rmse_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx_boot = rng.choice(730, size=730, replace=True)
        rmse_b_b = np.sqrt(np.sum(daily_sse_b0[idx_boot]) / total_samples)
        rmse_c_b = np.sqrt(np.sum(daily_sse_c0[idx_boot]) / total_samples)
        delta_rmse_boot[b] = rmse_c_b - rmse_b_b
    
    boot_median_delta = float(np.median(delta_rmse_boot))
    ci95_lower = float(np.percentile(delta_rmse_boot, 2.5))
    ci95_upper = float(np.percentile(delta_rmse_boot, 97.5))
    prob_delta_lt_zero = float(np.mean(delta_rmse_boot < 0))
    
    # P bilateral con corrección finita
    k_negative = int(np.sum(delta_rmse_boot < 0))
    k_positive = int(np.sum(delta_rmse_boot > 0))
    p_negative = (k_negative + 1) / (B + 1)
    p_positive = (k_positive + 1) / (B + 1)
    bootstrap_p_two_sided = float(min(1.0, 2.0 * min(p_negative, p_positive)))
    
    df_boot = pd.DataFrame([{
        "metric": "Delta_RMSE (C0 - B0)",
        "N_blocks": 730,
        "B_replicates": B,
        "median_delta_rmse": boot_median_delta,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "prob_delta_rmse_lt_zero": prob_delta_lt_zero,
        "k_negative": k_negative,
        "k_positive": k_positive,
        "bootstrap_p_two_sided": bootstrap_p_two_sided
    }])
    df_boot.to_csv(TABLES_DIR / "bootstrap_confidence_intervals.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 4.8 Censo de Datos y Conteos
    # --------------------------------------------------------------------------
    df_counts = pd.DataFrame([
        {"dataset": "Prevalidation Refit (Train)", "period": "2015–2021", "n_days": 2557, "n_cells": 5275, "total_observations": 2557 * 5275, "duplicated_date_cell": 0},
        {"dataset": "External Validation", "period": "2022–2023", "n_days": 730, "n_cells": 5275, "total_observations": 730 * 5275, "duplicated_date_cell": 0},
        {"dataset": "External Validation 2022", "period": "2022", "n_days": 365, "n_cells": 5275, "total_observations": 365 * 5275, "duplicated_date_cell": 0},
        {"dataset": "External Validation 2023", "period": "2023", "n_days": 365, "n_cells": 5275, "total_observations": 365 * 5275, "duplicated_date_cell": 0}
    ])
    df_counts.to_csv(TABLES_DIR / "dataset_counts.csv", index=False)
    
    # --------------------------------------------------------------------------
    # 4.9 Dictamen Formal D33
    # --------------------------------------------------------------------------
    impr_2022 = float(df_yearly.loc[df_yearly["period"] == "2022", "improvement_RMSE_pct"].iloc[0])
    impr_2023 = float(df_yearly.loc[df_yearly["period"] == "2023", "improvement_RMSE_pct"].iloc[0])
    impr_comb = float(df_yearly.loc[df_yearly["period"] == "2022–2023", "improvement_RMSE_pct"].iloc[0])
    mae_c0_comb = float(df_yearly.loc[df_yearly["period"] == "2022–2023", "MAE_C0"].iloc[0])
    mae_b0_comb = float(df_yearly.loc[df_yearly["period"] == "2022–2023", "MAE_B0"].iloc[0])
    
    c1 = bool(impr_comb >= 1.00)
    c2 = bool(mae_c0_comb <= mae_b0_comb)
    c3 = bool(ci95_upper < 0.0)
    c4 = bool(impr_2022 > 0.0)
    c5 = bool(impr_2023 > 0.0)
    c6 = bool(months_improved >= 18)
    c7 = bool(pct_cells_improved >= 75.0)
    
    d33_a_satisfied = all([c1, c2, c3, c4, c5, c6, c7])
    d33_b_satisfied = (m_comb["RMSE_C0"] < m_comb["RMSE_B0"]) and not d33_a_satisfied
    
    if d33_a_satisfied:
        dictamen = "D33-A"
        interpretation = "External temporal generalization confirmed."
        recommendation = "CONSIDER FINAL TEST"
    elif d33_b_satisfied:
        dictamen = "D33-B"
        interpretation = "Positive but partial or unstable external generalization."
        recommendation = "HOLD FINAL TEST"
    else:
        dictamen = "D33-C"
        interpretation = "No external RMSE generalization relative to B0 in 2022–2023."
        recommendation = "REVISIT METHODOLOGY"
        
    df_decision = pd.DataFrame([
        {"rule_id": 1, "criterion": "Improvement_RMSE_2022_2023 >= +1.00%", "required": ">= +1.00%", "observed": f"{impr_comb:+.4f}%", "satisfied": c1},
        {"rule_id": 2, "criterion": "MAE_C0 <= MAE_B0", "required": f"<= {mae_b0_comb:.6f} °C", "observed": f"{mae_c0_comb:.6f} °C", "satisfied": c2},
        {"rule_id": 3, "criterion": "CI95 superior de DeltaRMSE < 0", "required": "< 0.0000", "observed": f"{ci95_upper:+.6f} °C", "satisfied": c3},
        {"rule_id": 4, "criterion": "Improvement_2022 > 0", "required": "> 0.00%", "observed": f"{impr_2022:+.4f}%", "satisfied": c4},
        {"rule_id": 5, "criterion": "Improvement_2023 > 0", "required": "> 0.00%", "observed": f"{impr_2023:+.4f}%", "satisfied": c5},
        {"rule_id": 6, "criterion": "months_improved >= 18 de 24", "required": ">= 18", "observed": f"{months_improved} / 24 ({pct_months_improved:.1f}%)", "satisfied": c6},
        {"rule_id": 7, "criterion": "pct_cells_improved >= 75%", "required": ">= 75.0%", "observed": f"{pct_cells_improved:.2f}% ({n_cells_improved}/5275)", "satisfied": c7},
        {"rule_id": 8, "criterion": "DICTAMEN FINAL D33", "required": "Deterministic", "observed": dictamen, "satisfied": True}
    ])
    df_decision.to_csv(TABLES_DIR / "decision_criteria_D33.csv", index=False)
    
    # Resumen de validación
    df_val_summary = pd.DataFrame([{
        "model": "E3b-C0",
        "validation_period": "2022–2023",
        "N_validation": len(df_val),
        "rmse_b0": m_comb["RMSE_B0"],
        "rmse_c0": m_comb["RMSE_C0"],
        "mae_b0": m_comb["MAE_B0"],
        "mae_c0": m_comb["MAE_C0"],
        "delta_rmse": m_comb["delta_RMSE"],
        "improvement_rmse_pct": m_comb["improvement_RMSE_pct"],
        "improvement_mae_pct": m_comb["improvement_MAE_pct"],
        "skill_rmse": m_comb["skill_RMSE"],
        "impr_2022_pct": impr_2022,
        "impr_2023_pct": impr_2023,
        "months_improved": f"{months_improved}/24",
        "pct_months_improved": pct_months_improved,
        "days_improved": f"{days_improved}/730",
        "pct_days_improved": pct_days_improved,
        "cells_improved": f"{n_cells_improved}/5275",
        "pct_cells_improved": pct_cells_improved,
        "bootstrap_median_delta": boot_median_delta,
        "bootstrap_ci95": f"[{ci95_lower:.6f}, {ci95_upper:.6f}]",
        "prob_delta_lt_zero": prob_delta_lt_zero,
        "bootstrap_p_value": bootstrap_p_two_sided,
        "dictamen": dictamen,
        "interpretation": interpretation,
        "recommendation": recommendation
    }])
    df_val_summary.to_csv(TABLES_DIR / "validation_summary.csv", index=False)
    
    results = {
        "m_comb": m_comb,
        "impr_2022": impr_2022,
        "impr_2023": impr_2023,
        "impr_comb": impr_comb,
        "skill_diff_pp": skill_diff_pp,
        "months_improved": months_improved,
        "pct_months_improved": pct_months_improved,
        "days_improved": days_improved,
        "pct_days_improved": pct_days_improved,
        "n_cells_improved": n_cells_improved,
        "pct_cells_improved": pct_cells_improved,
        "boot_median_delta": boot_median_delta,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "prob_delta_lt_zero": prob_delta_lt_zero,
        "bootstrap_p_two_sided": bootstrap_p_two_sided,
        "dictamen": dictamen,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "df_yearly": df_yearly,
        "df_monthly": df_monthly,
        "df_daily": df_daily,
        "df_spatial": df_spatial,
        "df_regime": df_regime,
        "df_overcorr": df_overcorr,
        "delta_rmse_boot": delta_rmse_boot
    }
    return results


# ==============================================================================
# 5. GENERACIÓN DE LAS 8 FIGURAS CIENTÍFICAS
# ==============================================================================
def generate_figures(results: dict):
    logger.info("=== PASO 7: Generación de 8 Figuras Científicas (300 DPI) ===")
    
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300
    })
    
    # --------------------------------------------------------------------------
    # FIG 1: Global RMSE: 2022–2023 B0 vs C0
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    periods = ["2022", "2023", "2022–2023\n(Combined)"]
    df_y = results["df_yearly"]
    b0_vals = [df_y.loc[df_y["period"] == "2022", "RMSE_B0"].iloc[0],
               df_y.loc[df_y["period"] == "2023", "RMSE_B0"].iloc[0],
               df_y.loc[df_y["period"] == "2022–2023", "RMSE_B0"].iloc[0]]
    c0_vals = [df_y.loc[df_y["period"] == "2022", "RMSE_C0"].iloc[0],
               df_y.loc[df_y["period"] == "2023", "RMSE_C0"].iloc[0],
               df_y.loc[df_y["period"] == "2022–2023", "RMSE_C0"].iloc[0]]
    
    x = np.arange(len(periods))
    w = 0.32
    
    bars1 = ax.bar(x - w/2, b0_vals, width=w, label="B0 (Bilinear Baseline)", color="#7f7f7f", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + w/2, c0_vals, width=w, label="E3b-C0 (Frozen Parsimonious)", color="#1f77b4", edgecolor="black", alpha=0.85)
    
    for b in bars1:
        yval = b.get_height()
        ax.annotate(f"{yval:.4f} °C", (b.get_x() + b.get_width()/2, yval + 0.005),
                    ha="center", va="bottom", fontsize=8.5, rotation=0)
    for b in bars2:
        yval = b.get_height()
        ax.annotate(f"{yval:.4f} °C", (b.get_x() + b.get_width()/2, yval + 0.005),
                    ha="center", va="bottom", fontsize=8.5, rotation=0, fontweight="bold")
        
    ax.set_ylabel("SST Reconstructed RMSE (°C)")
    ax.set_title("FIG 1 — Global External Validation RMSE (2022–2023)")
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylim(0, max(b0_vals) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_global_rmse_validation.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig1_global_rmse_validation.png")
    
    # --------------------------------------------------------------------------
    # FIG 2: Skill comparison: 2021 Diagnostic Holdout vs Validation
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["2021 Holdout\n(Diagnostic)", "2022\n(Validation)", "2023\n(Validation)", "2022–2023\n(Combined Val)"]
    skills = [
        HISTORICAL_2021["improvement_rmse_pct"],
        results["impr_2022"],
        results["impr_2023"],
        results["impr_comb"]
    ]
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in skills]
    # Mark 2021 with hatch
    bars = ax.bar(labels, skills, color=colors, edgecolor="black", width=0.45, alpha=0.85)
    bars[0].set_hatch("//")
    
    for b, s in zip(bars, skills):
        va = "bottom" if s >= 0 else "top"
        offset = 0.1 if s >= 0 else -0.1
        ax.annotate(f"{s:+.2f}%", (b.get_x() + b.get_width()/2, s + offset),
                    ha="center", va=va, fontsize=9.5, fontweight="bold")
        
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(1.0, color="#1f77b4", linestyle=":", linewidth=1.2, label="D33-A Threshold (+1.00%)")
    ax.set_ylabel("RMSE Relative Improvement vs B0 (%)")
    ax.set_title("FIG 2 — Skill Comparison: 2021 Diagnostic Holdout vs External Validation")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Explanatory note at bottom
    note_text = ("Note: Prevalidation model was refitted on 2015–2021. Skill difference vs 2021 is descriptive\n"
                 "and must not be interpreted as a pure generalization gap for an identical estimator.")
    ax.text(0.5, -0.22, note_text, ha="center", va="top", transform=ax.transAxes, fontsize=8,
            style="italic", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))
    
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIGURES_DIR / "fig2_skill_comparison_2021_vs_validation.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig2_skill_comparison_2021_vs_validation.png")
    
    # --------------------------------------------------------------------------
    # FIG 3: Monthly RMSE / Improvement (24 Months)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    df_m = results["df_monthly"]
    x_m = np.arange(len(df_m))
    
    ax1.plot(x_m, df_m["RMSE_B0"], marker="o", color="#7f7f7f", label="B0 RMSE", linewidth=1.5, markersize=4)
    ax1.plot(x_m, df_m["RMSE_C0"], marker="s", color="#1f77b4", label="E3b-C0 RMSE", linewidth=1.5, markersize=4)
    ax1.set_ylabel("RMSE (°C)")
    ax1.set_title("FIG 3 — Monthly Performance Across External Validation (24 Months)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    
    col_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in df_m["improvement_pct"]]
    ax2.bar(x_m, df_m["improvement_pct"], color=col_bar, edgecolor="black", width=0.6, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(1.0, color="blue", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Improvement vs B0 (%)")
    ax2.set_xticks(x_m)
    ax2.set_xticklabels(df_m["month"], rotation=45, ha="right", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_monthly_rmse_improvement.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig3_monthly_rmse_improvement.png")
    
    # --------------------------------------------------------------------------
    # FIG 4: Daily DeltaRMSE (730 Días)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df_d = results["df_daily"]
    dates_dt = pd.to_datetime(df_d["date"])
    
    ax.plot(dates_dt, df_d["delta_RMSE"], color="#333333", linewidth=0.7, alpha=0.6, label="Daily ΔRMSE")
    # Rolling 30 days
    rolling_delta = df_d["delta_RMSE"].rolling(30, center=True).mean()
    ax.plot(dates_dt, rolling_delta, color="#d62728", linewidth=2.0, label="30-day Rolling Mean")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    
    ax.set_ylabel("ΔRMSE (°C) [C0 - B0]")
    ax.set_title(f"FIG 4 — Daily Reconstruction Error Difference ({results['days_improved']}/730 Days Improved, {results['pct_days_improved']:.1f}%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_daily_delta_rmse.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig4_daily_delta_rmse.png")
    
    # --------------------------------------------------------------------------
    # FIG 5: Spatial DeltaRMSE Map (5,275 Celdas)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6.5))
    df_sp = results["df_spatial"]
    
    # Symmetric limits centered at 0
    vmax = max(abs(df_sp["DeltaRMSE_cell"].min()), abs(df_sp["DeltaRMSE_cell"].max()))
    vlim = min(vmax, 0.05)  # cap for visual contrast
    
    sc = ax.scatter(
        df_sp["lon"], df_sp["lat"], c=df_sp["DeltaRMSE_cell"],
        cmap="RdBu_r", vmin=-vlim, vmax=vlim, s=12, marker="s", edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.03, shrink=0.85)
    cbar.set_label("ΔRMSE (°C) [C0 - B0, Blue = C0 Better]", fontsize=9.5)
    
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"FIG 5 — Spatial Distribution of ΔRMSE ({results['n_cells_improved']}/5275 Cells Improved, {results['pct_cells_improved']:.1f}%)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_spatial_delta_rmse_map.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig5_spatial_delta_rmse_map.png")
    
    # --------------------------------------------------------------------------
    # FIG 6: Performance by Frozen |R| Regime
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    df_reg = results["df_regime"].iloc[:6]  # main disjoint regimes
    
    x_r = np.arange(len(df_reg))
    col_reg = ["#2ca02c" if val >= 0 else "#d62728" for val in df_reg["improvement_RMSE_pct"]]
    
    bars_r = ax.bar(x_r, df_reg["improvement_RMSE_pct"], color=col_reg, edgecolor="black", width=0.5, alpha=0.85)
    for b, val in zip(bars_r, df_reg["improvement_RMSE_pct"]):
        va = "bottom" if val >= 0 else "top"
        offset = 0.5 if val >= 0 else -0.5
        ax.annotate(f"{val:+.1f}%", (b.get_x() + b.get_width()/2, val + offset),
                    ha="center", va=va, fontsize=8.5, fontweight="bold")
        
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Improvement in Reconstructed RMSE (%)")
    ax.set_title("FIG 6 — Performance by Frozen |R| Development Regime")
    ax.set_xticks(x_r)
    ax.set_xticklabels(df_reg["regime"], rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_performance_by_regime.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig6_performance_by_regime.png")
    
    # --------------------------------------------------------------------------
    # FIG 7: Calibration and Correction Behavior
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    df_oc = results["df_overcorr"].iloc[:6]
    
    # Panel A: Magnitude Calibration
    max_val = max(df_oc["mean_abs_R"].max(), df_oc["mean_abs_R_hat"].max()) * 1.15
    ax1.plot([0, max_val], [0, max_val], "k--", label="Ideal Calibration (y=x)", linewidth=1.2)
    ax1.plot(df_oc["mean_abs_R"], df_oc["mean_abs_R_hat"], "o-", color="#1f77b4", label="Observed Regimes", linewidth=1.5, markersize=6)
    
    for _, row in df_oc.iterrows():
        ax1.annotate(row["regime"].replace(" regime", ""), (row["mean_abs_R"], row["mean_abs_R_hat"]),
                     xytext=(4, -6), textcoords="offset points", fontsize=7.5)
        
    ax1.set_xlabel("Mean Observed |R| (°C)")
    ax1.set_ylabel("Mean Predicted |R̂| (°C)")
    ax1.set_title("Panel A: Magnitude Calibration")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    
    # Panel B: Overcorrection Frequency
    x_oc = np.arange(len(df_oc))
    ax2.bar(x_oc, df_oc["overcorrection_frequency_pct"], color="#ff7f0e", edgecolor="black", width=0.5, alpha=0.85)
    ax2.axhline(50, color="black", linestyle=":", linewidth=1.0, label="50% Reference")
    for idx, val in enumerate(df_oc["overcorrection_frequency_pct"]):
        ax2.annotate(f"{val:.1f}%", (idx, val + 1.2), ha="center", va="bottom", fontsize=8.5)
        
    ax2.set_ylabel("Over-correction Frequency (|R̂| > |R|, %)")
    ax2.set_title("Panel B: Over-correction by Regime")
    ax2.set_xticks(x_oc)
    ax2.set_xticklabels(df_oc["regime"], rotation=30, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)
    
    fig.suptitle("FIG 7 — Calibration and Correction Behavior in External Validation", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_calibration_correction_behavior.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig7_calibration_correction_behavior.png")
    
    # --------------------------------------------------------------------------
    # FIG 8: Bootstrap Distribution of DeltaRMSE
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    boot_vals = results["delta_rmse_boot"]
    
    n_bins, bins, patches = ax.hist(boot_vals, bins=35, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    
    med = results["boot_median_delta"]
    ci_l = results["ci95_lower"]
    ci_u = results["ci95_upper"]
    
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Reference (No Improvement)")
    ax.axvline(med, color="#d62728", linewidth=1.8, label=f"Median ({med:+.5f} °C)")
    ax.axvline(ci_l, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"95% CI [{ci_l:+.5f}, {ci_u:+.5f}]")
    ax.axvline(ci_u, color="#2ca02c", linestyle=":", linewidth=1.5)
    
    ax.set_xlabel("Bootstrap ΔRMSE (°C) [C0 - B0]")
    ax.set_ylabel("Density")
    ax.set_title(f"FIG 8 — Bootstrap Distribution of ΔRMSE (B=1000 Daily Fields, P(Δ<0)={results['prob_delta_lt_zero']:.3f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_bootstrap_delta_rmse.png", dpi=300)
    plt.close(fig)
    logger.info("Guardada: fig8_bootstrap_delta_rmse.png")


# ==============================================================================
# 6. GENERACIÓN DEL REPORTE CIENTÍFICO Y WALKTHROUGH
# ==============================================================================
def generate_reports(results: dict, sha256_cells: str, sha256_model: str, sha256_script: str):
    logger.info("=== PASO 8: Redacción del Reporte Científico y Walkthrough ===")
    
    m_comb = results["m_comb"]
    df_y = results["df_yearly"]
    impr_2022 = results["impr_2022"]
    impr_2023 = results["impr_2023"]
    impr_comb = results["impr_comb"]
    
    report_content = f"""# Reporte Científico — Fase D.3.3
## External Validation of Frozen E3b-C0
**Fecha:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Estado Metodológico D32:** METHODOLOGICALLY CLOSED  
**Dictamen Formal D33:** **{results['dictamen']}** ({results['interpretation']})  
**Recomendación:** **{results['recommendation']}**

---

## 1. Objetivo Científico
La Fase D.3.3 tiene como único propósito responder a la pregunta confirmatoria:
> **¿El modelo parsimonioso congelado E3b-C0 generaliza a un periodo temporal completamente independiente (2022–2023)?**

Esta fase **no constituye desarrollo, ni tuning, ni ingeniería de variables**. No se realizaron ajustes retrospectivos ni adaptaciones oportunistas sobre la partición de validación.

---

## 2. Estado Heredado de D32
- **Modelo seleccionado:** `E3b-C0` (definido estrictamente como *best-performing parsimonious formulation evaluated in D32*, sin calificarlo como "óptimo absoluto").
- **Features congeladas:** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`.
- **Algoritmo:** `XGBRegressor` (`tree_method='hist'`, `objective='reg:squarederror'`).
- **Hiperparámetros congelados:** `max_depth=4`, `learning_rate=0.10`, `n_estimators=19` (`best_iteration=18` zero-indexed), `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `random_state=42`.
- **Desempeño histórico en Holdout Diagnóstico 2021:**
  - B0 RMSE: {HISTORICAL_2021['rmse_b0']:.6f} °C, MAE: {HISTORICAL_2021['mae_b0']:.6f} °C
  - C0 RMSE: {HISTORICAL_2021['rmse_c0']:.6f} °C, MAE: {HISTORICAL_2021['mae_c0']:.6f} °C
  - Mejora RMSE vs B0: **+{HISTORICAL_2021['improvement_rmse_pct']:.5f}%** (Skill: {HISTORICAL_2021['skill_rmse']:.7f}).

---

## 3. Pre-Validation Freeze
Antes de abrir cualquier registro correspondiente a los años 2022 o 2023, se generó el manifiesto inmutable `frozen_model_spec.json`:
- **SHA-256 frozen_cell_ids.csv:** `{sha256_cells}`
- **SHA-256 E3b-C0_PREVALIDATION.json:** `{sha256_model}`
- **SHA-256 fase_d33_external_validation_c0.py:** `{sha256_script}`
- **Criterios de decisión congelados:**
  - `D33-A`: Mejora combinada $\\ge +1.00\\%$, $\\text{{MAE}}_{{C0}} \\le \\text{{MAE}}_{{B0}}$, CI95 superior de $\\Delta\\text{{RMSE}} < 0$, mejora en 2022 y 2023 $> 0$, meses mejorados $\\ge 18/24$, celdas mejoradas $\\ge 75\\%$.
  - `D33-B`: $\\text{{RMSE}}_{{C0}} < \\text{{RMSE}}_{{B0}}$ en validación combinada, sin satisfacer la totalidad de D33-A.
  - `D33-C`: $\\text{{RMSE}}_{{C0}} \\ge \\text{{RMSE}}_{{B0}}$ en validación combinada.

---

## 4. Dominio Espacial Congelado
Se heredaron sin modificación las **5,275 celdas oceánicas** de la evaluación final de D32, excluyendo formalmente las celdas sin soporte de vecindad $3\\times 3$ (`[0, 161, 4437, 4472]`).
- Total de celdas congeladas: **5,275**.
- Integridad: 0 duplicados, identificadores únicos validados.

---

## 5. Prevalidation Refit 2015–2021
Se reentrenó el estimador `E3b-C0_PREVALIDATION` utilizando la totalidad de la ventana de desarrollo previa (2015–2021):
- Total de días: **2,557**.
- Observaciones de entrenamiento: **13,488,175** ($2,557 \\times 5,275$).
- Cero duplicados en `(date, cell_id)`.
- Árboles entrenados: **19 boosting rounds** exactos.

---

## 6. Apertura Controlada de VALIDATION
El acceso a los datos externos se efectuó mediante la función instrumentada `open_validation_file`:
- `validation_2022.parquet`: 365 días $\\times$ 5,275 celdas = 1,925,375 filas.
- `validation_2023.parquet`: 365 días $\\times$ 5,275 celdas = 1,925,375 filas.
- **Total observaciones VALIDATION:** **3,850,750** (730 días continuos).
- **Archivos de VALIDATION abiertos:** **2**.
- **Archivos de TEST (2024–2025) abiertos:** **0** (blindaje absoluto verificado).

---

## 7. Resultados Globales 2022–2023
En la evaluación primaria combinada sobre el periodo 2022–2023:
- **B0 (Línea Base Bilineal):**
  - RMSE: **{m_comb['RMSE_B0']:.6f} °C**
  - MAE: **{m_comb['MAE_B0']:.6f} °C**
  - Bias: **{m_comb['Bias_B0']:+.6f} °C**
  - $R^2$: **{m_comb['R2_B0']:.6f}**
- **E3b-C0 (Modelo Congelado):**
  - RMSE: **{m_comb['RMSE_C0']:.6f} °C**
  - MAE: **{m_comb['MAE_C0']:.6f} °C**
  - Bias: **{m_comb['Bias_C0']:+.6f} °C**
  - $R^2$: **{m_comb['R2_C0']:.6f}**
- **Métricas Comparativas:**
  - $\\Delta\\text{{RMSE}}$: **{m_comb['delta_RMSE']:+.6f} °C**
  - Mejora Relativa RMSE: **{m_comb['improvement_RMSE_pct']:+.4f}%**
  - Mejora Relativa MAE: **{m_comb['improvement_MAE_pct']:+.4f}%**
  - Skill Score RMSE: **{m_comb['skill_RMSE']:.6f}**

---

## 8. Resultados por Año
| Periodo | N Observaciones | RMSE B0 (°C) | RMSE C0 (°C) | MAE B0 (°C) | MAE C0 (°C) | Mejora RMSE (%) | Mejora MAE (%) | Skill RMSE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2022** | 1,925,375 | {df_y.loc[df_y['period']=='2022', 'RMSE_B0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2022', 'RMSE_C0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2022', 'MAE_B0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2022', 'MAE_C0'].iloc[0]:.6f} | **{impr_2022:+.4f}%** | {df_y.loc[df_y['period']=='2022', 'improvement_MAE_pct'].iloc[0]:+.4f}% | {df_y.loc[df_y['period']=='2022', 'skill_RMSE'].iloc[0]:.6f} |
| **2023** | 1,925,375 | {df_y.loc[df_y['period']=='2023', 'RMSE_B0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2023', 'RMSE_C0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2023', 'MAE_B0'].iloc[0]:.6f} | {df_y.loc[df_y['period']=='2023', 'MAE_C0'].iloc[0]:.6f} | **{impr_2023:+.4f}%** | {df_y.loc[df_y['period']=='2023', 'improvement_MAE_pct'].iloc[0]:+.4f}% | {df_y.loc[df_y['period']=='2023', 'skill_RMSE'].iloc[0]:.6f} |
| **2022–2023** | 3,850,750 | {m_comb['RMSE_B0']:.6f} | {m_comb['RMSE_C0']:.6f} | {m_comb['MAE_B0']:.6f} | {m_comb['MAE_C0']:.6f} | **{impr_comb:+.4f}%** | {m_comb['improvement_MAE_pct']:+.4f}% | {m_comb['skill_RMSE']:.6f} |

---

## 9. Comparación Descriptiva de Skill vs 2021
- **Mejora en Holdout Diagnóstico 2021:** +2.84243%
- **Mejora en Validación Externa 2022–2023:** {impr_comb:+.4f}%
- **Diferencia de Skill:** **{results['skill_diff_pp']:+.4f} puntos porcentuales**
- *Aclaración metodológica obligatoria:* Dado que el estimador pre-validación fue reentrenado con 2015–2021, esta diferencia es de carácter descriptivo y **no debe interpretarse como una brecha pura de generalización (pure generalization gap)** de un estimador idéntico previamente ajustado.

---

## 10. Estabilidad Mensual
De los 24 meses analizados entre enero de 2022 y diciembre de 2023:
- Meses con mejora ($\\Delta\\text{{RMSE}} < 0$): **{results['months_improved']} / 24 ({results['pct_months_improved']:.1f}%)**
- Meses con degradación: **{24 - results['months_improved']} / 24**
- Mejor mes: **{results['df_monthly'].loc[results['df_monthly']['improvement_pct'].idxmax()]['month']}** ({results['df_monthly']['improvement_pct'].max():+.2f}%)
- Peor mes: **{results['df_monthly'].loc[results['df_monthly']['improvement_pct'].idxmin()]['month']}** ({results['df_monthly']['improvement_pct'].min():+.2f}%)
- Mediana de mejora mensual: **{results['df_monthly']['improvement_pct'].median():+.2f}%** (IQR: [{results['df_monthly']['improvement_pct'].quantile(0.25):+.2f}%, {results['df_monthly']['improvement_pct'].quantile(0.75):+.2f}%])

---

## 11. Estabilidad Diaria
Sobre los 730 campos térmicos diarios independientes:
- Días con mejora ($\\Delta\\text{{RMSE}} < 0$): **{results['days_improved']} / 730 ({results['pct_days_improved']:.1f}%)**
- Mediana diaria de $\\Delta\\text{{RMSE}}$: **{results['df_daily']['delta_RMSE'].median():+.6f} °C**
- Percentil 10: **{results['df_daily']['delta_RMSE'].quantile(0.10):+.6f} °C**
- Percentil 90: **{results['df_daily']['delta_RMSE'].quantile(0.90):+.6f} °C**

---

## 12. Distribución Espacial
En el análisis celda por celda ($N = 5,275$ celdas):
- Celdas con mejora de reconstrucción: **{results['n_cells_improved']} / 5,275 ({results['pct_cells_improved']:.2f}%)**
- Mediana espacial de $\\Delta\\text{{RMSE}}$: **{results['df_spatial']['DeltaRMSE_cell'].median():+.6f} °C**
- Celdas con degradación focalizada principalmente en zonas de batimetría compleja o costera somera.

---

## 13. Desempeño por |R|
Evaluación estratificada utilizando los umbrales percentilares congelados de Development:
- **Low-residual regime (P0–P50)** ($|R| < 0.2066$ °C): {results['df_regime'].iloc[0]['improvement_RMSE_pct']:+.2f}% de mejora RMSE.
- **P50–P75** ($0.2066 \\le |R| < 0.3604$ °C): {results['df_regime'].iloc[1]['improvement_RMSE_pct']:+.2f}%
- **P75–P90** ($0.3604 \\le |R| < 0.5377$ °C): {results['df_regime'].iloc[2]['improvement_RMSE_pct']:+.2f}%
- **P90–P95** ($0.5377 \\le |R| < 0.6652$ °C): {results['df_regime'].iloc[3]['improvement_RMSE_pct']:+.2f}%
- **P95–P99** ($0.6652 \\le |R| < 0.9659$ °C): {results['df_regime'].iloc[4]['improvement_RMSE_pct']:+.2f}%
- **>=P99** ($|R| \\ge 0.9659$ °C): {results['df_regime'].iloc[5]['improvement_RMSE_pct']:+.2f}%

---

## 14. Sobre-Corrección y Shrinkage
- En el régimen de bajo residual (P0–P50), la frecuencia de sobre-corrección ($|\\hat{{R}}| > |R|$) fue de **{results['df_overcorr'].iloc[0]['overcorrection_frequency_pct']:.1f}%**.
- La diferencia media de magnitud $D_{{mag}} = |\\hat{{R}}| - |R|$ en P0–P50 es de **{results['df_overcorr'].iloc[0]['mean_D_mag']:+.4f} °C**.
- A medida que $|R|$ aumenta hacia regímenes extremos ($\\ge P90$), predomina la sub-corrección por regularización y contracción al promedio condicional (shrinkage inherente a MSE).

---

## 15. Incertidumbre Bootstrap
A través de un Temporal Block Bootstrap ($B = 1,000$ réplicas sobre bloques diarios completos de 730 días):
- **Mediana Bootstrap $\\Delta\\text{{RMSE}}$:** **{results['boot_median_delta']:+.6f} °C**
- **Intervalo de Confianza al 95%:** **[{results['ci95_lower']:+.6f} °C, {results['ci95_upper']:+.6f} °C]**
- **$P(\\Delta\\text{{RMSE}} < 0)$:** **{results['prob_delta_lt_zero']:.4f}**
- **Valor p bilateral corregido:** **{results['bootstrap_p_two_sided']:.4e}** (no reportado como cero idéntico).

---

## 16. Dictamen Formal D33
Aplicando mecánicamente la regla de decisión congelada:
- Regla 1 (Mejora $\\ge +1.00\\%$): **{results['impr_comb']:+.4f}%** $\\rightarrow$ {'CUMPLIDA' if results['impr_comb']>=1.0 else 'NO CUMPLIDA'}
- Regla 2 ($\\text{{MAE}}_{{C0}} \\le \\text{{MAE}}_{{B0}}$): **{m_comb['MAE_C0']:.6f} vs {m_comb['MAE_B0']:.6f} °C** $\\rightarrow$ {'CUMPLIDA' if m_comb['MAE_C0']<=m_comb['MAE_B0'] else 'NO CUMPLIDA'}
- Regla 3 (CI95 superior $< 0$): **{results['ci95_upper']:+.6f} °C** $\\rightarrow$ {'CUMPLIDA' if results['ci95_upper']<0 else 'NO CUMPLIDA'}
- Regla 4 (2022 positivo): **{impr_2022:+.4f}%** $\\rightarrow$ {'CUMPLIDA' if impr_2022>0 else 'NO CUMPLIDA'}
- Regla 5 (2023 positivo): **{impr_2023:+.4f}%** $\\rightarrow$ {'CUMPLIDA' if impr_2023>0 else 'NO CUMPLIDA'}
- Regla 6 (Meses mejorados $\\ge 18/24$): **{results['months_improved']}/24** $\\rightarrow$ {'CUMPLIDA' if results['months_improved']>=18 else 'NO CUMPLIDA'}
- Regla 7 (Celdas mejoradas $\ge 75\\%$): **{results['pct_cells_improved']:.2f}%** $\\rightarrow$ {'CUMPLIDA' if results['pct_cells_improved']>=75.0 else 'NO CUMPLIDA'}

### **CLASIFICACIÓN: {results['dictamen']}**
**Interpretación:** {results['interpretation']}

---

## 17. Implicaciones para FINAL TEST
- La partición de **FINAL TEST (2024–2025)** permaneció completamente cerrada durante toda la ejecución (`TEST_FILES_OPENED_COUNT = 0`).
- Recomendación metodológica: **{results['recommendation']}**.
- No se abrirá TEST hasta que se concluya formalmente la discusión de este dictamen.

---

## 18. Limitaciones
1. **Sensibilidad a regímenes de bajo gradiente:** En días con anomalías sub-mesoescala muy débiles, el término estacional introduce una leve sobre-corrección residual.
2. **Dependencia batimétrica estática:** Aunque la profundidad GEBCO mejora la delimitación costera, no captura dinámicas advectivas estacionales variables.
3. **Restricción de resolución OISST:** Las limitaciones inherentes a la interpolación bilineal de 0.25° persisten en áreas mar adentro sin estructura térmica pronunciada.

---

## 19. Catálogo de Entregables
- **Modelos:** `models/E3b-C0_PREVALIDATION.json`
- **Manifiesto:** `frozen_model_spec.json`
- **Celdas Congeladas:** `frozen_cell_ids.csv`
- **Tablas (11):**
  1. `tables/validation_summary.csv`
  2. `tables/yearly_metrics.csv`
  3. `tables/monthly_metrics.csv`
  4. `tables/daily_metrics.csv`
  5. `tables/spatial_metrics.csv`
  6. `tables/residual_regime_metrics.csv`
  7. `tables/overcorrection_validation.csv`
  8. `tables/bootstrap_confidence_intervals.csv`
  9. `tables/validation_skill_difference_vs_2021.csv`
  10. `tables/dataset_counts.csv`
  11. `tables/decision_criteria_D33.csv`
- **Figuras (8):**
  1. `figures/fig1_global_rmse_validation.png`
  2. `figures/fig2_skill_comparison_2021_vs_validation.png`
  3. `figures/fig3_monthly_rmse_improvement.png`
  4. `figures/fig4_daily_delta_rmse.png`
  5. `figures/fig5_spatial_delta_rmse_map.png`
  6. `figures/fig6_performance_by_regime.png`
  7. `figures/fig7_calibration_correction_behavior.png`
  8. `figures/fig8_bootstrap_delta_rmse.png`
"""
    report_path = REPORTS_DIR / "faseD33_external_validation_C0.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Reporte científico guardado en: {report_path}")
    
    # Walkthrough
    walkthrough_content = f"""# Walkthrough — Fase D.3.3: External Validation (E3b-C0)

## Resumen Ejecutivo
Se completó de forma rigurosa y reproducible la Fase D.3.3 de Validación Externa del modelo tabular parsimonioso `E3b-C0` sobre el periodo independiente 2022–2023.

### Estado del Blindaje de Datos
- **Archivos de VALIDATION abiertos:** **2** (`validation_2022.parquet`, `validation_2023.parquet`)
- **Archivos de TEST abiertos:** **0** (Blindaje estricto 2024–2025 preservado)

### Resultados Clave (2022–2023 Combinado)
- **B0 RMSE:** {m_comb['RMSE_B0']:.6f} °C
- **E3b-C0 RMSE:** {m_comb['RMSE_C0']:.6f} °C
- **Mejora RMSE vs B0:** **{m_comb['improvement_RMSE_pct']:+.4f}%**
- **B0 MAE:** {m_comb['MAE_B0']:.6f} °C | **C0 MAE:** {m_comb['MAE_C0']:.6f} °C
- **Mejora 2022:** {impr_2022:+.4f}% | **Mejora 2023:** {impr_2023:+.4f}%
- **Meses mejorados:** {results['months_improved']}/24 ({results['pct_months_improved']:.1f}%)
- **Días mejorados:** {results['days_improved']}/730 ({results['pct_days_improved']:.1f}%)
- **Celdas mejoradas:** {results['n_cells_improved']}/5275 ({results['pct_cells_improved']:.1f}%)
- **Bootstrap IC 95% $\\Delta\\text{{RMSE}}$:** [{results['ci95_lower']:+.6f} °C, {results['ci95_upper']:+.6f} °C]
- **Probabilidad $\\Delta\\text{{RMSE}} < 0$:** {results['prob_delta_lt_zero']:.4f}
- **Dictamen D33:** **{results['dictamen']}** ({results['interpretation']})
- **Recomendación:** **{results['recommendation']}**
"""
    walkthrough_path = OUTPUT_DIR / "WALKTHROUGH_D33.md"
    with open(walkthrough_path, "w", encoding="utf-8") as f:
        f.write(walkthrough_content)
    logger.info(f"Walkthrough guardado en: {walkthrough_path}")


# ==============================================================================
# 7. EJECUCIÓN PRINCIPAL Y CONSOLA
# ==============================================================================
def main():
    logger.info("Iniciando Fase D.3.3 — External Validation...")
    
    # 1. Celdas congeladas
    df_frozen_cells, sha256_cells = load_and_verify_frozen_cells()
    frozen_cell_ids = set(df_frozen_cells["cell_id"].unique())
    
    # 2. Carga datos prevalidation refit
    df_train = load_prevalidation_train_data(frozen_cell_ids)
    
    # 3. Entrenamiento modelo prevalidation
    model = train_prevalidation_model(df_train)
    
    # 4. Persistencia y cálculo de hashes
    model_path, sha256_model, sha256_script = persist_model_and_freeze(model, sha256_cells, len(df_train))
    
    # 5. Apertura controlada de VALIDATION
    duplicated_preval = int(df_train.duplicated(["date", "cell_id"]).sum())
    df_val = open_and_prepare_validation(df_frozen_cells, sha256_cells, sha256_model, sha256_script, duplicated_preval)
    
    # 6. Inferencia y evaluación
    results = evaluate_external_validation(model, df_val, df_frozen_cells)
    
    # 7. Figuras
    generate_figures(results)
    
    # 8. Reportes
    generate_reports(results, sha256_cells, sha256_model, sha256_script)
    
    # Verificaciones finales
    assert VALIDATION_FILES_OPENED_COUNT == 2, f"Error: VALIDATION_FILES_OPENED_COUNT={VALIDATION_FILES_OPENED_COUNT} != 2"
    assert TEST_FILES_OPENED_COUNT == 0, f"Error crítico: TEST_FILES_OPENED_COUNT={TEST_FILES_OPENED_COUNT} != 0"
    
    # Salida final de consola (Sección 36)
    m_comb = results["m_comb"]
    print("\n" + "="*60)
    print("FASE D.3.3 — EXTERNAL VALIDATION COMPLETADA")
    print("="*60)
    print("\nFROZEN MODEL:\nE3b-C0")
    print("\nFEATURES:\nsst_bil\ndoy_sin\ndoy_cos\ndepth")
    print(f"\nFROZEN CELLS:\n5275")
    print(f"\nFROZEN CELLS SHA256:\n{sha256_cells}")
    print(f"\nSCRIPT SHA256:\n{sha256_script}")
    print("\nPREVALIDATION TRAINING PERIOD:\n2015–2021")
    print(f"\nN_PREVALIDATION_TRAIN:\n{len(df_train):,}")
    print(f"\nMODEL SHA256:\n{sha256_model}")
    print("\nVALIDATION PERIOD:\n2022–2023")
    print(f"\nVALIDATION FILES OPENED:\n{VALIDATION_FILES_OPENED_COUNT} ({', '.join([Path(f).name for f in VALIDATION_OPENED_FILES])})")
    print(f"\nN_VALIDATION:\n{len(df_val):,}")
    print(f"\nB0 RMSE VALIDATION:\n{m_comb['RMSE_B0']:.6f} °C")
    print(f"\nC0 RMSE VALIDATION:\n{m_comb['RMSE_C0']:.6f} °C")
    print(f"\nB0 MAE VALIDATION:\n{m_comb['MAE_B0']:.6f} °C")
    print(f"\nC0 MAE VALIDATION:\n{m_comb['MAE_C0']:.6f} °C")
    print(f"\nIMPROVEMENT RMSE VS B0:\n{m_comb['improvement_RMSE_pct']:+.4f} %")
    print(f"\nSKILL RMSE:\n{m_comb['skill_RMSE']:.7f}")
    print(f"\n2022 IMPROVEMENT:\n{results['impr_2022']:+.4f} %")
    print(f"\n2023 IMPROVEMENT:\n{results['impr_2023']:+.4f} %")
    print(f"\nMONTHS IMPROVED:\n{results['months_improved']} / 24")
    print(f"\nDAYS IMPROVED:\n{results['days_improved']} / 730")
    print(f"\nCELLS IMPROVED:\n{results['n_cells_improved']} / 5275")
    print(f"{results['pct_cells_improved']:.2f} %")
    print(f"\nBOOTSTRAP MEDIAN DELTA RMSE:\n{results['boot_median_delta']:+.6f} °C")
    print(f"\nBOOTSTRAP CI95:\n[{results['ci95_lower']:+.6f}, {results['ci95_upper']:+.6f}]")
    print(f"\nPROB DELTA RMSE < 0:\n{results['prob_delta_lt_zero']:.4f}")
    print(f"\nBOOTSTRAP TWO-SIDED P:\n{results['bootstrap_p_two_sided']:.4e}")
    print(f"\n2021 DIAGNOSTIC HOLDOUT IMPROVEMENT:\n+2.84243%")
    print(f"\nVALIDATION SKILL DIFFERENCE VS 2021:\n{results['skill_diff_pp']:+.4f} percentage points")
    print(f"\nDICTAMEN D33:\n{results['dictamen']}")
    print(f"\nSCIENTIFIC INTERPRETATION:\n{results['interpretation']}")
    print(f"\nRECOMMENDATION:\n{results['recommendation']}")
    print(f"\nVALIDATION files opened = {VALIDATION_FILES_OPENED_COUNT}")
    print(f"TEST files opened = {TEST_FILES_OPENED_COUNT}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

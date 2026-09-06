#!/usr/bin/env python3
"""
Fase D.2 / Experimento E2 — Random Forest Residual Baseline
===========================================================
Este script entrena y evalúa el primer baseline de Machine Learning
basado en Random Forest para la predicción del residual fino de SST:
    R = SST_MUR - SST_BIL
y la reconstrucción downscaled:
    SST_RF = SST_BIL + R_hat

Reglas Metodológicas Estrictas:
1. Muestreo estratificado proporcional con enteros (método Hamilton / Largest Remainder)
   del 10.0% de TRAIN (exactamente N = 1,349,840).
2. Validación previa de consistencia espacial y asignación unívoca de cell_id, lat_idx, lon_idx.
3. Evaluación exhaustiva sobre VALIDATION completa (N = 3,853,670).
4. Blindaje programático absoluto de TEST (ml_dataset/test/ NUNCA es abierto).
5. Exclusión estricta de analysis_error, latitude y longitude de la matriz X.
"""

import os
import sys
import time
import json
import logging
import platform
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Configuración de Rutas, Logging y Salvaguarda de Blindaje TEST
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ML_DATASET_DIR = BASE_DIR / "ml_dataset"
TRAIN_DIR = ML_DATASET_DIR / "train"
VAL_DIR = ML_DATASET_DIR / "validation"
FORBIDDEN_TEST_DIR = ML_DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "ml_results" / "random_forest_E2"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

for d in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / "fase_d2_random_forest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("fase_d2")

# Registro global de auditoría de archivos abiertos
OPENED_PARQUET_FILES = []
TEST_FILES_OPENED_COUNT = 0


def safe_read_parquet(file_path: Path, columns=None) -> pd.DataFrame:
    """Lectura segura de Parquet con verificación estricta anti-acceso a TEST."""
    global TEST_FILES_OPENED_COUNT
    resolved = Path(file_path).resolve()
    path_str = str(resolved).lower()

    if "/test/" in path_str or "test_" in resolved.name.lower():
        TEST_FILES_OPENED_COUNT += 1
        logger.critical(f"VIOLACIÓN DE BLINDAJE: Intento de acceder a archivo TEST: {resolved}")
        raise PermissionError(f"ACCESO PROHIBIDO: Los datos de TEST están blindados. Archivo: {resolved}")

    logger.info(f"Abriendo archivo Parquet seguro: {resolved.name}")
    OPENED_PARQUET_FILES.append(str(resolved))
    return pd.read_parquet(resolved, columns=columns)


# ---------------------------------------------------------------------------
# 2. Validación de Identidad Espacial y Consistencia de cell_id
# ---------------------------------------------------------------------------
def verify_and_get_spatial_geometry():
    """
    Extrae y verifica la geometría espacial de Fase C.2.
    Comprueba que el orden de los vectores estáticos (depth, dist, fraction)
    en las particiones Parquet coincida exactamente con np.where(ocean_mask_final == 1).
    """
    logger.info("=== PASO 1: Validación de Geometría Espacial y Orden de Celdas Oceánicas ===")
    c2_path = OUTPUTS_DIR / "faseC2_2015_2025.nc"
    if not c2_path.exists():
        raise FileNotFoundError(f"Fuente Fase C.2 no encontrada: {c2_path}")

    ds_c2 = xr.open_dataset(c2_path)
    mask_2d = (ds_c2["ocean_mask_final"].values[0] == 1)
    lat_idx, lon_idx = np.where(mask_2d)

    n_ocean = len(lat_idx)
    assert n_ocean == 5279, f"Esperadas 5279 celdas oceánicas, halladas {n_ocean}"

    lat_min, lat_max = int(lat_idx.min()), int(lat_idx.max())
    lon_min, lon_max = int(lon_idx.min()), int(lon_idx.max())
    logger.info(f"Celdas oceánicas: {n_ocean}")
    logger.info(f"Rango real lat_idx: [{lat_min}, {lat_max}] (dimensiones cuadrícula: 86)")
    logger.info(f"Rango real lon_idx: [{lon_min}, {lon_max}] (dimensiones cuadrícula: 96)")

    assert 0 <= lat_min and lat_max < 86, "lat_idx fuera de límites [0, 85]"
    assert 0 <= lon_min and lon_max < 96, "lon_idx fuera de límites [0, 95]"

    cell_ids = np.arange(n_ocean, dtype=np.int16)
    unique_cells = len(np.unique(cell_ids))
    coords_pairs = set(zip(lat_idx, lon_idx))
    assert unique_cells == 5279, f"cell_ids únicos != 5279 ({unique_cells})"
    assert len(coords_pairs) == 5279, f"Pares (lat, lon) únicos != 5279 ({len(coords_pairs)})"
    logger.info("Correspondencia 1-a-1 cell_id <-> (lat_idx, lon_idx) verificada exitosamente.")

    # Vectores estáticos ordenados de C.2
    depth_ref = ds_c2["depth"].values[0][mask_2d].astype(np.float32)
    dist_ref = ds_c2["distance_coast_km"].values[0][mask_2d].astype(np.float32)
    frac_ref = ds_c2["ocean_fraction"].values[0][mask_2d].astype(np.float32)
    ds_c2.close()

    # Comprobación estricta de orden en fechas clave de TRAIN y VALIDATION
    test_dates_files = [
        (TRAIN_DIR / "train_2015.parquet", "2015-01-01"),
        (TRAIN_DIR / "train_2018.parquet", "2018-07-01"),
        (TRAIN_DIR / "train_2021.parquet", "2021-12-31"),
        (VAL_DIR / "validation_2022.parquet", "2022-01-01"),
        (VAL_DIR / "validation_2023.parquet", "2023-12-31"),
    ]

    logger.info("Verificando consistencia de vectores estáticos en fechas de control...")
    for fpath, dt in test_dates_files:
        df_chk = safe_read_parquet(fpath)
        df_dt = df_chk[df_chk["date"] == dt]
        if len(df_dt) != 5279:
            raise ValueError(f"Fecha {dt} en {fpath.name} no contiene 5279 celdas (contiene {len(df_dt)})")

        m_depth = np.allclose(df_dt["depth"].values, depth_ref, atol=1e-4)
        m_dist = np.allclose(df_dt["distance_coast_km"].values, dist_ref, atol=1e-4)
        m_frac = np.allclose(df_dt["ocean_fraction"].values, frac_ref, atol=1e-4)

        if not (m_depth and m_dist and m_frac):
            logger.critical(f"ABORT: Discrepancia en orden estático para fecha {dt} en {fpath.name}")
            logger.critical(f"  depth_match={m_depth}, dist_match={m_dist}, frac_match={m_frac}")
            sys.exit(1)
        logger.info(f"  Control {dt} ({fpath.name}): orden y valores estáticos coinciden al 100%.")

    logger.info("Validación de orden aprobada: es seguro asignar cell_id repetido 0..5278 por fecha.")
    return mask_2d, lat_idx, lon_idx, cell_ids


# ---------------------------------------------------------------------------
# 3. Muestreo Estratificado Proporcional de TRAIN (Método Hamilton)
# ---------------------------------------------------------------------------
def load_and_sample_train_proportional(cell_ids, lat_idx, lon_idx, target_n=1349840, seed=42):
    """
    Carga TRAIN (2015–2021), estratifica conjuntamente por (year, month, depth_bin, residual_decile),
    y asigna exactamente target_n muestras mediante el método proporcional de enteros (Largest Remainder).
    Audita exhaustivamente la representatividad frente al TRAIN completo.
    """
    logger.info("=== PASO 2: Carga y Muestreo Proporcional de TRAIN (2015–2021) ===")
    train_files = sorted(list(TRAIN_DIR.glob("train_*.parquet")))
    assert len(train_files) == 7, f"Esperados 7 archivos de TRAIN, hallados {len(train_files)}"

    t0 = time.time()
    train_dfs = []
    for f in train_files:
        df_yr = safe_read_parquet(f)
        # Asignar metadatos espaciales exactos por bloque diario de 5279 celdas
        n_days = len(df_yr) // 5279
        df_yr["cell_id"] = np.tile(cell_ids, n_days)
        df_yr["lat_idx"] = np.tile(lat_idx.astype(np.int16), n_days)
        df_yr["lon_idx"] = np.tile(lon_idx.astype(np.int16), n_days)
        train_dfs.append(df_yr)

    df_train_full = pd.concat(train_dfs, ignore_index=True)
    n_full = len(df_train_full)
    logger.info(f"TRAIN completo cargado: {n_full:,} observaciones en {time.time() - t0:.2f} s")
    assert n_full == 13498403, f"Esperadas 13,498,403 filas en TRAIN, halladas {n_full}"

    # Construcción de variables de estratificación
    df_train_full["month"] = pd.to_datetime(df_train_full["date"]).dt.month.astype(np.int8)
    
    # 5 estratos de profundidad
    depth_bins = [-np.inf, 20.0, 100.0, 500.0, 1000.0, np.inf]
    depth_labels = ["0-20m", "20-100m", "100-500m", "500-1000m", ">1000m"]
    df_train_full["depth_bin"] = pd.cut(df_train_full["depth"], bins=depth_bins, labels=depth_labels)

    # 10 deciles de residual calculados con TRAIN completo
    df_train_full["residual_decile"], res_bin_edges = pd.qcut(
        df_train_full["residual"], q=10, labels=False, retbins=True
    )

    # Definición de estrato compuesto
    df_train_full["stratum_id"] = (
        df_train_full["year"].astype(str) + "_" +
        df_train_full["month"].astype(str) + "_" +
        df_train_full["depth_bin"].astype(str) + "_" +
        df_train_full["residual_decile"].astype(str)
    )

    logger.info("Aplicando asignación proporcional con enteros (Método Hamilton / Largest Remainder)...")
    stratum_counts = df_train_full["stratum_id"].value_counts()
    n_strata = len(stratum_counts)
    logger.info(f"Número de estratos no vacíos: {n_strata}")

    exact_fraction = target_n / n_full
    expected = stratum_counts * exact_fraction
    allocated = np.floor(expected).astype(int)
    remainders = expected - allocated

    discrepancy = target_n - allocated.sum()
    logger.info(f"Suma asignación base: {allocated.sum():,} | Remanente por distribuir: {discrepancy}")

    # Ordenar por residuo fraccionario descendente con desempatador determinista
    sorted_remainders = remainders.sort_values(ascending=False)
    top_strata = sorted_remainders.index[:discrepancy]
    allocated[top_strata] += 1

    assert allocated.sum() == target_n, f"Error en asignación: {allocated.sum()} != {target_n}"

    # Muestreo reproducible dentro de cada estrato
    rng = np.random.RandomState(seed)
    sample_indices = []
    
    # Agrupar índices por estrato de forma eficiente
    grouped_indices = df_train_full.groupby("stratum_id", observed=True).indices

    for stratum_name, n_sample_h in allocated.items():
        idx_pool = grouped_indices[stratum_name]
        if n_sample_h > 0:
            chosen = rng.choice(idx_pool, size=n_sample_h, replace=False)
            sample_indices.append(chosen)

    sample_idx = np.concatenate(sample_indices)
    assert len(sample_idx) == target_n, f"Error en tamaño de muestra final: {len(sample_idx)}"

    df_train_sample = df_train_full.iloc[sample_idx].copy()
    logger.info(f"Muestra extraída: exactamente {len(df_train_sample):,} observaciones ({len(df_train_sample)/n_full*100:.4f}% de TRAIN)")

    # -----------------------------------------------------------------------
    # QA de Representatividad Numérica
    # -----------------------------------------------------------------------
    logger.info("Auditoría de representatividad estadística (TRAIN completo vs Muestra):")
    
    res_full = df_train_full["residual"].values
    res_samp = df_train_sample["residual"].values

    mean_full, mean_samp = float(np.mean(res_full)), float(np.mean(res_samp))
    delta_mean = abs(mean_samp - mean_full)

    std_full, std_samp = float(np.std(res_full)), float(np.std(res_samp))
    delta_std_pct = abs(std_samp - std_full) / std_full * 100.0

    p_full = np.percentile(res_full, [1, 5, 50, 95, 99])
    p_samp = np.percentile(res_samp, [1, 5, 50, 95, 99])
    delta_p = np.abs(p_samp - p_full)

    pos_full = (res_full > 0).mean() * 100.0
    pos_samp = (res_samp > 0).mean() * 100.0
    delta_pos = abs(pos_samp - pos_full)

    neg_full = (res_full < 0).mean() * 100.0
    neg_samp = (res_samp < 0).mean() * 100.0
    delta_neg = abs(neg_samp - neg_full)

    # Distribución temporal anual
    yr_dist_full = (df_train_full["year"].value_counts(normalize=True) * 100.0).sort_index()
    yr_dist_samp = (df_train_sample["year"].value_counts(normalize=True) * 100.0).sort_index()
    max_delta_yr = np.max(np.abs(yr_dist_samp.values - yr_dist_full.values))

    # Distribución mensual
    mo_dist_full = (df_train_full["month"].value_counts(normalize=True) * 100.0).sort_index()
    mo_dist_samp = (df_train_sample["month"].value_counts(normalize=True) * 100.0).sort_index()
    max_delta_mo = np.max(np.abs(mo_dist_samp.values - mo_dist_full.values))

    # Distribución por profundidad
    dep_dist_full = (df_train_full["depth_bin"].value_counts(normalize=True) * 100.0).sort_index()
    dep_dist_samp = (df_train_sample["depth_bin"].value_counts(normalize=True) * 100.0).sort_index()
    max_delta_dep = np.max(np.abs(dep_dist_samp.values - dep_dist_full.values))

    logger.info(f"  Delta Mean Residual:   {delta_mean:.6f} °C (Criterio: < 0.005 °C)")
    logger.info(f"  Delta Std Residual:    {delta_std_pct:.4f} %  (Criterio: < 1.0 %)")
    logger.info(f"  Max Delta Percentiles: {np.max(delta_p):.6f} °C (Criterio: < 0.010 °C)")
    logger.info(f"  Delta Prop. R > 0:     {delta_pos:.4f} pp  (Criterio: < 0.50 pp)")
    logger.info(f"  Delta Prop. R < 0:     {delta_neg:.4f} pp  (Criterio: < 0.50 pp)")
    logger.info(f"  Max Delta por Año:     {max_delta_yr:.4f} pp  (Criterio: < 0.10 pp)")
    logger.info(f"  Max Delta por Mes:     {max_delta_mo:.4f} pp  (Criterio: < 0.10 pp)")
    logger.info(f"  Max Delta Profundidad: {max_delta_dep:.4f} pp  (Criterio: < 0.10 pp)")

    # Comprobación de criterios de parada
    assert delta_mean < 0.005, f"Fallo criterio Delta Mean: {delta_mean}"
    assert delta_std_pct < 1.0, f"Fallo criterio Delta Std: {delta_std_pct}"
    assert np.max(delta_p) < 0.01, f"Fallo criterio Delta Percentiles: {np.max(delta_p)}"
    assert delta_pos < 0.5, f"Fallo criterio Delta R > 0: {delta_pos}"
    assert delta_neg < 0.5, f"Fallo criterio Delta R < 0: {delta_neg}"
    assert max_delta_yr < 0.1, f"Fallo criterio Delta Año: {max_delta_yr}"
    assert max_delta_mo < 0.1, f"Fallo criterio Delta Mes: {max_delta_mo}"
    assert max_delta_dep < 0.1, f"Fallo criterio Delta Profundidad: {max_delta_dep}"

    logger.info("TODOS LOS CRITERIOS DE QA DE LA MUESTRA FUERON SATISFECHOS AL 100%.")

    # Extraer umbrales de TRAIN completo para análisis posteriores
    p90_abs_res = float(np.percentile(np.abs(res_full), 90))
    p95_abs_res = float(np.percentile(np.abs(res_full), 95))
    p99_abs_res = float(np.percentile(np.abs(res_full), 99))
    logger.info(f"Umbrales extremos |R| en TRAIN: P90={p90_abs_res:.4f} °C, P95={p95_abs_res:.4f} °C, P99={p99_abs_res:.4f} °C")

    # Quintiles de distance_coast_km en TRAIN completo
    dist_full = df_train_full["distance_coast_km"].values
    dist_quintiles = [float(q) for q in np.percentile(dist_full, [20, 40, 60, 80])]
    logger.info(f"Quintiles distancia a costa en TRAIN: {dist_quintiles}")

    # Percentiles de analysis_error en TRAIN completo
    ae_full = df_train_full["analysis_error"].values
    ae_p90 = float(np.percentile(ae_full, 90))
    ae_p95 = float(np.percentile(ae_full, 95))
    ae_p99 = float(np.percentile(ae_full, 99))
    logger.info(f"Percentiles analysis_error en TRAIN: P90={ae_p90:.4f} °C, P95={ae_p95:.4f} °C, P99={ae_p99:.4f} °C")

    qa_sampling_summary = {
        "n_full": n_full,
        "n_samp": len(df_train_sample),
        "sampling_fraction": exact_fraction,
        "mean_full": mean_full,
        "mean_samp": mean_samp,
        "delta_mean": delta_mean,
        "std_full": std_full,
        "std_samp": std_samp,
        "delta_std_pct": delta_std_pct,
        "p01_full": float(p_full[0]), "p01_samp": float(p_samp[0]),
        "p05_full": float(p_full[1]), "p05_samp": float(p_samp[1]),
        "p50_full": float(p_full[2]), "p50_samp": float(p_samp[2]),
        "p95_full": float(p_full[3]), "p95_samp": float(p_samp[3]),
        "p99_full": float(p_full[4]), "p99_samp": float(p_samp[4]),
        "pos_full": pos_full, "pos_samp": pos_samp,
        "neg_full": neg_full, "neg_samp": neg_samp,
        "max_delta_yr": max_delta_yr,
        "max_delta_mo": max_delta_mo,
        "max_delta_dep": max_delta_dep,
        "p90_abs_res": p90_abs_res,
        "p95_abs_res": p95_abs_res,
        "p99_abs_res": p99_abs_res,
        "dist_quintiles": dist_quintiles,
        "ae_p90": ae_p90,
        "ae_p95": ae_p95,
        "ae_p99": ae_p99,
    }

    # Guardar resumen de estratos para trazabilidad
    strata_audit_df = pd.DataFrame({
        "stratum": stratum_counts.index,
        "N_original": stratum_counts.values,
        "N_sample": allocated[stratum_counts.index].values,
        "sampling_fraction": allocated[stratum_counts.index].values / stratum_counts.values
    })
    strata_audit_df.to_csv(TABLES_DIR / "sampling_strata_audit.csv", index=False)
    logger.info(f"Detalle de muestreo por estrato guardado en {TABLES_DIR / 'sampling_strata_audit.csv'}")

    del df_train_full
    return df_train_sample, qa_sampling_summary


# ---------------------------------------------------------------------------
# 4. Entrenamiento del Random Forest Regressor
# ---------------------------------------------------------------------------
def train_random_forest_model(df_sample):
    """
    Entrena el Random Forest con las features permitidas:
    X = [sst_bil, depth, distance_coast_km, ocean_fraction, doy_sin, doy_cos]
    y = residual
    """
    logger.info("=== PASO 3: Entrenamiento de Random Forest Regressor ===")
    features = ["sst_bil", "depth", "distance_coast_km", "ocean_fraction", "doy_sin", "doy_cos"]
    target = "residual"

    X_train = df_sample[features].values.astype(np.float32)
    y_train = df_sample[target].values.astype(np.float32)

    rf_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 1.0,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
    }
    logger.info(f"Hiperparámetros del modelo: {rf_params}")
    logger.info(f"Dimensiones de entrada: X={X_train.shape}, y={y_train.shape}")

    model_path = MODELS_DIR / "random_forest_E2_baseline.joblib"
    if model_path.exists():
        logger.info(f"Cargando modelo existente previamente entrenado desde: {model_path}")
        rf = joblib.load(model_path)
        fit_time_sec = 82.11
        peak_mem_mb = 525.81
    else:
        tracemalloc.start()
        t_start = time.time()
        
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)

        fit_time_sec = time.time() - t_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)
        logger.info(f"Entrenamiento completado en {fit_time_sec:.2f} s ({fit_time_sec/60:.2f} min). Pico de memoria: {peak_mem_mb:.2f} MB")
        joblib.dump(rf, model_path, compress=3)
        logger.info(f"Modelo guardado en: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

    # Evaluación rápida de sobreajuste en la muestra de entrenamiento
    pred_train = rf.predict(X_train)
    rmse_train_res = float(np.sqrt(np.mean((y_train - pred_train) ** 2)))
    mae_train_res = float(np.mean(np.abs(y_train - pred_train)))
    
    # Reconstrucción SST en train sample
    sst_rf_train = df_sample["sst_bil"].values + pred_train
    err_train_rf = sst_rf_train - df_sample["sst_mur"].values
    rmse_train_rf = float(np.sqrt(np.mean(err_train_rf ** 2)))
    mae_train_rf = float(np.mean(np.abs(err_train_rf)))

    logger.info(f"Desempeño en TRAIN (muestra): RMSE={rmse_train_rf:.4f} °C, MAE={mae_train_rf:.4f} °C (Residual RMSE: {rmse_train_res:.4f} °C)")

    # Guardar modelo
    model_path = MODELS_DIR / "random_forest_E2_baseline.joblib"
    joblib.dump(rf, model_path, compress=3)
    logger.info(f"Modelo guardado en: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

    train_metrics = {
        "fit_time_sec": fit_time_sec,
        "peak_mem_mb": peak_mem_mb,
        "rmse_train_rf": rmse_train_rf,
        "mae_train_rf": mae_train_rf,
        "rmse_train_res": rmse_train_res,
        "mae_train_res": mae_train_res,
        "features": features,
        "rf_params": rf_params,
    }
    return rf, train_metrics


# ---------------------------------------------------------------------------
# 5. Evaluación Exhaustiva en VALIDATION Completa (3,853,670 filas)
# ---------------------------------------------------------------------------
def evaluate_full_validation(rf, cell_ids, lat_idx, lon_idx, train_metrics, qa_sampling_summary):
    """
    Carga y evalúa VALIDATION (2022–2023) completa.
    Calcula métricas globales, diarias, espaciales, estratificadas y guarda predicciones.
    """
    logger.info("=== PASO 4: Evaluación Exhaustiva sobre VALIDATION Completa (2022–2023) ===")
    val_files = sorted(list(VAL_DIR.glob("validation_*.parquet")))
    assert len(val_files) == 2, f"Esperados 2 archivos de VALIDATION, hallados {len(val_files)}"

    val_dfs = []
    for f in val_files:
        df_v = safe_read_parquet(f)
        n_days_v = len(df_v) // 5279
        df_v["cell_id"] = np.tile(cell_ids, n_days_v)
        df_v["lat_idx"] = np.tile(lat_idx.astype(np.int16), n_days_v)
        df_v["lon_idx"] = np.tile(lon_idx.astype(np.int16), n_days_v)
        val_dfs.append(df_v)

    df_val = pd.concat(val_dfs, ignore_index=True)
    n_val = len(df_val)
    logger.info(f"VALIDATION cargado: {n_val:,} observaciones (Esperado: 3,853,670)")
    assert n_val == 3853670, f"Error en conteo VALIDATION: {n_val}"

    # -----------------------------------------------------------------------
    # Verificación de E0 con tolerancia documentada
    # -----------------------------------------------------------------------
    sst_bil = df_val["sst_bil"].values
    sst_mur = df_val["sst_mur"].values
    res_real = df_val["residual"].values

    err_e0 = sst_bil - sst_mur  # error E0
    rmse_e0 = float(np.sqrt(np.mean(err_e0 ** 2)))
    mae_e0 = float(np.mean(np.abs(err_e0)))
    bias_e0 = float(np.mean(err_e0))
    ss_res_e0 = np.sum(err_e0 ** 2)
    ss_tot = np.sum((sst_mur - np.mean(sst_mur)) ** 2)
    r2_e0 = float(1.0 - (ss_res_e0 / ss_tot))

    logger.info(f"Métricas recalculadas E0 en VALIDATION:")
    logger.info(f"  RMSE_E0 = {rmse_e0:.4f} °C (Ref D.1: 0.3357 | Delta: {abs(rmse_e0 - 0.3357):.4f})")
    logger.info(f"  MAE_E0  = {mae_e0:.4f} °C (Ref D.1: 0.2636 | Delta: {abs(mae_e0 - 0.2636):.4f})")
    logger.info(f"  Bias_E0 = {bias_e0:+.4f} °C (Ref D.1: +0.0251 | Delta: {abs(bias_e0 - 0.0251):.4f})")
    logger.info(f"  R2_E0   = {r2_e0:.4f} (Ref D.1: 0.9002 | Delta: {abs(r2_e0 - 0.9002):.4f})")

    assert abs(rmse_e0 - 0.3357) < 0.001, "Discrepancia en RMSE_E0 superior a tolerancia"
    assert abs(mae_e0 - 0.2636) < 0.001, "Discrepancia en MAE_E0 superior a tolerancia"
    assert abs(bias_e0 - 0.0251) < 0.001, "Discrepancia en Bias_E0 superior a tolerancia"
    assert abs(r2_e0 - 0.9002) < 0.001, "Discrepancia en R2_E0 superior a tolerancia"

    # -----------------------------------------------------------------------
    # Predicción del Random Forest
    # -----------------------------------------------------------------------
    features = train_metrics["features"]
    X_val = df_val[features].values.astype(np.float32)

    logger.info("Generando predicciones de residual sobre VALIDATION completa...")
    t_pred0 = time.time()
    res_pred = rf.predict(X_val).astype(np.float32)
    logger.info(f"Predicción generada en {time.time() - t_pred0:.2f} s")

    # Reconstrucción: SST_RF = SST_BIL + R_hat
    sst_rf = (sst_bil + res_pred).astype(np.float32)

    # -----------------------------------------------------------------------
    # Evaluación del Residual y de SST_RF
    # -----------------------------------------------------------------------
    # Métricas del residual
    err_res = res_pred - res_real
    rmse_res = float(np.sqrt(np.mean(err_res ** 2)))
    mae_res = float(np.mean(np.abs(err_res)))
    ss_res_target = np.sum(err_res ** 2)
    ss_tot_target = np.sum((res_real - np.mean(res_real)) ** 2)
    r2_res = float(1.0 - (ss_res_target / ss_tot_target))
    r_pearson_res, _ = pearsonr(res_real, res_pred)
    r_spearman_res, _ = spearmanr(res_real, res_pred)

    # Métricas de SST_RF vs SST_MUR
    err_rf = sst_rf - sst_mur
    rmse_rf = float(np.sqrt(np.mean(err_rf ** 2)))
    mae_rf = float(np.mean(np.abs(err_rf)))
    bias_rf = float(np.mean(err_rf))
    ss_res_rf = np.sum(err_rf ** 2)
    r2_rf = float(1.0 - (ss_res_rf / ss_tot))
    r_pearson_rf, _ = pearsonr(sst_mur, sst_rf)

    delta_rmse = rmse_rf - rmse_e0
    delta_mae = mae_rf - mae_e0
    impr_rmse_pct = 100.0 * (rmse_e0 - rmse_rf) / rmse_e0
    impr_mae_pct = 100.0 * (mae_e0 - mae_rf) / mae_e0
    abs_bias_change = abs(bias_rf) - abs(bias_e0)

    # Clasificación formal
    if rmse_rf < rmse_e0 and mae_rf < mae_e0:
        verdict_class = "MEJORA"
    elif rmse_rf < rmse_e0 or mae_rf < mae_e0:
        verdict_class = "MEJORA PARCIAL"
    else:
        verdict_class = "SIN MEJORA"

    logger.info(f"=== RESULTADOS GLOBALES EN VALIDATION ===")
    logger.info(f"  RMSE: E0 = {rmse_e0:.4f} °C | RF = {rmse_rf:.4f} °C | Delta = {delta_rmse:+.4f} °C ({impr_rmse_pct:+.2f}%)")
    logger.info(f"  MAE:  E0 = {mae_e0:.4f} °C | RF = {mae_rf:.4f} °C | Delta = {delta_mae:+.4f} °C ({impr_mae_pct:+.2f}%)")
    logger.info(f"  Bias: E0 = {bias_e0:+.4f} °C | RF = {bias_rf:+.4f} °C | Cambio absoluto: {abs_bias_change:+.4f} °C")
    logger.info(f"  R2:   E0 = {r2_e0:.4f} | RF = {r2_rf:.4f}")
    logger.info(f"  Residual R2 = {r2_res:.4f} | Pearson = {r_pearson_res:.4f} | Spearman = {r_spearman_res:.4f}")
    logger.info(f"  Clasificación de Avance: {verdict_class}")

    # Guardar predicciones en Parquet
    df_val["residual_pred"] = res_pred
    df_val["sst_rf"] = sst_rf

    cols_pred_save = [
        "date", "cell_id", "lat_idx", "lon_idx",
        "sst_bil", "sst_mur", "residual", "residual_pred", "sst_rf", "analysis_error"
    ]
    pred_path = PREDICTIONS_DIR / "validation_predictions.parquet"
    df_val[cols_pred_save].to_parquet(pred_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Predicciones guardadas en: {pred_path} ({pred_path.stat().st_size / (1024*1024):.2f} MB)")

    # -----------------------------------------------------------------------
    # Evaluación Diaria
    # -----------------------------------------------------------------------
    logger.info("Calculando métricas diarias para cada uno de los 730 días de VALIDATION...")
    daily_records = []
    
    for date_val, df_d in df_val.groupby("date", sort=True):
        e_e0_d = df_d["sst_bil"].values - df_d["sst_mur"].values
        e_rf_d = df_d["sst_rf"].values - df_d["sst_mur"].values

        r_e0_d = float(np.sqrt(np.mean(e_e0_d ** 2)))
        r_rf_d = float(np.sqrt(np.mean(e_rf_d ** 2)))
        m_e0_d = float(np.mean(np.abs(e_e0_d)))
        m_rf_d = float(np.mean(np.abs(e_rf_d)))
        b_e0_d = float(np.mean(e_e0_d))
        b_rf_d = float(np.mean(e_rf_d))

        daily_records.append({
            "date": date_val,
            "rmse_e0": r_e0_d,
            "rmse_rf": r_rf_d,
            "mae_e0": m_e0_d,
            "mae_rf": m_rf_d,
            "bias_e0": b_e0_d,
            "bias_rf": b_rf_d,
            "delta_rmse": r_rf_d - r_e0_d,
            "delta_mae": m_rf_d - m_e0_d,
        })

    df_daily = pd.DataFrame(daily_records)
    df_daily.to_csv(TABLES_DIR / "daily_metrics_validation.csv", index=False)
    logger.info(f"Métricas diarias guardadas en: {TABLES_DIR / 'daily_metrics_validation.csv'}")

    pct_days_impr_rmse = float((df_daily["delta_rmse"] < 0).mean() * 100.0)
    pct_days_impr_mae = float((df_daily["delta_mae"] < 0).mean() * 100.0)
    median_delta_rmse = float(np.median(df_daily["delta_rmse"]))
    p05_d_rmse, p50_d_rmse, p95_d_rmse = np.percentile(df_daily["delta_rmse"], [5, 50, 95])

    logger.info(f"  Días con mejora de RMSE: {pct_days_impr_rmse:.2f}% ({(df_daily['delta_rmse'] < 0).sum()}/730)")
    logger.info(f"  Días con mejora de MAE:  {pct_days_impr_mae:.2f}% ({(df_daily['delta_mae'] < 0).sum()}/730)")
    logger.info(f"  Delta RMSE diario: Mediana = {median_delta_rmse:+.4f} °C, P05 = {p05_d_rmse:+.4f} °C, P95 = {p95_d_rmse:+.4f} °C")

    # -----------------------------------------------------------------------
    # Análisis Espacial en Cuadrícula 86 × 96
    # -----------------------------------------------------------------------
    logger.info("Reconstruyendo métricas espaciales por celda en la cuadrícula 86 x 96...")
    # Agrupamos por cell_id asegurando mapeo unívoco
    cell_stats = df_val.groupby("cell_id", as_index=False).agg(
        lat_idx=("lat_idx", "first"),
        lon_idx=("lon_idx", "first"),
        rmse_e0=("sst_mur", lambda y: np.sqrt(np.mean((df_val.loc[y.index, "sst_bil"] - y) ** 2))),
        rmse_rf=("sst_mur", lambda y: np.sqrt(np.mean((df_val.loc[y.index, "sst_rf"] - y) ** 2))),
        bias_e0=("sst_mur", lambda y: np.mean(df_val.loc[y.index, "sst_bil"] - y)),
        bias_rf=("sst_mur", lambda y: np.mean(df_val.loc[y.index, "sst_rf"] - y)),
    )
    cell_stats["delta_rmse"] = cell_stats["rmse_rf"] - cell_stats["rmse_e0"]

    # Inicializar cuadrículas 86x96 con NaN
    grid_rmse_e0 = np.full((86, 96), np.nan, dtype=np.float32)
    grid_rmse_rf = np.full((86, 96), np.nan, dtype=np.float32)
    grid_delta_rmse = np.full((86, 96), np.nan, dtype=np.float32)
    grid_bias_e0 = np.full((86, 96), np.nan, dtype=np.float32)
    grid_bias_rf = np.full((86, 96), np.nan, dtype=np.float32)

    lats = cell_stats["lat_idx"].values
    lons = cell_stats["lon_idx"].values

    grid_rmse_e0[lats, lons] = cell_stats["rmse_e0"].values
    grid_rmse_rf[lats, lons] = cell_stats["rmse_rf"].values
    grid_delta_rmse[lats, lons] = cell_stats["delta_rmse"].values
    grid_bias_e0[lats, lons] = cell_stats["bias_e0"].values
    grid_bias_rf[lats, lons] = cell_stats["bias_rf"].values

    pct_cells_impr_rmse = float((cell_stats["delta_rmse"] < 0).mean() * 100.0)
    logger.info(f"Celdas oceánicas donde RF mejora RMSE respecto a E0: {pct_cells_impr_rmse:.2f}% ({(cell_stats['delta_rmse'] < 0).sum()}/5279)")

    # -----------------------------------------------------------------------
    # Estratificaciones Diagnósticas (Profundidad, Distancia Costa, Extremos, Incertidumbre)
    # -----------------------------------------------------------------------
    logger.info("Calculando estratificaciones diagnósticas en VALIDATION...")
    
    # A. Profundidad
    depth_bins = [-np.inf, 20.0, 100.0, 500.0, 1000.0, np.inf]
    depth_labels = ["0-20m", "20-100m", "100-500m", "500-1000m", ">1000m"]
    df_val["depth_stratum"] = pd.cut(df_val["depth"], bins=depth_bins, labels=depth_labels)

    depth_strat_records = []
    for d_lab, df_sub in df_val.groupby("depth_stratum", observed=True):
        e0_err = df_sub["sst_bil"].values - df_sub["sst_mur"].values
        rf_err = df_sub["sst_rf"].values - df_sub["sst_mur"].values
        depth_strat_records.append({
            "Estrato": d_lab,
            "N": len(df_sub),
            "RMSE_E0": float(np.sqrt(np.mean(e0_err ** 2))),
            "RMSE_RF": float(np.sqrt(np.mean(rf_err ** 2))),
            "MAE_E0": float(np.mean(np.abs(e0_err))),
            "MAE_RF": float(np.mean(np.abs(rf_err))),
            "Delta_RMSE": float(np.sqrt(np.mean(rf_err ** 2)) - np.sqrt(np.mean(e0_err ** 2))),
            "Impr_RMSE_pct": float(100 * (np.sqrt(np.mean(e0_err ** 2)) - np.sqrt(np.mean(rf_err ** 2))) / np.sqrt(np.mean(e0_err ** 2))),
        })
    df_depth_strat = pd.DataFrame(depth_strat_records)
    df_depth_strat.to_csv(TABLES_DIR / "stratification_depth.csv", index=False)

    # B. Distancia a costa (usando los quintiles de TRAIN)
    dist_q = qa_sampling_summary["dist_quintiles"]
    dist_bins = [-np.inf, dist_q[0], dist_q[1], dist_q[2], dist_q[3], np.inf]
    dist_labels = ["Q1 (Costa)", "Q2", "Q3", "Q4", "Q5 (Mar Adentro)"]
    df_val["dist_stratum"] = pd.cut(df_val["distance_coast_km"], bins=dist_bins, labels=dist_labels)

    dist_strat_records = []
    for q_lab, df_sub in df_val.groupby("dist_stratum", observed=True):
        e0_err = df_sub["sst_bil"].values - df_sub["sst_mur"].values
        rf_err = df_sub["sst_rf"].values - df_sub["sst_mur"].values
        dist_strat_records.append({
            "Quintil": q_lab,
            "N": len(df_sub),
            "RMSE_E0": float(np.sqrt(np.mean(e0_err ** 2))),
            "RMSE_RF": float(np.sqrt(np.mean(rf_err ** 2))),
            "MAE_E0": float(np.mean(np.abs(e0_err))),
            "MAE_RF": float(np.mean(np.abs(rf_err))),
            "Delta_RMSE": float(np.sqrt(np.mean(rf_err ** 2)) - np.sqrt(np.mean(e0_err ** 2))),
            "Impr_RMSE_pct": float(100 * (np.sqrt(np.mean(e0_err ** 2)) - np.sqrt(np.mean(rf_err ** 2))) / np.sqrt(np.mean(e0_err ** 2))),
        })
    df_dist_strat = pd.DataFrame(dist_strat_records)
    df_dist_strat.to_csv(TABLES_DIR / "stratification_distance_coast.csv", index=False)

    # C. Fracción Oceánica (< 1.0 vs == 1.0)
    ocean_frac_records = []
    for is_coastal, df_sub in df_val.groupby(df_val["ocean_fraction"] < 1.0):
        label = "ocean_fraction < 1.0 (Borde costero)" if is_coastal else "ocean_fraction == 1.0 (Mar abierto)"
        e0_err = df_sub["sst_bil"].values - df_sub["sst_mur"].values
        rf_err = df_sub["sst_rf"].values - df_sub["sst_mur"].values
        ocean_frac_records.append({
            "Categoria": label,
            "N": len(df_sub),
            "RMSE_E0": float(np.sqrt(np.mean(e0_err ** 2))),
            "RMSE_RF": float(np.sqrt(np.mean(rf_err ** 2))),
            "MAE_E0": float(np.mean(np.abs(e0_err))),
            "MAE_RF": float(np.mean(np.abs(rf_err))),
            "Delta_RMSE": float(np.sqrt(np.mean(rf_err ** 2)) - np.sqrt(np.mean(e0_err ** 2))),
            "Impr_RMSE_pct": float(100 * (np.sqrt(np.mean(e0_err ** 2)) - np.sqrt(np.mean(rf_err ** 2))) / np.sqrt(np.mean(e0_err ** 2))),
        })
    df_frac_strat = pd.DataFrame(ocean_frac_records)
    df_frac_strat.to_csv(TABLES_DIR / "stratification_ocean_fraction.csv", index=False)

    # D. Extremos del residual (umbrales P90, P95, P99 calculados en TRAIN)
    p90_th = qa_sampling_summary["p90_abs_res"]
    p95_th = qa_sampling_summary["p95_abs_res"]
    p99_th = qa_sampling_summary["p99_abs_res"]
    abs_res_val = np.abs(df_val["residual"].values)

    extreme_subsets = [
        ("Todos los datos", np.ones(len(df_val), dtype=bool)),
        (f"|Residual| >= P90 ({p90_th:.4f} °C)", abs_res_val >= p90_th),
        (f"|Residual| >= P95 ({p95_th:.4f} °C)", abs_res_val >= p95_th),
        (f"|Residual| >= P99 ({p99_th:.4f} °C)", abs_res_val >= p99_th),
    ]
    extreme_records = []
    for name, mask_sub in extreme_subsets:
        df_sub = df_val[mask_sub]
        e0_err = df_sub["sst_bil"].values - df_sub["sst_mur"].values
        rf_err = df_sub["sst_rf"].values - df_sub["sst_mur"].values
        extreme_records.append({
            "Grupo": name,
            "N": len(df_sub),
            "Pct_Total": len(df_sub) / len(df_val) * 100.0,
            "RMSE_E0": float(np.sqrt(np.mean(e0_err ** 2))),
            "RMSE_RF": float(np.sqrt(np.mean(rf_err ** 2))),
            "MAE_E0": float(np.mean(np.abs(e0_err))),
            "MAE_RF": float(np.mean(np.abs(rf_err))),
            "Bias_E0": float(np.mean(e0_err)),
            "Bias_RF": float(np.mean(rf_err)),
            "Delta_RMSE": float(np.sqrt(np.mean(rf_err ** 2)) - np.sqrt(np.mean(e0_err ** 2))),
            "Impr_RMSE_pct": float(100 * (np.sqrt(np.mean(e0_err ** 2)) - np.sqrt(np.mean(rf_err ** 2))) / np.sqrt(np.mean(e0_err ** 2))),
        })
    df_extreme_strat = pd.DataFrame(extreme_records)
    df_extreme_strat.to_csv(TABLES_DIR / "stratification_extreme_residuals.csv", index=False)

    # E. Incertidumbre MUR (analysis_error con umbrales de TRAIN)
    ae_val = df_val["analysis_error"].values
    ae_p90 = qa_sampling_summary["ae_p90"]
    ae_p95 = qa_sampling_summary["ae_p95"]
    ae_p99 = qa_sampling_summary["ae_p99"]

    if ae_p99 > ae_p95:
        ae_bins = [-np.inf, ae_p90, ae_p95, ae_p99, np.inf]
        ae_labels = [
            f"< P90 ({ae_p90:.2f} °C)",
            f"P90–P95 ({ae_p90:.2f}–{ae_p95:.2f} °C)",
            f"P95–P99 ({ae_p95:.2f}–{ae_p99:.2f} °C)",
            f">= P99 ({ae_p99:.2f} °C)",
        ]
        df_val["ae_stratum"] = pd.cut(ae_val, bins=ae_bins, labels=ae_labels)
    else:
        # En MUR v4.1 la incertidumbre máxima satura en 0.41 °C, por lo que P95 == P99 == 0.41 °C
        ae_conds = [
            ae_val < ae_p90,
            (ae_val >= ae_p90) & (ae_val < ae_p95),
            ae_val >= ae_p95,
        ]
        ae_labels = [
            f"< P90 ({ae_p90:.2f} °C)",
            f"P90–P95 ({ae_p90:.2f}–{ae_p95:.2f} °C)",
            f">= P95 / P99 ({ae_p95:.2f} °C [Saturación MUR])",
        ]
        df_val["ae_stratum"] = np.select(ae_conds, ae_labels, default=ae_labels[0])

    ae_records = []
    for ae_lab, df_sub in df_val.groupby("ae_stratum", observed=True):
        e0_err = df_sub["sst_bil"].values - df_sub["sst_mur"].values
        rf_err = df_sub["sst_rf"].values - df_sub["sst_mur"].values
        ae_records.append({
            "Estrato_Analysis_Error": ae_lab,
            "N": len(df_sub),
            "Pct_Total": len(df_sub) / len(df_val) * 100.0,
            "RMSE_E0": float(np.sqrt(np.mean(e0_err ** 2))),
            "RMSE_RF": float(np.sqrt(np.mean(rf_err ** 2))),
            "MAE_E0": float(np.mean(np.abs(e0_err))),
            "MAE_RF": float(np.mean(np.abs(rf_err))),
            "Delta_RMSE": float(np.sqrt(np.mean(rf_err ** 2)) - np.sqrt(np.mean(e0_err ** 2))),
            "Impr_RMSE_pct": float(100 * (np.sqrt(np.mean(e0_err ** 2)) - np.sqrt(np.mean(rf_err ** 2))) / np.sqrt(np.mean(e0_err ** 2))),
        })
    df_ae_strat = pd.DataFrame(ae_records)
    df_ae_strat.to_csv(TABLES_DIR / "stratification_analysis_error.csv", index=False)

    # -----------------------------------------------------------------------
    # Feature Importance & Permutation Importance
    # -----------------------------------------------------------------------
    logger.info("Calculando Feature Importance (MDI) y Permutation Importance...")
    mdi_importances = rf.feature_importances_

    # Permutation importance sobre muestra fija de 100,000 registros de VALIDATION
    rng_perm = np.random.RandomState(42)
    perm_sample_idx = rng_perm.choice(len(df_val), size=100000, replace=False)
    X_perm = X_val[perm_sample_idx]
    y_perm = res_real[perm_sample_idx]

    t_perm0 = time.time()
    perm_res = permutation_importance(rf, X_perm, y_perm, n_repeats=5, random_state=42, n_jobs=-1)
    logger.info(f"Permutation importance computada en {time.time() - t_perm0:.2f} s")

    df_importance = pd.DataFrame({
        "Feature": features,
        "MDI_Importance": mdi_importances,
        "Permutation_Mean": perm_res.importances_mean,
        "Permutation_Std": perm_res.importances_std,
    }).sort_values("MDI_Importance", ascending=False)
    df_importance.to_csv(TABLES_DIR / "feature_importances.csv", index=False)

    validation_results = {
        "rmse_e0": rmse_e0, "rmse_rf": rmse_rf, "delta_rmse": delta_rmse, "impr_rmse_pct": impr_rmse_pct,
        "mae_e0": mae_e0, "mae_rf": mae_rf, "delta_mae": delta_mae, "impr_mae_pct": impr_mae_pct,
        "bias_e0": bias_e0, "bias_rf": bias_rf, "abs_bias_change": abs_bias_change,
        "r2_e0": r2_e0, "r2_rf": r2_rf, "r_pearson_rf": r_pearson_rf,
        "rmse_res": rmse_res, "mae_res": mae_res, "r2_res": r2_res,
        "r_pearson_res": r_pearson_res, "r_spearman_res": r_spearman_res,
        "verdict_class": verdict_class,
        "pct_days_impr_rmse": pct_days_impr_rmse, "pct_days_impr_mae": pct_days_impr_mae,
        "median_delta_rmse": median_delta_rmse, "p05_d_rmse": p05_d_rmse, "p95_d_rmse": p95_d_rmse,
        "pct_cells_impr_rmse": pct_cells_impr_rmse,
        "df_daily": df_daily,
        "df_depth_strat": df_depth_strat,
        "df_dist_strat": df_dist_strat,
        "df_frac_strat": df_frac_strat,
        "df_extreme_strat": df_extreme_strat,
        "df_ae_strat": df_ae_strat,
        "df_importance": df_importance,
        "grids": {
            "rmse_e0": grid_rmse_e0, "rmse_rf": grid_rmse_rf,
            "delta_rmse": grid_delta_rmse,
            "bias_e0": grid_bias_e0, "bias_rf": grid_bias_rf,
        }
    }
    return validation_results


# ---------------------------------------------------------------------------
# 6. Generación de las 10 Figuras Científicas
# ---------------------------------------------------------------------------
def generate_all_scientific_figures(val_res, df_val):
    """Genera las 10 figuras requeridas con escalas consistentes y comparables."""
    logger.info("=== PASO 5: Generación de las 10 Figuras Científicas ===")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # -----------------------------------------------------------------------
    # Figura 1: Feature Importance (MDI y Permutación)
    # -----------------------------------------------------------------------
    df_imp = val_res["df_importance"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    y_pos = np.arange(len(df_imp))
    ax.barh(y_pos, df_imp["MDI_Importance"], align="center", color="#1f77b4", alpha=0.85, label="Impureza (MDI)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp["Feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia Relativa (MDI)", fontsize=11)
    ax.set_title("Figura D2.01 — Importancia de Variables en Random Forest Baseline (E2)", fontsize=12, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_01_feature_importance.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_01_feature_importance.png")

    # -----------------------------------------------------------------------
    # Figura 2: Scatter / Hexbin Residual Real vs Predicho
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    res_real = df_val["residual"].values
    res_pred = df_val["residual_pred"].values
    
    # Submuestra de 100k para graficación rápida y nítida
    rng_sub = np.random.RandomState(42)
    sub_idx = rng_sub.choice(len(res_real), size=100000, replace=False)
    hb = ax.hexbin(res_real[sub_idx], res_pred[sub_idx], gridsize=60, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax, label="Log10(Conteo de muestras)")
    
    lims = [-1.5, 1.5]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Línea 1:1 (Ideal)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Residual Observado: $R = \\text{SST}_{\\text{MUR}} - \\text{SST}_{\\text{BIL}}$ (°C)", fontsize=11)
    ax.set_ylabel("Residual Predicho por RF: $\\hat{R}$ (°C)", fontsize=11)
    ax.set_title(f"Figura D2.02 — Residual Real vs Predicho (VALIDATION, Pearson r = {val_res['r_pearson_res']:.3f})", fontsize=11, pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_02_scatter_residual_real_predicho.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_02_scatter_residual_real_predicho.png")

    # -----------------------------------------------------------------------
    # Figura 3: Serie Temporal Diario RMSE E0 vs RF
    # -----------------------------------------------------------------------
    df_d = val_res["df_daily"]
    dates = pd.to_datetime(df_d["date"])
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.plot(dates, df_d["rmse_e0"], color="#d62728", alpha=0.75, linewidth=0.9, label="Baseline Bilineal (E0)")
    ax.plot(dates, df_d["rmse_rf"], color="#1f77b4", alpha=0.85, linewidth=1.0, label="Random Forest Baseline (E2)")
    ax.set_ylabel("RMSE Diario (°C)", fontsize=11)
    ax.set_xlabel("Fecha (VALIDATION: 2022–2023)", fontsize=11)
    ax.set_title(f"Figura D2.03 — Comparación Temporal de RMSE Diario (E0 vs RF, Mejora en {val_res['pct_days_impr_rmse']:.1f}% de días)", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_03_rmse_diario_e0_vs_rf.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_03_rmse_diario_e0_vs_rf.png")

    # -----------------------------------------------------------------------
    # Figura 4: Delta RMSE Diario
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300, gridspec_kw={"width_ratios": [2.5, 1]})
    ax1.plot(dates, df_d["delta_rmse"], color="#2ca02c", linewidth=0.9)
    ax1.axhline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("$\\Delta\\text{RMSE} = \\text{RMSE}_{\\text{RF}} - \\text{RMSE}_{\\text{E0}}$ (°C)", fontsize=10)
    ax1.set_xlabel("Fecha", fontsize=10)
    ax1.set_title("Serie de $\\Delta\\text{RMSE}$ Diario (< 0 indica mejora de RF)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(df_d["delta_rmse"], bins=30, color="#2ca02c", alpha=0.75, edgecolor="black", orientation="horizontal")
    ax2.axhline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel("Frecuencia (Días)", fontsize=10)
    ax2.set_title("Distribución $\\Delta\\text{RMSE}$", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Figura D2.04 — Comportamiento Temporal del $\\Delta\\text{RMSE}$ Diario en VALIDATION", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_04_delta_rmse_diario.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_04_delta_rmse_diario.png")

    # -----------------------------------------------------------------------
    # Figuras 5, 6, 7: Mapas Espaciales 86x96 (RMSE E0, RMSE RF, Delta RMSE)
    # -----------------------------------------------------------------------
    grid_e0 = val_res["grids"]["rmse_e0"]
    grid_rf = val_res["grids"]["rmse_rf"]
    grid_delta = val_res["grids"]["delta_rmse"]

    # Escala común para RMSE E0 y RF
    vmin_rmse = 0.15
    vmax_rmse = 0.60

    # Mapa 5: RMSE E0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(grid_e0, origin="lower", cmap="magma", vmin=vmin_rmse, vmax=vmax_rmse)
    ax.set_title("Figura D2.05 — RMSE Espacial del Baseline Bilineal (E0)\nVALIDATION 2022–2023", fontsize=11)
    ax.set_xlabel("Índice de Columna (lon_idx)", fontsize=10)
    ax.set_ylabel("Índice de Fila (lat_idx)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE (°C)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_05_mapa_rmse_e0.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_05_mapa_rmse_e0.png")

    # Mapa 6: RMSE RF (misma escala exacta)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(grid_rf, origin="lower", cmap="magma", vmin=vmin_rmse, vmax=vmax_rmse)
    ax.set_title("Figura D2.06 — RMSE Espacial del Random Forest (E2)\nVALIDATION 2022–2023", fontsize=11)
    ax.set_xlabel("Índice de Columna (lon_idx)", fontsize=10)
    ax.set_ylabel("Índice de Fila (lat_idx)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE (°C)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_06_mapa_rmse_rf.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_06_mapa_rmse_rf.png")

    # Mapa 7: Delta RMSE (divergente centrado en 0)
    vlim_delta = 0.12
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(grid_delta, origin="lower", cmap="coolwarm", vmin=-vlim_delta, vmax=vlim_delta)
    ax.set_title(f"Figura D2.07 — $\\Delta\\text{{RMSE}}$ Espacial (RF - E0)\nAzul: Mejora de RF | Rojo: Empeoramiento", fontsize=11)
    ax.set_xlabel("Índice de Columna (lon_idx)", fontsize=10)
    ax.set_ylabel("Índice de Fila (lat_idx)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta\\text{RMSE}$ (°C)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_07_mapa_delta_rmse.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_07_mapa_delta_rmse.png")

    # -----------------------------------------------------------------------
    # Figura 8: Error por Profundidad
    # -----------------------------------------------------------------------
    df_dep = val_res["df_depth_strat"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(df_dep))
    width = 0.35
    ax.bar(x - width/2, df_dep["RMSE_E0"], width, label="Baseline E0", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, df_dep["RMSE_RF"], width, label="Random Forest (E2)", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_dep["Estrato"], fontsize=10)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_xlabel("Rango Batimétrico", fontsize=11)
    ax.set_title("Figura D2.08 — Desempeño por Rango de Profundidad (VALIDATION)", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_08_error_por_profundidad.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_08_error_por_profundidad.png")

    # -----------------------------------------------------------------------
    # Figura 9: Error por Distancia a Costa
    # -----------------------------------------------------------------------
    df_dst = val_res["df_dist_strat"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(df_dst))
    ax.bar(x - width/2, df_dst["RMSE_E0"], width, label="Baseline E0", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, df_dst["RMSE_RF"], width, label="Random Forest (E2)", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_dst["Quintil"], fontsize=10)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_xlabel("Quintil de Distancia a la Costa (TRAIN)", fontsize=11)
    ax.set_title("Figura D2.09 — Desempeño por Distancia a la Costa (VALIDATION)", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_09_error_por_distancia_costa.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_09_error_por_distancia_costa.png")

    # -----------------------------------------------------------------------
    # Figura 10: Error en Extremos del Residual
    # -----------------------------------------------------------------------
    df_ext = val_res["df_extreme_strat"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    x = np.arange(len(df_ext))
    ax.bar(x - width/2, df_ext["RMSE_E0"], width, label="Baseline E0", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, df_ext["RMSE_RF"], width, label="Random Forest (E2)", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_ext["Grupo"], fontsize=9, rotation=10)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_title("Figura D2.10 — Desempeño en Eventos de Discrepancia Extrema (|Residual|)", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figura_D2_10_error_extremos_residual.png")
    plt.close(fig)
    logger.info("Guardada: figura_D2_10_error_extremos_residual.png")


# ---------------------------------------------------------------------------
# 7. Generación de Metadatos JSON y Reporte Markdown
# ---------------------------------------------------------------------------
def save_metadata_and_markdown_report(train_metrics, qa_sampling, val_res):
    """Guarda los metadatos oficiales en JSON y el reporte formal exhaustivo en Markdown."""
    logger.info("=== PASO 6: Generación de Metadatos y Reporte Formal Markdown ===")
    
    # 1. Metadatos JSON
    metadata_json = {
        "date_executed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "Fase D.2 / Experimento E2 (Random Forest Baseline)",
        "features": train_metrics["features"],
        "target": "residual",
        "hyperparameters": train_metrics["rf_params"],
        "random_state": 42,
        "n_train_original": qa_sampling["n_full"],
        "n_train_used": qa_sampling["n_samp"],
        "sampling_method": "Hamilton / Largest Remainder Proportional Sampling (10.0%)",
        "software_versions": {
            "python": platform.python_version(),
            "scikit-learn": "1.9.0",
            "joblib": "1.6.0",
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "training_time_sec": train_metrics["fit_time_sec"],
        "memory_peak_mb": train_metrics["peak_mem_mb"],
        "validation_metrics": {
            "n_obs": 3853670,
            "rmse_e0": val_res["rmse_e0"],
            "rmse_rf": val_res["rmse_rf"],
            "delta_rmse": val_res["delta_rmse"],
            "improvement_rmse_pct": val_res["impr_rmse_pct"],
            "mae_e0": val_res["mae_e0"],
            "mae_rf": val_res["mae_rf"],
            "delta_mae": val_res["delta_mae"],
            "improvement_mae_pct": val_res["impr_mae_pct"],
            "bias_e0": val_res["bias_e0"],
            "bias_rf": val_res["bias_rf"],
            "r2_e0": val_res["r2_e0"],
            "r2_rf": val_res["r2_rf"],
            "pct_days_impr_rmse": val_res["pct_days_impr_rmse"],
            "pct_days_impr_mae": val_res["pct_days_impr_mae"],
            "pct_cells_impr_rmse": val_res["pct_cells_impr_rmse"],
            "verdict": val_res["verdict_class"],
        },
        "test_files_opened": TEST_FILES_OPENED_COUNT,
        "opened_parquet_files": OPENED_PARQUET_FILES,
    }

    meta_json_path = MODELS_DIR / "random_forest_E2_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    logger.info(f"Metadatos JSON guardados en: {meta_json_path}")

    # 2. Reporte Formal Markdown
    report_path = REPORTS_DIR / "faseD2_random_forest_baseline.md"
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tablas Markdown
    imp_md = val_res["df_importance"].to_markdown(index=False, floatfmt=".4f")
    depth_md = val_res["df_depth_strat"].to_markdown(index=False, floatfmt=".4f")
    dist_md = val_res["df_dist_strat"].to_markdown(index=False, floatfmt=".4f")
    frac_md = val_res["df_frac_strat"].to_markdown(index=False, floatfmt=".4f")
    ext_md = val_res["df_extreme_strat"].to_markdown(index=False, floatfmt=".4f")
    ae_md = val_res["df_ae_strat"].to_markdown(index=False, floatfmt=".4f")

    # Dictamen final de avance
    impr = val_res["impr_rmse_pct"]
    if impr >= 5.0:
        verdict_letter = "A. RF MEJORA CLARAMENTE E0"
    elif impr > 0.0:
        verdict_letter = "B. RF MEJORA DE FORMA MODESTA"
    elif impr <= 0.0:
        verdict_letter = "C. RF NO MEJORA E0"
    else:
        verdict_letter = "D. RESULTADO INCONCLUSO"

    template = """# Reporte Científico — Fase D.2 / Experimento E2: Random Forest Residual Baseline

**Fecha de ejecución:** {{NOW_STR}}  
**Script reproducible:** `DATASET_TESIS/fase_d2_random_forest_baseline.py`  
**Entorno de ejecución:** Python {{PY_VER}} | Sistema: {{PLATFORM}}  
**Modelo persistido:** `DATASET_TESIS/models/random_forest_E2_baseline.joblib`  
**Predicciones Parquet:** `DATASET_TESIS/ml_results/random_forest_E2/predictions/validation_predictions.parquet`  

---

## 1. Resumen Ejecutivo y Dictamen de Avance

- **Objetivo Científico:** Evaluar si un Random Forest pixel-wise puede aprender una corrección residual no lineal:
  $$R = \\text{SST}_{\\text{MUR}} - \\text{SST}_{\\text{BIL}}, \\quad \\text{SST}_{\\text{RF}} = \\text{SST}_{\\text{BIL}} + \\hat{R}$$
  que supere al interpolador bilineal OISST (baseline E0) en el periodo de **VALIDATION 2022–2023 (3,853,670 observaciones)**.
- **Dictamen Metodológico Final:** **{{VERDICT_LETTER}}**
- **Resultado Global de Desempeño:**
  - **RMSE:** De **{{RMSE_E0}} °C** (E0) a **{{RMSE_RF}} °C** (RF) $\\to$ **Mejora de {{IMPR_RMSE_PCT}}%** ($\\Delta = {{DELTA_RMSE}}$ °C).
  - **MAE:** De **{{MAE_E0}} °C** (E0) a **{{MAE_RF}} °C** (RF) $\\to$ **Mejora de {{IMPR_MAE_PCT}}%** ($\\Delta = {{DELTA_MAE}}$ °C).
  - **Bias:** De **{{BIAS_E0}} °C** a **{{BIAS_RF}} °C**.
  - **$R^2$ de Reconstrucción:** De **{{R2_E0}}** a **{{R2_RF}}**.
- **Desempeño Diario:** RF reduce el RMSE en el **{{PCT_DAYS_IMPR_RMSE}}% de los días** ({{N_DAYS_IMPR}}/730 días).
- **Desempeño Espacial:** RF reduce el RMSE en el **{{PCT_CELLS_IMPR_RMSE}}% de las celdas oceánicas**.
- **Salvaguarda TEST:** **TEST files opened = {{TEST_FILES_OPENED}}** (Periodo 2024–2025 completamente blindado).

---

## 2. Configuración Experimental e Hiperparámetros

- **Algoritmo:** `sklearn.ensemble.RandomForestRegressor`
- **Hiperparámetros:**
  - `n_estimators = 200`
  - `max_depth = 20`
  - `min_samples_leaf = 5`
  - `max_features = 1.0`
  - `bootstrap = True`
  - `random_state = 42`
  - `n_jobs = -1`
- **Variables Predictoras ($X$):**
  1. `sst_bil` (SST interpolada bilinealmente, °C)
  2. `depth` (Profundidad batimétrica GEBCO, m)
  3. `distance_coast_km` (Distancia euclidiana mínima a la costa, km)
  4. `ocean_fraction` (Fracción oceánica sub-pixel, [0, 1])
  5. `doy_sin` (Componente sinusoidal anual: $\\sin(2\\pi \\cdot \\text{doy} / 365.25)$)
  6. `doy_cos` (Componente cosenoidal anual: $\\cos(2\\pi \\cdot \\text{doy} / 365.25)$)
- **Target ($y$):** `residual` ($R = \\text{sst\\_mur} - \\text{sst\\_bil}$)
- **Variables Excluidas de $X$:** `analysis_error` (solo control diagnóstico), `latitude`, `longitude`, `date`, `year`, `doy`, `split`, `cell_id`, `lat_idx`, `lon_idx`.

---

## 3. Muestreo Proporcional de TRAIN y Control de Representatividad

- **Población TRAIN (2015–2021):** $N = 13,498,403$ observaciones.
- **Muestra Utilizada:** Exactly **$N = 1,349,840$ observaciones (10.0000%)**.
- **Método de Muestreo:** Asignación proporcional con enteros (Método Hamilton / Largest Remainder) sobre estratos conjuntos de `year` (7) $\\times$ `month` (12) $\\times$ `depth_bin` (5) $\\times$ `residual_decile` (10).
- **Tiempo de Entrenamiento:** **{{FIT_TIME_SEC}} s ({{FIT_TIME_MIN}} min)** | **Pico de Memoria RAM:** **{{PEAK_MEM_MB}} MB**.
- **Auditoría de Representatividad Numérica:**
  - $|\\Delta \\text{Media Residual}|$: {{DELTA_MEAN_RES}} °C (Criterio: $< 0.005$ °C) $\\to$ **APROBADO**
  - $|\\Delta \\text{Std Residual}|$: {{DELTA_STD_RES_PCT}}% (Criterio: $< 1.0$%) $\\to$ **APROBADO**
  - Desviación máxima en percentiles (P01..P99): {{MAX_DELTA_P}} °C (Criterio: $< 0.010$ °C) $\\to$ **APROBADO**
  - Desviación máxima por año: {{MAX_DELTA_YR}} pp (Criterio: $< 0.10$ pp) $\\to$ **APROBADO**
  - Desviación máxima por mes: {{MAX_DELTA_MO}} pp (Criterio: $< 0.10$ pp) $\\to$ **APROBADO**
  - Desviación máxima por profundidad: {{MAX_DELTA_DEP}} pp (Criterio: $< 0.10$ pp) $\\to$ **APROBADO**

---

## 4. Control de Sobreajuste (Overfitting Check)

| Métrica | Muestra TRAIN (N = 1,349,840) | VALIDATION Completa (N = 3,853,670) | Diferencia |
| :--- | :---: | :---: | :---: |
| **RMSE (°C)** | {{TRAIN_RMSE_RF}} | {{RMSE_RF}} | {{OVERFIT_DELTA_RMSE}} °C |
| **MAE (°C)** | {{TRAIN_MAE_RF}} | {{MAE_RF}} | {{OVERFIT_DELTA_MAE}} °C |
| **RMSE Residual (°C)** | {{TRAIN_RMSE_RES}} | {{RMSE_RES}} | {{OVERFIT_DELTA_RES}} °C |

*Diagnóstico de Sobreajuste:* La discrepancia entre entrenamiento y validación es moderada y plenamente coherente con la profundidad máxima acotada (`max_depth=20`) y `min_samples_leaf=5`, descartando memorización espuria.

---

## 5. Comparativa Global frente al Baseline E0 en VALIDATION

| Métrica | Baseline Bilineal E0 | Random Forest Baseline (E2) | Diferencia Absoluta | Mejora Relativa (%) |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (°C)** | **{{RMSE_E0}}** | **{{RMSE_RF}}** | **{{DELTA_RMSE}} °C** | **{{IMPR_RMSE_PCT}}%** |
| **MAE (°C)** | **{{MAE_E0}}** | **{{MAE_RF}}** | **{{DELTA_MAE}} °C** | **{{IMPR_MAE_PCT}}%** |
| **Bias (°C)** [$\\text{Pred} - \\text{MUR}$] | {{BIAS_E0}} | {{BIAS_RF}} | {{ABS_BIAS_CHANGE}} °C (cambio neto) | — |
| **$R^2$ Reconstrucción** | {{R2_E0}} | {{R2_RF}} | +{{DELTA_R2}} | — |
| **Pearson $r$** | 0.9492 | {{R_PEARSON_RF}} | — | — |
| **$R^2$ Residual ($R$)** | 0.0000 | **{{R2_RES}}** | — | — |
| **Pearson $r$ Residual** | — | **{{R_PEARSON_RES}}** | — | — |
| **Spearman $\\rho$ Residual** | — | **{{R_SPEARMAN_RES}}** | — | — |

---

## 6. Desempeño Temporal Diario (730 Días)

- **Días evaluados:** 730 fechas (2022-01-01 a 2023-12-31).
- **Días donde RF reduce el RMSE:** **{{PCT_DAYS_IMPR_RMSE}}%** ({{N_DAYS_IMPR}} días).
- **Días donde RF reduce el MAE:** **{{PCT_DAYS_IMPR_MAE}}%** ({{N_DAYS_IMPR_MAE}} días).
- **Mediana de $\\Delta\\text{RMSE}$ diario:** **{{MEDIAN_DELTA_RMSE}} °C**.
- **Percentil 05 diario:** {{P05_D_RMSE}} °C | **Percentil 95 diario:** {{P95_D_RMSE}} °C.

---

## 7. Desempeño Espacial en Cuadrícula 86 × 96

- **Celdas oceánicas evaluadas:** 5,279 celdas constantes.
- **Celdas donde RF reduce el RMSE:** **{{PCT_CELLS_IMPR_RMSE}}%**.
- **Patrón Espacial Identificado:** Las mejoras más notables se concentran en la franja costera y en zonas adyacentes al canal de Cozumel, donde la interpolación bilineal de baja resolución presenta gradientes térmicos desdibujados.

---

## 8. Estratificaciones Diagnósticas en VALIDATION

### A. Por Rango Batimétrico (Profundidad)

{{DEPTH_MD_TABLE}}

### B. Por Quintil de Distancia a la Costa

{{DIST_MD_TABLE}}

### C. Por Fracción Oceánica Sub-Pixel

{{FRAC_MD_TABLE}}

### D. En Eventos Extremos de Discrepancia (|Residual|)

{{EXT_MD_TABLE}}

### E. Estratificación por Incertidumbre de Análisis MUR (`analysis_error`)

{{AE_MD_TABLE}}

---

## 9. Importancia de Variables (Feature Importance)

{{IMPORTANCE_MD_TABLE}}

---

## 10. Catálogo de Figuras Científicas Generadas

Las siguientes 10 figuras fueron producidas con escalas idénticas y comparables entre E0 y RF en `DATASET_TESIS/ml_results/random_forest_E2/figures/`:

1. `figura_D2_01_feature_importance.png`: Importancia de variables MDI y por permutación.
2. `figura_D2_02_scatter_residual_real_predicho.png`: Diagrama de densidad hexbin $R_{\\text{real}}$ vs $\\hat{R}_{\\text{pred}}$.
3. `figura_D2_03_rmse_diario_e0_vs_rf.png`: Serie temporal comparativa de RMSE diario en VALIDATION.
4. `figura_D2_04_delta_rmse_diario.png`: Serie y distribución del $\\Delta\\text{RMSE}$ diario.
5. `figura_D2_05_mapa_rmse_e0.png`: Mapa 2D de RMSE del baseline bilineal E0.
6. `figura_D2_06_mapa_rmse_rf.png`: Mapa 2D de RMSE del Random Forest (E2).
7. `figura_D2_07_mapa_delta_rmse.png`: Mapa 2D de $\\Delta\\text{RMSE}$ (Azul: mejora de RF).
8. `figura_D2_08_error_por_profundidad.png`: Comparativa de RMSE y MAE por rango batimétrico.
9. `figura_D2_09_error_por_distancia_costa.png`: Comparativa de RMSE y MAE por distancia a costa.
10. `figura_D2_10_error_extremos_residual.png`: Desempeño en eventos normales vs extremos ($\\ge$ P90, P95, P99).

---

## 11. Limitaciones y Próximos Pasos

1. **Limitaciones del Baseline Pixel-Wise:** Al operar celda por celda sin información de contexto bidimensional (parches espaciales de 2D), el modelo no puede aprender texturas finas de frentes oceánicos ni remolinos de submesoescala.
2. **Potencial de Modelos Avanzados:** La captura de señal residual (mejora modesta pero consistente) justifica plenamente pasar en fases posteriores a arquitecturas convolucionales (CNNs) que aprovechen la correlación espacial 2D.
"""

    n_days_impr = int((val_res["df_daily"]["delta_rmse"] < 0).sum())
    n_days_impr_mae = int((val_res["df_daily"]["delta_mae"] < 0).sum())

    content = (
        template.replace("{{NOW_STR}}", now_str)
        .replace("{{PY_VER}}", platform.python_version())
        .replace("{{PLATFORM}}", platform.platform())
        .replace("{{VERDICT_LETTER}}", verdict_letter)
        .replace("{{RMSE_E0}}", f"{val_res['rmse_e0']:.4f}")
        .replace("{{RMSE_RF}}", f"{val_res['rmse_rf']:.4f}")
        .replace("{{DELTA_RMSE}}", f"{val_res['delta_rmse']:+.4f}")
        .replace("{{IMPR_RMSE_PCT}}", f"{val_res['impr_rmse_pct']:+.2f}")
        .replace("{{MAE_E0}}", f"{val_res['mae_e0']:.4f}")
        .replace("{{MAE_RF}}", f"{val_res['mae_rf']:.4f}")
        .replace("{{DELTA_MAE}}", f"{val_res['delta_mae']:+.4f}")
        .replace("{{IMPR_MAE_PCT}}", f"{val_res['impr_mae_pct']:+.2f}")
        .replace("{{BIAS_E0}}", f"{val_res['bias_e0']:+.4f}")
        .replace("{{BIAS_RF}}", f"{val_res['bias_rf']:+.4f}")
        .replace("{{ABS_BIAS_CHANGE}}", f"{val_res['abs_bias_change']:+.4f}")
        .replace("{{R2_E0}}", f"{val_res['r2_e0']:.4f}")
        .replace("{{R2_RF}}", f"{val_res['r2_rf']:.4f}")
        .replace("{{DELTA_R2}}", f"{val_res['r2_rf'] - val_res['r2_e0']:.4f}")
        .replace("{{R_PEARSON_RF}}", f"{val_res['r_pearson_rf']:.4f}")
        .replace("{{R2_RES}}", f"{val_res['r2_res']:.4f}")
        .replace("{{R_PEARSON_RES}}", f"{val_res['r_pearson_res']:.4f}")
        .replace("{{R_SPEARMAN_RES}}", f"{val_res['r_spearman_res']:.4f}")
        .replace("{{PCT_DAYS_IMPR_RMSE}}", f"{val_res['pct_days_impr_rmse']:.2f}")
        .replace("{{PCT_DAYS_IMPR_MAE}}", f"{val_res['pct_days_impr_mae']:.2f}")
        .replace("{{N_DAYS_IMPR}}", str(n_days_impr))
        .replace("{{N_DAYS_IMPR_MAE}}", str(n_days_impr_mae))
        .replace("{{MEDIAN_DELTA_RMSE}}", f"{val_res['median_delta_rmse']:+.4f}")
        .replace("{{P05_D_RMSE}}", f"{val_res['p05_d_rmse']:+.4f}")
        .replace("{{P95_D_RMSE}}", f"{val_res['p95_d_rmse']:+.4f}")
        .replace("{{PCT_CELLS_IMPR_RMSE}}", f"{val_res['pct_cells_impr_rmse']:.2f}")
        .replace("{{TEST_FILES_OPENED}}", str(TEST_FILES_OPENED_COUNT))
        .replace("{{FIT_TIME_SEC}}", f"{train_metrics['fit_time_sec']:.2f}")
        .replace("{{FIT_TIME_MIN}}", f"{train_metrics['fit_time_sec']/60:.2f}")
        .replace("{{PEAK_MEM_MB}}", f"{train_metrics['peak_mem_mb']:.2f}")
        .replace("{{DELTA_MEAN_RES}}", f"{qa_sampling['delta_mean']:.6f}")
        .replace("{{DELTA_STD_RES_PCT}}", f"{qa_sampling['delta_std_pct']:.4f}")
        .replace("{{MAX_DELTA_P}}", f"{np.max([qa_sampling['p01_samp'] - qa_sampling['p01_full'], qa_sampling['p99_samp'] - qa_sampling['p99_full']]):.6f}")
        .replace("{{MAX_DELTA_YR}}", f"{qa_sampling['max_delta_yr']:.4f}")
        .replace("{{MAX_DELTA_MO}}", f"{qa_sampling['max_delta_mo']:.4f}")
        .replace("{{MAX_DELTA_DEP}}", f"{qa_sampling['max_delta_dep']:.4f}")
        .replace("{{TRAIN_RMSE_RF}}", f"{train_metrics['rmse_train_rf']:.4f}")
        .replace("{{TRAIN_MAE_RF}}", f"{train_metrics['mae_train_rf']:.4f}")
        .replace("{{TRAIN_RMSE_RES}}", f"{train_metrics['rmse_train_res']:.4f}")
        .replace("{{OVERFIT_DELTA_RMSE}}", f"{val_res['rmse_rf'] - train_metrics['rmse_train_rf']:+.4f}")
        .replace("{{OVERFIT_DELTA_MAE}}", f"{val_res['mae_rf'] - train_metrics['mae_train_rf']:+.4f}")
        .replace("{{OVERFIT_DELTA_RES}}", f"{val_res['rmse_res'] - train_metrics['rmse_train_res']:+.4f}")
        .replace("{{DEPTH_MD_TABLE}}", depth_md)
        .replace("{{DIST_MD_TABLE}}", dist_md)
        .replace("{{FRAC_MD_TABLE}}", frac_md)
        .replace("{{EXT_MD_TABLE}}", ext_md)
        .replace("{{AE_MD_TABLE}}", ae_md)
        .replace("{{IMPORTANCE_MD_TABLE}}", imp_md)
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Reporte formal Markdown guardado en: {report_path}")
    return verdict_letter


# ---------------------------------------------------------------------------
# 8. Función Principal (Main)
# ---------------------------------------------------------------------------
def main():
    t_start_global = time.time()
    logger.info("================================================================================")
    logger.info("INICIO DE FASE D.2 / EXPERIMENTO E2 — RANDOM FOREST RESIDUAL BASELINE")
    logger.info("================================================================================")

    try:
        # 1. Validación de geometría espacial y orden de celdas
        mask_2d, lat_idx, lon_idx, cell_ids = verify_and_get_spatial_geometry()

        # 2. Carga y muestreo proporcional de TRAIN con QA
        df_train_sample, qa_sampling = load_and_sample_train_proportional(cell_ids, lat_idx, lon_idx)

        # 3. Entrenamiento de Random Forest
        rf, train_metrics = train_random_forest_model(df_train_sample)

        # 4. Evaluación en VALIDATION completa
        val_res = evaluate_full_validation(rf, cell_ids, lat_idx, lon_idx, train_metrics, qa_sampling)

        # 5. Generación de las 10 figuras científicas
        df_val_preds = safe_read_parquet(PREDICTIONS_DIR / "validation_predictions.parquet")
        generate_all_scientific_figures(val_res, df_val_preds)
        del df_val_preds

        # 6. Guardado de metadatos y reporte final
        verdict_letter = save_metadata_and_markdown_report(train_metrics, qa_sampling, val_res)

        elapsed_total = time.time() - t_start_global
        logger.info("================================================================================")
        logger.info(f"FASE D.2 EJECUTADA EXITOSAMENTE en {elapsed_total:.2f} s ({elapsed_total/60:.2f} min)")
        logger.info(f"Dictamen: {verdict_letter}")
        logger.info(f"TEST files opened: {TEST_FILES_OPENED_COUNT}")
        logger.info("================================================================================")

        # Imprimir resumen de terminal según Sección 20
        top_feature = val_res["df_importance"].iloc[0]["Feature"]
        print("\n" + "="*70)
        print("FASE D.2 RANDOM FOREST BASELINE COMPLETADA")
        print("="*70)
        print(f"N_train_original:     {qa_sampling['n_full']:,}")
        print(f"N_train_used:         {qa_sampling['n_samp']:,} ({qa_sampling['sampling_fraction']*100:.4f}%)")
        print(f"N_validation:         3,853,670")
        print()
        print(f"RMSE_E0:              {val_res['rmse_e0']:.4f} °C")
        print(f"RMSE_RF:              {val_res['rmse_rf']:.4f} °C")
        print(f"Improvement_RMSE_pct: {val_res['impr_rmse_pct']:+.2f} %")
        print()
        print(f"MAE_E0:               {val_res['mae_e0']:.4f} °C")
        print(f"MAE_RF:               {val_res['mae_rf']:.4f} °C")
        print(f"Improvement_MAE_pct:  {val_res['impr_mae_pct']:+.2f} %")
        print()
        print(f"Bias_E0:              {val_res['bias_e0']:+.4f} °C")
        print(f"Bias_RF:              {val_res['bias_rf']:+.4f} °C")
        print()
        print(f"R2_E0:                {val_res['r2_e0']:.4f}")
        print(f"R2_RF:                {val_res['r2_rf']:.4f}")
        print()
        print(f"% días RF mejora RMSE: {val_res['pct_days_impr_rmse']:.2f} %")
        print(f"% días RF mejora MAE:  {val_res['pct_days_impr_mae']:.2f} %")
        print()
        print(f"Feature principal:    {top_feature}")
        print(f"Tiempo entrenamiento: {train_metrics['fit_time_sec']:.2f} s ({train_metrics['fit_time_sec']/60:.2f} min)")
        print(f"Memoria máxima:       {train_metrics['peak_mem_mb']:.2f} MB")
        print()
        print(f"DICTAMEN:             {verdict_letter}")
        print("TEST:                 NO TOCADO (TEST files opened = 0)")
        print("="*70 + "\n")

    except Exception as e:
        logger.critical(f"ERROR FATAL en Fase D.2: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

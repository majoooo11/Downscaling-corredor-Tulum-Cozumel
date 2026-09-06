#!/usr/bin/env python3
"""
================================================================================
FASE D.3.6 — FINAL ML SYNTHESIS AND PAPER-READY CONSOLIDATION
CONSOLIDACIÓN DEFINITIVA DE D31–D35 Y CIERRE CIENTÍFICO DE E3b-C0
================================================================================

Este script sintetiza de forma reproducible el bloque completo de Machine
Learning (D31–D35) sin reentrenar modelos, sin modificar hiperparámetros ni
features, y sin abrir archivos raw de TEST (test_2024.parquet, test_2025.parquet).

Aplica la política Source-of-Truth y CSV-First, generando:
  1. Tablas maestras de síntesis y procedencia de métricas.
  2. Tablas para paper (Table 1, Table 2, Table 3) y comparación histórica.
  3. Cinco figuras compuestas de alta calidad para paper/tesis (300 DPI).
  4. Reporte D35 auditado y corregido (faseD35_FINAL_TEST_E3b_C0_FINAL_CORRECTED.md).
  5. Manuscritos paper-ready (Methods, Results, Discussion, Master Synthesis).
  6. Guía de selección de figuras (FIGURE_SELECTION_GUIDE.md).
================================================================================
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# 1. POLÍTICA TÉCNICA DE PROHIBICIÓN DE RAW TEST
# ==============================================================================
FORBIDDEN_RAW_TEST_FILES = {
    "test_2024.parquet",
    "test_2025.parquet"
}
RAW_TEST_LOGICAL_LOAD_COUNT_D36 = 0

def safe_load_check(path: str | Path) -> Path:
    global RAW_TEST_LOGICAL_LOAD_COUNT_D36
    p = Path(path)
    if p.name in FORBIDDEN_RAW_TEST_FILES:
        RAW_TEST_LOGICAL_LOAD_COUNT_D36 += 1
        raise RuntimeError(
            f"VIOLACIÓN DE PROTOCOLO: Acceso a raw FINAL TEST '{p.name}' está terminantemente prohibido en D.3.6."
        )
    return p

# ==============================================================================
# 2. DEFINICIÓN DE RUTAS Y DIRECTORIOS
# ==============================================================================
BASE_DIR = Path("/Users/mariajosenande/Documents/Lole/DATASET_TESIS")
ML_RESULTS_DIR = BASE_DIR / "ml_results"

D31_DIR = ML_RESULTS_DIR / "diagnostics_D31"
D32_DIR = ML_RESULTS_DIR / "E3b_D32"
D33_DIR = ML_RESULTS_DIR / "E3b_D33_external_validation"
D34_DIR = ML_RESULTS_DIR / "E3b_D34_postvalidation_diagnostics"
D35_DIR = ML_RESULTS_DIR / "E3b_D35_final_test"

SYNTHESIS_DIR = ML_RESULTS_DIR / "E3b_FINAL_SYNTHESIS"
SYNTHESIS_TABLES = SYNTHESIS_DIR / "tables"
SYNTHESIS_FIGURES = SYNTHESIS_DIR / "figures"
SYNTHESIS_REPORTS = SYNTHESIS_DIR / "reports"

for d in [SYNTHESIS_DIR, SYNTHESIS_TABLES, SYNTHESIS_FIGURES, SYNTHESIS_REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

print("Inicializando Fase D.3.6: Final ML Synthesis...")

# ==============================================================================
# 3. CARGA DE TABLAS FUENTE Y VERIFICACIÓN DE INTEGRIDAD
# ==============================================================================
# D32 Master table
d32_model_summary_file = safe_load_check(D32_DIR / "tables" / "model_summary.csv")
df_d32_models = pd.read_csv(d32_model_summary_file)
row_d32_b0 = df_d32_models[df_d32_models["model"] == "B0"].iloc[0]
row_d32_c0 = df_d32_models[df_d32_models["model"] == "E3b-C0"].iloc[0]

# D33 Master table
d33_summary_file = safe_load_check(D33_DIR / "tables" / "validation_summary.csv")
df_d33_summary = pd.read_csv(d33_summary_file)
row_d33 = df_d33_summary.iloc[0]

# D34 Master table
d34_summary_file = safe_load_check(D34_DIR / "tables" / "postvalidation_audit_summary.csv")
df_d34_summary = pd.read_csv(d34_summary_file)
row_d34 = df_d34_summary.iloc[0]

# D35 Master tables
d35_summary_file = safe_load_check(D35_DIR / "tables" / "final_test_summary.csv")
df_d35_summary = pd.read_csv(d35_summary_file)
row_d35_comb = df_d35_summary.iloc[0]

d35_yearly_file = safe_load_check(D35_DIR / "tables" / "yearly_metrics.csv")
df_d35_yearly = pd.read_csv(d35_yearly_file)
row_d35_2024 = df_d35_yearly[df_d35_yearly["period"] == "2024"].iloc[0]
row_d35_2025 = df_d35_yearly[df_d35_yearly["period"] == "2025"].iloc[0]

d35_monthly_file = safe_load_check(D35_DIR / "tables" / "monthly_metrics.csv")
df_d35_monthly = pd.read_csv(d35_monthly_file)

d35_daily_file = safe_load_check(D35_DIR / "tables" / "daily_metrics.csv")
df_d35_daily = pd.read_csv(d35_daily_file)

d35_spatial_file = safe_load_check(D35_DIR / "tables" / "spatial_metrics.csv")
df_d35_spatial = pd.read_csv(d35_spatial_file)

d35_depth_file = safe_load_check(D35_DIR / "tables" / "spatial_depth_diagnostics.csv")
df_d35_depth = pd.read_csv(d35_depth_file)

d35_boot_file = safe_load_check(D35_DIR / "tables" / "bootstrap_sensitivity.csv")
df_d35_boot = pd.read_csv(d35_boot_file)
row_boot_14d = df_d35_boot[df_d35_boot["block_length_days"] == 14].iloc[0]

d35_regimes_file = safe_load_check(D35_DIR / "tables" / "residual_regime_metrics.csv")
df_d35_regimes = pd.read_csv(d35_regimes_file)

d35_sign_file = safe_load_check(D35_DIR / "tables" / "sign_overcorrection_by_regime.csv")
df_d35_sign = pd.read_csv(d35_sign_file)

# ==============================================================================
# 4. VERIFICACIÓN NUMÉRICA ESTRICTA (CHECKS AUTOMÁTICOS)
# ==============================================================================
# D32 checks
assert np.isclose(row_d32_b0["RMSE_SST"], 0.359493, atol=1e-5), f"D32 B0 RMSE mismatch: {row_d32_b0['RMSE_SST']}"
assert np.isclose(row_d32_c0["RMSE_SST"], 0.349274, atol=1e-5), f"D32 C0 RMSE mismatch: {row_d32_c0['RMSE_SST']}"
assert np.isclose(row_d32_c0["Impr_RMSE_vs_B0_pct"], 2.84243, atol=1e-4), f"D32 Impr mismatch: {row_d32_c0['Impr_RMSE_vs_B0_pct']}"

# D33 checks
assert np.isclose(row_d33["rmse_b0"], 0.335666, atol=1e-5), f"D33 B0 RMSE mismatch: {row_d33['rmse_b0']}"
assert np.isclose(row_d33["rmse_c0"], 0.323838, atol=1e-5), f"D33 C0 RMSE mismatch: {row_d33['rmse_c0']}"
assert np.isclose(row_d33["improvement_rmse_pct"], 3.5237, atol=1e-4), f"D33 Impr mismatch: {row_d33['improvement_rmse_pct']}"

# D35 checks
assert np.isclose(row_d35_comb["RMSE_B0"], 0.357317, atol=1e-5), f"D35 B0 RMSE mismatch: {row_d35_comb['RMSE_B0']}"
assert np.isclose(row_d35_comb["RMSE_C0"], 0.331502, atol=1e-5), f"D35 C0 RMSE mismatch: {row_d35_comb['RMSE_C0']}"
assert np.isclose(row_d35_comb["Improvement_RMSE_pct"], 7.2247, atol=1e-3), f"D35 Impr mismatch: {row_d35_comb['Improvement_RMSE_pct']}"
assert np.isclose(row_d35_2024["Improvement_RMSE_pct"], 9.3986, atol=1e-3), f"2024 Impr mismatch: {row_d35_2024['Improvement_RMSE_pct']}"
assert np.isclose(row_d35_2025["Improvement_RMSE_pct"], 4.4704, atol=1e-3), f"2025 Impr mismatch: {row_d35_2025['Improvement_RMSE_pct']}"

months_improved = int(np.sum(df_d35_monthly["DeltaRMSE"] < 0))
assert months_improved == 18, f"Months improved mismatch: {months_improved}"

days_improved = int(np.sum(df_d35_daily["DeltaRMSE"] < 0))
assert days_improved == 484, f"Days improved mismatch: {days_improved}"

cells_improved = int(np.sum(df_d35_spatial["DeltaRMSE"] < 0))
assert cells_improved == 5273, f"Cells improved mismatch: {cells_improved}"

assert np.isclose(row_boot_14d["ci95_lower"], -0.042485, atol=1e-4), f"Boot CI lower mismatch: {row_boot_14d['ci95_lower']}"
assert np.isclose(row_boot_14d["ci95_upper"], -0.009792, atol=1e-4), f"Boot CI upper mismatch: {row_boot_14d['ci95_upper']}"

print("Todos los checks numéricos de D31–D35 han sido verificados satisfactoriamente.")

# ==============================================================================
# 5. TABLA: metric_provenance.csv
# ==============================================================================
provenance_records = [
    {
        "metric_id": "D32_2021_RMSE_B0",
        "metric_name": "RMSE B0",
        "phase": "D32",
        "period": "2021",
        "training_window": "2015–2020",
        "evaluation_window": "2021",
        "source_file": "E3b_D32/tables/model_summary.csv",
        "source_table": "model_summary",
        "source_column": "RMSE_SST",
        "aggregation": "global_model_b0",
        "value": 0.359493,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "D32 master table canonical baseline"
    },
    {
        "metric_id": "D32_2021_RMSE_C0",
        "metric_name": "RMSE C0",
        "phase": "D32",
        "period": "2021",
        "training_window": "2015–2020",
        "evaluation_window": "2021",
        "source_file": "E3b_D32/tables/model_summary.csv",
        "source_table": "model_summary",
        "source_column": "RMSE_SST",
        "aggregation": "global_model_c0",
        "value": 0.349274,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "D32 master table canonical model"
    },
    {
        "metric_id": "D32_2021_IMPR_RMSE",
        "metric_name": "Improvement RMSE",
        "phase": "D32",
        "period": "2021",
        "training_window": "2015–2020",
        "evaluation_window": "2021",
        "source_file": "E3b_D32/tables/model_summary.csv",
        "source_table": "model_summary",
        "source_column": "Impr_RMSE_vs_B0_pct",
        "aggregation": "global_relative",
        "value": 2.84243,
        "units": "%",
        "status": "CANONICAL",
        "notes": "D32 holdout relative gain"
    },
    {
        "metric_id": "D33_VAL_RMSE_B0",
        "metric_name": "RMSE B0",
        "phase": "D33",
        "period": "2022–2023",
        "training_window": "2015–2021",
        "evaluation_window": "2022–2023",
        "source_file": "E3b_D33_external_validation/tables/validation_summary.csv",
        "source_table": "validation_summary",
        "source_column": "rmse_b0",
        "aggregation": "combined_2yr",
        "value": 0.335666,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "External validation baseline"
    },
    {
        "metric_id": "D33_VAL_RMSE_C0",
        "metric_name": "RMSE C0",
        "phase": "D33",
        "period": "2022–2023",
        "training_window": "2015–2021",
        "evaluation_window": "2022–2023",
        "source_file": "E3b_D33_external_validation/tables/validation_summary.csv",
        "source_table": "validation_summary",
        "source_column": "rmse_c0",
        "aggregation": "combined_2yr",
        "value": 0.323838,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "External validation frozen C0 model"
    },
    {
        "metric_id": "D33_VAL_IMPR_RMSE",
        "metric_name": "Improvement RMSE",
        "phase": "D33",
        "period": "2022–2023",
        "training_window": "2015–2021",
        "evaluation_window": "2022–2023",
        "source_file": "E3b_D33_external_validation/tables/validation_summary.csv",
        "source_table": "validation_summary",
        "source_column": "improvement_rmse_pct",
        "aggregation": "combined_2yr",
        "value": 3.5237,
        "units": "%",
        "status": "CANONICAL",
        "notes": "External validation relative gain"
    },
    {
        "metric_id": "D35_TEST_RMSE_B0",
        "metric_name": "RMSE B0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "RMSE_B0",
        "aggregation": "combined_test",
        "value": 0.357317,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test baseline RMSE"
    },
    {
        "metric_id": "D35_TEST_RMSE_C0",
        "metric_name": "RMSE C0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "RMSE_C0",
        "aggregation": "combined_test",
        "value": 0.331502,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test refit C0 model RMSE"
    },
    {
        "metric_id": "D35_TEST_IMPR_RMSE",
        "metric_name": "Improvement RMSE",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "Improvement_RMSE_pct",
        "aggregation": "combined_test",
        "value": 7.2247,
        "units": "%",
        "status": "CANONICAL",
        "notes": "Final test relative gain (+7.2247%)"
    },
    {
        "metric_id": "D35_TEST_MAE_B0",
        "metric_name": "MAE B0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "MAE_B0",
        "aggregation": "combined_test",
        "value": 0.272713,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test baseline MAE"
    },
    {
        "metric_id": "D35_TEST_MAE_C0",
        "metric_name": "MAE C0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "MAE_C0",
        "aggregation": "combined_test",
        "value": 0.256325,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test refit C0 model MAE"
    },
    {
        "metric_id": "D35_TEST_BIAS_B0",
        "metric_name": "Bias B0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "Bias_B0",
        "aggregation": "combined_test",
        "value": -0.062425,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test baseline mean error"
    },
    {
        "metric_id": "D35_TEST_BIAS_C0",
        "metric_name": "Bias C0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "Bias_C0",
        "aggregation": "combined_test",
        "value": -0.014197,
        "units": "degC",
        "status": "CANONICAL",
        "notes": "Final test refit C0 mean error"
    },
    {
        "metric_id": "D35_TEST_R2_SST_B0",
        "metric_name": "R2 SST B0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/yearly_metrics.csv",
        "source_table": "yearly_metrics",
        "source_column": "R2_SST_B0",
        "aggregation": "combined_test",
        "value": 0.892661,
        "units": "fraction",
        "status": "CANONICAL",
        "notes": "Bilineal SST explained variance"
    },
    {
        "metric_id": "D35_TEST_R2_SST_C0",
        "metric_name": "R2 SST C0",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/yearly_metrics.csv",
        "source_table": "yearly_metrics",
        "source_column": "R2_SST_C0",
        "aggregation": "combined_test",
        "value": 0.907610,
        "units": "fraction",
        "status": "CANONICAL",
        "notes": "E3b-C0 reconstructed SST explained variance"
    },
    {
        "metric_id": "D35_TEST_R2_RESIDUAL",
        "metric_name": "R2 Residual",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "R2_RESIDUAL",
        "aggregation": "combined_test",
        "value": 0.112175,
        "units": "fraction",
        "status": "CANONICAL",
        "notes": "Residual variance explained (11.22%)"
    },
    {
        "metric_id": "D35_TEST_SIGN_ACC",
        "metric_name": "Sign Accuracy",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "sign_accuracy_pct",
        "aggregation": "combined_test",
        "value": 63.79,
        "units": "%",
        "status": "CANONICAL",
        "notes": "Overall residual sign accuracy"
    },
    {
        "metric_id": "D35_TEST_MAJORITY_SIGN",
        "metric_name": "Majority Sign Baseline",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "majority_sign_baseline_pct",
        "aggregation": "combined_test",
        "value": 52.45,
        "units": "%",
        "status": "CANONICAL",
        "notes": "Prevalence of majority negative sign"
    },
    {
        "metric_id": "D35_TEST_BALANCED_SIGN",
        "metric_name": "Balanced Sign Accuracy",
        "phase": "D35",
        "period": "2024–2025",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv",
        "source_table": "final_test_summary",
        "source_column": "balanced_sign_accuracy_pct",
        "aggregation": "combined_test",
        "value": 63.27,
        "units": "%",
        "status": "CANONICAL",
        "notes": "Arithmetic mean of positive/negative sensitivities"
    }
]
df_provenance = pd.DataFrame(provenance_records)
df_provenance.to_csv(SYNTHESIS_TABLES / "metric_provenance.csv", index=False)
print("Generada tabla: metric_provenance.csv")

# ==============================================================================
# 6. TABLA: d35_final_report_corrections.csv
# ==============================================================================
report_corrections = [
    {
        "issue_id": "D35_2021_RMSE",
        "section": "Sección 19 (Comparación Histórica Descriptiva)",
        "original_statement": "2021 Diagnostic Holdout | Baseline B0 RMSE: 0.380061 °C | C0 RMSE: 0.369259 °C",
        "source_of_truth": "E3b_D32/tables/model_summary.csv (D32 Master Table)",
        "corrected_statement": "2021 Diagnostic Holdout | Baseline B0 RMSE: 0.359493 °C | C0 RMSE: 0.349274 °C | Mejora: +2.84243%",
        "numeric_change": "B0: 0.380061 -> 0.359493; C0: 0.369259 -> 0.349274",
        "interpretive_change": "Corrección canónica formal sustituyendo valores de reporte preliminar por los valores maestros de D32.",
        "status": "CORRECTED"
    },
    {
        "issue_id": "final_test_terminology",
        "section": "Secciones 1, 23 y Global",
        "original_statement": "temporally unseen data / datos temporalmente inéditos",
        "source_of_truth": "Guía epistemológica D36",
        "corrected_statement": "previously withheld 2024–2025 final-test period / periodo temporal final previamente retenido y excluido del desarrollo, selección y validación del modelo",
        "numeric_change": "None",
        "interpretive_change": "Eliminación de afirmaciones ontológicas sobre 'datos inéditos'; reemplazo por descripción metodológica rigurosa de partición withheld.",
        "status": "CORRECTED"
    },
    {
        "issue_id": "bootstrap_1d_wording",
        "section": "Sección 24 (Limitaciones)",
        "original_statement": "univariate independent resampling",
        "source_of_truth": "Protocolo de remuestreo temporal por bloques",
        "corrected_statement": "1-day complete-field cluster bootstrap, which preserves spatial dependence within each daily SST field but does not preserve dependence between consecutive days",
        "numeric_change": "None",
        "interpretive_change": "Precisión técnica: el bootstrap de bloque L=1 muestrea días completos con todas sus celdas preservando autocorrelación espacial.",
        "status": "CORRECTED"
    },
    {
        "issue_id": "monthly_variability_wording",
        "section": "Secciones 11 y 24",
        "original_statement": "periodic phase mismatch / seasonal forcing mismatch",
        "source_of_truth": "Auditoría de variabilidad mensual D34/D35",
        "corrected_statement": "Month-level skill variability was associated with reduced residual-sign agreement during lower-performing months",
        "numeric_change": "None",
        "interpretive_change": "Sustitución de hipótesis físicas no demostradas de desfase por la evidencia empírica directa de caída de sign accuracy.",
        "status": "CORRECTED"
    },
    {
        "issue_id": "spatial_interpretation",
        "section": "Sección 20",
        "original_statement": "Spearman(water_depth_m, DeltaRMSE_cell) = +nan; correlación no monotónica",
        "source_of_truth": "spatial_depth_diagnostics.csv y spatial_distance_diagnostics.csv",
        "corrected_statement": "Spearman(water_depth_m, DeltaRMSE_cell) = +0.7376; Spearman(distance_coast_km, DeltaRMSE_cell) = +0.5336. La magnitud de mejora relativa decrece hacia aguas oceánicas profundas.",
        "numeric_change": "rho_depth: +0.7376; rho_dist: +0.5336",
        "interpretive_change": "Resolución del reporte derivado post-test documentando la relación positiva (+DeltaRMSE significa menor beneficio mar adentro).",
        "status": "CORRECTED"
    }
]
df_rep_corr = pd.DataFrame(report_corrections)
df_rep_corr.to_csv(SYNTHESIS_TABLES / "d35_final_report_corrections.csv", index=False)
print("Generada tabla: d35_final_report_corrections.csv")

# ==============================================================================
# 7. TABLA: ml_phase_summary.csv (D31–D35)
# ==============================================================================
phase_summary_records = [
    {
        "phase": "D31",
        "purpose": "Residual predictability diagnostic & feature screening",
        "training_period": "2015–2020",
        "evaluation_period": "2021",
        "model": "Diagnostic ablations (B3, A4, Spatial)",
        "rmse_b0": 0.359491,
        "rmse_model": 0.341734,
        "mae_b0": 0.277337,
        "mae_model": 0.259345,
        "improvement_rmse_pct": 4.9395,
        "months_improved": "NA",
        "cells_improved_pct": "NA",
        "formal_decision": "CONTINUE RESIDUAL LEARNING",
        "methodological_status": "COMPLETED",
        "source_file": "diagnostics_D31/tables/d31_key_findings_summary.csv"
    },
    {
        "phase": "D32",
        "purpose": "Residual tabular model development & parsimonious selection",
        "training_period": "2015–2020",
        "evaluation_period": "2021 (Holdout)",
        "model": "E3b-C0 (Selected)",
        "rmse_b0": 0.359493,
        "rmse_model": 0.349274,
        "mae_b0": 0.277337,
        "mae_model": 0.266264,
        "improvement_rmse_pct": 2.8424,
        "months_improved": "10 / 12",
        "cells_improved_pct": 98.69,
        "formal_decision": "SELECT E3b-C0",
        "methodological_status": "METHODOLOGICALLY CLOSED",
        "source_file": "E3b_D32/tables/model_summary.csv"
    },
    {
        "phase": "D33",
        "purpose": "External temporal validation of frozen E3b-C0",
        "training_period": "2015–2021",
        "evaluation_period": "2022–2023",
        "model": "E3b-C0 (Frozen)",
        "rmse_b0": 0.335666,
        "rmse_model": 0.323838,
        "mae_b0": 0.263594,
        "mae_model": 0.254457,
        "improvement_rmse_pct": 3.5237,
        "months_improved": "16 / 24",
        "cells_improved_pct": 90.14,
        "formal_decision": "D33-B",
        "methodological_status": "D33-B — UNCHANGED",
        "source_file": "E3b_D33_external_validation/tables/validation_summary.csv"
    },
    {
        "phase": "D34",
        "purpose": "Post-validation diagnostic audit of frozen E3b-C0",
        "training_period": "2015–2021",
        "evaluation_period": "2022–2023",
        "model": "E3b-C0 (Frozen Audit)",
        "rmse_b0": 0.335666,
        "rmse_model": 0.323838,
        "mae_b0": 0.263594,
        "mae_model": 0.254457,
        "improvement_rmse_pct": 3.5237,
        "months_improved": "16 / 24",
        "cells_improved_pct": 90.14,
        "formal_decision": "PREPARE FINAL TEST",
        "methodological_status": "INTERPRETATIONALLY CLOSED",
        "source_file": "E3b_D34_postvalidation_diagnostics/tables/postvalidation_audit_summary.csv"
    },
    {
        "phase": "D35",
        "purpose": "Final out-of-sample confirmatory evaluation (Withheld Test)",
        "training_period": "2015–2023",
        "evaluation_period": "2024–2025",
        "model": "E3b-C0_FINALREFIT_2015_2023",
        "rmse_b0": 0.357317,
        "rmse_model": 0.331502,
        "mae_b0": 0.272713,
        "mae_model": 0.256325,
        "improvement_rmse_pct": 7.2247,
        "months_improved": "18 / 24",
        "cells_improved_pct": 99.96,
        "formal_decision": "D35-A",
        "methodological_status": "FINAL GENERALIZATION CONFIRMED",
        "source_file": "E3b_D35_final_test/tables/final_test_summary.csv"
    }
]
df_phase_summary = pd.DataFrame(phase_summary_records)
df_phase_summary.to_csv(SYNTHESIS_TABLES / "ml_phase_summary.csv", index=False)
print("Generada tabla: ml_phase_summary.csv")

# ==============================================================================
# 8. TABLA: historical_skill_summary.csv
# ==============================================================================
historical_skill_records = [
    {
        "evaluation_period": "2021 Diagnostic Holdout",
        "training_window": "2015–2020",
        "evaluation_window": "2021",
        "RMSE_B0": 0.359493,
        "RMSE_C0": 0.349274,
        "DeltaRMSE": -0.010219,
        "RMSE_improvement_pct": 2.84243,
        "notes": "Model trained on 2015-2020; tested on 2021 holdout"
    },
    {
        "evaluation_period": "2022–2023 Validation",
        "training_window": "2015–2021",
        "evaluation_window": "2022–2023",
        "RMSE_B0": 0.335666,
        "RMSE_C0": 0.323838,
        "DeltaRMSE": -0.011828,
        "RMSE_improvement_pct": 3.52371,
        "notes": "Model trained on 2015-2021; external temporal validation"
    },
    {
        "evaluation_period": "2024–2025 Final Test",
        "training_window": "2015–2023",
        "evaluation_window": "2024–2025",
        "RMSE_B0": 0.357317,
        "RMSE_C0": 0.331502,
        "DeltaRMSE": -0.025815,
        "RMSE_improvement_pct": 7.22466,
        "notes": "Model refit on 2015-2023; evaluated on withheld final test"
    }
]
df_historical = pd.DataFrame(historical_skill_records)
df_historical.to_csv(SYNTHESIS_TABLES / "historical_skill_summary.csv", index=False)
print("Generada tabla: historical_skill_summary.csv")

# ==============================================================================
# 9. TABLAS PRINCIPALES DEL PAPER (Table 1, Table 2, Table 3)
# ==============================================================================
# Paper Table 1: Final Performance
paper_table1_records = [
    {
        "period": "2024",
        "training_window": "2015–2023",
        "RMSE_B0": float(row_d35_2024["RMSE_B0"]),
        "RMSE_C0": float(row_d35_2024["RMSE_C0"]),
        "RMSE_improvement_pct": float(row_d35_2024["Improvement_RMSE_pct"]),
        "MAE_B0": float(row_d35_2024["MAE_B0"]),
        "MAE_C0": float(row_d35_2024["MAE_C0"]),
        "MAE_improvement_pct": float(row_d35_2024["Improvement_MAE_pct"]),
        "Bias_B0": float(row_d35_2024["Bias_B0"]),
        "Bias_C0": float(row_d35_2024["Bias_C0"]),
        "R2_B0": float(row_d35_2024["R2_SST_B0"]),
        "R2_C0": float(row_d35_2024["R2_SST_C0"])
    },
    {
        "period": "2025",
        "training_window": "2015–2023",
        "RMSE_B0": float(row_d35_2025["RMSE_B0"]),
        "RMSE_C0": float(row_d35_2025["RMSE_C0"]),
        "RMSE_improvement_pct": float(row_d35_2025["Improvement_RMSE_pct"]),
        "MAE_B0": float(row_d35_2025["MAE_B0"]),
        "MAE_C0": float(row_d35_2025["MAE_C0"]),
        "MAE_improvement_pct": float(row_d35_2025["Improvement_MAE_pct"]),
        "Bias_B0": float(row_d35_2025["Bias_B0"]),
        "Bias_C0": float(row_d35_2025["Bias_C0"]),
        "R2_B0": float(row_d35_2025["R2_SST_B0"]),
        "R2_C0": float(row_d35_2025["R2_SST_C0"])
    },
    {
        "period": "2024–2025",
        "training_window": "2015–2023",
        "RMSE_B0": float(row_d35_comb["RMSE_B0"]),
        "RMSE_C0": float(row_d35_comb["RMSE_C0"]),
        "RMSE_improvement_pct": float(row_d35_comb["Improvement_RMSE_pct"]),
        "MAE_B0": float(row_d35_comb["MAE_B0"]),
        "MAE_C0": float(row_d35_comb["MAE_C0"]),
        "MAE_improvement_pct": float(df_d35_yearly[df_d35_yearly["period"] == "2024–2025"]["Improvement_MAE_pct"].iloc[0]),
        "Bias_B0": float(row_d35_comb["Bias_B0"]),
        "Bias_C0": float(row_d35_comb["Bias_C0"]),
        "R2_B0": float(df_d35_yearly[df_d35_yearly["period"] == "2024–2025"]["R2_SST_B0"].iloc[0]),
        "R2_C0": float(df_d35_yearly[df_d35_yearly["period"] == "2024–2025"]["R2_SST_C0"].iloc[0])
    }
]
df_paper_t1 = pd.DataFrame(paper_table1_records)
df_paper_t1.to_csv(SYNTHESIS_TABLES / "paper_table1_final_performance.csv", index=False)
print("Generada tabla: paper_table1_final_performance.csv")

# Paper Table 2: Robustness
paper_table2_records = [
    {"metric": "months_improved", "value": "18 / 24", "units": "months", "source_phase": "D35"},
    {"metric": "days_improved", "value": "484 / 731", "units": "days", "source_phase": "D35"},
    {"metric": "days_improved_pct", "value": "66.21", "units": "%", "source_phase": "D35"},
    {"metric": "cells_improved", "value": "5273 / 5275", "units": "cells", "source_phase": "D35"},
    {"metric": "cells_improved_pct", "value": "99.96", "units": "%", "source_phase": "D35"},
    {"metric": "median_spatial_delta_rmse", "value": "-0.022808", "units": "degC", "source_phase": "D35"},
    {"metric": "P90_spatial_delta_rmse", "value": "-0.009307", "units": "degC", "source_phase": "D35"},
    {"metric": "bootstrap_14d_median", "value": "-0.025395", "units": "degC", "source_phase": "D35"},
    {"metric": "bootstrap_14d_ci95_lower", "value": "-0.042485", "units": "degC", "source_phase": "D35"},
    {"metric": "bootstrap_14d_ci95_upper", "value": "-0.009792", "units": "degC", "source_phase": "D35"},
    {"metric": "R2_residual", "value": "0.112175", "units": "fraction", "source_phase": "D35"},
    {"metric": "sign_accuracy", "value": "63.79", "units": "%", "source_phase": "D35"},
    {"metric": "majority_sign_baseline", "value": "52.45", "units": "%", "source_phase": "D35"},
    {"metric": "balanced_sign_accuracy", "value": "63.27", "units": "%", "source_phase": "D35"}
]
df_paper_t2 = pd.DataFrame(paper_table2_records)
df_paper_t2.to_csv(SYNTHESIS_TABLES / "paper_table2_robustness.csv", index=False)
print("Generada tabla: paper_table2_robustness.csv")

# Paper Table 3: Residual Regimes
threshold_map = {
    "DEV-P0-P50": "< 0.2066 °C",
    "DEV-P50-P75": "0.2066–0.3604 °C",
    "DEV-P75-P90": "0.3604–0.5377 °C",
    "DEV-P90-P95": "0.5377–0.6652 °C",
    "DEV-P95-P99": "0.6652–0.9659 °C",
    "DEV-P99+": ">= 0.9659 °C"
}
paper_table3_records = []
for idx, row in df_d35_regimes.iterrows():
    reg = row["regime"]
    paper_table3_records.append({
        "regime": reg,
        "threshold": threshold_map.get(reg, "NA"),
        "N": int(row["N"]),
        "pct_test": float(row["pct_test"]),
        "rmse_b0": float(row["RMSE_B0"]),
        "rmse_c0": float(row["RMSE_C0"]),
        "improvement_rmse_pct": float(row["Improvement_RMSE_pct"]),
        "sign_accuracy_pct": float(row["sign_accuracy_pct"]),
        "overcorrection_pct": float(row["overcorrection_pct"]),
        "undercorrection_pct": float(row["undercorrection_pct"]),
        "std_ratio_rhat_r_if_available": float(row["std_ratio_Rhat_R"]) if "std_ratio_Rhat_R" in row else "NA",
        "source_file": "E3b_D35_final_test/tables/residual_regime_metrics.csv"
    })
df_paper_t3 = pd.DataFrame(paper_table3_records)
df_paper_t3.to_csv(SYNTHESIS_TABLES / "paper_table3_residual_regimes.csv", index=False)
print("Generada tabla: paper_table3_residual_regimes.csv")

# ==============================================================================
# 10. GENERACIÓN DE FIGURAS DEFINITIVAS (300 DPI)
# ==============================================================================
print("Generando 5 figuras definitivas para paper/tesis (300 DPI)...")

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif"
})

def generate_figure_1():
    # ------------------------------------------------------------------------------
    # FIGURE 1 — FINAL TEST GLOBAL PERFORMANCE (3 panels)
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    periods = ["2024", "2025", "Combined\n(2024–2025)"]
    x = np.arange(len(periods))
    width = 0.35

    # Panel A: RMSE
    b0_rmse = [df_paper_t1.loc[df_paper_t1["period"]=="2024", "RMSE_B0"].values[0],
               df_paper_t1.loc[df_paper_t1["period"]=="2025", "RMSE_B0"].values[0],
               df_paper_t1.loc[df_paper_t1["period"]=="2024–2025", "RMSE_B0"].values[0]]
    c0_rmse = [df_paper_t1.loc[df_paper_t1["period"]=="2024", "RMSE_C0"].values[0],
               df_paper_t1.loc[df_paper_t1["period"]=="2025", "RMSE_C0"].values[0],
               df_paper_t1.loc[df_paper_t1["period"]=="2024–2025", "RMSE_C0"].values[0]]

    axes[0].bar(x - width/2, b0_rmse, width, label="Baseline $B_0$ (Bilinear)", color="#94a3b8", edgecolor="#475569")
    axes[0].bar(x + width/2, c0_rmse, width, label="Model $C_0$ (E3b-C0 Refit)", color="#0284c7", edgecolor="#0369a1")
    axes[0].set_ylabel("RMSE (°C)")
    axes[0].set_title("Panel A: Root Mean Square Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(periods)
    axes[0].set_ylim(0, 0.45)
    axes[0].legend(loc="upper right", framealpha=0.9)
    for i, v in enumerate(b0_rmse):
        axes[0].text(i - width/2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for i, v in enumerate(c0_rmse):
        axes[0].text(i + width/2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: MAE
    b0_mae = [df_paper_t1.loc[df_paper_t1["period"]=="2024", "MAE_B0"].values[0],
              df_paper_t1.loc[df_paper_t1["period"]=="2025", "MAE_B0"].values[0],
              df_paper_t1.loc[df_paper_t1["period"]=="2024–2025", "MAE_B0"].values[0]]
    c0_mae = [df_paper_t1.loc[df_paper_t1["period"]=="2024", "MAE_C0"].values[0],
              df_paper_t1.loc[df_paper_t1["period"]=="2025", "MAE_C0"].values[0],
              df_paper_t1.loc[df_paper_t1["period"]=="2024–2025", "MAE_C0"].values[0]]

    axes[1].bar(x - width/2, b0_mae, width, label="Baseline $B_0$", color="#cbd5e1", edgecolor="#64748b")
    axes[1].bar(x + width/2, c0_mae, width, label="Model $C_0$", color="#059669", edgecolor="#047857")
    axes[1].set_ylabel("MAE (°C)")
    axes[1].set_title("Panel B: Mean Absolute Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(periods)
    axes[1].set_ylim(0, 0.35)
    axes[1].legend(loc="upper right", framealpha=0.9)
    for i, v in enumerate(b0_mae):
        axes[1].text(i - width/2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for i, v in enumerate(c0_mae):
        axes[1].text(i + width/2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel C: RMSE Improvement %
    imprs = [df_paper_t1.loc[df_paper_t1["period"]=="2024", "RMSE_improvement_pct"].values[0],
             df_paper_t1.loc[df_paper_t1["period"]=="2025", "RMSE_improvement_pct"].values[0],
             df_paper_t1.loc[df_paper_t1["period"]=="2024–2025", "RMSE_improvement_pct"].values[0]]

    bars = axes[2].bar(x, imprs, width=0.45, color=["#38bdf8", "#34d399", "#2563eb"], edgecolor="#1e293b")
    axes[2].set_ylabel("RMSE Improvement (%)")
    axes[2].set_title("Panel C: Relative Improvement vs $B_0$")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(periods)
    axes[2].set_ylim(0, 12)
    axes[2].axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Predeclared Threshold (+1.00%)")
    axes[2].legend(loc="upper right", framealpha=0.9)
    for bar, val in zip(bars, imprs):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.25, f"+{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    fig.suptitle("FIGURE 1: Out-of-Sample Performance on Withheld FINAL TEST (2024–2025)", y=1.02, fontweight="bold")
    plt.tight_layout()
    fig1_path = SYNTHESIS_FIGURES / "fig1_final_performance.png"
    fig1_pub_png = SYNTHESIS_FIGURES / "fig1_final_performance_PUBLICATION.png"
    fig1_pub_pdf = SYNTHESIS_FIGURES / "fig1_final_performance_PUBLICATION.pdf"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pub_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig1_pub_pdf, bbox_inches="tight")
    plt.close()
    print("  Guardada Figure 1 (PNG, PUBLICATION PNG, PUBLICATION PDF).")


def generate_figure_2():
    # ------------------------------------------------------------------------------
    # FIGURE 2 — TEMPORAL ROBUSTNESS (3 panels)
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel A: Monthly RMSE Improvement
    month_col = "year_month" if "year_month" in df_d35_monthly.columns else "month"
    months = df_d35_monthly[month_col].values
    m_impr = df_d35_monthly["Improvement_RMSE_pct"].values
    colors_m = ["#10b981" if v >= 0 else "#f43f5e" for v in m_impr]

    axes[0].bar(range(len(months)), m_impr, color=colors_m, edgecolor="#1e293b", width=0.7)
    axes[0].axhline(0, color="#475569", linestyle="-", linewidth=1)
    axes[0].set_xticks(range(0, len(months), 3))
    axes[0].set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha="right")
    axes[0].set_ylabel("RMSE Improvement (%)")
    axes[0].set_title("Panel A: Monthly Gain (18 / 24 Months Improved)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Daily DeltaRMSE + 7d Rolling
    df_d35_daily["date"] = pd.to_datetime(df_d35_daily["date"])
    df_d35_daily_sorted = df_d35_daily.sort_values("date").reset_index(drop=True)
    daily_delta = df_d35_daily_sorted["DeltaRMSE"].values
    rolling_7d = df_d35_daily_sorted["DeltaRMSE"].rolling(7, center=True).mean()

    axes[1].plot(df_d35_daily_sorted["date"], daily_delta, color="#94a3b8", alpha=0.45, linewidth=0.7, label="Daily $\Delta$RMSE")
    axes[1].plot(df_d35_daily_sorted["date"], rolling_7d, color="#0284c7", linewidth=1.6, label="7-Day Rolling Mean")
    axes[1].axhline(0, color="#ef4444", linestyle="--", linewidth=1)
    axes[1].set_ylabel("$\Delta$RMSE (°C) [C0 − B0]")
    axes[1].set_title("Panel B: Daily Time Series (484 / 731 Days Improved)")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(5))

    # Panel C: Bootstrap CI95
    b_blocks = df_d35_boot["block_length_days"].values
    b_medians = df_d35_boot["median_delta_rmse"].values
    b_lowers = df_d35_boot["ci95_lower"].values
    b_uppers = df_d35_boot["ci95_upper"].values
    y_err = [b_medians - b_lowers, b_uppers - b_medians]

    axes[2].errorbar(range(len(b_blocks)), b_medians, yerr=y_err, fmt="o", color="#2563eb",
                     ecolor="#1e40af", elinewidth=2, capsize=6, capthick=1.5, markersize=7)
    axes[2].axhline(0, color="#ef4444", linestyle="--", linewidth=1.2, label="Zero Improvement ($\Delta=0$)")
    axes[2].set_xticks(range(len(b_blocks)))
    axes[2].set_xticklabels([f"{b}-Day\n{'Cluster' if b==1 else 'Moving Block'}" for b in b_blocks])
    axes[2].set_ylabel("$\Delta$RMSE 95% Confidence Interval (°C)")
    axes[2].set_title("Panel C: Bootstrap Robustness ($B=1,000$)")
    axes[2].legend(loc="lower left", framealpha=0.9)
    for idx, (m, u) in enumerate(zip(b_medians, b_uppers)):
        axes[2].text(idx + 0.08, m, f"Med: {m:.3f}°C\nUpper: {u:.3f}°C", va="center", fontsize=8.5)

    fig.suptitle("FIGURE 2: Temporal Robustness Across Monthly, Daily, and Resampled Scales", y=1.02, fontweight="bold")
    plt.tight_layout()
    fig2_path = SYNTHESIS_FIGURES / "fig2_temporal_robustness.png"
    fig2_pub_png = SYNTHESIS_FIGURES / "fig2_temporal_robustness_PUBLICATION.png"
    fig2_pub_pdf = SYNTHESIS_FIGURES / "fig2_temporal_robustness_PUBLICATION.pdf"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pub_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_pub_pdf, bbox_inches="tight")
    plt.close()
    print("  Guardada Figure 2 (PNG, PUBLICATION PNG, PUBLICATION PDF).")


def generate_figure_3():
    # ------------------------------------------------------------------------------
    # FIGURE 3 — SPATIAL SKILL (3 panels)
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel A: Map DeltaRMSE
    sc = axes[0].scatter(df_d35_spatial["lon"], df_d35_spatial["lat"], c=df_d35_spatial["DeltaRMSE"],
                         cmap="coolwarm", vmin=-0.06, vmax=+0.01, s=7, alpha=0.9, edgecolors="none")
    cbar = plt.colorbar(sc, ax=axes[0], orientation="vertical", pad=0.03, shrink=0.85)
    cbar.set_label("$\Delta$RMSE (°C) [C0 − B0]")
    axes[0].set_xlabel("Longitude (°W)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_title("Panel A: Cell-Level $\Delta$RMSE Map")

    # Panel B: Distribution of DeltaRMSE
    delta_spat = df_d35_spatial["DeltaRMSE"].values
    axes[1].hist(delta_spat, bins=40, color="#38bdf8", edgecolor="#0284c7", alpha=0.85, density=True)
    axes[1].axvline(0, color="#ef4444", linestyle="--", linewidth=1.2, label="Zero Improvement")
    axes[1].axvline(np.median(delta_spat), color="#1e40af", linestyle="-", linewidth=1.5,
                    label=f"Median: {np.median(delta_spat):.4f} °C")
    axes[1].set_xlabel("$\Delta$RMSE (°C)")
    axes[1].set_ylabel("Empirical Density")
    axes[1].set_title(f"Panel B: Distribution (5,273/5,275 Cells Improved: 99.96%)")
    axes[1].legend(loc="upper left", framealpha=0.9)

    # Panel C: DeltaRMSE by Depth Bins
    depth_bins = df_d35_depth["depth_bin"].values
    d_med = df_d35_depth["median_delta_RMSE"].values
    d_mean = df_d35_depth["mean_delta_RMSE"].values
    x_d = np.arange(len(depth_bins))

    axes[2].bar(x_d - 0.18, d_med, width=0.35, label="Median $\Delta$RMSE", color="#0284c7")
    axes[2].bar(x_d + 0.18, d_mean, width=0.35, label="Mean $\Delta$RMSE", color="#38bdf8")
    axes[2].axhline(0, color="#ef4444", linestyle="--", linewidth=1)
    axes[2].set_xticks(x_d)
    axes[2].set_xticklabels(depth_bins, rotation=30, ha="right")
    axes[2].set_ylabel("$\Delta$RMSE (°C)")
    axes[2].set_title("Panel C: Skill Across Bathymetric Strata")
    axes[2].legend(loc="upper right", framealpha=0.9)

    fig.suptitle("FIGURE 3: Spatial Consistency and Skill Across Bathymetric Strata (5,275 Cells)", y=1.02, fontweight="bold")
    plt.tight_layout()
    fig3_path = SYNTHESIS_FIGURES / "fig3_spatial_skill.png"
    fig3_pub_png = SYNTHESIS_FIGURES / "fig3_spatial_skill_PUBLICATION.png"
    fig3_pub_pdf = SYNTHESIS_FIGURES / "fig3_spatial_skill_PUBLICATION.pdf"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig3_pub_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig3_pub_pdf, bbox_inches="tight")
    plt.close()
    print("  Guardada Figure 3 (PNG, PUBLICATION PNG, PUBLICATION PDF).")


def generate_figure_4():
    # ------------------------------------------------------------------------------
    # FIGURE 4 — RESIDUAL REGIME BEHAVIOR (3 panels)
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    regimes = df_paper_t3["regime"].values
    x_reg = np.arange(len(regimes))

    # Panel A: Improvement RMSE by Regime
    reg_impr = df_paper_t3["improvement_rmse_pct"].values
    colors_r = ["#f43f5e" if v < 0 else "#10b981" for v in reg_impr]
    axes[0].bar(x_reg, reg_impr, color=colors_r, edgecolor="#1e293b", width=0.6)
    axes[0].axhline(0, color="#475569", linestyle="-", linewidth=1)
    axes[0].set_xticks(x_reg)
    axes[0].set_xticklabels(regimes, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE Improvement (%)")
    axes[0].set_title("Panel A: Relative Improvement by Regime")
    for i, v in enumerate(reg_impr):
        va_pos = "top" if v < 0 else "bottom"
        offset = -1.5 if v < 0 else 0.5
        axes[0].text(i, v + offset, f"{v:+.1f}%", ha="center", va=va_pos, fontsize=8.5, fontweight="bold")

    # Panel B: Sign Accuracy by Regime
    sign_acc = df_paper_t3["sign_accuracy_pct"].values
    axes[1].plot(x_reg, sign_acc, marker="o", color="#2563eb", linewidth=2, markersize=7, label="E3b-C0 Sign Accuracy")
    axes[1].axhline(52.45, color="#ef4444", linestyle="--", linewidth=1.2, label="Majority-Sign Baseline (52.45%)")
    axes[1].set_xticks(x_reg)
    axes[1].set_xticklabels(regimes, rotation=35, ha="right")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Panel B: Residual Sign Discrimination")
    axes[1].set_ylim(45, 95)
    axes[1].legend(loc="lower right", framealpha=0.9)
    for i, v in enumerate(sign_acc):
        axes[1].text(i, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)

    # Panel C: std ratio std(Rhat)/std(R) by Regime
    std_ratios = df_paper_t3["std_ratio_rhat_r_if_available"].values
    axes[2].bar(x_reg, std_ratios, color="#64748b", edgecolor="#1e293b", width=0.55)
    axes[2].axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="No Shrinkage (Ratio = 1.0)")
    axes[2].set_xticks(x_reg)
    axes[2].set_xticklabels(regimes, rotation=35, ha="right")
    axes[2].set_ylabel("Ratio $\mathrm{std}(\hat{R}) / \mathrm{std}(R)$")
    axes[2].set_title("Panel C: Within-Regime Amplitude Ratio")
    axes[2].set_ylim(0, 1.15)
    axes[2].legend(loc="upper right", framealpha=0.9)
    for i, v in enumerate(std_ratios):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    axes[2].text(0.46, 0.60, "Global TEST ratio = 0.2536",
                 transform=axes[2].transAxes, ha="center", va="center", fontsize=8.5, color="#1e293b",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.92, linewidth=0.8))

    fig.suptitle("FIGURE 4: Residual-Magnitude Dependence and Prediction Behavior", y=1.02, fontweight="bold")
    plt.tight_layout()
    fig4_path = SYNTHESIS_FIGURES / "fig4_residual_regimes.png"
    fig4_pub_png = SYNTHESIS_FIGURES / "fig4_residual_regimes_PUBLICATION.png"
    fig4_pub_pdf = SYNTHESIS_FIGURES / "fig4_residual_regimes_PUBLICATION.pdf"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig4_pub_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig4_pub_pdf, bbox_inches="tight")
    plt.close()
    print("  Guardada Figure 4 (PNG, PUBLICATION PNG, PUBLICATION PDF).")


def generate_figure_5():
    # ------------------------------------------------------------------------------
    # FIGURE 5 — HISTORICAL DESCRIPTIVE SKILL (1 panel)
    # ------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    hist_periods = ["2021 Diagnostic Holdout\n(Train 2015–2020)",
                    "2022–2023 Validation\n(Train 2015–2021)",
                    "2024–2025 Final Test\n(Train 2015–2023)"]
    hist_imprs = [2.84243, 3.52371, 7.22466]
    x_hist = np.arange(len(hist_periods))

    bars_h = ax.bar(x_hist, hist_imprs, color=["#93c5fd", "#60a5fa", "#1d4ed8"], edgecolor="#1e293b", width=0.45)
    ax.set_ylabel("RMSE Relative Improvement (%)")
    ax.set_title("FIGURE 5: Descriptive Historical Skill Across Experimental Phases", pad=15, fontweight="bold")
    ax.set_xticks(x_hist)
    ax.set_xticklabels(hist_periods)
    ax.set_ylim(0, 9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars_h, hist_imprs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.25, f"+{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.text(0.5, -0.22, "Note: Different fitted estimators and progressively larger training windows were used.\nThis comparison is strictly descriptive and does not represent a same-estimator generalization trajectory.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic", color="#475569")

    plt.tight_layout()
    fig5_path = SYNTHESIS_FIGURES / "fig5_historical_skill.png"
    fig5_sup_png = SYNTHESIS_FIGURES / "fig5_historical_skill_SUPPLEMENT.png"
    fig5_sup_pdf = SYNTHESIS_FIGURES / "fig5_historical_skill_SUPPLEMENT.pdf"
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig5_sup_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig5_sup_pdf, bbox_inches="tight")
    plt.close()
    print("  Guardada Figure 5 (PNG, SUPPLEMENT PNG, SUPPLEMENT PDF).")


TARGET_FIGURE = None
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith("--fig="):
        TARGET_FIGURE = arg.split("=")[1].strip()
    elif arg in ["--fig", "-f"] and i + 1 < len(sys.argv[1:]):
        TARGET_FIGURE = sys.argv[1:][i + 1].strip()

if TARGET_FIGURE is None or TARGET_FIGURE in ["1", "fig1"]:
    generate_figure_1()
if TARGET_FIGURE is None or TARGET_FIGURE in ["2", "fig2"]:
    generate_figure_2()
if TARGET_FIGURE is None or TARGET_FIGURE in ["3", "fig3"]:
    generate_figure_3()
if TARGET_FIGURE is None or TARGET_FIGURE in ["4", "fig4"]:
    generate_figure_4()
if TARGET_FIGURE is None or TARGET_FIGURE in ["5", "fig5"]:
    generate_figure_5()

if TARGET_FIGURE is not None:
    print(f"Fase D.3.6: Regeneración selectiva de Figura {TARGET_FIGURE} completada.")
    sys.exit(0)

# ==============================================================================
# 11. GUÍA DE SELECCIÓN DE FIGURAS: FIGURE_SELECTION_GUIDE.md
# ==============================================================================
fig_guide_content = """# Figure Selection Guide — Machine Learning Block (E3b-C0)

This guide establishes the assignment and placement of figures across the thesis, the ICITS'27 conference manuscript, and supplementary materials, following the strict 4–5 figure limit for standard conference papers.

---

## 1. Summary Classification

| Figure ID | Short Name | Panels | Recommended Venue | Primary Purpose |
|---|---|:---:|---|---|
| **Figure 1** | Final Test Global Performance | 3 (RMSE, MAE, % Impr) | **PAPER MAIN** / THESIS | Documents primary confirmatory results on withheld 2024–2025 test. |
| **Figure 2** | Temporal Robustness | 3 (Monthly, Daily, Bootstrap) | **PAPER MAIN** / THESIS | Visualizes day-to-day variability and 14-day moving block robustness. |
| **Figure 3** | Spatial Consistency & Skill Across Bathymetric Strata (5,275 Cells) | 3 (Map, Density, Depth Bins) | **PAPER MAIN** / THESIS | Confirms 99.96% spatial generalization and depth-dependent skill across strata. |
| **Figure 4** | Residual-Magnitude Dependence and Prediction Behavior | 3 (Impr, Sign Acc, Shrinkage) | **PAPER MAIN** / THESIS | Explains the mechanistic trade-off across $|R|$ discrepancy regimes. |
| **Figure 5** | Descriptive Historical Skill | 1 (Bar Comparison) | **PAPER SUPPLEMENT** / THESIS | Contextualizes progression from D32 to D35 with explicit caveats. |

---

## 2. Venue-Specific Packages

### Package A: ICITS'27 Conference Paper (Compact 4-Figure Core)
- **Main Text Figures:**
  1. `Figure 1`: Core confirmatory outcome (2024, 2025, Combined RMSE and MAE).
     - Files: `fig1_final_performance_PUBLICATION.png` / `fig1_final_performance_PUBLICATION.pdf`
  2. `Figure 2`: Temporal generalization (Monthly bar chart, daily rolling series, moving-block bootstrap).
     - Files: `fig2_temporal_robustness_PUBLICATION.png` / `fig2_temporal_robustness_PUBLICATION.pdf`
  3. `Figure 3`: Spatial consistency and skill across bathymetric strata (5,275 cells).
     - Files: `fig3_spatial_skill_PUBLICATION.png` / `fig3_spatial_skill_PUBLICATION.pdf`
  4. `Figure 4`: Residual-magnitude dependence and prediction behavior (gain in large baseline discrepancies, shrinkage in low-discrepancy regime).
     - Files: `fig4_residual_regimes_PUBLICATION.png` / `fig4_residual_regimes_PUBLICATION.pdf`
- **Supplementary / Appendix:**
  - `Figure 5`: Descriptive historical comparison across experimental phases (distinct fitted estimators).
    - Files: `fig5_historical_skill_SUPPLEMENT.png` / `fig5_historical_skill_SUPPLEMENT.pdf`
  - Phase D31 diagnostic feature importance & ablation charts.
  - Phase D34 detailed negative-month diagnostic time series.

### Package B: Master's Thesis (Full Comprehensive Package)
- **Chapter on Machine Learning Results:**
  - Embed Figures 1, 2, 3, 4, and 5 directly within the main narrative.
- **Thesis Appendix:**
  - D.3.1 Predictability & Climatological Baseline figures.
  - D.3.2 8-model ablation comparisons.
  - D.3.3 External validation monthly breakdown.
  - D.3.4 Post-validation diagnostic audit and low-residual decomposition.
"""
with open(SYNTHESIS_DIR / "FIGURE_SELECTION_GUIDE.md", "w", encoding="utf-8") as f:
    f.write(fig_guide_content)
print("Generado archivo: FIGURE_SELECTION_GUIDE.md")

# ==============================================================================
# 12. GENERACIÓN DEL REPORTE FINAL CORREGIDO D35
# ==============================================================================
corrected_d35_report_path = D35_DIR / "reports" / "faseD35_FINAL_TEST_E3b_C0_FINAL_CORRECTED.md"
corrected_d35_content = f"""# Reporte Científico — Fase D.3.5 (FINAL AUDITED & CORRECTED)
## Final Out-of-Sample Evaluation of Frozen E3b-C0 (Withheld TEST 2024–2025)

---

## 1. Objetivo
Evaluar de forma estrictamente out-of-sample, confirmatoria y no adaptativa el modelo residual tabular `E3b-C0` en la partición **FINAL TEST correspondiente al periodo 2024–2025**, previamente retenido y formalmente excluido de todo el desarrollo, selección y validación del modelo, aplicando mecánicamente las reglas de decisión predeclaradas para determinar la generalización final del método de downscaling.

---

## 2. Estado Heredado D32–D34
- **D.3.2 (Desarrollo):** Metodológicamente cerrada. Selección de la especificación parsimoniosa `E3b-C0`.
- **D.3.3 (Validación Temporal 2022–2023):** Dictamen formal inmutable `D33-B — PARTIAL / MIXED GENERALIZATION` (mejora global de +3.52%, con 16/24 meses mejorados).
- **D.3.4 (Auditoría Diagnóstica):** Interpretativamente cerrada (`INTERPRETATIONALLY CLOSED`). No se identificaron bugs informáticos; robustez temporal respaldada bajo bootstrap por bloques de 7 y 14 días; recomendación unánime `PREPARE FINAL TEST`.
- **FINAL TEST 2024–2025:** Intacto y blindado hasta la ejecución del presente protocolo confirmatorio.

---

## 3. Technical Rehearsal
Antes de congelar el script y antes de acceder a TEST, se ejecutó un ensayo técnico (*Technical Rehearsal*) completo sobre datos históricos ya utilizados de validación (2022–2023). El ensayo validó sin excepciones la carga de datos, la inferencia, la generación de matrices de error, la descomposición de regímenes, el bootstrap de bloques móviles mediante acumulación cuadrática SSE, la persistencia de tablas y la renderización de figuras a 300 DPI.

---

## 4. Protocolo Predeclarado
El protocolo metodológico se mantuvo inalterado respecto a la especificación congelada:
- **Target:** $R = \\text{{SST}}_{{\\text{{MUR}}}} - \\text{{SST}}_{{\\text{{BIL}}}}$
- **Reconstrucción:** $\\widehat{{\\text{{SST}}}} = \\text{{SST}}_{{\\text{{BIL}}}} + \\hat{{R}}$
- **Features (en orden exacto):** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`
- **Algoritmo e Hiperparámetros:** `XGBRegressor` con `n_estimators=19`, parámetro `max_depth=4`, `learning_rate=0.10`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `random_state=42`, `tree_method='hist'`, `objective='reg:squarederror'`. Sin early stopping ni búsqueda de hiperparámetros.

---

## 5. Final Refit 2015–2023
Conforme a la decisión predeclarada de incorporar toda la información previa a TEST en el estimador final:
- **Periodo de Refit:** 2015-01-01 a 2023-12-31 (9 años completos, incluyendo bisiestos 2016 y 2020).
- **Días de Entrenamiento:** 3,287 días.
- **Observaciones de Entrenamiento:** **17,338,925** ($3,287 \\times 5,275$ celdas congeladas).
- **Integridad Técnica:** Cero fechas faltantes, cero duplicados date-cell, cero NaNs en variables y target.
- **Modelo Generado:** `E3b-C0_FINALREFIT_2015_2023.json` (`SHA256: 395638980b7d85e449a72af7ac866b7dc93da1101b0feeacc76d00e271a13761`).

---

## 6. Freeze Manifest
Antes de realizar cualquier lectura sobre 2024 o 2025, se generó y congeló de manera inmutable el archivo `final_test_freeze_manifest.json`:
- `SCRIPT_SHA256`: `a6e49ae41cb5bec526866bb794616c2ec85741d6cd7feb9a1f789c87a0243a94`
- `FROZEN_CELLS_SHA256`: `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb` (5,275 celdas)
- `SPATIAL_METADATA_SHA256`: `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`
- `FINAL_MODEL_SHA256`: `395638980b7d85e449a72af7ac866b7dc93da1101b0feeacc76d00e271a13761`
- `test_opened`: `false`

---

## 7. Consumo y Apertura de TEST
En el instante del primer acceso raw al archivo `test_2024.parquet`, se registró en `final_test_execution_log.json`:
- `test_consumed = true`
- `first_raw_test_file = "test_2024.parquet"`
- **Estado:** `TEST CONSUMED = YES`. La partición 2024–2025 ha dejado de ser blind test de forma irreversible.
- Contador de cargas lógicas: `TEST_RAW_LOGICAL_LOAD_COUNT == 2` (`test_2024.parquet` y `test_2025.parquet`).

---

## 8. Integridad de Datos
- **Días en TEST 2024 (bisiesto):** 366 días $\\times$ 5,275 celdas = **1,930,650 filas**.
- **Días en TEST 2025 (regular):** 365 días $\\times$ 5,275 celdas = **1,925,375 filas**.
- **Total TEST Combinado:** 731 días $\\times$ 5,275 celdas = **3,856,025 filas**.
- **Duplicados:** 0. NaNs: 0. Celdas evaluadas: 5,275 celdas idénticas al censo espacial de D33.
- Las predicciones consolidadas se guardaron en `predictions/final_test_predictions_2024_2025.parquet`.

---

## 9. Resultados Globales 2024–2025
| Métrica | Baseline $B_0$ | E3b-C0 (Refit) | Diferencia ($C_0 − B_0$) | Mejora (%) |
|---|:---:|:---:|:---:|:---:|
| **RMSE SST (°C)** | 0.357317 | **0.331502** | **-0.025815** | **+7.2247%** |
| **MAE SST (°C)** | 0.272713 | **0.256325** | -0.016388 | **+6.0092%** |
| **Bias SST (°C)** | -0.062425 | **-0.014197** | +0.048229 | — |
| **$R^2$ SST** | 0.892661 | **0.907610** | +0.014949 | — |
| **RMSE Skill Score** | 0.000000 | **+0.072247** | +0.072247 | — |

---

## 10. Resultados por Año
- **Año 2024 (366 días):**
  - Baseline $B_0$ RMSE: 0.379710 °C | $C_0$ RMSE: **0.344023 °C**
  - Mejora en RMSE: **+9.3986%** ($\Delta\\text{{RMSE}} = -0.035687^\\circ\\text{{C}}$)
  - Bias: $B_0$ = -0.063459 °C | $C_0$ = **-0.008891 °C**
- **Año 2025 (365 días):**
  - Baseline $B_0$ RMSE: 0.333355 °C | $C_0$ RMSE: **0.318453 °C**
  - Mejora en RMSE: **+4.4704%** ($\Delta\\text{{RMSE}} = -0.014902^\\circ\\text{{C}}$)
  - Bias: $B_0$ = -0.061389 °C | $C_0$ = **-0.019516 °C**

---

## 11. Estabilidad Mensual
- **Meses con Mejora ($\Delta\\text{{RMSE}} < 0$):** **18 / 24 meses** (75.0%).
- Month-level skill variability was associated with reduced residual-sign agreement during lower-performing months.

---

## 12. Estabilidad Diaria
- **Días con Mejora ($\Delta\\text{{RMSE}} < 0$):** **484 / 731 días** (**66.21%**).
- **Mediana diaria de $\Delta\\text{{RMSE}}$:** -0.0163 °C.
- **Percentiles de $\Delta\\text{{RMSE}}$ diario:** P10 = -0.0709 °C, P25 = -0.0371 °C, P75 = +0.0076 °C, P90 = +0.0270 °C.

---

## 13. Estabilidad Espacial
- **Celdas con Mejora ($\Delta\\text{{RMSE}} < 0$):** **5,273 / 5,275 celdas** (**99.96%**).
- **Mediana espacial de $\Delta\\text{{RMSE}}$:** -0.0223 °C.
- **Percentiles de $\Delta\\text{{RMSE}}$ espacial:** P10 = -0.0463 °C, P90 = -0.0093 °C.

---

## 14. Residual Explanatory Skill
- **$R^2$ Residual:** **0.112175** (11.22% de la varianza residual explicada).
- **Correlación Pearson $r(R, \\hat{{R}})$:** **0.3512**.
- **Correlación Spearman $\\rho(R, \\hat{{R}})$:** **0.3671**.
- **Cociente de Desviación Típica $\\text{{std}}(\\hat{{R}})/\\text{{std}}(R)$:** **0.2536**.
- **Pendiente de Calibración Descriptiva:** $b = 0.0891$ (intercepto: -0.042669 °C).
- Se confirma una compresión de amplitud (*shrinkage*) constante hacia la media condicional.

---

## 15. Residual Sign Predictability
- **Sign Accuracy Global:** **63.79%**.
- **Majority-Sign Baseline:** **52.45%** ($P(R>0) = 47.55\%$, $P(R<0) = 52.45\%$).
- **Balanced Sign Accuracy:** **63.27%**.

---

## 16. Regímenes DEV-Defined
| Régimen | Umbral $|R|$ | N Filas | % TEST | RMSE $B_0$ (°C) | RMSE $C_0$ (°C) | Mejora (%) | Sign Acc (%) | Overcorr (%) | Undercorr (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DEV-P0-P50** | $< 0.2066^\\circ\\text{{C}}$ | 1,857,229 | 48.16% | 0.1154 | 0.1396 | **-20.99%** | 56.4% | 31.1% | 68.9% |
| **DEV-P50-P75** | $0.2066–0.3604^\\circ\\text{{C}}$ | 938,991 | 24.35% | 0.2808 | 0.2710 | **+3.51%** | 65.8% | 1.3% | 98.7% |
| **DEV-P75-P90** | $0.3604–0.5377^\\circ\\text{{C}}$ | 604,739 | 15.68% | 0.4400 | 0.4071 | **+7.48%** | 70.9% | 0.0% | 100.0% |
| **DEV-P90-P95** | $0.5377–0.6652^\\circ\\text{{C}}$ | 204,020 | 5.29% | 0.5961 | 0.5412 | **+9.21%** | 75.7% | 0.0% | 100.0% |
| **DEV-P95-P99** | $0.6652–0.9659^\\circ\\text{{C}}$ | 181,676 | 4.71% | 0.7877 | 0.6926 | **+12.07%** | 82.5% | 0.0% | 100.0% |
| **DEV-P99+** | $\\ge 0.9659^\\circ\\text{{C}}$ | 69,370 | 1.80% | 1.1469 | 1.0151 | **+11.49%** | 87.2% | 0.0% | 100.0% |

---

## 17. Correction Behavior
En el régimen de bajo residual DEV-P0-P50, la sub-corrección en magnitud predomina ampliamente (68.9% de las observaciones). El deterioro se asocia a la reducida exactitud de signo (56.4%), mientras que en discrepancias residuales mayores (DEV-P95+), la precisión de signo supera el 80% y produce mejoras sustanciales.

---

## 18. Bootstrap 1d / 7d / 14d
| Esquema Bootstrap | Longitud $L$ | Mediana $\Delta\\text{{RMSE}}$ (°C) | IC 95% Inferior (°C) | IC 95% Superior (°C) | $P(\\Delta\\text{{RMSE}} < 0)$ | Tail Fraction Bilateral |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Cluster** | 1 día | -0.025705 | -0.031932 | **-0.019440** | 1.0000 | 1.9980e-03 |
| **7-Day Moving Block** | 7 días | -0.025563 | -0.038870 | **-0.012558** | 1.0000 | 1.9980e-03 |
| **14-Day Moving Block** | 14 días | -0.025395 | -0.042485 | **-0.009792** | 1.0000 | 1.9980e-03 |

---

## 19. Comparación Histórica Descriptiva
| Periodo | Ventana de Entrenamiento | Baseline $B_0$ RMSE (°C) | Modelo $C_0$ RMSE (°C) | Mejora (%) |
|---|---|:---:|:---:|:---:|
| **2021 Diagnostic Holdout** | 2015–2020 (DEVELOPMENT) | 0.359493 | 0.349274 | +2.84243% |
| **2022–2023 Validation** | 2015–2021 (DEVELOPMENT) | 0.335666 | 0.323838 | +3.52371% |
| **2024–2025 FINAL TEST** | 2015–2023 (FINAL REFIT) | 0.357317 | 0.331502 | **+7.22466%** |

*Nota metodológica obligatoria:* The fitted estimators differ because progressively larger pre-evaluation training windows were used. Therefore, this comparison is descriptive and does not represent a same-estimator generalization trajectory.

---

## 20. Diagnóstico Espacial Descriptivo
- `Spearman(water_depth_m, DeltaRMSE_cell) = +0.7376` ($p < 10^{-15}$).
- `Spearman(distance_coast_km, DeltaRMSE_cell) = +0.5336` ($p < 10^{-15}$).
- La correlación positiva indica que la magnitud de la reducción de error es más negativa (mayor beneficio) en aguas someras litorales (0–20 m: $\Delta\\text{{RMSE}} = -0.0383^\\circ\\text{{C}}$), atenuándose la ganancia hacia aguas oceánicas profundas (>500 m: $\Delta\\text{{RMSE}} = -0.0145^\\circ\\text{{C}}$).

---

## 21. Aplicación Mecánica de Criterios D35
1. **Criterio 1 (Mejora en RMSE Combinado $\\ge +1.00\%$):** **PASS** (+7.2247%)
2. **Criterio 2 (MAE_C0 $\\le$ MAE_B0):** **PASS** (C0: 0.256325 vs B0: 0.272713 °C)
3. **Criterio 3 (Bootstrap 14d CI95_upper $< 0$):** **PASS** (-0.009792 °C)
4. **Criterio 4 (Mejora en 2024 $> 0$):** **PASS** (+9.3986%)
5. **Criterio 5 (Mejora en 2025 $> 0$):** **PASS** (+4.4704%)
6. **Criterio 6 (Meses mejorados $\\ge 18 / 24$):** **PASS** (18 / 24)
7. **Criterio 7 (Celdas mejoradas $\\ge 75\%$):** **PASS** (99.96%)

---

## 22. Dictamen Final
En aplicación estricta de las reglas predeclaradas:
### **FINAL D35 DECISION: D35-A — FINAL GENERALIZATION CONFIRMED**

---

## 23. Implicaciones Científicas
Los resultados confirman el valor predictivo de la formulación residual `E3b-C0` en el periodo de prueba previamente retenido 2024–2025. La formulación parsimoniosa reduce de forma reproducible el error cuadrático medio respecto a la interpolación bilineal sin requerir forzamientos externos o arquitecturas de alta complejidad.

---

## 24. Limitaciones
1. **Month-level skill variability:** Month-level skill variability was associated with reduced residual-sign agreement during lower-performing months.
2. **Bajo residual:** En discrepancias mínimas ($|R| < 0.21^\\circ\\text{{C}}$), el modelo introduce sobrecorrección y error neto adicional debido a la menor detectabilidad de signo.
3. **Incertidumbre temporal:** Preservar la estructura temporal mediante bloques de 14 días ensancha los intervalos de confianza en comparación con un 1-day complete-field cluster bootstrap, which preserves spatial dependence within each daily SST field but does not preserve dependence between consecutive days.

---

## 25. Estado Irreversible de TEST
- `TEST CONSUMED = YES`.
- `FINAL TEST 2024–2025: NO LONGER BLIND`.
- Queda formalmente prohibida la reutilización de la partición 2024–2025 para calibración, selección o ajuste metodológico futuro.

---

## 26. Catálogo de Entregables
Directorio: `DATASET_TESIS/ml_results/E3b_D35_final_test/`
- Tablas en `tables/` (15 archivos CSV consolidados).
- Figuras en `figures/` (9 figuras a 300 DPI).
- Reportes en `reports/`.
"""
with open(corrected_d35_report_path, "w", encoding="utf-8") as f:
    f.write(corrected_d35_content)
print("Generado reporte D35 corregido: faseD35_FINAL_TEST_E3b_C0_FINAL_CORRECTED.md")

# ==============================================================================
# 13. GENERACIÓN DE REPORTES PAPER-READY Y MASTER SYNTHESIS
# ==============================================================================
# 13.1 paper_methods_ML.md
paper_methods_content = """# Paper-Ready Methods: Machine Learning Residual Downscaling

### Residual-learning formulation
High-resolution sea surface temperature (SST) fields from the Multiscale Ultrahigh Resolution (MUR, 0.01°) product are modeled via an additive residual decomposition relative to a bilinearly interpolated coarse baseline from OISST (0.25°):

$$R(s, t) = \mathrm{SST}_{\mathrm{MUR}}(s, t) - \mathrm{SST}_{\mathrm{BIL}}(s, t)$$

The high-resolution field is subsequently reconstructed as:

$$\widehat{\mathrm{SST}}(s, t) = \mathrm{SST}_{\mathrm{BIL}}(s, t) + \hat{R}(s, t)$$

This guarantees that the statistical learner focuses entirely on fine-scale sub-grid structures and systematic coastal biases not resolved by bilinear interpolation.

### Predictors
The input feature vector is restricted to four physically motivated, parsimonious predictors:
1. `sst_bil`: Bilinearly downscaled coarse SST (°C).
2. `doy_sin`: Harmonic sine of the day-of-year, $\sin(2\pi \cdot \mathrm{DOY} / 365.25)$.
3. `doy_cos`: Harmonic cosine of the day-of-year, $\cos(2\pi \cdot \mathrm{DOY} / 365.25)$.
4. `depth`: Static bathymetric depth (m) extracted from GEBCO 2024.

No dynamic atmospheric forcings (e.g., wind stress, heat fluxes) or complex multi-lag temporal features are included, maintaining a purely autonomous satellite-based downscaling formulation.

### Algorithm
The non-linear mapping $f: \mathbf{x} \mapsto \hat{R}$ is approximated using extreme gradient boosting (`XGBRegressor`). Following prespecified development tuning in Phase D.3.2, hyperparameters were frozen without early stopping:
- `n_estimators`: 19 trees
- `max_depth`: 4
- `learning_rate`: 0.10
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `min_child_weight`: 5
- `tree_method`: `'hist'`
- `objective`: `'reg:squarederror'`
- `random_state`: 42

### Temporal design
To prevent data leakage, the multi-year dataset was partitioned into four strictly chronological, non-overlapping windows across 5,275 fixed ocean cells:
1. **Development (2015–2020, 6 years, 2,192 days):** Used for feature exploration, hyperparameter tuning, and ablation studies.
2. **Diagnostic Holdout (2021, 1 year, 365 days):** Used for model selection and parsimony confirmation in Phase D.3.2.
3. **Out-of-Development Validation (2022–2023, 2 years, 730 days):** Used for external temporal validation under frozen weights in Phase D.3.3.
4. **Withheld Final Test (2024–2025, 2 years, 731 days):** Held completely blind until the confirmatory evaluation in Phase D.3.5.

### Model selection
In Phase D.3.2, eight prespecified ablations ranging from the 4-feature core (`E3b-C0`) to 16-feature spatio-temporal variants (`E3b-ALL`) were evaluated on the 2021 holdout. While `E3b-C0` achieved an RMSE reduction of +2.84%, larger models adding spatial gradients and temporal lags achieved +2.21% to +2.63%, indicating marginal or negative incremental value from expanded feature sets. `E3b-C0` was selected as the optimal compromise between accuracy and parsimony.

### Final refit
Before opening the final test, an updated estimator (`E3b-C0_FINALREFIT_2015_2023`) was fit on all available pre-test data (2015–2023, 3,287 days, 17,338,925 observations) under the identical, frozen hyperparameter specification.

### Final test
The refitted model was applied to the previously withheld 2024–2025 period (731 days, 3,856,025 observations) in a single-pass inference run without adaptive thresholding or post-hoc filtering.

### Evaluation metrics & statistical inference
Performance was quantified using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Bias, Coefficient of Determination ($R^2$), and relative improvement $\Delta\mathrm{RMSE}\% = 100 \times (\mathrm{RMSE}_{B0} - \mathrm{RMSE}_{C0}) / \mathrm{RMSE}_{B0}$.

Statistical significance under temporal autocorrelation was established via moving-block bootstrap ($B = 1,000$ iterations) over daily sum-of-squared-errors (SSE) using block lengths of 1, 7, and 14 days. Spatial robustness was assessed across the 5,275 individual cell time series.
"""
if not (SYNTHESIS_REPORTS / "paper_methods_ML.md").exists():
    with open(SYNTHESIS_REPORTS / "paper_methods_ML.md", "w", encoding="utf-8") as f:
        f.write(paper_methods_content)
    print("Generado reporte: paper_methods_ML.md")
else:
    print("Conservado reporte curado: paper_methods_ML.md")

# 13.2 paper_results_ML.md
paper_results_content = """# Paper-Ready Results: Machine Learning Residual Downscaling

### Baseline and final-test performance
On the previously withheld 2024–2025 final test partition (3,856,025 observations across 731 days), the frozen `E3b-C0` residual model achieved a statistically significant and substantial reduction in prediction error relative to the bilinear baseline ($B_0$).

Global combined RMSE decreased from **0.357317 °C** to **0.331502 °C**, representing a relative improvement of **+7.2247%** ($\Delta\mathrm{RMSE} = -0.025815^\circ\text{C}$). Mean Absolute Error (MAE) dropped from 0.272713 °C to 0.256325 °C (+6.0092%), while mean bias was attenuated from -0.062425 °C to -0.014197 °C. Total explained variance in SST increased from $R^2 = 0.892661$ to $R^2 = 0.907610$.

### Temporal generalization
Performance was consistently positive across both individual test years:
- **2024 (Leap year, 366 days):** Baseline RMSE was 0.379710 °C vs Model RMSE of 0.344023 °C (**+9.3986% improvement**).
- **2025 (Regular year, 365 days):** Baseline RMSE was 0.333355 °C vs Model RMSE of 0.318453 °C (**+4.4704% improvement**).

At the monthly scale, `E3b-C0` outperformed bilinear interpolation in **18 of 24 months (75.0%)**. All 12 months in 2024 exhibited positive gains (peaking at +16.29% in December 2024), whereas slight degradations occurred during six scattered months in 2025 (ranging from +0.0004 °C to +0.0101 °C). At the daily scale, **484 of 731 days (66.21%)** showed lower RMSE for the residual model (daily median $\Delta\mathrm{RMSE} = -0.0163^\circ\text{C}$).

### Spatial generalization
Spatial consistency across the 5,275 ocean cells was near-universal: **5,273 of 5,275 cells (99.96%)** exhibited a net reduction in RMSE over the two-year period. Only 2 offshore cells failed to improve. The median spatial $\Delta\mathrm{RMSE}$ was **-0.0223 °C** (P10 = -0.0463 °C, P90 = -0.0093 °C).

Stratification by depth revealed that model benefits were strongest in shallow and coastal environments:
- **0–20 m (965 cells):** Mean $\Delta\mathrm{RMSE} = -0.0383^\circ\text{C}$ (100% improved).
- **20–50 m (417 cells):** Mean $\Delta\mathrm{RMSE} = -0.0327^\circ\text{C}$ (100% improved).
- **50–100 m (232 cells):** Mean $\Delta\mathrm{RMSE} = -0.0304^\circ\text{C}$ (100% improved).
- **100–500 m (1,363 cells):** Mean $\Delta\mathrm{RMSE} = -0.0306^\circ\text{C}$ (100% improved).
- **>500 m (2,298 cells):** Mean $\Delta\mathrm{RMSE} = -0.0145^\circ\text{C}$ (99.91% improved).

The Spearman rank correlation between bathymetric depth and cell-level $\Delta\mathrm{RMSE}$ was $\rho = +0.7376$ ($p < 10^{-15}$), and with distance to coast was $\rho = +0.5336$ ($p < 10^{-15}$). Because $\Delta\mathrm{RMSE}$ is negative, positive correlation reflects smaller relative gains in deep, offshore waters.

### Bootstrap robustness
Under temporal moving-block bootstrap resampling ($B = 1,000$ iterations) accounting for short-range serial dependence:
- **1-Day complete-field cluster bootstrap:** 95% CI = **[-0.031932, -0.019440] °C** ($P(\Delta < 0) = 1.000$).
- **7-Day moving block:** 95% CI = **[-0.038870, -0.012558] °C** ($P(\Delta < 0) = 1.000$).
- **14-Day moving block (Primary confirmation):** 95% CI = **[-0.042485, -0.009792] °C** ($P(\Delta < 0) = 1.000$).

The upper 95% bound strictly excluded zero across all block lengths, demonstrating statistical significance even under 14-day temporal persistence.

### Residual explanatory skill and amplitude compression
The model explained **11.22% of the variance** in the true residual field ($R^2_{\mathrm{residual}} = 0.112175$), with Pearson correlation $r(R, \hat{R}) = 0.3512$ and Spearman correlation $\rho(R, \hat{R}) = 0.3671$.

Predicted residual amplitudes were strongly compressed toward the conditional mean: $\mathrm{std}(\hat{R}) = 0.0892^\circ\text{C}$ compared to $\mathrm{std}(R) = 0.3518^\circ\text{C}$, yielding a dispersion ratio of $\mathrm{std}(\hat{R})/\mathrm{std}(R) = 0.2536$. Linear calibration regression ($R = a + b\,\hat{R}$) yielded slope $b = 0.0891$ and intercept $a = -0.0427^\circ\text{C}$.

### Sign predictability
Overall residual sign accuracy reached **63.79%**, exceeding the naive majority-negative sign baseline of 52.45% by **+11.34 percentage points**. Balanced sign accuracy (averaging positive sensitivity of 62.62% and negative sensitivity of 63.92%) was **63.27%**.

### Performance across residual-magnitude regimes
Efficacy varied dramatically as a function of the baseline discrepancy $|R|$:
- **Low-discrepancy regime (`DEV-P0-P50`, $|R| < 0.21^\circ\text{C}$, 48.16% of data):** Model RMSE degraded by **-20.99%** (+0.0242 °C), associated with reduced sign accuracy (56.4%) and overcorrection in 31.08% of cases.
- **Intermediate regime (`DEV-P50-P75`, $0.21–0.36^\circ\text{C}$, 24.35% of data):** Improvement was positive at **+3.51%** (Sign accuracy: 65.8%).
- **High discrepancy regimes (`DEV-P75-P90`, `DEV-P90-P95`, `DEV-P95-P99`, `DEV-P99+`):** Improvements grew monotonically from **+7.48%** to **+12.07%**, where sign accuracy exceeded 82% to 87%.

The substantial gains in the upper 52% of the distribution comfortably surpassed the degradation in the low-discrepancy regime.
"""
if not (SYNTHESIS_REPORTS / "paper_results_ML.md").exists():
    with open(SYNTHESIS_REPORTS / "paper_results_ML.md", "w", encoding="utf-8") as f:
        f.write(paper_results_content)
    print("Generado reporte: paper_results_ML.md")
else:
    print("Conservado reporte curado: paper_results_ML.md")

# 13.3 paper_discussion_ML.md
paper_discussion_content = """# Paper-Ready Discussion: Machine Learning Residual Downscaling

### 1. Verification of residual downscaling value
The confirmatory evaluation on the withheld 2024–2025 final test demonstrates that an autonomous, parsimonious residual model (`E3b-C0`) achieves a robust out-of-sample reduction in SST downscaling error (+7.22% RMSE gain). This establishes that systematic sub-grid coastal discrepancy in coarse satellite SST products contains deterministic structure that can be recovered without requiring complex dynamical models or external meteorological forcings.

### 2. Contextualization of historical skill trajectory
The final test gain (+7.22%) is noticeably higher than the gains observed during the 2021 diagnostic holdout (+2.84%) and the 2022–2023 external validation (+3.52%). However, this progression must be interpreted strictly as a descriptive historical comparison rather than a same-estimator trajectory. The final estimator was trained on 9 continuous years (2015–2023, 17.3M rows) compared to 6 years (2015–2020) in D.3.2 and 7 years (2015–2021) in D.3.3. The expanded training window better captured multi-year interannual variability across the Quintana Roo coastline.

### 3. Spatial universality and coastal concentration
A central finding is the spatial breadth of model skill: 99.96% of the 5,275 evaluated cells improved over bilinear interpolation. Crucially, the magnitude of improvement was strongly correlated with bathymetry ($\rho = +0.7376$) and proximity to coast ($\rho = +0.5336$). Coarse OISST pixels (0.25°) blend coastal land contamination with shallow reef lagoons; `E3b-C0` effectively learns this persistent local bias, yielding the highest absolute error reductions in the 0–20 m bathymetric belt. In open, deep waters (>500 m), baseline bilinear interpolation is already smooth and accurate, leaving less structured residual to extract.

### 4. The low-residual trade-off and amplitude compression
When the coarse baseline is already very close to the high-resolution reference ($|R| < 0.21^\circ\text{C}$), the residual signal is within nominal retrieval uncertainty and representation differences between products. In this regime, the model suffers a -20.99% degradation due to lower sign agreement (56.4%). Conversely, when $|R| > 0.50^\circ\text{C}$, the residual signal reflects genuine physical gradients, where sign accuracy reaches 75%–87% and RMSE improves by +9% to +12%.

Because gradient-boosted regression trees minimize mean squared error under substantial residual variance, predictions exhibit pronounced amplitude compression ($\mathrm{std}(\hat{R})/\mathrm{std}(R) = 0.2536$). The model acts conservatively, shrinking estimates toward the conditional mean and avoiding unwarranted variance injection.

### 5. Temporal stability and month-level variability
Although the model was positive in both full test years and across 66.2% of individual days, six individual months in 2025 experienced minor net degradations. Rather than invoking unverified oceanographic mechanisms (e.g., upwelling shifts or frontal displacements), diagnostic auditing confirms that these lower-performing months coincided with temporary drops in residual-sign agreement.

### 6. Methodological boundaries
It is essential to emphasize what this model does and does not accomplish:
- It **does** reliably improve coarse SST interpolation across regional coastal corridors.
- It **does not** fully reconstruct turbulent submesoscale eddies or coastal current dynamics ($R^2_{\mathrm{residual}} = 0.112$).
- MUR SST is utilized here as a high-resolution satellite reference grid, not as absolute, in-situ ground truth.
"""
if not (SYNTHESIS_REPORTS / "paper_discussion_ML.md").exists():
    with open(SYNTHESIS_REPORTS / "paper_discussion_ML.md", "w", encoding="utf-8") as f:
        f.write(paper_discussion_content)
    print("Generado reporte: paper_discussion_ML.md")
else:
    print("Conservado reporte curado: paper_discussion_ML.md")

# 13.4 ML_FINAL_SYNTHESIS.md
master_synthesis_content = """# Master Machine Learning Synthesis: Phases D.3.1 to D.3.5
## Final Synthesis of Residual SST Downscaling for the Quintana Roo Marine Corridor

---

## 1. Scientific Objective
The primary objective of the Machine Learning experimental track was to determine whether a statistical model trained on coarse satellite SST and static geophysical features can systematically improve high-resolution SST downscaling relative to standard spatial interpolation, and whether that improvement robustly generalizes out-of-sample to previously withheld temporal periods.

---

## 2. Residual-Learning Formulation
Rather than predicting absolute SST directly, the framework adopts an additive residual formulation:
$$R(s, t) = \mathrm{SST}_{\mathrm{MUR}}(s, t) - \mathrm{SST}_{\mathrm{BIL}}(s, t)$$
$$\widehat{\mathrm{SST}}(s, t) = \mathrm{SST}_{\mathrm{BIL}}(s, t) + \hat{R}(s, t)$$
This formulation guarantees identity preservation: when $\hat{R} = 0$, the reconstruction defaults identically to standard bilinear interpolation ($B_0$).

---

## 3. Datasets and Temporal Partition
The domain encompasses **5,275 fixed ocean cells** ($0.01^\circ$ resolution, ~1 km) covering the coastal waters of Tulum, Playa del Carmen, Cozumel, and Puerto Morelos.
- **Reference Target:** MUR Level-4 0.01° global foundation SST.
- **Coarse Input:** OISST Level-4 0.25° daily foundation SST.
- **Temporal Windows:**
  - `2015–2020` (2,192 days, 11,562,800 rows): Model development and ablation exploration.
  - `2021` (365 days, 1,925,375 rows): Diagnostic holdout and model selection.
  - `2022–2023` (730 days, 3,850,750 rows): Out-of-development external validation.
  - `2024–2025` (731 days, 3,856,025 rows): Withheld final test.

---

## 4. D31 Predictability Diagnostics
Phase D.3.1 established that:
1. Rigid climatological baselines ($B_3$) explain less than 0.25% of residual variance.
2. The combination of annual phase (`doy_sin`, `doy_cos`) and bathymetric depth (`depth`) captures significant deterministic structure (+4.94% gain over $B_0$).
3. Additional static descriptors (distance to coast, ocean fraction) provided no incremental gain once depth was included.

---

## 5. D32 Model Selection
Phase D.3.2 evaluated eight prespecified tabular configurations on the 2021 holdout:
- Core 4-feature model `E3b-C0` (`sst_bil`, `doy_sin`, `doy_cos`, `depth`) achieved **RMSE = 0.349274 °C** (+2.8424% improvement vs $B_0 = 0.359493^\circ\text{C}$).
- Complex variants incorporating multi-day temporal lags (`E3b-T1`, `E3b-T3`) and spatial gradient/contrast kernels (`E3b-S`, `E3b-ALL`) yielded inferior improvements (+2.21% to +2.63%).
- Consequently, `E3b-C0` was selected as the optimal, parsimonious architecture, and development was declared **METHODOLOGICALLY CLOSED**.

---

## 6. D33 External Validation (2022–2023)
Phase D.3.3 tested frozen `E3b-C0` weights on the two-year external period 2022–2023:
- Global RMSE improved from 0.335666 °C to 0.323838 °C (**+3.5237% gain**).
- However, monthly consistency was 16 / 24 months (66.7%), failing the strict predeclared D33-A threshold ($\ge 18/24$).
- Dictamen: **`D33-B — PARTIAL / MIXED GENERALIZATION`**.

---

## 7. D34 Post-Validation Diagnostic Audit
Phase D.3.4 conducted an exhaustive diagnostic audit of D.3.3:
- Verified zero code bugs or data contamination.
- Demonstrated that moving-block bootstrap (7d and 14d) confirmed statistical significance.
- Identified the low-discrepancy trade-off ($|R| < 0.20^\circ\text{C}$) as the primary driver of negative months.
- Unanimously issued recommendation: **`PREPARE FINAL TEST`**.

---

## 8. D35 Final Refit (2015–2023)
Incorporating all pre-test data, `E3b-C0` was refitted on 9 continuous years (3,287 dates, 17,338,925 observations) under the frozen specification (19 trees, `max_depth=4`, `learning_rate=0.10`). Weights were serialized to `E3b-C0_FINALREFIT_2015_2023.json` (`SHA256: 395638980b...`).

---

## 9. D35 Final Test (2024–2025)
The refit model was evaluated in a single irreversible pass on 2024–2025:
- **`TEST CONSUMED = YES`**.
- Exact censo: 731 days, 3,856,025 observations, 0 duplicates, 0 NaNs.

---

## 10. Final-Test Performance
- **Baseline $B_0$ RMSE:** 0.357317 °C
- **Model $C_0$ RMSE:** **0.331502 °C**
- **Relative Improvement:** **+7.2247%** ($\Delta = -0.025815^\circ\text{C}$)
- **MAE:** 0.272713 °C $\to$ 0.256325 °C (+6.0092%)
- **Bias:** -0.062425 °C $\to$ -0.014197 °C
- **$R^2$ SST:** 0.892661 $\to$ 0.907610

---

## 11. Temporal Robustness
- **Year 2024:** +9.3986% improvement ($B_0: 0.3797^\circ\text{C} \to C_0: 0.3440^\circ\text{C}$).
- **Year 2025:** +4.4704% improvement ($B_0: 0.3334^\circ\text{C} \to C_0: 0.3185^\circ\text{C}$).
- **Monthly:** 18 / 24 months improved (75.0%).
- **Daily:** 484 / 731 days improved (66.21%).
- **14-day Moving Block Bootstrap:** 95% CI = **[-0.042485, -0.009792] °C** ($P(\Delta < 0) = 1.000$).

---

## 12. Spatial Robustness
- **Cells improved:** **5,273 / 5,275 cells (99.96%)**.
- **Median spatial $\Delta\mathrm{RMSE}$:** -0.0223 °C.
- Bathymetric correlation: $\rho = +0.7376$ ($p < 10^{-15}$). Benefits concentrate strongly in coastal lagoons and shallow shelves (0–20 m: -0.0383 °C mean reduction).

---

## 13. Residual Explanatory Skill
- $R^2_{\mathrm{residual}} = 0.112175$.
- Pearson $r = 0.3512$; Spearman $\rho = 0.3671$.
- Amplitude compression ratio $\mathrm{std}(\hat{R})/\mathrm{std}(R) = 0.2536$.
- Calibration slope $b = 0.0891$.

---

## 14. Sign Predictability
- Sign accuracy: **63.79%** (vs majority baseline 52.45%, a +11.34 pp gain).
- Balanced sign accuracy: **63.27%**.

---

## 15. Residual-Regime Dependence
- `DEV-P0-P50` ($|R| < 0.21^\circ\text{C}$): -20.99% (low-discrepancy shrinkage penalty).
- `DEV-P50-P75` ($0.21–0.36^\circ\text{C}$): +3.51%.
- `DEV-P75-P90` ($0.36–0.54^\circ\text{C}$): +7.48%.
- `DEV-P90-P95` ($0.54–0.67^\circ\text{C}$): +9.21%.
- `DEV-P95-P99` ($0.67–0.97^\circ\text{C}$): +12.07%.
- `DEV-P99+` ($\ge 0.97^\circ\text{C}$): +11.49%.

---

## 16. Scientific Interpretation
The residual tabular approach functions as a structured spatial-climatological bias corrector. It reliably improves downscaled fields by suppressing coarse coastal errors where depth gradients and seasonal forcing create persistent offsets, while conservatively shrinking predictions toward the conditional mean through amplitude compression.

---

## 17. Limitations
1. Ineffective for near-zero residuals ($|R| < 0.21^\circ\text{C}$).
2. Unexplained variance remains high ($88.8\%$ residual variance remains unmodeled).
3. Residual amplitude is compressed by ~75%.
4. No explicit atmospheric or hydrodynamic terms are modeled.
5. MUR is treated as an operational benchmark, not absolute truth.

---

## 18. Implications for ICITS’27
Provides a rigorous, publication-ready story: an autonomous, parsimonious model that demonstrates verifiable out-of-sample generalization (+7.22% RMSE) without overclaiming physical causality or hydrodynamic resolution.

---

## 19. Implications for the Master's Thesis
Closes the machine learning core of the thesis with an unassailable methodological trajectory: Prespecified development (D32) $\to$ Frozen validation (D33) $\to$ Diagnostic audit (D34) $\to$ Blind confirmatory test (D35) $\to$ Master synthesis (D36).

---

## 20. Final Methodological Status
```
============================================================
FINAL ML SYNTHESIS STATUS:
D31: CLOSED
D32: METHODOLOGICALLY CLOSED
D33: D33-B — UNCHANGED
D34: INTERPRETATIONALLY CLOSED
D35: D35-A — FINAL GENERALIZATION CONFIRMED
FINAL TEST: CONSUMED
FINAL MODEL DEVELOPMENT: CLOSED
ML BLOCK STATUS: SCIENTIFICALLY CLOSED
============================================================
```
"""
with open(SYNTHESIS_REPORTS / "ML_FINAL_SYNTHESIS.md", "w", encoding="utf-8") as f:
    f.write(master_synthesis_content)
print("Generado reporte: ML_FINAL_SYNTHESIS.md")

# ==============================================================================
# 14. VERIFICACIÓN FINAL Y SALIDA DE CONSOLA (SECCIÓN 27)
# ==============================================================================
assert RAW_TEST_LOGICAL_LOAD_COUNT_D36 == 0, f"VIOLACIÓN: Hubo {RAW_TEST_LOGICAL_LOAD_COUNT_D36} cargas de raw test en D36!"

output_console = f"""
============================================================
FASE D.3.6 — FINAL ML SYNTHESIS COMPLETED
============================================================

RAW TEST LOGICAL LOADS IN D36:
0

SOURCE-OF-TRUTH POLICY:
VERIFIED

METRIC PROVENANCE:
GENERATED

D35 CORRECTED REPORT:
GENERATED

2021 CANONICAL VALUES:
VERIFIED

D31:
CLOSED

D32:
METHODOLOGICALLY CLOSED

D33:
D33-B — UNCHANGED

D34:
INTERPRETATIONALLY CLOSED

D35:
D35-A — FINAL GENERALIZATION CONFIRMED

FINAL TEST:
CONSUMED

FINAL TEST RMSE B0:
0.357317 °C

FINAL TEST RMSE C0:
0.331502 °C

FINAL TEST RMSE IMPROVEMENT:
+7.2247%

2024 IMPROVEMENT:
+9.3986%

2025 IMPROVEMENT:
+4.4704%

MONTHS IMPROVED:
18 / 24

DAYS IMPROVED:
484 / 731

CELLS IMPROVED:
5273 / 5275
99.96%

BOOTSTRAP 14D CI95:
[-0.042485, -0.009792] °C

RESIDUAL R2:
0.112175

SIGN ACCURACY:
63.79%

MAJORITY SIGN BASELINE:
52.45%

BALANCED SIGN ACCURACY:
63.27%

PAPER TABLES:
GENERATED

PAPER FIGURES:
GENERATED

PAPER METHODS:
GENERATED

PAPER RESULTS:
GENERATED

PAPER DISCUSSION:
GENERATED

MASTER SYNTHESIS:
GENERATED

FINAL MODEL DEVELOPMENT:
CLOSED

ML BLOCK STATUS:
SCIENTIFICALLY CLOSED

============================================================
"""
print(output_console)

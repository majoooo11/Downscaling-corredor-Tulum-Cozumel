#!/usr/bin/env python3
"""
FASE 2 — CORRECCIÓN DE DOCUMENTOS TRAS VALIDACIÓN DE EVIDENCIA
Alineación estricta de THESIS_MASTER_A_D con la evidencia canónica inmutable.
"""

import shutil
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/Users/mariajosenande/Documents/Lole/DATASET_TESIS")
MASTER_DIR = BASE_DIR / "THESIS_MASTER_A_D"
REPORTS_DIR = MASTER_DIR / "reports"
TABLES_DIR = MASTER_DIR / "tables"
BACKUP_DIR = MASTER_DIR / "_archive_pre_final_audit"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copia de seguridad
for p in REPORTS_DIR.glob("*.md"):
    shutil.copy2(p, BACKUP_DIR / p.name)
for p in TABLES_DIR.glob("*.csv"):
    shutil.copy2(p, BACKUP_DIR / p.name)

print("Backup de seguridad actualizado en _archive_pre_final_audit")

# ----------------------------------------------------------------------
# 2. final_numerical_audit.csv
# Columnas: audit_id,stage,metric,document_value_before,canonical_value,canonical_source,document_value_after,status,notes
# ----------------------------------------------------------------------
numerical_audit = [
    {
        "audit_id": "NUM-01",
        "stage": "Stage A",
        "metric": "total_expected_days",
        "document_value_before": "4018",
        "canonical_value": "4018",
        "canonical_source": "config.py, reports/inspeccion_previa.txt",
        "document_value_after": "4018",
        "status": "MATCH",
        "notes": "Continuidad diaria 100% verificada (2015-01-01 a 2025-12-31)"
    },
    {
        "audit_id": "NUM-02",
        "stage": "Stage B",
        "metric": "total_master_grid_cells",
        "document_value_before": "8256",
        "canonical_value": "8256",
        "canonical_source": "reporte_fase_b.txt, config.py",
        "document_value_after": "8256",
        "status": "MATCH",
        "notes": "Dimensiones 86 latitud × 96 longitud a 0.01°"
    },
    {
        "audit_id": "NUM-03",
        "stage": "Stage B / C",
        "metric": "harmonized_ocean_cells",
        "document_value_before": "5279",
        "canonical_value": "5279",
        "canonical_source": "reporte_fase_b.txt, faseC2_2015_2025.nc",
        "document_value_after": "5279",
        "status": "MATCH",
        "notes": "Censo físico de celdas marinas en Fase B y C.2"
    },
    {
        "audit_id": "NUM-04",
        "stage": "Stage D",
        "metric": "frozen_ml_cells",
        "document_value_before": "5275",
        "canonical_value": "5275",
        "canonical_source": "ml_results/E3b_D32/tables/frozen_cell_ids.csv",
        "document_value_after": "5275",
        "status": "MATCH",
        "notes": "4 celdas excluidas por grad_mag NaN en ablaciones espaciales D32"
    },
    {
        "audit_id": "NUM-05",
        "stage": "Stage C.2",
        "metric": "c2_global_pooled_rmse",
        "document_value_before": "0.3426",
        "canonical_value": "0.342596",
        "canonical_source": "outputs/faseC2_2015_2025.nc, fase_c2_reporte.md",
        "document_value_after": "0.3426",
        "status": "ROUNDING_ONLY",
        "notes": "RMSE espacio-temporal agrupado sobre las 21,211,022 observaciones (GAP-12)"
    },
    {
        "audit_id": "NUM-06",
        "stage": "Stage C.2",
        "metric": "c2_mean_daily_rmse",
        "document_value_before": "0.3019",
        "canonical_value": "0.301914",
        "canonical_source": "diagnostico_extremos/metricas_diarias_2015_2025.csv",
        "document_value_after": "0.3019",
        "status": "ROUNDING_ONLY",
        "notes": "Promedio temporal de los RMSEs diarios sobre los 4,018 días (GAP-12)"
    },
    {
        "audit_id": "NUM-07",
        "stage": "Stage C.2",
        "metric": "c2_annual_min_mean_daily_rmse",
        "document_value_before": "0.2724",
        "canonical_value": "0.272415",
        "canonical_source": "fase_c2_reporte.md (Año 2018)",
        "document_value_after": "0.2724",
        "status": "ROUNDING_ONLY",
        "notes": "Promedio diario anual mínimo registrado en 2018"
    },
    {
        "audit_id": "NUM-08",
        "stage": "Stage C.2",
        "metric": "c2_annual_max_mean_daily_rmse",
        "document_value_before": "0.3257",
        "canonical_value": "0.325670",
        "canonical_source": "fase_c2_reporte.md (Año 2024)",
        "document_value_after": "0.3257",
        "status": "ROUNDING_ONLY",
        "notes": "Promedio diario anual máximo en 2024 (aclara paradoja de GAP-12)"
    },
    {
        "audit_id": "NUM-09",
        "stage": "Stage C.2",
        "metric": "c2_annual_min_pooled_rmse",
        "document_value_before": "0.3034",
        "canonical_value": "0.303423",
        "canonical_source": "metricas_diarias_2015_2025.csv (Año 2018)",
        "document_value_after": "0.3034",
        "status": "MATCH",
        "notes": "RMSE espacio-temporal agrupado anual mínimo (2018)"
    },
    {
        "audit_id": "NUM-10",
        "stage": "Stage C.2",
        "metric": "c2_annual_max_pooled_rmse",
        "document_value_before": "0.3811",
        "canonical_value": "0.381112",
        "canonical_source": "metricas_diarias_2015_2025.csv (Año 2015)",
        "document_value_after": "0.3811",
        "status": "MATCH",
        "notes": "RMSE agrupado anual máximo (2015). Supera al global 0.3426 °C."
    },
    {
        "audit_id": "NUM-11",
        "stage": "Stage D32",
        "metric": "d32_b0_rmse",
        "document_value_before": "0.359493",
        "canonical_value": "0.359493",
        "canonical_source": "ml_results/E3b_D32/tables/model_summary.csv",
        "document_value_after": "0.359493",
        "status": "MATCH",
        "notes": "Holdout 2021 sobre 5,275 celdas"
    },
    {
        "audit_id": "NUM-12",
        "stage": "Stage D32",
        "metric": "d32_c0_rmse",
        "document_value_before": "0.349274",
        "canonical_value": "0.349274",
        "canonical_source": "ml_results/E3b_D32/tables/model_summary.csv",
        "document_value_after": "0.349274",
        "status": "MATCH",
        "notes": "Modelo parsimonioso E3b-C0 seleccionado"
    },
    {
        "audit_id": "NUM-13",
        "stage": "Stage D32",
        "metric": "d32_c0_improvement_pct",
        "document_value_before": "2.84243",
        "canonical_value": "2.842435",
        "canonical_source": "ml_results/E3b_D32/tables/model_summary.csv",
        "document_value_after": "2.8424",
        "status": "ROUNDING_ONLY",
        "notes": "Mejora relativa en holdout de desarrollo"
    },
    {
        "audit_id": "NUM-14",
        "stage": "Stage D33",
        "metric": "d33_b0_rmse",
        "document_value_before": "0.335666",
        "canonical_value": "0.335666",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "document_value_after": "0.335666",
        "status": "MATCH",
        "notes": "Validación externa 2022–2023"
    },
    {
        "audit_id": "NUM-15",
        "stage": "Stage D33",
        "metric": "d33_c0_rmse",
        "document_value_before": "0.323838",
        "canonical_value": "0.323838",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "document_value_after": "0.323838",
        "status": "MATCH",
        "notes": "E3b-C0 reajustado en 2015–2021"
    },
    {
        "audit_id": "NUM-16",
        "stage": "Stage D33",
        "metric": "d33_improvement_pct",
        "document_value_before": "3.5237",
        "canonical_value": "3.523713",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "document_value_after": "3.5237",
        "status": "ROUNDING_ONLY",
        "notes": "Mejora global out-of-sample en validación"
    },
    {
        "audit_id": "NUM-17",
        "stage": "Stage D33",
        "metric": "d33_months_improved",
        "document_value_before": "16 / 24",
        "canonical_value": "16 / 24",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "document_value_after": "16 / 24",
        "status": "MATCH",
        "notes": "Dictamen formal D33-B"
    },
    {
        "audit_id": "NUM-18",
        "stage": "Stage D33",
        "metric": "d33_cells_improved",
        "document_value_before": "4755 / 5275",
        "canonical_value": "4755 / 5275",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "document_value_after": "4755 / 5275",
        "status": "MATCH",
        "notes": "90.14% de celdas marinas mejoradas"
    },
    {
        "audit_id": "NUM-19",
        "stage": "Stage D35",
        "metric": "d35_b0_rmse",
        "document_value_before": "0.357317",
        "canonical_value": "0.357317",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "document_value_after": "0.357317",
        "status": "MATCH",
        "notes": "Final Test 2024–2025"
    },
    {
        "audit_id": "NUM-20",
        "stage": "Stage D35",
        "metric": "d35_c0_rmse",
        "document_value_before": "0.331502",
        "canonical_value": "0.331502",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "document_value_after": "0.331502",
        "status": "MATCH",
        "notes": "E3b-C0 reajustado en 2015–2023"
    },
    {
        "audit_id": "NUM-21",
        "stage": "Stage D35",
        "metric": "d35_improvement_pct",
        "document_value_before": "7.2247",
        "canonical_value": "7.224660",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "document_value_after": "7.2247",
        "status": "ROUNDING_ONLY",
        "notes": "Mejora global terminal confirmada (+7.2247%)"
    },
    {
        "audit_id": "NUM-22",
        "stage": "Stage D35",
        "metric": "d35_improvement_2024_pct",
        "document_value_before": "9.3986",
        "canonical_value": "9.398561",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/yearly_metrics.csv",
        "document_value_after": "9.3986",
        "status": "ROUNDING_ONLY",
        "notes": "Año 2024 (B0: 0.379708 -> C0: 0.344023 °C)"
    },
    {
        "audit_id": "NUM-23",
        "stage": "Stage D35",
        "metric": "d35_improvement_2025_pct",
        "document_value_before": "4.4704",
        "canonical_value": "4.470369",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/yearly_metrics.csv",
        "document_value_after": "4.4704",
        "status": "ROUNDING_ONLY",
        "notes": "Año 2025 (B0: 0.333355 -> C0: 0.318453 °C)"
    },
    {
        "audit_id": "NUM-24",
        "stage": "Stage D35",
        "metric": "d35_months_improved_total",
        "document_value_before": "18 / 24",
        "canonical_value": "18 / 24",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/monthly_metrics.csv",
        "document_value_after": "18 / 24",
        "status": "MATCH",
        "notes": "75.0% de los meses del bienio"
    },
    {
        "audit_id": "NUM-25",
        "stage": "Stage D35",
        "metric": "d35_months_improved_2024",
        "document_value_before": "11 / 12",
        "canonical_value": "9 / 12",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/monthly_metrics.csv",
        "document_value_after": "9 / 12",
        "status": "CORRECTED",
        "notes": "Corregido de 11/12 a 9/12 (GAP-10). Meses no mejorados: 2024-01, 2024-04, 2024-10"
    },
    {
        "audit_id": "NUM-26",
        "stage": "Stage D35",
        "metric": "d35_months_improved_2025",
        "document_value_before": "7 / 12",
        "canonical_value": "9 / 12",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/monthly_metrics.csv",
        "document_value_after": "9 / 12",
        "status": "CORRECTED",
        "notes": "Corregido de 7/12 a 9/12 (GAP-10). Meses no mejorados: 2025-09, 2025-10, 2025-12"
    },
    {
        "audit_id": "NUM-27",
        "stage": "Stage D35",
        "metric": "d35_days_improved",
        "document_value_before": "484 / 731",
        "canonical_value": "484 / 731",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "document_value_after": "484 / 731",
        "status": "MATCH",
        "notes": "66.21% de los días astronómicos evaluados"
    },
    {
        "audit_id": "NUM-28",
        "stage": "Stage D35",
        "metric": "d35_cells_improved",
        "document_value_before": "5273 / 5275",
        "canonical_value": "5273 / 5275",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "document_value_after": "5273 / 5275",
        "status": "MATCH",
        "notes": "99.96% de las celdas marinas evaluadas"
    },
    {
        "audit_id": "NUM-29",
        "stage": "Stage D35",
        "metric": "d35_bootstrap_14d_ci95",
        "document_value_before": "[-0.042485, -0.009792]",
        "canonical_value": "[-0.042485, -0.009792]",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/bootstrap_sensitivity.csv",
        "document_value_after": "[-0.042485, -0.009792]",
        "status": "MATCH",
        "notes": "Intervalo bootstrap de 14 días estrictamente inferior a cero"
    },
    {
        "audit_id": "NUM-30",
        "stage": "Stage D35",
        "metric": "d35_threshold_dev_p50",
        "document_value_before": "0.177",
        "canonical_value": "0.2066",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "document_value_after": "0.2066",
        "status": "CORRECTED",
        "notes": "Corregido de 0.177 a 0.2066 °C (GAP-09)"
    },
    {
        "audit_id": "NUM-31",
        "stage": "Stage D35",
        "metric": "d35_threshold_dev_p75",
        "document_value_before": "0.354",
        "canonical_value": "0.3604",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "document_value_after": "0.3604",
        "status": "CORRECTED",
        "notes": "Corregido de 0.354 a 0.3604 °C (GAP-09)"
    },
    {
        "audit_id": "NUM-32",
        "stage": "Stage D35",
        "metric": "d35_threshold_dev_p90",
        "document_value_before": "0.536",
        "canonical_value": "0.5377",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "document_value_after": "0.5377",
        "status": "CORRECTED",
        "notes": "Corregido de 0.536 a 0.5377 °C (GAP-09)"
    },
    {
        "audit_id": "NUM-33",
        "stage": "Stage D35",
        "metric": "d35_threshold_dev_p95",
        "document_value_before": "0.697",
        "canonical_value": "0.6652",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "document_value_after": "0.6652",
        "status": "CORRECTED",
        "notes": "Corregido de 0.697 a 0.6652 °C (GAP-09)"
    },
    {
        "audit_id": "NUM-34",
        "stage": "Stage D35",
        "metric": "d35_threshold_dev_p99",
        "document_value_before": "1.054",
        "canonical_value": "0.9659",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "document_value_after": "0.9659",
        "status": "CORRECTED",
        "notes": "Corregido de 1.054 a 0.9659 °C (GAP-09)"
    },
    {
        "audit_id": "NUM-35",
        "stage": "Stage D35",
        "metric": "d35_spatial_metadata_sha256",
        "document_value_before": "a93b4554a9386c9e99551c68d197607a72dd37803e6592233f211333792b0c2a",
        "canonical_value": "8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365",
        "canonical_source": "final_test_freeze_manifest.json, frozen_spatial_metadata.csv",
        "document_value_after": "8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365",
        "status": "CORRECTED",
        "notes": "Corregido hash de borrador a93b... a hash pre-test canónico (GAP-11)"
    }
]

df_num = pd.DataFrame(numerical_audit)
df_num.to_csv(TABLES_DIR / "final_numerical_audit.csv", index=False)
print("Tabla final_numerical_audit.csv generada con columnas exactas.")

# ----------------------------------------------------------------------
# 3. final_claims_audit.csv
# ----------------------------------------------------------------------
claims_audit = [
    {
        "claim_id": "CLM-01",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "original_claim": "lags introduce spurious seasonal persistence, while spatial neighborhoods amplify coastal-edge noise",
        "claim_type": "Mechanistic causal assertion",
        "evidence_source": "feature_ablation.csv (D32)",
        "support_level": "SPECULATIVE",
        "replacement_text": "The evaluated temporal-lag and spatial-neighborhood extensions did not provide incremental predictive skill relative to E3b-C0. The specific mechanism responsible for their lower performance was not isolated.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-02",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "original_claim": "discrepancia residual dominada por ruido radiométrico de sensor y piso de ruido",
        "claim_type": "Physical sensor noise claim",
        "evidence_source": "residual_regime_metrics.csv (D35)",
        "support_level": "SPECULATIVE",
        "replacement_text": "When the MUR–BIL discrepancy was small, there was less margin for a beneficial correction and lower residual-sign agreement was associated with relative degradation.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-03",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "original_claim": "cuando existen gradientes térmicos verdaderos / genuinos",
        "claim_type": "True physical gradient assertion",
        "evidence_source": "residual_regime_metrics.csv (D35)",
        "support_level": "SPECULATIVE",
        "replacement_text": "Sign agreement and RMSE improvement increased across larger MUR–BIL discrepancy regimes.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-04",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "original_claim": "Due to squared loss, the model compresses amplitude / MSE causes shrinkage",
        "claim_type": "Causal loss assertion",
        "evidence_source": "residual_metrics.csv (D35)",
        "support_level": "REQUIRES_LITERATURE",
        "replacement_text": "The fitted model exhibited pronounced amplitude compression (std(Rhat)/std(R) ≈ 0.25), consistent with regression toward the conditional mean under the selected regularized MSE-based formulation.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-05",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "original_claim": "bathymetry causes skill / coastal geometry explains skill / land contamination causes improvement",
        "claim_type": "Causal spatial claim",
        "evidence_source": "spatial_depth_diagnostics.csv (D35)",
        "support_level": "DIRECTLY_SUPPORTED (as association only)",
        "replacement_text": "The magnitude of improvement decreased toward deeper and more offshore cells. These associations are descriptive and do not establish physical causality.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-06",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "5. Jury Q&A",
        "original_claim": "OISST bilineal ya explica el 89.3% de la varianza total, por lo que el modelo opera exclusivamente sobre el 10.7% restante",
        "claim_type": "Variance decomposition claim",
        "evidence_source": "final_test_summary.csv (D35)",
        "support_level": "UNSUPPORTED (in exact variance terms)",
        "replacement_text": "The bilinear baseline already showed high agreement with MUR (R² ≈ 0.893), so the residual learner was evaluated against a strong baseline. The 7.22% final-test RMSE reduction therefore represents incremental improvement over an already competitive interpolation.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-07",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "5. Jury Q&A",
        "original_claim": "La presencia de geometrías continentales complejas e islas (Cozumel) produce problemas severos de convolución en fronteras",
        "claim_type": "CNN convolution defect claim",
        "evidence_source": "D31 & D32 gates",
        "support_level": "SPECULATIVE",
        "replacement_text": "CNNs were not pursued because the prespecified predictability diagnostics and tabular ablation results supported a parsimonious tabular formulation. The incremental gain from engineered spatial predictors did not exceed the predefined threshold required to justify substantially greater model complexity.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-08",
        "document": "THESIS_LIMITATIONS_A_D.md",
        "section": "4. Stage D",
        "original_claim": "La mayor parte de la varianza residual instantánea diaria permanece no explicada debido a la falta de forzantes dinámicos de alta resolución",
        "claim_type": "Attribution of unexplained variance",
        "evidence_source": "residual_metrics.csv (D35)",
        "support_level": "SPECULATIVE",
        "replacement_text": "Most residual variance remained unexplained by the selected predictor set. Potential contributors include unrepresented dynamic processes, differences among SST products, retrieval uncertainty, and variability not captured by the available predictors.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-09",
        "document": "THESIS_LIMITATIONS_A_D.md",
        "section": "4. Stage D",
        "original_claim": "No es adecuado para predecir anomalías térmicas extremas instantáneas aisladas",
        "claim_type": "Extreme anomaly suitability assertion without metric",
        "evidence_source": "residual_metrics.csv (D35)",
        "support_level": "UNSUPPORTED (without dedicated extreme metric)",
        "replacement_text": "The model underrepresented residual amplitude, which may limit its ability to reproduce large instantaneous MUR–BIL discrepancies.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-10",
        "document": "THESIS_LIMITATIONS_A_D.md",
        "section": "4. Stage D",
        "original_claim": "reflejando sensibilidad a ciclos interanuales o anomalías térmicas regionales no capturadas",
        "claim_type": "Interannual physical cycle assertion",
        "evidence_source": "yearly_metrics.csv (D35)",
        "support_level": "SPECULATIVE",
        "replacement_text": "Month-level skill remained variable, indicating that aggregate final-test improvement was not temporally uniform.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-11",
        "document": "THESIS_METHODS_MASTER_A_D.md",
        "section": "1. Study Area",
        "original_claim": "El dominio abarca el canal de Yucatán",
        "claim_type": "Geographic coordinate mismatch",
        "evidence_source": "config.py (Lat 19.90–20.75°N)",
        "support_level": "UNSUPPORTED",
        "replacement_text": "Tulum–Cozumel sector, Cozumel Channel, Mexican Caribbean (the Yucatán Channel proper is located north of 21.5°N and is not within the domain).",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-12",
        "document": "THESIS_METHODS_MASTER_A_D.md",
        "section": "11. Residual Formulation",
        "original_claim": "garantiza que en ausencia de señal aprendible el modelo converja naturalmente al baseline",
        "claim_type": "Theoretical convergence guarantee",
        "evidence_source": "Mathematical algebraic identity",
        "support_level": "REQUIRES_LITERATURE",
        "replacement_text": "If Rhat = 0, the reconstructed SST is exactly equal to the bilinear baseline.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-13",
        "document": "THESIS_METHODS_MASTER_A_D.md",
        "section": "14. Predictability",
        "original_claim": "evaluó si el residual fino exhibe estructura determinística",
        "claim_type": "Deterministic physical claim",
        "evidence_source": "D31 reports",
        "support_level": "SPECULATIVE",
        "replacement_text": "evaluated whether the fine residual R exhibits reproducible predictive structure beyond a null/no-skill baseline.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-14",
        "document": "THESIS_METHODS_MASTER_A_D.md",
        "section": "20. Bootstrap",
        "original_claim": "bloques temporales continuos de 14 días para estimar el intervalo de confianza preservando la dependencia serial de mesoescala",
        "claim_type": "Physical mesoscale scale assertion",
        "evidence_source": "bootstrap_sensitivity.csv (D35)",
        "support_level": "SPECULATIVE",
        "replacement_text": "moving block bootstrap with 14-day blocks to preserve short-range temporal dependence under the evaluated block lengths.",
        "status": "CORRECTED"
    },
    {
        "claim_id": "CLM-15",
        "document": "THESIS_RESULTS_MASTER_A_D.md",
        "section": "12. Bootstrap",
        "original_claim": "confirmando significancia estadística al 95%",
        "claim_type": "Classical hypothesis testing phrasing",
        "evidence_source": "bootstrap_sensitivity.csv (D35)",
        "support_level": "REQUIRES_LITERATURE",
        "replacement_text": "The 95% bootstrap interval remained entirely below zero, supporting the robustness of the aggregate RMSE improvement under short-range temporal dependence at the evaluated block length.",
        "status": "CORRECTED"
    }
]

df_claims = pd.DataFrame(claims_audit)
df_claims.to_csv(TABLES_DIR / "final_claims_audit.csv", index=False)
print("Tabla final_claims_audit.csv generada.")

# ----------------------------------------------------------------------
# 4. final_documentation_corrections.csv
# Columnas: correction_id,document,section,issue,original,corrected,canonical_source,experimental_result_changed,status
# ----------------------------------------------------------------------
corrections_log = [
    {
        "correction_id": "CORR-01",
        "document": "DOCUMENTATION_GAPS_A_D.md",
        "section": "2. Tabla Maestra de Gaps",
        "issue": "Inclusión formal de GAP-09 a GAP-12 con estado RESOLVED & DOCUMENTED",
        "original": "GAPs 01 a 08",
        "corrected": "GAPs 01 a 12 (GAP-09 DEV thresholds, GAP-10 monthly allocation, GAP-11 spatial hash, GAP-12 C2 RMSE)",
        "canonical_source": "final_test_freeze_manifest.json, monthly_metrics.csv, metricas_diarias_2015_2025.csv",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-02",
        "document": "THESIS_RESULTS_MASTER_A_D.md",
        "section": "14. Residual-Regime Dependence",
        "issue": "Umbrales no canónicos en descripción de regímenes (GAP-09)",
        "original": "0.177, 0.354, 0.536, 0.697, 1.054 °C",
        "corrected": "0.2066, 0.3604, 0.5377, 0.6652, 0.9659 °C",
        "canonical_source": "final_test_freeze_manifest.json, residual_regime_metrics.csv",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-03",
        "document": "THESIS_RESULTS_MASTER_A_D.md",
        "section": "12. Temporal Robustness",
        "issue": "Desglose anual erróneo de meses mejorados (GAP-10)",
        "original": "11 de 12 en 2024, 7 de 12 en 2025",
        "corrected": "9 de 12 en 2024 (75.0%), 9 de 12 en 2025 (75.0%) [Total: 18/24 (75.0%)]",
        "canonical_source": "monthly_metrics.csv (columna DeltaRMSE)",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-04",
        "document": "REPRODUCIBILITY_MAP_A_D.md",
        "section": "3. Cryptographic Hashes",
        "issue": "Hash de frozen_spatial_metadata.csv incorrecto (GAP-11)",
        "original": "a93b4554a9386c9e99551c68d197607a72dd37803e6592233f211333792b0c2a",
        "corrected": "8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365",
        "canonical_source": "final_test_freeze_manifest.json, frozen_spatial_metadata.csv",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-05",
        "document": "THESIS_RESULTS_MASTER_A_D.md",
        "section": "5. Full Harmonization Results",
        "issue": "Aclaración de C2 Pooled RMSE vs Mean Daily RMSE (GAP-12)",
        "original": "RMSE = 0.3426 °C; osciló entre 0.2724 °C (2018) y 0.3257 °C (2024)",
        "corrected": "RMSE global agrupado (pooled) = 0.3426 °C; promedio diario decenal = 0.3019 °C; promedio diario anual osciló entre 0.2724 °C (2018) y 0.3257 °C (2024); RMSE agrupado anual osciló entre 0.3034 °C (2018) y 0.3811 °C (2015)",
        "canonical_source": "fase_c2_reporte.md, metricas_diarias_2015_2025.csv",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-06",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I",
        "issue": "Eliminación de lenguaje mecanicista no demostrado en parsimonia (D32)",
        "original": "lags introducen persistencia estacional espuria, vecindades amplifican ruido costero",
        "corrected": "The evaluated temporal-lag and spatial-neighborhood extensions did not provide incremental predictive skill relative to E3b-C0. The specific mechanism responsible for their lower performance was not isolated.",
        "canonical_source": "feature_ablation.csv (D32)",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-07",
        "document": "THESIS_DISCUSSION_NOTES_A_D.md",
        "section": "2. Category I & 5. Jury Q&A",
        "issue": "Sustitución de afirmaciones de 'ruido' y respuestas imprecisas a jurado",
        "original": "ruido radiométrico de sensor, piso de ruido, gradientes físicos verdaderos, OISST explica 89.3% operando sobre 10.7%",
        "corrected": "Régimen de discrepancia reducida MUR–BIL, discrepancias mayores MUR–BIL, baseline bilineal altamente competitivo (R² ≈ 0.893)",
        "canonical_source": "residual_regime_metrics.csv, final_test_summary.csv",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-08",
        "document": "THESIS_METHODS_MASTER_A_D.md",
        "section": "1. Study Area",
        "issue": "Inclusión errónea del Canal de Yucatán en el área de estudio",
        "original": "El dominio abarca el canal de Yucatán",
        "corrected": "Sector Tulum–Cozumel, canal de Cozumel y Caribe mexicano (el canal de Yucatán propiamente dicho se sitúa al norte de 21.5°N y no forma parte del dominio)",
        "canonical_source": "config.py",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    },
    {
        "correction_id": "CORR-09",
        "document": "REPRODUCIBILITY_MAP_A_D.md",
        "section": "4. Documentation Layer Status",
        "issue": "Actualización del estado formal de congelamiento documental",
        "original": "No especificado / PENDING",
        "corrected": "EXPERIMENTAL LAYER: CLOSED | FINAL TEST: CONSUMED | DOCUMENTATION LAYER: AUDITED AND FROZEN",
        "canonical_source": "Auditoría final de evidencia",
        "experimental_result_changed": "NO",
        "status": "APPLIED"
    }
]

df_corr = pd.DataFrame(corrections_log)
df_corr.to_csv(TABLES_DIR / "final_documentation_corrections.csv", index=False)
print("Tabla final_documentation_corrections.csv generada con columnas exactas.")

print("Completado procesamiento de tablas de Fase 2.")

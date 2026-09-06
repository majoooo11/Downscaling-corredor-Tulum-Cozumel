# Master Thesis Writing Map (Stages A to D)
## Guía de Procedencia Documental para Redacción de Capítulos de Tesis

**Documento:** `THESIS_WRITING_MAP.md`  
**Estado:** AUDITED & FROZEN  
**Propósito:** Proporcionar un índice de navegación unívoco para que el autor de la tesis pueda redactar directamente cada capítulo y subsección sin necesidad de rebuscar en logs, scripts antiguos o directorios dispersos.

---

## 1. Capítulo: Introducción y Área de Estudio

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada | CANONICAL STATUS |
| :--- | :--- | :--- | :--- |
| **Descripción geográfica del corredor** | `reports/THESIS_METHODS_MASTER_A_D.md` (Sección 1) | `THESIS_MASTER_A_D/figures/fig_master_pipeline_A_D.png` | **VERIFIED** |
| **Límites de coordenadas y batimetría regional** | `config.py`, `reports/reporte_fase_b.txt` | `tables/master_phase_summary.csv` | **VERIFIED** |

---

## 2. Capítulo: Datos y Metodología

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada | CANONICAL STATUS |
| :--- | :--- | :--- | :--- |
| **Producto MUR SST v4.1 y control de calidad** | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 2, 3) | `tables/dataset_provenance.csv` (Fila 1) | **VERIFIED** |
| **Producto NOAA OISST v2.1 y Halo Espacial** | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 2, 7) | `tables/dataset_provenance.csv` (Fila 3) | **VERIFIED** |
| **Modelo batimétrico GEBCO 2026** | `reports/THESIS_METHODS_MASTER_A_D.md` (Sección 2, 6) | `tables/dataset_provenance.csv` (Fila 4) | **RESOLVED** (GAP-02) |
| **Definición de grilla maestra y máscara corregida** | `reports/reporte_fase_b.txt`, `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 4, 5) | `tables/master_decision_log.csv` (DEC-01, DEC-02) | **VERIFIED** |
| **Problema de NaNs costeros y soporte Estrategia A** | `reports/reporte_faseC1b_2015-01-01.txt` | `figures/faseC1b_comparacion_estrategias_2015-01-01.png` | **VERIFIED** |
| **Armonización temporal completa 2015–2025 (C.2)** | `reports/fase_c2_reporte.md` | `figures/faseC2_serie_rmse_diario.png` | **RESOLVED** (GAP-12) |
| **Auditoría satelital independiente VIIRS/MODIS** | `reports/validacion_infrarroja_integrada_FINAL.md` | `reports/detalle_flags_viirs_FINAL.txt`, `modis` | **VERIFIED** |
| **Auditoría de incertidumbre MUR analysis_error** | `auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md` | `tables/master_decision_log.csv` (DEC-05) | **VERIFIED** |
| **Construcción tabular y partición anti-fuga (D.1)** | `ml_dataset/METADATA.md` | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 12, 13) | **VERIFIED** |
| **Selección de modelo y ablaciones E3b (D32)** | `ml_results/E3b_D32/tables/feature_ablation.csv` | `ml_results/E3b_D32/figures/fig2_relative_improvement_vs_b0.png` | **VERIFIED** |
| **Validación externa 2022–2023 (D33)** | `ml_results/E3b_D33_external_validation/tables/validation_summary.csv` | `tables/master_decision_log.csv` (DEC-08) | **VERIFIED** |
| **Protocolo de congelamiento y Final Test (D35)** | `ml_results/E3b_D35_final_test/final_test_freeze_manifest.json` | `tables/master_decision_log.csv` (DEC-09, DEC-10) | **RESOLVED** (GAP-11) |

---

## 3. Capítulo: Resultados

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada | CANONICAL STATUS |
| :--- | :--- | :--- | :--- |
| **Línea base de armonización decenal (Baseline E0)** | `reports/fase_c2_reporte.md` | `tables/metric_provenance_A_D.csv` (MET-C-05 a MET-C-09) | **RESOLVED** (GAP-12) |
| **Resultados de Holdout 2021 y selección E3b-C0** | `ml_results/E3b_D32/tables/model_summary.csv` | `tables/metric_provenance_A_D.csv` (MET-D32-01 a MET-D32-03) | **VERIFIED** |
| **Resultados de Validación Externa 2022–2023** | `ml_results/E3b_D33_external_validation/tables/validation_summary.csv` | `tables/metric_provenance_A_D.csv` (MET-D33-01 a MET-D33-05) | **VERIFIED** |
| **Desempeño final en Test 2024–2025** | `ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table1_final_performance.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig1_final_performance.png` | **VERIFIED** |
| **Robustez temporal (desglose anual 9/12 y 9/12)** | `ml_results/E3b_D35_final_test/tables/monthly_metrics.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig2_temporal_robustness.png` | **RESOLVED** (GAP-10) |
| **Robustez espacial (celdas mejoradas y batimetría)** | `ml_results/E3b_D35_final_test/tables/spatial_depth_diagnostics.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig3_spatial_skill.png` | **VERIFIED** |
| **Análisis de regímenes con thresholds canónicos** | `ml_results/E3b_D35_final_test/tables/residual_regime_metrics.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig4_residual_regimes.png` | **RESOLVED** (GAP-09) |

---

## 4. Capítulo: Discusión y Conclusiones

| Subsección de Tesis | Archivo de Procedencia Primaria | Notas Metodológicas | CANONICAL STATUS |
| :--- | :--- | :--- | :--- |
| **Interpretación del modelo parsimonioso** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Evidencia de parsimonia y estabilidad | **VERIFIED** |
| **Análisis de compresión de amplitud y calibración** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Explicar $b = 0.0891$ en $\hat{R} = a + b R$ | **RESOLVED** (GAP-03) |
| **Fronteras en régimen bajo ($|R| < 0.2066^\circ\text{C}$)** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Comportamiento en DEV-P0–P50 | **RESOLVED** (GAP-09) |
| **Limitaciones del estudio** | `reports/THESIS_LIMITATIONS_A_D.md` | Limitaciones por etapas A, B, C, D | **VERIFIED** |
| **Gaps metodológicos resueltos** | `reports/DOCUMENTATION_GAPS_A_D.md` | Trazabilidad de GAPs 01 a 12 | **RESOLVED** |
| **Conclusiones terminales de tesis** | `reports/THESIS_MASTER_SYNTHESIS_A_D.md` (Sección 28) | Síntesis conclusiva definitiva | **VERIFIED** |

---

## 5. Documentos Históricos con Cifras Preliminares No Canónicas

| Documento Histórico / Borrador | Cifra Incorrecta / Discrepante | Cifra Canónica Correcta | CANONICAL STATUS |
| :--- | :--- | :--- | :--- |
| **Borradores preliminares con percentiles aproximados** | $P_{50} \approx 0.177, P_{75} \approx 0.354, P_{90} \approx 0.536, P_{95} \approx 0.697, P_{99} \approx 1.054^\circ\text{C}$ | $0.2066, 0.3604, 0.5377, 0.6652, 0.9659^\circ\text{C}$ (`final_test_freeze_manifest.json`) | **DO NOT USE** |
| **Borradores con desglose mensual 11/12 y 7/12** | 2024 = 11/12, 2025 = 7/12 | 2024 = 9/12, 2025 = 9/12 (18/24 total; `monthly_metrics.csv`) | **DO NOT USE** |
| **Borradores con hash espacial `a93b4554...`** | `a93b4554a9386c9e99551c68d197607a72dd37803e6592233f211333792b0c2a` | `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365` (`final_test_freeze_manifest.json`) | **DO NOT USE** |
| **Borradores comparando RMSE global con max mean daily** | Comparación directa de 0.3426 °C con 0.3257 °C sin distinguir métricas | Pooled decenal = 0.3426 °C; Mean daily decenal = 0.3019 °C; Pooled anual máx = 0.3811 °C | **DO NOT USE** |

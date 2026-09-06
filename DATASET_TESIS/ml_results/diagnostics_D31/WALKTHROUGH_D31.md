# Walkthrough — Fase D.3.1: Diagnóstico de Predictibilidad del Residual

Se completó formalmente la **Fase D.3.1**, diagnosticando de forma exhaustiva las causas por las cuales E3 exhibe correlación predictiva con el residual pero no supera a E0 en términos de error cuadrático medio global.

## 1. Resumen de la Ejecución Científica
- **Auditoría Espacial:** PASS (5,279 celdas oceánicas validadas, grilla 86x96, orientación latitud detectada).
- **Datasets:** DEVELOPMENT 2015–2020 (N = 11,571,568) y DIAGNOSTIC HOLDOUT 2021 (N = 1,926,835).
- **Blindaje Estricto:** VALIDATION files opened = 0 | TEST files opened = 0.

## 2. Dictamen Oficial
- **Dictamen:** **D31-B — EVIDENCIA PARA E3b TABULAR**
- **Justificación cuantitativa:**
  Una formulación tabular predefinida logra una mejora en RMSE >= 1.0% respecto a B0 (+3.94% para BASE, +4.83% para SPATIAL) con estabilidad temporal (11/12 meses) sin deteriorar el MAE (0.2597 <= 0.2773), pero el incremento marginal de las features espaciales 2D sobre la base tabular es de +0.93%, no alcanzando el umbral de +1.00% requerido por el Criterio 1 de D31-A.

## 3. Catálogo de Tablas y Figuras
- 15 tablas CSV generadas en `DATASET_TESIS/ml_results/diagnostics_D31/tables/`.
- 11 figuras PNG de alta resolución generadas en `DATASET_TESIS/ml_results/diagnostics_D31/figures/`.

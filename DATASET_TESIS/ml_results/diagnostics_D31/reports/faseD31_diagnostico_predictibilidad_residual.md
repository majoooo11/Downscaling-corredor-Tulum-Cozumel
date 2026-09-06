# Reporte Científico — Fase D.3.1: Diagnóstico de Predictibilidad del Residual

**Fecha de ejecución:** 2026-09-05 18:23:26  
**Script reproducible:** `DATASET_TESIS/fase_d31_diagnostico_predictibilidad_residual.py`  
**Diseño temporal:** DEVELOPMENT (2015–2020, N=11,571,568) | HOLDOUT (2021, N=1,926,835)  
**Salvaguarda de blindaje:** VALIDATION files opened = 0 | TEST files opened = 0  

---

## 1. Dictamen Metodológico Formal de la Fase D.3.1

### **DICTAMEN: D31-B — EVIDENCIA PARA E3b TABULAR**

**Justificación basada estrictamente en los criterios cuantitativos predefinidos:**  
Una formulación tabular predefinida logra una mejora en RMSE >= 1.0% respecto a B0 (+3.94% para BASE, +4.83% para SPATIAL) con estabilidad temporal (11/12 meses) sin deteriorar el MAE (0.2597 <= 0.2773), pero el incremento marginal de las features espaciales 2D sobre la base tabular es de +0.93%, no alcanzando el umbral de +1.00% requerido por el Criterio 1 de D31-A.

---

## 2. Resultados Clave por Ejes Analíticos

### A. Baselines Climatológicos en HOLDOUT 2021 (Análisis B)
| baseline                   |   RMSE_SST |   MAE_SST |   Bias_SST |     R2_RES |   Impr_RMSE_vs_B0_pct |
|:---------------------------|-----------:|----------:|-----------:|-----------:|----------------------:|
| B0 (Bilineal E0)           |   0.359491 |  0.277336 | -0.0501231 | -0.0198256 |              0        |
| B1 (Media Global)          |   0.360012 |  0.278089 | -0.0537299 | -0.0227814 |             -0.14482  |
| B2 (Climatología Mensual)  |   0.360277 |  0.278075 | -0.0537585 | -0.0242896 |             -0.21863  |
| B3 (Armónicos Anuales OLS) |   0.358692 |  0.277293 | -0.0537501 | -0.0152953 |              0.222345 |
| B4 (Pixel-Mes Fijo)        |   0.358928 |  0.276719 | -0.0537586 | -0.0166311 |              0.156734 |

*Hallazgo:* La corrección puramente climatológica reduce el RMSE en menos de 0.3% sobre 2021, demostrando que la mayor parte del residual no es un ciclo estacional medio rígido.

### B. Ablación de Features Tabulares A1..A5 en HOLDOUT 2021 (Análisis A)
| ablation   | features                                                            |   RMSE_SST |   MAE_SST |   Impr_RMSE_vs_B0_pct |
|:-----------|:--------------------------------------------------------------------|-----------:|----------:|----------------------:|
| A1         | sst_bil                                                             |   0.356735 |  0.27404  |              0.766827 |
| A2         | sst_bil, doy_sin, doy_cos                                           |   0.345224 |  0.263131 |              3.96864  |
| A3         | sst_bil, depth, distance_coast_km, ocean_fraction                   |   0.35522  |  0.272824 |              1.18812  |
| A4         | sst_bil, doy_sin, doy_cos, depth                                    |   0.341734 |  0.259343 |              4.93952  |
| A5         | sst_bil, doy_sin, doy_cos, depth, distance_coast_km, ocean_fraction |   0.341929 |  0.259871 |              4.88529  |

*Hallazgo:* Las combinaciones que integran componentes temporales armónicas (doy_sin, doy_cos) y batimetría (A2, A4, A5) logran mejoras del 3.97% al 4.94% sobre B0 en 2021, mientras que A1 (sst_bil puro) apenas aporta 0.77%.

### C. Modelos Diagnósticos con Contexto Espacial sobre `SPATIAL_VALID_MASK` (Análisis E)
| model               | features                                                                                           |   N_evaluated |   RMSE_SST |   MAE_SST |   Impr_RMSE_vs_B0_pct |   Impr_RMSE_vs_BASE_pct |
|:--------------------|:---------------------------------------------------------------------------------------------------|--------------:|-----------:|----------:|----------------------:|------------------------:|
| B0 (Bilineal E0)    | none (R_hat=0)                                                                                     |       1925375 |   0.359493 |  0.277337 |               0       |                0        |
| E-DIAG-BASE         | sst_bil, doy_sin, doy_cos                                                                          |       1925375 |   0.345326 |  0.2633   |               3.94064 |                0        |
| E-DIAG-SPATIAL      | sst_bil, doy_sin, doy_cos, grad_mag_sst_bil, local_std_3x3, local_contrast                         |       1925375 |   0.34212  |  0.259665 |               4.83266 |                0.928609 |
| E-DIAG-SPATIAL-PLUS | sst_bil, doy_sin, doy_cos, depth, grad_mag_sst_bil, local_std_3x3, local_range_3x3, local_contrast |       1925375 |   0.342679 |  0.259967 |               4.67708 |                0.766651 |

### D. Métricas de Estabilidad Temporal y Distribución Espacial
- **Estabilidad Temporal:** E-DIAG-SPATIAL mejora a B0 en **11 de 12 meses** en 2021.
- **Distribución Espacial:** E-DIAG-SPATIAL mejora a E-DIAG-BASE en **94.56% de las celdas evaluables** (4988 de 5275 celdas).

### E. Diagnóstico Central vs Colas de Discrepancia (Análisis F)
| regime                    |       N |   pct_total |   RMSE_B0 |   RMSE_SPATIAL |   Impr_SPATIAL_vs_B0_pct |
|:--------------------------|--------:|------------:|----------:|---------------:|-------------------------:|
| 0 - P50                   |  901459 |    46.8199  |  0.116394 |       0.135956 |                -16.8069  |
| P50 - P75                 |  465912 |    24.1985  |  0.281325 |       0.263528 |                  6.32587 |
| P75 - P90                 |  322436 |    16.7467  |  0.442944 |       0.406559 |                  8.21445 |
| P90 - P95                 |  121383 |     6.30438 |  0.595257 |       0.550455 |                  7.52651 |
| P95 - P99                 |   84902 |     4.40963 |  0.774436 |       0.739592 |                  4.49929 |
| >= P99                    |   29283 |     1.5209  |  1.2054   |       1.16616  |                  3.25546 |
| Régimen Central |R| < P90 | 1689807 |    87.7651  |  0.257849 |       0.246065 |                  4.56995 |
| Cola Extrema |R| >= P90   |  235568 |    12.2349  |  0.761153 |       0.722719 |                  5.04946 |
| Cola Extrema |R| >= P99   |   29283 |     1.5209  |  1.2054   |       1.16616  |                  3.25546 |
| Total Máscara             | 1925375 |   100       |  0.359493 |       0.34212  |                  4.83266 |

### F. Discriminación Prospectiva de Grandes Discrepancias (Análisis G)
| event              | score            |    AUROC |     AUPRC |   prevalence |   enrichment_ratio |
|:-------------------|:-----------------|---------:|----------:|-------------:|-------------------:|
| P90 (>= 0.5435 °C) | |R_hat_SPATIAL|  | 0.540333 | 0.139573  |    0.122349  |            1.14077 |
| P90 (>= 0.5435 °C) | grad_mag_sst_bil | 0.524816 | 0.138702  |    0.122349  |            1.13366 |
| P90 (>= 0.5435 °C) | local_std_3x3    | 0.526817 | 0.140939  |    0.122349  |            1.15195 |
| P95 (>= 0.6698 °C) | |R_hat_SPATIAL|  | 0.494913 | 0.0644369 |    0.0593053 |            1.08653 |
| P95 (>= 0.6698 °C) | grad_mag_sst_bil | 0.541698 | 0.0708033 |    0.0593053 |            1.19388 |
| P95 (>= 0.6698 °C) | local_std_3x3    | 0.544647 | 0.0725509 |    0.0593053 |            1.22335 |
| P99 (>= 0.9810 °C) | |R_hat_SPATIAL|  | 0.376987 | 0.0278258 |    0.015209  |            1.82956 |
| P99 (>= 0.9810 °C) | grad_mag_sst_bil | 0.566206 | 0.0193464 |    0.015209  |            1.27204 |
| P99 (>= 0.9810 °C) | local_std_3x3    | 0.570402 | 0.0198658 |    0.015209  |            1.30619 |

### G. Persistencia Temporal y Memoria Residual (Análisis H)
|   lag_days |   mean_corr_R |   median_corr_R |   P10_corr_R |   P90_corr_R |   mean_corr_abs_R |
|-----------:|--------------:|----------------:|-------------:|-------------:|------------------:|
|          1 |      0.68528  |        0.685583 |    0.670251  |     0.69934  |         0.515011  |
|          2 |      0.362243 |        0.359936 |    0.338258  |     0.387653 |         0.179286  |
|          3 |      0.220596 |        0.218725 |    0.190855  |     0.253623 |         0.0801062 |
|          7 |      0.132782 |        0.136913 |    0.0894344 |     0.166988 |         0.0452028 |

---

## 3. Catálogo de Entregables Generados

- **Tablas CSV (15 archivos):** en `DATASET_TESIS/ml_results/diagnostics_D31/tables/`
- **Figuras Científicas (11 archivos):** en `DATASET_TESIS/ml_results/diagnostics_D31/figures/`
- **Reportes:**
  - `faseD31_diagnostico_predictibilidad_residual.md`
  - `WALKTHROUGH_D31.md`

---
*Blindaje final confirmado:* `VALIDATION files opened = 0` | `TEST files opened = 0`

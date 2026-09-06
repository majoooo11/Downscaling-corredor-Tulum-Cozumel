# Reporte Científico Final — Fase D.3.2: E3b Tabular
## Modelado del Residual con Contexto Temporal y Espacial (Fase D.3.2-FINAL)

**Fecha de cierre metodológico:** 2026-09-05 19:23:55  
**Script reproducible:** `DATASET_TESIS/fase_d32_e3b_tabular.py`  
**Entrenamiento auditado:** FULL DEVELOPMENT 2015–2020 ($N = 11,546,975$ observaciones válidas bajo COMMON_VALID_MASK, de $11,571,568$ brutas)  
**Evaluación Holdout:** DIAGNOSTIC HOLDOUT 2021 ($N = 1,925,375$ observaciones pareadas, 99.92% de 2021)  
**Blindaje temporal:** `VALIDATION files opened = 0` | `TEST files opened = 0`  
**Estado:** **METHODOLOGICALLY CLOSED**

---

## 1. Resumen Ejecutivo y Dictámenes Formales Post-Auditoría

Tras reentrenar todas las formulaciones tabulares sobre la totalidad de DEVELOPMENT 2015–2020 ($N = 11,546,975$) y superar la microauditoría de integridad:

### **DICTAMEN TEMPORAL:** D32-C — SIN EVIDENCIA DE VALOR TEMPORAL
- *Evidencia:* El mejor modelo con memoria temporal (`E3b-T3`) obtiene $\text{RMSE} = 0.350475^\circ\text{C}$, empeorando al núcleo CORE (`E3b-C0`, $0.349274^\circ\text{C}$, cambio relativo de -0.34\% vs CORE). En las formulaciones evaluadas, los lags causales de $\text{SST\_BIL}$ no aportaron mejora incremental fuera de muestra.

### **DICTAMEN ESPACIAL:** SPATIAL-NO
- *Evidencia:* El modelo espacio-temporal (`E3b-TS`) obtiene $\text{RMSE} = 0.351551^\circ\text{C}$, lo que representa una diferencia de -0.31\% vs `E3b-T3`. Las features espaciales manuales evaluadas no proporcionan mejora incremental dentro de XGBoost.

### **RECOMENDACIÓN CIENTÍFICA POST-AUDITORÍA:** STOP COMPLEXIFICATION / REVISIT FORCING (Revisar formulación del residual o integrar forzamiento dinámico atmosférico)
- *Recomendación:* D32 no aporta evidencia empírica suficiente para priorizar mayor complejidad basada únicamente en $\text{SST\_BIL}$ (como una CNN 2D estándar). Se recomienda revisar la formulación del target residual o incorporar forzamiento dinámico atmosférico independiente.

---

## 2. Resultados Globales sobre COMMON_VALID_MASK en HOLDOUT 2021
Modelos entrenados sobre todo DEVELOPMENT 2015–2020 ($N = 11,546,975$ observaciones válidas):

| model   | features                                                                                                                                                                                                                  |   N_evaluated |   RMSE_SST |   MAE_SST |   Bias_SST |   Impr_RMSE_vs_B0_pct |   Impr_RMSE_vs_CORE_pct |   Skill_RMSE_vs_B0 |
|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|-----------:|----------:|-----------:|----------------------:|------------------------:|-------------------:|
| B0      | none                                                                                                                                                                                                                      |       1925375 |   0.359493 |  0.277337 | -0.0501454 |               0       |                0        |          0         |
| E3b-C0  | sst_bil, doy_sin, doy_cos, depth                                                                                                                                                                                          |       1925375 |   0.349274 |  0.266264 | -0.0375205 |               2.84243 |                0        |          0.0284243 |
| E3b-T1  | sst_bil, doy_sin, doy_cos, depth, sst_bil_lag1, delta_sst_1d                                                                                                                                                              |       1925375 |   0.350676 |  0.266832 | -0.0432643 |               2.45264 |               -0.401197 |          0.0245264 |
| E3b-T3  | sst_bil, doy_sin, doy_cos, depth, sst_bil_lag1, sst_bil_lag2, sst_bil_lag3, delta_sst_1d, delta_sst_2d, delta_sst_3d                                                                                                      |       1925375 |   0.350475 |  0.266631 | -0.0495217 |               2.50843 |               -0.343778 |          0.0250843 |
| E3b-S   | sst_bil, doy_sin, doy_cos, depth, grad_mag_sst_bil, local_std_3x3, local_contrast                                                                                                                                         |       1925375 |   0.350802 |  0.267686 | -0.0401471 |               2.41756 |               -0.437305 |          0.0241756 |
| E3b-TS  | sst_bil, doy_sin, doy_cos, depth, sst_bil_lag1, sst_bil_lag2, sst_bil_lag3, delta_sst_1d, delta_sst_2d, delta_sst_3d, grad_mag_sst_bil, local_std_3x3, local_contrast                                                     |       1925375 |   0.351551 |  0.267783 | -0.0501012 |               2.20904 |               -0.651926 |          0.0220904 |
| E3b-ALL | sst_bil, doy_sin, doy_cos, depth, sst_bil_lag1, sst_bil_lag2, sst_bil_lag3, delta_sst_1d, delta_sst_2d, delta_sst_3d, grad_mag_sst_bil, local_std_3x3, local_contrast, distance_coast_km, ocean_fraction, local_range_3x3 |       1925375 |   0.350048 |  0.266346 | -0.0494683 |               2.62712 |               -0.221612 |          0.0262712 |

---

## 3. Análisis de Incertidumbre Estadística (Temporal Block Bootstrap, B = 1,000)
Remuestreo por campos diarios completos de SST como unidades estadísticas:

| comparison       |   median_delta_rmse |   ci95_lower |   ci95_upper |   prob_delta_rmse_lt_zero |   bootstrap_p_two_sided |
|:-----------------|--------------------:|-------------:|-------------:|--------------------------:|------------------------:|
| E3b-C0 vs B0     |         -0.0101854  | -0.0146814   |  -0.0058642  |                     1     |                   0     |
| E3b-T3 vs E3b-C0 |          0.00114529 | -0.0040632   |   0.0064568  |                     0.353 |                   0.706 |
| E3b-TS vs E3b-T3 |          0.00107356 | -0.000184024 |   0.00251594 |                     0.049 |                   0.098 |
| E3b-TS vs B0     |         -0.00793103 | -0.0135046   |  -0.00210051 |                     0.995 |                   0.01  |
| E3b-ALL vs B0    |         -0.00956048 | -0.0155011   |  -0.00319392 |                     0.999 |                   0.002 |

*Interpretación estadística rigurosa:* Para la comparación clave entre modelos espaciales y temporales (`E3b-TS vs E3b-T3`), el intervalo de confianza al 95% incluye el cero ($[-0.00018, +0.00252]^\circ\text{C}$). La probabilidad de mejora bootstrap es $\text{prob\_delta\_rmse\_lt\_zero} = 0.049$ (sólo aproximadamente 5% de las réplicas favorecieron a E3b-TS) y el p-value bilateral correspondiente es $\text{bootstrap\_p\_two\_sided} = 0.098$ ($> 0.05$). No se encontró evidencia suficiente de una diferencia incremental que justifique la adición de predictores espaciales.

---

## 4. Diagnóstico por Régimen de Residual y Sobre-Corrección (E3b-C0)

| regime                    |       N |   pct_total |   RMSE_B0 |   RMSE_CORE |   RMSE_model |   impr_vs_B0_pct |   impr_vs_CORE_pct |   overcorrection_freq_pct |
|:--------------------------|--------:|------------:|----------:|------------:|-------------:|-----------------:|-------------------:|--------------------------:|
| 0 - P50                   |  901568 |    46.8256  |  0.116409 |    0.120958 |     0.120958 |         -3.90741 |                  0 |                 17.5641   |
| P50 - P75                 |  465851 |    24.1953  |  0.281349 |    0.268159 |     0.268159 |          4.68826 |                  0 |                  0.124718 |
| P75 - P90                 |  322425 |    16.7461  |  0.442968 |    0.422205 |     0.422205 |          4.68711 |                  0 |                  0        |
| P90 - P95                 |  121358 |     6.30308 |  0.595281 |    0.572138 |     0.572138 |          3.88776 |                  0 |                  0        |
| P95 - P99                 |   84888 |     4.40891 |  0.774445 |    0.756126 |     0.756126 |          2.36547 |                  0 |                  0        |
| >= P99                    |   29285 |     1.521   |  1.20538  |    1.2041   |     1.2041   |          0.10636 |                  0 |                  0        |
| Central (|R| < P90)       | 1689844 |    87.767   |  0.257858 |    0.248277 |     0.248277 |          3.71579 |                  0 |                  9.40519  |
| Extreme Tail (|R| >= P90) |  235531 |    12.233   |  0.761183 |    0.744976 |     0.744976 |          2.12921 |                  0 |                  0        |
| Tail (|R| >= P99)         |   29285 |     1.521   |  1.20538  |    1.2041   |     1.2041   |          0.10636 |                  0 |                  0        |
| Total                     | 1925375 |   100       |  0.359493 |    0.349274 |     0.349274 |          2.84243 |                  0 |                  8.25465  |

*Hallazgo empírico clave:*
1. **Régimen de bajo residual ($0 - \text{P50}$, $|R| < 0.2066^\circ\text{C}$):** Los modelos ML sobre-corrigen en un 17.6\% de los casos (deteriorando el RMSE vs B0 en -3.9\%), mientras que en el restante 82.4\% sub-corrigen la magnitud del residual.
2. **Régimen de alto residual ($|R| \ge \text{P75}$):** Los modelos reducen efectivamente el RMSE de B0, pero presentan un marcado encogimiento de magnitud (*strong shrinkage toward zero / regression toward the mean*), prediciendo correcciones de magnitud promedio $\approx 0.04 - 0.07^\circ\text{C}$ cuando el residuo real supera $0.50^\circ\text{C}$.

---

## 5. Interpretabilidad Post-Hoc (Feature Importance)

| feature   |   gain_relative |   permutation_delta_mse |   permutation_relative |
|:----------|----------------:|------------------------:|-----------------------:|
| doy_sin   |        0.31578  |             0.0293543   |              0.607514  |
| depth     |        0.241988 |             0.000845864 |              0.0175059 |
| doy_cos   |        0.224967 |             0           |              0         |
| sst_bil   |        0.217265 |             0.0181185   |              0.37498   |

*Aclaración conceptual:* La interpretación post-hoc del modelo seleccionado E3b-C0 muestra que la capacidad predictiva del ensamble está dominada por la representación estacional, SST bilineal y batimetría GEBCO. The near-zero individual permutation importance of doy_cos should not be interpreted as absence of seasonal information because doy_sin and doy_cos jointly encode annual phase.

---

## 6. Microauditoría Final de Integridad

### 1. Integridad date-cell
Se auditó la totalidad de la serie temporal 2015–2021 (7 años, 2,557 días continuos). Cada día contiene exactamente 5,279 celdas oceánicas de la cuadrícula nominal C.2:
- Filas totales brutas: 13,498,403 (exactamente igual a $2,557 \times 5,279$).
- Duplicados en combinación `(date, cell_id)`: **0** en todos los años 2015 a 2021.
- Mínimo, mediana y máximo de celdas únicas por día: exactamente 5,279 en todos los días de 2015 a 2021.
- Mínimo, mediana y máximo de filas por día: exactamente 5,279 en todos los días de 2015 a 2021.
- Máximo teórico anual vs registros observados: diferencia exactamente igual a **0**.
- Tabla generada y persistida: `tables/date_cell_integrity_audit.csv`.

### 2. Resolución del censo 2020
- **Problema identificado:** El borrador previo del reporte auditado indicaba para 2020 un valor de $N = 1,933,109$ observaciones, superando el máximo teórico ($366 \times 5,279 = 1,932,114$) en 995 registros.
- **Auditoría de datos fuente y de model.fit():**
  - El archivo `train_2020.parquet` tiene exactamente 1,932,114 observaciones brutas y 0 duplicados.
  - Al aplicar `COMMON_VALID_MASK` (que excluye 4 celdas costeras sin contexto 3×3 completo), las observaciones válidas en 2020 son exactamente 1,930,650 ($366 \times 5,275$).
  - Las observaciones utilizadas en `model.fit()` fueron exactamente 1,930,650 para 2020, y 11,546,975 para todo DEVELOPMENT (2015–2020), con 0 duplicados.
- **Dictamen:** *"Reporting-only error; model training data were unaffected."* La discrepancia se debió exclusivamente a un error tipográfico en la plantilla de texto del reporte preliminar.

### 3. Convención best_iteration
- **Problema identificado:** Determinar si `best_iteration = 18` en XGBoost 3.2.0 es 0-indexado y si los modelos finales sufrieron de un error off-by-one (18 vs 19 árboles).
- **Auditoría de API y de archivos de modelo:**
  - En XGBoost 3.2.0, `best_iteration` es 0-indexado (iteración 18 corresponde a la 19ª ronda de boosting).
  - La implementación del código utilizó explícitamente `n_estimators = best_config["best_iteration"] + 1` (19 árboles).
  - La inspección directa del archivo serializado `models/E3b-C0.json` confirmó que el modelo persistido contiene exactamente 19 árboles.
- **Dictamen:** No existió error off-by-one; la convención 0-indexada está correctamente verificada e implementada.

### 4. Nomenclatura bootstrap
- **Problema identificado:** La columna `p_value_improvement` en la tabla de bootstrap representaba la proporción de réplicas con $\Delta\text{RMSE} < 0$, lo cual no es un p-value estadístico de contraste de hipótesis.
- **Corrección aplicada:** Se renombró la columna a `prob_delta_rmse_lt_zero` y se incorporó el p-value bootstrap bilateral `bootstrap_p_two_sided = 2 * min(P(Δ<0), P(Δ>0))`.
- **Interpretación estadística rigurosa:**
  - Para `E3b-TS vs E3b-T3`: $\text{prob\_delta\_rmse\_lt\_zero} = 0.049$ (sólo ~5% de las réplicas favorecieron a E3b-TS) y $\text{bootstrap\_p\_two\_sided} = 0.098$ ($> 0.05$).
  - El intervalo de confianza al 95% cruza el cero ($[-0.00018, +0.00252]^\circ\text{C}$).
  - Conclusión rigurosa: *"No se encontró evidencia suficiente de una diferencia incremental."*

### 5. Corrección Feature Importance
- **Problema identificado:** El reporte preliminar discutía variables espaciales en la sección de importancia de `E3b-C0`, cuando este modelo consta únicamente de 4 predictores (`sst_bil`, `doy_sin`, `doy_cos`, `depth`).
- **Corrección aplicada:** Se eliminó la mención a variables espaciales. La capacidad predictiva de `E3b-C0` está dominada por la representación estacional, SST bilineal y batimetría GEBCO.
- **Aclaración sobre doy_cos:** The near-zero individual permutation importance of doy_cos should not be interpreted as absence of seasonal information because doy_sin and doy_cos jointly encode annual phase.

### 6. Impacto sobre métricas
- Dado que los datos de entrenamiento y los modelos de 19 árboles eran ya 100% correctos y libres de duplicados, las métricas sobre HOLDOUT 2021 ($N = 1,925,375$) son estrictamente invariantes:
  - B0 RMSE: **0.359493 °C**
  - E3b-C0 RMSE: **0.349274 °C** (+2.84% vs B0)
  - E3b-T1 RMSE: **0.350676 °C** (-0.40% vs C0)
  - E3b-T3 RMSE: **0.350475 °C** (-0.34% vs C0)
  - E3b-S RMSE: **0.350802 °C** (-0.44% vs C0)
  - E3b-TS RMSE: **0.351551 °C** (-0.65% vs C0)
  - E3b-ALL RMSE: **0.350048 °C** (-0.22% vs C0)

### 7. Dictamen final D32
- **DICTAMEN TEMPORAL:** `D32-C` (Neither T1 nor T3 improves over C0).
- **DICTAMEN ESPACIAL:** `SPATIAL-NO` (TS degrades vs T3; S degrades vs C0; 95% CI crosses 0).
- **SELECCIÓN DE MODELO:** `E3b-C0` seleccionado bajo el principio de parsimonia.
- **ESTADO DE LA FASE D.3.2:** **METHODOLOGICALLY CLOSED**.

---

## 7. Reconciliación D31 A4 vs D32 C0
- **Comparación cuantitativa:** D31 A4 obtuvo $\text{RMSE} = 0.341734^\circ\text{C}$ (+4.94% vs B0), mientras D32 C0 auditado obtuvo $\text{RMSE} = 0.349274^\circ\text{C}$ (+2.84\% vs B0), una diferencia de $\approx 0.0075^\circ\text{C}$.
- **Auditoría de consistencia:** Se verificó exhaustivamente que algoritmo, target, reconstrucción, variables, coordenadas batimétricas GEBCO, fechas de evaluación y máscara común ($N = 1,925,375$) son **100% idénticos**.
- **Causa demostrada:** The remaining difference is attributable to the different XGBoost configurations used in D31 and D32. The D32 configuration is substantially more regularized and exhibits stronger shrinkage of predicted residuals. D31 A4 utilizó 352 árboles de profundidad 5 (`learning_rate = 0.02`), mientras D32 C0 utilizó 19 árboles de profundidad 4 (`learning_rate = 0.10`) producto del tuning con parada temprana en 2020.

---

## 8. Coste Computacional y Tiempos de Ejecución
| Fase                                        | Segundos   |
|:--------------------------------------------|:-----------|
| Auditoría coordenadas                       | 0.22 s     |
| Carga y feature engineering                 | 6.83 s     |
| Muestreo Hamilton DEV (10%)                 | 1.34 s     |
| Tuning hiperparámetros CORE                 | 2.98 s     |
| Ajuste y predicción 6 ablaciones (Full DEV) | 15.99 s    |
| Estabilidad mensual                         | 0.06 s     |
| Distribución espacial celdas                | 0.03 s     |
| Regímenes y sobre-corrección                | 0.47 s     |
| Feature importance post-hoc                 | 0.07 s     |
| Temporal block bootstrap (B=1000)           | 2.46 s     |
| Generación 10 figuras científicas           | 1.23 s     |
| Tiempo total ejecución                      | 31.69 s    |

---

## 9. Catálogo de Entregables Persistidos

- **15 Tablas CSV:** en `DATASET_TESIS/ml_results/E3b_D32/tables/`
  - `date_cell_integrity_audit.csv` *(Nueva tabla de integridad date-cell)*
  - `model_summary.csv`
  - `computational_cost.csv`
  - `feature_ablation.csv`
  - `monthly_metrics.csv`
  - `spatial_metrics.csv`
  - `residual_regime_metrics.csv`
  - `overcorrection_diagnostic.csv`
  - `audit_changes.csv`
  - `d31_vs_d32_reconciliation.csv`
  - `pipeline_execution_times.csv`
  - `decision_criteria.csv`
  - `feature_importance.csv`
  - `bootstrap_confidence_intervals.csv`
  - `dataset_counts.csv`
- **11 Figuras Científicas:** en `DATASET_TESIS/ml_results/E3b_D32/figures/`
  - `fig1_rmse_comparison.png`
  - `fig2_relative_improvement_vs_b0.png`
  - `fig3_monthly_rmse_2021.png`
  - `fig4_spatial_map_delta_rmse_vs_b0.png`
  - `fig5_spatial_map_delta_rmse_vs_core.png`
  - `fig6_performance_by_regime.png`
  - `fig7_observed_vs_predicted_residual.png`
  - `fig8_feature_importance.png`
  - `fig9_distribution_r_rhat.png`
  - `fig10_overcorrection_diagnostic_AUDITED.png`
  - `fig10_overcorrection_diagnostic.png`
- **6 Modelos XGBoost JSON:** en `DATASET_TESIS/ml_results/E3b_D32/models/*.json`

---
*Blindaje verificado:* `VALIDATION files opened = 0` | `TEST files opened = 0`

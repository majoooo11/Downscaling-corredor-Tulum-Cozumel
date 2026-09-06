# Walkthrough — Fase D.3.4: Post-Validation Diagnostic Audit
## Diagnóstico No Adaptativo de E3b-C0 tras VALIDATION 2022–2023

Se completó formalmente la **Fase D.3.4 (Post-Validation Diagnostic Audit)**, diseñada para auditar y documentar estadísticamente las causas de la inestabilidad mensual observada en el dictamen **`D33-B — PARTIAL / MIXED GENERALIZATION`** (16/24 meses con mejora), **sin modificar el modelo congelado, sin reentrenamiento, sin ingeniería de variables y manteniendo FINAL TEST 2024–2025 completamente cerrado**.

---

## 1. Verificación de Integridad y Salvaguardas de Blindaje

- **Modelo congelado verificado:** `models/E3b-C0_PREVALIDATION.json`
  - SHA-256: `fb151cafd466613d0bda2cadfee61e9ebc1f8e0f889d9a646cc010ba6436ef5d` (**YES — Verificado**)
- **Celdas congeladas verificadas:** `frozen_cell_ids.csv`
  - SHA-256: `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb` (**YES — Verificado**)
- **Modelo modificado:** **NO** | **Modelo reentrenado:** **NO** | **Directorio `models/` en D34:** **NO creado**.
- **Reproducción exacta de D33:** **YES** ($B_0 \text{ RMSE} = 0.335666^\circ\text{C}$, $C_0 \text{ RMSE} = 0.323838^\circ\text{C}$, Mejora $= +3.5237\%$, 16/24 meses mejorados, 4,755/5,275 celdas mejoradas).
- **Bug computacional detectado:** **NO**.
- **FINAL TEST (2024–2025) abierto:** **0** (`assert TEST_FILES_OPENED_COUNT == 0`).

---

## 2. Hallazgos Diagnósticos Principales

### 1. Identidad Algebraica y Métricas Directas del Residual
- Se verificó que el error de reconstrucción de SST es idéntico al error de predicción del residual:
  $$\text{RMSE}(\text{SST}_{\text{hat}}, \text{SST}_{\text{MUR}}) \equiv \text{RMSE}(\hat{R}, R) = 0.323838^\circ\text{C}$$
- El $R^2$ residual combinado es modesto (**0.0640**), pero con correlación positiva robusta ($r = 0.2534, \rho = 0.3410$). En residual learning, un $R^2$ residual de esta magnitud es suficiente para reducir el error de SST en un $+3.52\%$, ya que $B_0$ ya explica el 90.02% de la varianza térmica total.
- La pendiente de calibración ($b = 0.0677$) y la razón de desviaciones estándar ($\text{std}(\hat{R})/\text{std}(R) = 0.2672$) documentan *a pattern consistent with regression toward the conditional mean in the MSE-trained and regularized model*.

### 2. Estabilidad Interanual del Sesgo
- El sesgo medio global cercano a cero ($-0.000470^\circ\text{C}$) no es producto de una cancelación extrema entre signos opuestos, sino que en ambos años $C_0$ reduce notablemente el sesgo de $B_0$:
  - 2022: Bias B0 = $-0.005902^\circ\text{C}$ $\rightarrow$ Bias C0 = **$-0.005682^\circ\text{C}$**.
  - 2023: Bias B0 = $+0.055959^\circ\text{C}$ $\rightarrow$ Bias C0 = **$+0.004742^\circ\text{C}$**.

### 3. Origen Estadístico de los 8 Meses Negativos
- Los 8 meses con degradación relativa (`2022-06`, `2022-09`, `2022-10`, `2022-12`, `2023-01`, `2023-06`, `2023-10`, `2023-11`) coinciden con anomalías residuales observadas $R$ cuyo signo real se desfasa sistemáticamente del ciclo climatológico medio predicho por las componentes armónicas (`doy_sin`, `doy_cos`).
- En estos meses adversos, la tasa de acierto de signo cae a una media de **47.6%** (inferior al azar).

### 4. Caso 2023-06 (-23.55%) vs 2023-07 (+18.58%)
- La auditoría técnica diaria (123 días de mayo a agosto 2023) confirmó continuidad absoluta de fechas, 5,275 celdas exactas, 0 duplicados, 0 NaNs y consistencia total en DOY y batimetría.
- El contraste no obedece a un error computacional, sino a **variabilidad temporal real del skill del modelo (*temporal variability in model skill*)**: en junio 2023 se presentó una anomalía residual fría que desfasó la climatología, mientras que en julio–agosto 2023 se presentaron *larger MUR–BIL residual discrepancies* fuertemente alineadas con la formulación del modelo, aportando mejoras superiores al $+18\%$.

### 5. Descomposición del Régimen de Bajo Residual (DEV-P0-P50)
- En $|R| < 0.2066^\circ\text{C}$ (donde el error relativo empeora en $-19.66\%$):
  - **42.85% (Categoría A):** Signo correcto + sub-corrección $\rightarrow$ **Mejora $\Delta\text{RMSE} = -0.0264^\circ\text{C}$**.
  - **12.96% (Categoría B):** Signo correcto + sobre-corrección $\rightarrow$ Empeora $\Delta\text{RMSE} = +0.0369^\circ\text{C}$.
  - **41.68% (Categoría C):** Signo incorrecto + magnitud pequeña $\rightarrow$ Empeora $\Delta\text{RMSE} = +0.0360^\circ\text{C}$.
  - **2.50% (Categoría D):** Signo incorrecto + magnitud grande $\rightarrow$ Empeora $\Delta\text{RMSE} = +0.2820^\circ\text{C}$.
- **Conclusión:** El deterioro en P0–P50 se debe predominantemente a **error de signo (44.18% en Cats C y D)** al intentar corregir *small MUR–BIL discrepancies* donde $B_0$ ya es casi perfecto ($\text{RMSE} = 0.1157^\circ\text{C}$), no a una sobre-corrección masiva de amplitud (el 76.2% de los casos presentan sub-corrección en magnitud).

### 6. Robustez de la Cola Alta (Discrepancias Grandes)
- La mejora relativa escala monótonamente en discrepancias grandes: DEV-P50-P75 (+1.93%), DEV-P75-P90 (+4.65%), DEV-P90-P95 (+5.86%), DEV-P95-P99 (+6.77%), DEV-P99+ (**+7.45%**).
- No es un artefacto de muestras reducidas: DEV-P99+ cuenta con 30,400 observaciones distribuidas en 623 días y en todas las 5,275 celdas.

### 7. Sensibilidad Bootstrap a Dependencia Temporal
- **1-Day Cluster (D33):** Mediana = $-0.011902^\circ\text{C}$, IC 95% = **$[-0.016446, -0.007224]^\circ\text{C}$**, $P(\Delta < 0) = 1.000$.
- **7-Day Moving Block:** Mediana = $-0.011774^\circ\text{C}$, IC 95% = **$[-0.021985, -0.002782]^\circ\text{C}$**, $P(\Delta < 0) = 0.989$.
- **14-Day Moving Block:** Mediana = $-0.011778^\circ\text{C}$, IC 95% = **$[-0.023610, -0.001051]^\circ\text{C}$**, $P(\Delta < 0) = 0.978$.
- **Conclusión de Robustez:** *The global RMSE improvement is robust to short-range temporal dependence under the evaluated bootstrap block lengths.* Incluso preservando bloques de 14 días con memoria sinóptica/subestacional, el límite superior permanece estrictamente negativo.

### 8. Diagnóstico Espacial vs Profundidad y Distancia a Costa
- La ganancia de precisión ocurre en más del 95% de las celdas en aguas costeras y someras (<100 m de profundidad, <20 km de la costa).
- En aguas abiertas profundas (>500 m), el 79.9% de las celdas mejoran, con menor amplitud de $\Delta\text{RMSE}$ debido a que el residual mar adentro es inherentemente más homogéneo.
- La degradación espacial es heterogénea (*spatially heterogeneous degradation*), refutando la idea de que la batimetría somera concentre las fallas del modelo.

---

## 3. Dictamen y Recomendación respecto a FINAL TEST

- **Dictamen D33:** **`D33-B — UNCHANGED`** (*Positive external generalization with insufficient month-level stability to satisfy the predeclared D33-A criterion*).
- **Regla Predeclarada de Recomendación:** Cumple los 7 criterios simultáneos (hash verificado, métricas reproducidas, sin bugs, auditoría junio–julio limpia, bootstrap de 7 y 14 días con límite superior < 0).
- **Recomendación Formal:** **`PREPARE FINAL TEST`**.
- *Aclaración vinculante:* Esta recomendación **NO autoriza la apertura automática de FINAL TEST (2024–2025)**. Dicha apertura exigirá un protocolo de congelamiento formal e independiente análogo al de D33.

---

## 4. Catálogo de Entregables Generados

Directorio: `DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/`

### A. 13 Tablas CSV (`tables/`)
1. [`residual_metrics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/residual_metrics.csv)
2. [`yearly_bias_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/yearly_bias_diagnostics.csv)
3. [`negative_months_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/negative_months_diagnostics.csv)
4. [`daily_june_july_2023_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/daily_june_july_2023_diagnostics.csv)
5. [`residual_regime_population_validation.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/residual_regime_population_validation.csv)
6. [`sign_accuracy_by_regime.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/sign_accuracy_by_regime.csv)
7. [`overcorrection_by_regime.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/overcorrection_by_regime.csv)
8. [`low_residual_error_decomposition.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/low_residual_error_decomposition.csv)
9. [`high_residual_regime_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/high_residual_regime_diagnostics.csv)
10. [`bootstrap_sensitivity.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/bootstrap_sensitivity.csv)
11. [`spatial_depth_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/spatial_depth_diagnostics.csv)
12. [`spatial_distance_diagnostics.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/spatial_distance_diagnostics.csv)
13. [`postvalidation_audit_summary.csv`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/tables/postvalidation_audit_summary.csv)

### B. 8 Figuras Científicas (300 DPI, `figures/`)
1. `figures/fig_d34_1_monthly_improvement_negative_months.png`
2. `figures/fig_d34_2_daily_delta_rmse_rolling7d.png`
3. `figures/fig_d34_3_may_august_2023_daily_diagnostics.png`
4. `figures/fig_d34_4_sign_accuracy_by_regime.png`
5. `figures/fig_d34_5_over_under_correction_by_regime.png`
6. `figures/fig_d34_6_improvement_by_regime_with_counts.png`
7. `figures/fig_d34_7_bootstrap_sensitivity_comparison.png`
8. `figures/fig_d34_8_spatial_delta_rmse_vs_depth.png`

### C. Reporte Formal
- [`faseD34_postvalidation_diagnostics.md`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/reports/faseD34_postvalidation_diagnostics.md)

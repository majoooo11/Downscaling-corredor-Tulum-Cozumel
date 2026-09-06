# Reporte Científico — Fase D.3.4
## Post-Validation Diagnostic Audit of Frozen E3b-C0
**Fecha de Ejecución:** 2026-09-06 02:14:04 UTC  
**Fase Previa:** Fase D.3.3 External Validation (DICTAMEN FORMAL = **D33-B — UNCHANGED**)  
**Estado Metodológico D32:** METHODOLOGICALLY CLOSED  
**Recomendación respecto a FINAL TEST:** **PREPARE FINAL TEST**

---

## 1. Objetivo
El propósito de la Fase D.3.4 es auditar de forma **estrictamente no adaptativa** el comportamiento del estimador congelado `E3b-C0` en la partición de validación externa 2022–2023, con el fin de diagnosticar por qué el modelo produce una mejora cuantitativa global (+3.52%), interanual (2022 y 2023 positivos) y espacialmente extensa (90.14% de las celdas), pero manifiesta inestabilidad a escala mensual (16/24 meses con mejora).

---

## 2. Estado Heredado de D33
- **Modelo:** `E3b-C0` (*best-performing parsimonious formulation evaluated in D32*).
- **Features congeladas:** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`.
- **Hiperparámetros congelados:** `max_depth=4`, `learning_rate=0.10`, `n_estimators=19`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `tree_method='hist'`, `objective='reg:squarederror'`.
- **Dictamen D33 Formal:** **`D33-B — PARTIAL / MIXED GENERALIZATION`**.
- **Interpretación canónica:** *Positive external generalization with insufficient month-level stability to satisfy the predeclared D33-A criterion.*

---

## 3. Principio de No Adaptación
Esta fase **no autoriza ni contempla la modificación del modelo**, adición de variables (e.g. ERA5/CMEMS), reentrenamiento, ajuste de hiperparámetros ni optimización dirigida a revertir los 8 meses adversos. Un resultado adverso se asume como información científica válida sobre los límites de generalización de la formulación parsimoniosa. La partición de **FINAL TEST (2024–2025) permaneció 100% blindada y cerrada** (`TEST_FILES_OPENED_COUNT = 0`).

---

## 4. Integridad del Modelo y Celdas Congeladas
- **SHA-256 del modelo cargado:** `fb151cafd466613d0bda2cadfee61e9ebc1f8e0f889d9a646cc010ba6436ef5d` (coincidencia idéntica con `frozen_model_spec.json`).
- **SHA-256 de celdas congeladas:** `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb` (5,275 celdas únicas idénticas a D32/D33).
- **Modelo modificado:** NO.
- **Modelo reentrenado:** NO.
- **Directorio `models/` en D34:** NO creado.

---

## 5. Reproducción Exacta de Métricas D33
El pipeline reprodujo con precisión de punto flotante los resultados maestros de D33 sobre las 3,850,750 observaciones de 2022–2023:
- B0 RMSE: **0.335666 °C** (idéntico a D33).
- C0 RMSE: **0.323838 °C** (idéntico a D33).
- Mejora RMSE combinada: **+3.5237%** (idéntico a D33).
- Meses mejorados: **16 / 24** (idéntico a D33).
- Celdas mejoradas: **4,755 / 5,275 (90.14%)** (idéntico a D33).
- **BUG DETECTED:** **NO**.

---

## 6. Identidad Algebraica SST–Residual
Se verificó analítica y computacionalmente la equivalencia:
$$\text{SST}_{\text{hat}} - \text{SST}_{\text{MUR}} = (\text{SST}_{\text{BIL}} + \hat{R}) - \text{SST}_{\text{MUR}} = \hat{R} - R$$
Por tanto:
$$\text{RMSE}(\text{SST}_{\text{hat}}, \text{SST}_{\text{MUR}}) \equiv \text{RMSE}(\hat{R}, R)$$
$$\text{MAE}(\text{SST}_{\text{hat}}, \text{SST}_{\text{MUR}}) \equiv \text{MAE}(\hat{R}, R)$$
Las métricas directas sobre el residual confirman que la reconstrucción térmica no es una entidad desacoplada, sino la traslación lineal de la predicción de $\hat{R}$.

---

## 7. Métricas Directas del Residual
| Periodo | RMSE Residual (°C) | MAE Residual (°C) | Bias Residual (°C) | $R^2$ Residual | Pearson $r(R, \hat{R})$ | Spearman $\rho$ | $\text{std}(\hat{R})/\text{std}(R)$ | Pendiente Calibración $b$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2022** | 0.314512 | 0.241482 | -0.005682 | **0.043928** | 0.2140 | 0.2655 | 0.1744 | 0.0373 |
| **2023** | 0.332904 | 0.267432 | +0.004742 | **0.066461** | 0.2627 | 0.3680 | 0.3116 | 0.0819 |
| **2022–2023** | 0.323838 | 0.254457 | -0.000470 | **0.064029** | 0.2534 | 0.3410 | 0.2672 | 0.0677 |

- **Explicación del $R^2$ residual:** Aunque el $R^2$ residual es bajo (0.0640), la correlación es positiva y estadísticamente significativa ($r = 0.2534$). En residual learning, un $R^2$ modesto sobre la anomalía de alta frecuencia es matemáticamente compatible con una reducción de error global de SST del $+3.52\%$, ya que $B_0$ ya explica el 90.02% de la varianza total de SST.
- **Contracción de amplitud:** El cociente $\text{std}(\hat{R})/\text{std}(R) = 0.2672$ y la pendiente $b = 0.0677$ documentan *a pattern consistent with regression toward the conditional mean in the MSE-trained and regularized model*.

---

## 8. Bias por Año
- **2022:** Bias B0 = -0.005902 °C | Bias C0 = **-0.005682 °C** (Mediana = -0.040602 °C).
- **2023:** Bias B0 = +0.055959 °C | Bias C0 = **+0.004742 °C** (Mediana = -0.029566 °C).
- **Combinado:** Bias C0 = **-0.000470 °C**.
- **Diagnóstico:** El sesgo medio cercano a cero no es un artefacto de cancelación extrema, sino que en ambos años individuales C0 reduce sustancialmente el sesgo respecto a B0 (en 2023, B0 presentaba $+0.055959^\circ\text{C}$ de sobrecalentamiento que C0 corrige a $+0.004742^\circ\text{C}$).

---

## 9. Auditoría de los 8 Meses Negativos
Los 8 meses con degradación relativa ($\Delta\text{RMSE} > 0$) fueron:
`2022-06` (-3.17%), `2022-09` (-1.12%), `2022-10` (-11.78%), `2022-12` (-4.07%), `2023-01` (-2.81%), `2023-06` (-23.55%), `2023-10` (-1.02%), `2023-11` (-0.64%).
- **Patrón Común Estadístico:** Los meses con degradación severa (e.g. 2023-06 y 2022-10) coinciden con anomalías residuales observadas $R$ de signo persistente opuesto al ciclo climatológico medio predicho por las componentes armónicas (`doy_sin`, `doy_cos`).
- La tasa media de acierto de signo en los meses negativos cae a un promedio de 47.6%, comparado con >65% en los meses positivos.

---

## 10. Diagnóstico Mayo–Agosto 2023
Se auditó día a día la transición entre **junio 2023 (-23.55%)** y **julio 2023 (+18.58%)**:
- Continuidad temporal: 123 días continuos, sin fechas faltantes.
- Celdas por día: 5,275 celdas exactas, 0 duplicados, 0 NaNs.
- DOY y batimetría: Cálculos armónicos y profundidades idénticos y continuos.
- **Diagnóstico:** No se identificó ninguna anomalía informática ni discontinuidad de datos. El brusco salto responde a *temporal variability in model skill*: en junio de 2023, el residual observado experimentó un enfriamiento anómalo desacoplado de la climatología, mientras que en julio–agosto se presentaron *larger MUR–BIL residual discrepancies* bien alineadas con el ciclo térmico donde el modelo aportó una ganancia superior al $+18\%$.

---

## 11. Población por Regímenes DEV
| Régimen | Umbral $|R|$ | N Muestras | % Validación | Media $|R|$ (°C) | Mediana $|R|$ (°C) | P90 $|R|$ (°C) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DEV-P0-P50** | $< 0.2066^\circ\text{C}$ | 1,827,764 | 47.47% | 0.0994 | 0.0976 | 0.1833 |
| **DEV-P50-P75** | $0.2066–0.3604^\circ\text{C}$ | 994,741 | 25.83% | 0.2781 | 0.2754 | 0.3414 |
| **DEV-P75-P90** | $0.3604–0.5377^\circ\text{C}$ | 630,094 | 16.36% | 0.4374 | 0.4319 | 0.5117 |
| **DEV-P90-P95** | $0.5377–0.6652^\circ\text{C}$ | 206,608 | 5.37% | 0.5938 | 0.5900 | 0.6474 |
| **DEV-P95-P99** | $0.6652–0.9659^\circ\text{C}$ | 161,143 | 4.18% | 0.7742 | 0.7579 | 0.8985 |
| **DEV-P99+** | $\ge 0.9659^\circ\text{C}$ | 30,400 | 0.79% | 1.1060 | 1.0673 | 1.2931 |

La distribución en VALIDATION es altamente representativa de DEVELOPMENT (el régimen DEV-P0-P50 contiene 47.47% de las muestras, muy cercano al 50% teórico).

---

## 12. Sign Accuracy
- En el régimen **DEV-P0-P50**, la tasa de acierto de signo es de tan solo **55.81%** (cercana al azar).
- Conforme $|R|$ aumenta hacia la cola alta, la precisión de signo escala monótonamente:
  - DEV-P50-P75: 62.37%
  - DEV-P75-P90: 66.45%
  - DEV-P90-P95: 73.75%
  - DEV-P95-P99: 80.45%
  - DEV-P99+: **88.87%**.

---

## 13. Over/Under-Correction
- Frecuencia de sobre-corrección en DEV-P0-P50: **23.80%**.
- Frecuencia de sub-corrección en DEV-P0-P50: **76.20%**.
- Diferencia de magnitud $D_{mag} = |\hat{R}| - |R|$ media en DEV-P0-P50: **-0.0475 °C**.
- **Diagnóstico:** El deterioro de P0–P50 no proviene de una sobre-corrección masiva de amplitud (el 76.2% de los casos son sub-correcciones en magnitud), sino de intentar ajustar discrepancias mínimas donde el signo estimado es frecuentemente erróneo.

---

## 14. Descomposición del Régimen DEV-P0-P50
| Categoría | N | % Régimen | RMSE B0 (°C) | RMSE C0 (°C) | $\Delta\text{RMSE}$ (°C) | Media $|R|$ (°C) | Media $|\hat{R}|$ (°C) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Signo Correcto + Sub-corrección** | 783,240 | 42.85% | 0.1283 | 0.1018 | -0.0264 | 0.1164 | 0.0295 |
| **B. Signo Correcto + Sobre-corrección** | 236,821 | 12.96% | 0.0784 | 0.1153 | +0.0369 | 0.0579 | 0.1336 |
| **C. Signo Incorrecto + Magnitud Pequeña** | 761,763 | 41.68% | 0.1121 | 0.1481 | +0.0360 | 0.0952 | 0.0352 |
| **D. Signo Incorrecto + Magnitud Grande** | 45,778 | 2.50% | 0.1112 | 0.3932 | +0.2820 | 0.0952 | 0.2901 |
| **E. Signo Cero / Magnitud Igual** | 162 | 0.01% | 0.0001 | 0.1139 | +0.1138 | 0.0000 | 0.0636 |

**Causa Fundamental del Deterioro en P0–P50:** Cuando el residual real es muy pequeño ($|R| < 0.2066^\circ\text{C}$, con media de apenas $0.099^\circ\text{C}$), $B_0$ ya es casi perfecto ($\text{RMSE} = 0.1157^\circ\text{C}$). En el 44.2% de los casos (Categoría C), el modelo predice en la dirección incorrecta; aunque la magnitud predicha sea modesta ($\sim 0.05^\circ\text{C}$), sumar un desplazamiento de signo contrario a una discrepancia mínima incrementa matemáticamente el error cuadrático.

---

## 15. Diagnóstico de Grandes Discrepancias MUR–BIL (Cola Alta)
En los regímenes de discrepancia residual moderada a alta:
- **DEV-P50-P75:** Mejora de **+1.93%** (63.2% de días mejoran).
- **DEV-P75-P90:** Mejora de **+4.65%** (67.2% de días mejoran).
- **DEV-P90-P95:** Mejora de **+5.86%** (72.1% de días mejoran).
- **DEV-P95-P99:** Mejora de **+6.77%** (78.0% de días mejoran).
- **DEV-P99+:** Mejora de **+7.45%** (82.1% de días mejoran).
- **Soporte Muestral:** No se trata de un artefacto de unos pocos días u outliers aislados: DEV-P99+ cuenta con 30,400 observaciones distribuidas en 623 días y en la totalidad de las 5,275 celdas. La ganancia en la cola alta es robusta y sistemática.

---

## 16. Sensibilidad Bootstrap (1d / 7d / 14d)
| Tipo de Bloque | Longitud $L$ | Mediana $\Delta\text{RMSE}$ (°C) | IC 95% Inferior (°C) | IC 95% Superior (°C) | $P(\Delta\text{RMSE} < 0)$ | Tail Fraction Bilateral |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Cluster (D33)** | 1 día | -0.011902 | -0.016446 | **-0.007224** | 1.0000 | 1.9980e-03 |
| **7-Day Moving Block** | 7 días | -0.011774 | -0.021985 | **-0.002782** | 0.9890 | 2.3976e-02 |
| **14-Day Moving Block** | 14 días | -0.011778 | -0.023610 | **-0.001051** | 0.9780 | 4.5954e-02 |

**Veredicto de Sensibilidad:**
*The global RMSE improvement is robust to short-range temporal dependence under the evaluated bootstrap block lengths.*
Incluso preservando bloques continuos de 7 y 14 días con autocorrelación temporal serial, el límite superior del intervalo de confianza al 95% permanece estrictamente negativo (-0.002782 °C y -0.001051 °C), confirmando que la ganancia global de +3.52% no es un artefacto de independencia temporal asumida.

---

## 17. Diagnóstico Espacial Descriptivo
- **Estratificación por Profundidad GEBCO:**
  - 0–20 m: 96.3% celdas mejoran (Mediana $\Delta\text{RMSE} = -0.0184^\circ\text{C}$).
  - 20–50 m: 95.7% celdas mejoran (Mediana $\Delta\text{RMSE} = -0.0116^\circ\text{C}$).
  - 50–100 m: 99.1% celdas mejoran (Mediana $\Delta\text{RMSE} = -0.0103^\circ\text{C}$).
  - 100–500 m: 99.8% celdas mejoran (Mediana $\Delta\text{RMSE} = -0.0132^\circ\text{C}$).
  - >500 m: 79.9% celdas mejoran (Mediana $\Delta\text{RMSE} = -0.0039^\circ\text{C}$).
- **Correlaciones Descriptivas:**
  - Spearman(water_depth_m, $\Delta\text{RMSE}$) = -0.1084 ($p < 10^{-5}$).
  - Spearman(distance_coast_km, $\Delta\text{RMSE}$) = -0.1691 ($p < 10^{-5}$).
- **Interpretación no causal:** La degradación espacial es heterogénea (*spatially heterogeneous degradation*); existe una correlación débil-moderada que indica menor ganancia relativa en aguas muy costeras y someras, pero no se infiere que la profundidad sea una causa directa del error térmico.

---

## 18. Correcciones de Interpretación Científica
Se incorporan las siguientes precisiones conceptuales vinculantes:
1. Reemplazo de "regímenes de bajo gradiente" por **"low-residual regime"**.
2. Reemplazo de "anomalías submesoescala débiles" por **"small MUR–BIL discrepancies"**.
3. Reemplazo de "shrinkage inherente a MSE" por **"a pattern consistent with regression toward the conditional mean in the MSE-trained and regularized model"**.
4. Reemplazo de "degradación en batimetría compleja" por **"spatially heterogeneous degradation"**.
5. Denominación estricta de la validación como **"out-of-development temporal validation under the frozen D33 protocol"**.

---

## 19. Dictamen D33 Heredado
El dictamen formal de la Fase D.3.3 permanece inalterado:
### **D33 FORMAL DECISION: D33-B — UNCHANGED**
*(Positive external generalization with insufficient month-level stability to satisfy the predeclared D33-A criterion).*

---

## 20. Recomendación respecto a FINAL TEST
Aplicando la regla predeclarada congelada:
1. `MODEL HASH VERIFIED` = **YES**
2. `FROZEN CELLS HASH VERIFIED` = **YES**
3. `D33 METRICS REPRODUCED` = **YES**
4. `NO DATA OR PREPROCESSING BUG` = **YES** (Verificado)
5. `JUNE-JULY AUDIT IDENTIFIES NO COMPUTATIONAL DISCONTINUITY` = **YES** (Verificado)
6. `7-DAY BOOTSTRAP CI95 UPPER < 0` = **YES** (-0.002782 °C)
7. `14-DAY BOOTSTRAP CI95 UPPER < 0` = **YES** (-0.001051 °C)

### **RECOMENDACIÓN FORMAL: PREPARE FINAL TEST**
*Aclaración de blindaje:* Esta recomendación **NO autoriza la apertura automática de FINAL TEST (2024–2025)**. Dicha apertura exigirá un protocolo de congelamiento formal independiente análogo al de D33.

---

## 21. Limitaciones
1. **Inestabilidad Estacional Intrínseca:** El modelo `E3b-C0` carece de forzamiento atmosférico/dinámico explícito; en meses donde el forzamiento real se desfasa del ciclo armónico estacional, la formulación no puede anticipar el signo del residual.
2. **Penalización en Discrepancias Mínimas:** En la mitad de las observaciones donde el residual real es inferior a $0.20^\circ\text{C}$, el estimador introduce un error cuadrático agregado por imprecisión de signo.
3. **Dependencia Temporal de Corto Rango:** Aunque los intervalos bootstrap a 7 y 14 días permanecen negativos, la amplitud del intervalo se ensancha, reflejando mayor incertidumbre al respetar la memoria térmica sinóptica.

---

## 22. Catálogo de Entregables
Directorio: `DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/`
- **Tablas (13):**
  1. `tables/residual_metrics.csv`
  2. `tables/yearly_bias_diagnostics.csv`
  3. `tables/negative_months_diagnostics.csv`
  4. `tables/daily_june_july_2023_diagnostics.csv`
  5. `tables/residual_regime_population_validation.csv`
  6. `tables/sign_accuracy_by_regime.csv`
  7. `tables/overcorrection_by_regime.csv`
  8. `tables/low_residual_error_decomposition.csv`
  9. `tables/high_residual_regime_diagnostics.csv`
  10. `tables/bootstrap_sensitivity.csv`
  11. `tables/spatial_depth_diagnostics.csv`
  12. `tables/spatial_distance_diagnostics.csv`
  13. `tables/postvalidation_audit_summary.csv`
- **Figuras (8):**
  1. `figures/fig_d34_1_monthly_improvement_negative_months.png`
  2. `figures/fig_d34_2_daily_delta_rmse_rolling7d.png`
  3. `figures/fig_d34_3_may_august_2023_daily_diagnostics.png`
  4. `figures/fig_d34_4_sign_accuracy_by_regime.png`
  5. `figures/fig_d34_5_over_under_correction_by_regime.png`
  6. `figures/fig_d34_6_improvement_by_regime_with_counts.png`
  7. `figures/fig_d34_7_bootstrap_sensitivity_comparison.png`
  8. `figures/fig_d34_8_spatial_delta_rmse_vs_depth.png`

# Reporte Científico — Fase D.3.5 (FINAL AUDITED & CORRECTED)
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
- **Target:** $R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$
- **Reconstrucción:** $\widehat{\text{SST}} = \text{SST}_{\text{BIL}} + \hat{R}$
- **Features (en orden exacto):** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`
- **Algoritmo e Hiperparámetros:** `XGBRegressor` con `n_estimators=19`, parámetro `max_depth=4`, `learning_rate=0.10`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `random_state=42`, `tree_method='hist'`, `objective='reg:squarederror'`. Sin early stopping ni búsqueda de hiperparámetros.

---

## 5. Final Refit 2015–2023
Conforme a la decisión predeclarada de incorporar toda la información previa a TEST en el estimador final:
- **Periodo de Refit:** 2015-01-01 a 2023-12-31 (9 años completos, incluyendo bisiestos 2016 y 2020).
- **Días de Entrenamiento:** 3,287 días.
- **Observaciones de Entrenamiento:** **17,338,925** ($3,287 \times 5,275$ celdas congeladas).
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
- **Días en TEST 2024 (bisiesto):** 366 días $\times$ 5,275 celdas = **1,930,650 filas**.
- **Días en TEST 2025 (regular):** 365 días $\times$ 5,275 celdas = **1,925,375 filas**.
- **Total TEST Combinado:** 731 días $\times$ 5,275 celdas = **3,856,025 filas**.
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
  - Mejora en RMSE: **+9.3986%** ($\Delta\text{RMSE} = -0.035687^\circ\text{C}$)
  - Bias: $B_0$ = -0.063459 °C | $C_0$ = **-0.008891 °C**
- **Año 2025 (365 días):**
  - Baseline $B_0$ RMSE: 0.333355 °C | $C_0$ RMSE: **0.318453 °C**
  - Mejora en RMSE: **+4.4704%** ($\Delta\text{RMSE} = -0.014902^\circ\text{C}$)
  - Bias: $B_0$ = -0.061389 °C | $C_0$ = **-0.019516 °C**

---

## 11. Estabilidad Mensual
- **Meses con Mejora ($\Delta\text{RMSE} < 0$):** **18 / 24 meses** (75.0%).
- Month-level skill variability was associated with reduced residual-sign agreement during lower-performing months.

---

## 12. Estabilidad Diaria
- **Días con Mejora ($\Delta\text{RMSE} < 0$):** **484 / 731 días** (**66.21%**).
- **Mediana diaria de $\Delta\text{RMSE}$:** -0.0163 °C.
- **Percentiles de $\Delta\text{RMSE}$ diario:** P10 = -0.0709 °C, P25 = -0.0371 °C, P75 = +0.0076 °C, P90 = +0.0270 °C.

---

## 13. Estabilidad Espacial
- **Celdas con Mejora ($\Delta\text{RMSE} < 0$):** **5,273 / 5,275 celdas** (**99.96%**).
- **Mediana espacial de $\Delta\text{RMSE}$:** -0.0223 °C.
- **Percentiles de $\Delta\text{RMSE}$ espacial:** P10 = -0.0463 °C, P90 = -0.0093 °C.

---

## 14. Residual Explanatory Skill
- **$R^2$ Residual:** **0.112175** (11.22% de la varianza residual explicada).
- **Correlación Pearson $r(R, \hat{R})$:** **0.3512**.
- **Correlación Spearman $\rho(R, \hat{R})$:** **0.3671**.
- **Cociente de Desviación Típica $\text{std}(\hat{R})/\text{std}(R)$:** **0.2536**.
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
| **DEV-P0-P50** | $< 0.2066^\circ\text{C}$ | 1,857,229 | 48.16% | 0.1154 | 0.1396 | **-20.99%** | 56.4% | 31.1% | 68.9% |
| **DEV-P50-P75** | $0.2066–0.3604^\circ\text{C}$ | 938,991 | 24.35% | 0.2808 | 0.2710 | **+3.51%** | 65.8% | 1.3% | 98.7% |
| **DEV-P75-P90** | $0.3604–0.5377^\circ\text{C}$ | 604,739 | 15.68% | 0.4400 | 0.4071 | **+7.48%** | 70.9% | 0.0% | 100.0% |
| **DEV-P90-P95** | $0.5377–0.6652^\circ\text{C}$ | 204,020 | 5.29% | 0.5961 | 0.5412 | **+9.21%** | 75.7% | 0.0% | 100.0% |
| **DEV-P95-P99** | $0.6652–0.9659^\circ\text{C}$ | 181,676 | 4.71% | 0.7877 | 0.6926 | **+12.07%** | 82.5% | 0.0% | 100.0% |
| **DEV-P99+** | $\ge 0.9659^\circ\text{C}$ | 69,370 | 1.80% | 1.1469 | 1.0151 | **+11.49%** | 87.2% | 0.0% | 100.0% |

---

## 17. Correction Behavior
En el régimen de bajo residual DEV-P0-P50, la sub-corrección en magnitud predomina ampliamente (68.9% de las observaciones). El deterioro se asocia a la reducida exactitud de signo (56.4%), mientras que en discrepancias residuales mayores (DEV-P95+), la precisión de signo supera el 80% y produce mejoras sustanciales.

---

## 18. Bootstrap 1d / 7d / 14d
| Esquema Bootstrap | Longitud $L$ | Mediana $\Delta\text{RMSE}$ (°C) | IC 95% Inferior (°C) | IC 95% Superior (°C) | $P(\Delta\text{RMSE} < 0)$ | Tail Fraction Bilateral |
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
- `Spearman(water_depth_m, DeltaRMSE_cell) = +0.7376` ($p < 10^-15$).
- `Spearman(distance_coast_km, DeltaRMSE_cell) = +0.5336` ($p < 10^-15$).
- La correlación positiva indica que la magnitud de la reducción de error es más negativa (mayor beneficio) en aguas someras litorales (0–20 m: $\Delta\text{RMSE} = -0.0383^\circ\text{C}$), atenuándose la ganancia hacia aguas oceánicas profundas (>500 m: $\Delta\text{RMSE} = -0.0145^\circ\text{C}$).

---

## 21. Aplicación Mecánica de Criterios D35
1. **Criterio 1 (Mejora en RMSE Combinado $\ge +1.00\%$):** **PASS** (+7.2247%)
2. **Criterio 2 (MAE_C0 $\le$ MAE_B0):** **PASS** (C0: 0.256325 vs B0: 0.272713 °C)
3. **Criterio 3 (Bootstrap 14d CI95_upper $< 0$):** **PASS** (-0.009792 °C)
4. **Criterio 4 (Mejora en 2024 $> 0$):** **PASS** (+9.3986%)
5. **Criterio 5 (Mejora en 2025 $> 0$):** **PASS** (+4.4704%)
6. **Criterio 6 (Meses mejorados $\ge 18 / 24$):** **PASS** (18 / 24)
7. **Criterio 7 (Celdas mejoradas $\ge 75\%$):** **PASS** (99.96%)

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
2. **Bajo residual:** En discrepancias mínimas ($|R| < 0.21^\circ\text{C}$), el modelo introduce sobrecorrección y error neto adicional debido a la menor detectabilidad de signo.
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

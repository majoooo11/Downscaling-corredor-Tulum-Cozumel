# Reporte Científico — Fase D.3.3
## External Validation of Frozen E3b-C0
**Fecha:** 2026-09-06 01:41:03 UTC  
**Estado Metodológico D32:** METHODOLOGICALLY CLOSED  
**Dictamen Formal D33:** **D33-B** (Positive but partial or unstable external generalization.)  
**Recomendación:** **HOLD FINAL TEST**

---

## 1. Objetivo Científico
La Fase D.3.3 tiene como único propósito responder a la pregunta confirmatoria:
> **¿El modelo parsimonioso congelado E3b-C0 generaliza a un periodo temporal completamente independiente (2022–2023)?**

Esta fase **no constituye desarrollo, ni tuning, ni ingeniería de variables**. No se realizaron ajustes retrospectivos ni adaptaciones oportunistas sobre la partición de validación.

---

## 2. Estado Heredado de D32
- **Modelo seleccionado:** `E3b-C0` (definido estrictamente como *best-performing parsimonious formulation evaluated in D32*, sin calificarlo como "óptimo absoluto").
- **Features congeladas:** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`.
- **Algoritmo:** `XGBRegressor` (`tree_method='hist'`, `objective='reg:squarederror'`).
- **Hiperparámetros congelados:** `max_depth=4`, `learning_rate=0.10`, `n_estimators=19` (`best_iteration=18` zero-indexed), `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `random_state=42`.
- **Desempeño histórico en Holdout Diagnóstico 2021:**
  - B0 RMSE: 0.359493 °C, MAE: 0.277337 °C
  - C0 RMSE: 0.349274 °C, MAE: 0.266264 °C
  - Mejora RMSE vs B0: **+2.84243%** (Skill: 0.0284243).

---

## 3. Pre-Validation Freeze
Antes de abrir cualquier registro correspondiente a los años 2022 o 2023, se generó el manifiesto inmutable `frozen_model_spec.json`:
- **SHA-256 frozen_cell_ids.csv:** `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb`
- **SHA-256 E3b-C0_PREVALIDATION.json:** `fb151cafd466613d0bda2cadfee61e9ebc1f8e0f889d9a646cc010ba6436ef5d`
- **SHA-256 fase_d33_external_validation_c0.py:** `2f1d971b0b152bf9f402b493e46def6aa3ab64842c825e07b447b65e402251f9`
- **Criterios de decisión congelados:**
  - `D33-A`: Mejora combinada $\ge +1.00\%$, $\text{MAE}_{C0} \le \text{MAE}_{B0}$, CI95 superior de $\Delta\text{RMSE} < 0$, mejora en 2022 y 2023 $> 0$, meses mejorados $\ge 18/24$, celdas mejoradas $\ge 75\%$.
  - `D33-B`: $\text{RMSE}_{C0} < \text{RMSE}_{B0}$ en validación combinada, sin satisfacer la totalidad de D33-A.
  - `D33-C`: $\text{RMSE}_{C0} \ge \text{RMSE}_{B0}$ en validación combinada.

---

## 4. Dominio Espacial Congelado
Se heredaron sin modificación las **5,275 celdas oceánicas** de la evaluación final de D32, excluyendo formalmente las celdas sin soporte de vecindad $3\times 3$ (`[0, 161, 4437, 4472]`).
- Total de celdas congeladas: **5,275**.
- Integridad: 0 duplicados, identificadores únicos validados.

---

## 5. Prevalidation Refit 2015–2021
Se reentrenó el estimador `E3b-C0_PREVALIDATION` utilizando la totalidad de la ventana de desarrollo previa (2015–2021):
- Total de días: **2,557**.
- Observaciones de entrenamiento: **13,488,175** ($2,557 \times 5,275$).
- Cero duplicados en `(date, cell_id)`.
- Árboles entrenados: **19 boosting rounds** exactos.

---

## 6. Apertura Controlada de VALIDATION
El acceso a los datos externos se efectuó mediante la función instrumentada `open_validation_file`:
- `validation_2022.parquet`: 365 días $\times$ 5,275 celdas = 1,925,375 filas.
- `validation_2023.parquet`: 365 días $\times$ 5,275 celdas = 1,925,375 filas.
- **Total observaciones VALIDATION:** **3,850,750** (730 días continuos).
- **Archivos de VALIDATION abiertos:** **2**.
- **Archivos de TEST (2024–2025) abiertos:** **0** (blindaje absoluto verificado).

---

## 7. Resultados Globales 2022–2023
En la evaluación primaria combinada sobre el periodo 2022–2023:
- **B0 (Línea Base Bilineal):**
  - RMSE: **0.335666 °C**
  - MAE: **0.263594 °C**
  - Bias: **+0.025029 °C**
  - $R^2$: **0.900208**
- **E3b-C0 (Modelo Congelado):**
  - RMSE: **0.323838 °C**
  - MAE: **0.254457 °C**
  - Bias: **-0.000470 °C**
  - $R^2$: **0.907117**
- **Métricas Comparativas:**
  - $\Delta\text{RMSE}$: **-0.011828 °C**
  - Mejora Relativa RMSE: **+3.5237%**
  - Mejora Relativa MAE: **+3.4663%**
  - Skill Score RMSE: **0.035237**

---

## 8. Resultados por Año
| Periodo | N Observaciones | RMSE B0 (°C) | RMSE C0 (°C) | MAE B0 (°C) | MAE C0 (°C) | Mejora RMSE (%) | Mejora MAE (%) | Skill RMSE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2022** | 1,925,375 | 0.321710 | 0.314512 | 0.247039 | 0.241482 | **+2.2375%** | +2.2491% | 0.022375 |
| **2023** | 1,925,375 | 0.349065 | 0.332904 | 0.280150 | 0.267432 | **+4.6298%** | +4.5396% | 0.046298 |
| **2022–2023** | 3,850,750 | 0.335666 | 0.323838 | 0.263594 | 0.254457 | **+3.5237%** | +3.4663% | 0.035237 |

---

## 9. Comparación Descriptiva de Skill vs 2021
- **Mejora en Holdout Diagnóstico 2021:** +2.84243%
- **Mejora en Validación Externa 2022–2023:** +3.5237%
- **Diferencia de Skill:** **+0.6813 puntos porcentuales**
- *Aclaración metodológica obligatoria:* Dado que el estimador pre-validación fue reentrenado con 2015–2021, esta diferencia es de carácter descriptivo y **no debe interpretarse como una brecha pura de generalización (pure generalization gap)** de un estimador idéntico previamente ajustado.

---

## 10. Estabilidad Mensual
De los 24 meses analizados entre enero de 2022 y diciembre de 2023:
- Meses con mejora ($\Delta\text{RMSE} < 0$): **16 / 24 (66.7%)**
- Meses con degradación: **8 / 24**
- Mejor mes: **2023-07** (+18.58%)
- Peor mes: **2023-06** (-23.55%)
- Mediana de mejora mensual: **+1.58%** (IQR: [-1.04%, +6.47%])

---

## 11. Estabilidad Diaria
Sobre los 730 campos térmicos diarios independientes:
- Días con mejora ($\Delta\text{RMSE} < 0$): **478 / 730 (65.5%)**
- Mediana diaria de $\Delta\text{RMSE}$: **-0.006046 °C**
- Percentil 10: **-0.071587 °C**
- Percentil 90: **+0.027951 °C**

---

## 12. Distribución Espacial
En el análisis celda por celda ($N = 5,275$ celdas):
- Celdas con mejora de reconstrucción: **4755 / 5,275 (90.14%)**
- Mediana espacial de $\Delta\text{RMSE}$: **-0.009361 °C**
- Celdas con degradación focalizada principalmente en zonas de batimetría compleja o costera somera.

---

## 13. Desempeño por |R|
Evaluación estratificada utilizando los umbrales percentilares congelados de Development:
- **Low-residual regime (P0–P50)** ($|R| < 0.2066$ °C): -19.66% de mejora RMSE.
- **P50–P75** ($0.2066 \le |R| < 0.3604$ °C): +1.93%
- **P75–P90** ($0.3604 \le |R| < 0.5377$ °C): +4.65%
- **P90–P95** ($0.5377 \le |R| < 0.6652$ °C): +5.86%
- **P95–P99** ($0.6652 \le |R| < 0.9659$ °C): +6.77%
- **>=P99** ($|R| \ge 0.9659$ °C): +7.45%

---

## 14. Sobre-Corrección y Shrinkage
- En el régimen de bajo residual (P0–P50), la frecuencia de sobre-corrección ($|\hat{R}| > |R|$) fue de **23.8%**.
- La diferencia media de magnitud $D_{mag} = |\hat{R}| - |R|$ en P0–P50 es de **-0.0475 °C**.
- A medida que $|R|$ aumenta hacia regímenes extremos ($\ge P90$), predomina la sub-corrección por regularización y contracción al promedio condicional (shrinkage inherente a MSE).

---

## 15. Incertidumbre Bootstrap
A través de un Temporal Block Bootstrap ($B = 1,000$ réplicas sobre bloques diarios completos de 730 días):
- **Mediana Bootstrap $\Delta\text{RMSE}$:** **-0.011902 °C**
- **Intervalo de Confianza al 95%:** **[-0.016446 °C, -0.007224 °C]**
- **$P(\Delta\text{RMSE} < 0)$:** **1.0000**
- **Valor p bilateral corregido:** **1.9980e-03** (no reportado como cero idéntico).

---

## 16. Dictamen Formal D33
Aplicando mecánicamente la regla de decisión congelada:
- Regla 1 (Mejora $\ge +1.00\%$): **+3.5237%** $\rightarrow$ CUMPLIDA
- Regla 2 ($\text{MAE}_{C0} \le \text{MAE}_{B0}$): **0.254457 vs 0.263594 °C** $\rightarrow$ CUMPLIDA
- Regla 3 (CI95 superior $< 0$): **-0.007224 °C** $\rightarrow$ CUMPLIDA
- Regla 4 (2022 positivo): **+2.2375%** $\rightarrow$ CUMPLIDA
- Regla 5 (2023 positivo): **+4.6298%** $\rightarrow$ CUMPLIDA
- Regla 6 (Meses mejorados $\ge 18/24$): **16/24** $\rightarrow$ NO CUMPLIDA
- Regla 7 (Celdas mejoradas $\ge 75\%$): **90.14%** $\rightarrow$ CUMPLIDA

### **CLASIFICACIÓN: D33-B**
**Interpretación:** Positive but partial or unstable external generalization.

---

## 17. Implicaciones para FINAL TEST
- La partición de **FINAL TEST (2024–2025)** permaneció completamente cerrada durante toda la ejecución (`TEST_FILES_OPENED_COUNT = 0`).
- Recomendación metodológica: **HOLD FINAL TEST**.
- No se abrirá TEST hasta que se concluya formalmente la discusión de este dictamen.

---

## 18. Limitaciones
1. **Sensibilidad a regímenes de bajo gradiente:** En días con anomalías sub-mesoescala muy débiles, el término estacional introduce una leve sobre-corrección residual.
2. **Dependencia batimétrica estática:** Aunque la profundidad GEBCO mejora la delimitación costera, no captura dinámicas advectivas estacionales variables.
3. **Restricción de resolución OISST:** Las limitaciones inherentes a la interpolación bilineal de 0.25° persisten en áreas mar adentro sin estructura térmica pronunciada.

---

## 19. Catálogo de Entregables
- **Modelos:** `models/E3b-C0_PREVALIDATION.json`
- **Manifiesto:** `frozen_model_spec.json`
- **Celdas Congeladas:** `frozen_cell_ids.csv`
- **Tablas (11):**
  1. `tables/validation_summary.csv`
  2. `tables/yearly_metrics.csv`
  3. `tables/monthly_metrics.csv`
  4. `tables/daily_metrics.csv`
  5. `tables/spatial_metrics.csv`
  6. `tables/residual_regime_metrics.csv`
  7. `tables/overcorrection_validation.csv`
  8. `tables/bootstrap_confidence_intervals.csv`
  9. `tables/validation_skill_difference_vs_2021.csv`
  10. `tables/dataset_counts.csv`
  11. `tables/decision_criteria_D33.csv`
- **Figuras (8):**
  1. `figures/fig1_global_rmse_validation.png`
  2. `figures/fig2_skill_comparison_2021_vs_validation.png`
  3. `figures/fig3_monthly_rmse_improvement.png`
  4. `figures/fig4_daily_delta_rmse.png`
  5. `figures/fig5_spatial_delta_rmse_map.png`
  6. `figures/fig6_performance_by_regime.png`
  7. `figures/fig7_calibration_correction_behavior.png`
  8. `figures/fig8_bootstrap_delta_rmse.png`

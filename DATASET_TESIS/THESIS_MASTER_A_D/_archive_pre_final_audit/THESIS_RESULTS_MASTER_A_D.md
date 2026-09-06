# Master Results Synthesis (Stages A to D)
## Compilación Numérica y Narrativa de Evidencia Empírica para Tesis

**Documento:** `THESIS_RESULTS_MASTER_A_D.md`  
**Estado:** AUDITED & FROZEN  
**Regla Epistemológica:** Estricta fidelidad a las tablas maestras consolidadas (CSV-First Policy). Este documento reúne los hallazgos empíricos verificados desde la etapa de datos hasta la evaluación terminal de Machine Learning, empleando lenguaje estrictamente descriptivo sin especulaciones mecanicistas no demostradas.

---

## 1. Data Completeness and Acquisition Audits (Stage A)

- **Completitud temporal decenal:** La auditoría sobre el periodo 2015-01-01 a 2025-12-31 confirmó la disponibilidad de **4,018 días astronómicos continuos** en los productos MUR SST v4.1 y NOAA OISST v2.1. No se registró ningún día faltante ni archivo duplicado ($N_{\text{días}} = 4,018 / 4,018$, 100.0% de integridad temporal).
- **Consistencia física de unidades:** Tras la conversión de Kelvin a Celsius ($T_{\text{C}} = T_{\text{K}} - 273.15$), las temperaturas superficiales del mar de MUR se mantuvieron en rangos oceanográficamente realistas para el Caribe occidental (mínimo decenal: ~24.0 °C; máximo decenal: ~32.5 °C).
- **Resolución batimétrica:** El modelo GEBCO 2026 Grid proporcionó cobertura continua sobre el 100% de la cuadrícula objetivo sin celdas nulas en el dominio geográfico delimitado.

---

## 2. Spatial-Domain Construction and Ocean Masking (Stage B)

- **Censo de la cuadrícula maestra:** El mallado maestro de 0.01° (86 celdas en latitud × 96 celdas en longitud) arrojó un total de **8,256 celdas espaciales**.
- **Censo de máscara oceánica corregida:**
  - Máscara nativa MUR: 5,662 celdas de océano y 2,594 de tierra.
  - Aplicación de $\text{ocean\_fraction} \ge 0.5$ y corrección insular de Cozumel: **5,279 celdas oceánicas** y **2,977 celdas terrestres**.
  - Exactamente **383 celdas** fueron reclasificadas de agua a tierra por representar superficies continentales mixtas con fracción acuática inferior al 50%.

---

## 3. Coastal Interpolation Problem (Stage C.1)

- En la prueba piloto del 2015-01-01, la interpolación bilineal directa de OISST v2.1 con halo regular de 0.5° arrojó **1,292 celdas oceánicas con valor NaN**.
- Esta pérdida afectó al **24.47% del dominio marino** del corredor.
- El análisis geométrico confirmó que la pérdida se debió a que 20 nodos del halo occidental de OISST (0.25°) coinciden con la Península de Yucatán y están enmascarados como tierra en el producto nativo, imposibilitando el cierre de los cuadriláteros bilineales en la franja costera continental.

---

## 4. Coastal Strategy Comparison (Stage C.1b)

La comparación de métodos para recuperar la cobertura en 2015-01-01 arrojó:
- **Estrategia A (Soporte costero regular nearest-ocean):** 5,279 / 5,279 celdas válidas (100.0%). Baseline E0: $\text{RMSE} = 0.2367^\circ\text{C}$, $\text{MAE} = 0.2004^\circ\text{C}$, $\text{Bias} = +0.1872^\circ\text{C}$.
- **Estrategia B (Triangulación Delaunay 2D):** 5,279 / 5,279 celdas válidas (100.0%). Baseline E0: $\text{RMSE} = 0.2363^\circ\text{C}$, $\text{MAE} = 0.1996^\circ\text{C}$, $\text{Bias} = +0.1862^\circ\text{C}$.
- **Discrepancia numérica entre Estrategia A y B:**
  $$\text{MAE} = 0.0038^\circ\text{C}, \quad P_{95} = 0.0214^\circ\text{C}, \quad \text{Máxima} = 0.0626^\circ\text{C}$$
- La cuasi-identidad estadística justificó la selección de la Estrategia A para el pipeline decenal.

---

## 5. Full Harmonization Results (Stage C.2)

La armonización continua del periodo 2015–2025 generó el cubo de datos consolidado `faseC2_2015_2025.nc`:
- **Volumen observacional:** 4,018 días × 5,279 celdas = **21,211,022 puntos espaciotemporales**.
- **Métricas decenales del Baseline E0 (OISST bilineal vs MUR SST):**
  - **RMSE agrupado decenal (*Pooled Spatiotemporal RMSE*):** **$0.3426^\circ\text{C}$** ($0.342596^\circ\text{C}$)
  - **Promedio decenal de los RMSEs diarios (*Mean Daily RMSE*):** **$0.3019^\circ\text{C}$** ($0.301914^\circ\text{C}$)
  - **MAE decenal:** $0.2631^\circ\text{C}$ | **Bias decenal:** $+0.0133^\circ\text{C}$ | **$R^2$ decenal:** $0.9016$
- **Desglose de variabilidad anual:**
  - El promedio anual de los RMSEs diarios osciló entre un mínimo de $0.2724^\circ\text{C}$ (2018) y un máximo de $0.3257^\circ\text{C}$ (2024).
  - El RMSE agrupado anual osciló entre $0.3034^\circ\text{C}$ (2018) y $0.3811^\circ\text{C}$ (2015), situándose 2024 en $0.3797^\circ\text{C}$ y 2025 en $0.3334^\circ\text{C}$.

---

## 6. Satellite and Uncertainty Audits (Stage C.Sat & C.AE)

- **Auditoría satelital independiente (VIIRS + MODIS L2P):** En los 4 días críticos del evento anómalo de octubre 2015, MODIS Aqua registró 0 observaciones de calidad $QL=5$ (100% de píxeles rechazados por nubes). VIIRS registró observaciones limpias en solo 1 de 8 pasos orbitales (10 píxeles aislados, 0.1% del dominio). La evidencia radiométrica infrarroja fue formalmente clasificada como **inconclusa** por bloqueo nuboso generalizado, ratificando la decisión de no alterar el cubo C.2.
- **Auditoría de MUR Analysis Error (2015–2025):**
  - Se confirmó una correlación positiva moderada entre `analysis_error` y el RMSE diario de discrepancia MUR–BIL:
    $$\text{Spearman } \rho = +0.2853 \, (p = 4.25 \times 10^{-76}), \quad \text{Pearson } r = +0.3338 \, (p = 3.56 \times 10^{-105})$$
  - El 100% del dominio oceánico alcanzó el techo asintótico de $0.4100^\circ\text{C}$ durante eventos nubosos severos (E1 en 2015, E2 en 2021 y E3 en 2024).
  - El evento E4 (2015-08-03) actuó como contraejemplo: discrepancia severa ($\text{RMSE} = 1.102^\circ\text{C}$) con `analysis_error` nominal ($0.3908^\circ\text{C}$).

---

## 7. Residual Predictability Diagnostics (Stage D31)

- En Development (2015–2020), el residual $R$ mostró una autocorrelación temporal a 1 día de $r \approx 0.68$, decreciendo a $r \approx 0.35$ a 3 días.
- La correlación espacial de Spearman entre el residual y la profundidad marina ($depth$) en el canal de Cozumel reveló estructura sistemática persistente, supporting reproducible predictive structure beyond a null baseline.

---

## 8. Model-Selection Results (Stage D32)

Evaluación de las seis configuraciones candidatas sobre Diagnostic Holdout 2021 (5,275 celdas comunes, $1,925,375$ observaciones):
- **Baseline B0:** $\text{RMSE} = 0.359493^\circ\text{C}, \quad \text{MAE} = 0.277336^\circ\text{C}$
- **`E3b-C0` (Seleccionado):** $\text{RMSE} = 0.349274^\circ\text{C}$ (Mejora: **+2.84243%**, $\text{MAE} = 0.266264^\circ\text{C}$)
- **`E3b-T1`:** $\text{RMSE} = 0.350676^\circ\text{C}$ (Mejora: +2.45264%)
- **`E3b-T3`:** $\text{RMSE} = 0.350475^\circ\text{C}$ (Mejora: +2.50843%)
- **`E3b-S`:** $\text{RMSE} = 0.350802^\circ\text{C}$ (Mejora: +2.41756%)
- **`E3b-TS`:** $\text{RMSE} = 0.351551^\circ\text{C}$ (Mejora: +2.20904%)
- **`E3b-ALL`:** $\text{RMSE} = 0.350048^\circ\text{C}$ (Mejora: +2.62712%)
- `E3b-C0` superó a todas las extensiones temporales y espaciales complejas, siendo congelado formalmente.

---

## 9. External Validation Results (Stage D33)

Evaluación de la especificación congelada de `E3b-C0` (reajustada en 2015–2021) en el periodo independiente 2022–2023 ($3,850,750$ observaciones):
- **Baseline B0:** $\text{RMSE} = 0.335666^\circ\text{C}, \quad \text{MAE} = 0.263594^\circ\text{C}$
- **Modelo C0:** $\text{RMSE} = 0.323838^\circ\text{C}, \quad \text{MAE} = 0.254457^\circ\text{C}$
- **Mejora global:** **+3.5237%** de reducción en RMSE ($\Delta\text{RMSE} = -0.011828^\circ\text{C}$).
- **Desempeño temporal:** **16 de 24 meses mejorados** (66.7%).
- **Desempeño espacial:** **4,755 de 5,275 celdas mejoradas** (**90.14%**).
- **Dictamen formal:** **D33-B — Partial/Mixed Generalization** (mejora global y espacial sólida, pero estabilidad mensual sub-umbral de 18 meses).

---

## 10. Diagnostic Findings (Stage D34)

- **Compresión de amplitud:** El modelo `E3b-C0` predice un residual con desviación estándar reducida frente a la observada: $\text{std}(\hat{R}) / \text{std}(R) \approx 0.25$, consistente con regresión hacia la media condicional bajo formulación regularizada de MSE.
- **Comportamiento por regímenes de discrepancia (Umbrales canónicos congelados):**
  - DEV-P50 = $0.2066^\circ\text{C}$
  - DEV-P75 = $0.3604^\circ\text{C}$
  - DEV-P90 = $0.5377^\circ\text{C}$
  - DEV-P95 = $0.6652^\circ\text{C}$
  - DEV-P99 = $0.9659^\circ\text{C}$
  Cuando la discrepancia original entre MUR y OISST es pequeña ($|R| < 0.2066^\circ\text{C}$, percentiles 0–50 de Development), el modelo degrada ligeramente el error cuadrático ($\sim -20\%$). En cambio, en discrepancias mayores (P75–P99+), el modelo alcanza mejoras de RMSE de +7% a +12%.

---

## 11. Final-Test Performance (Stage D35)

Evaluación en el conjunto ciego Final Test 2024–2025 ($3,856,025$ observaciones, 731 días):
- **Métricas primarias consolidadas:**
  - **Baseline B0 RMSE:** $0.357317^\circ\text{C}$
  - **Modelo C0 RMSE:** $0.331502^\circ\text{C}$
  - **Mejora relativa global de RMSE:** **+7.2247%** ($\Delta\text{RMSE} = -0.025815^\circ\text{C}$)
  - **MAE:** Reducción de $0.272713^\circ\text{C}$ a $0.256325^\circ\text{C}$ (+6.01% de mejora)
  - **Sesgo medio (*Bias*):** Reducción de $-0.062425^\circ\text{C}$ a $-0.014197^\circ\text{C}$ (reducción del sesgo en 77.3%)
  - **$R^2$ de SST reconstruida:** Aumento de $0.892661$ (B0) a **$0.907610$** (C0)
- **Desglose anual:**
  - Año 2024: B0 RMSE = $0.379708^\circ\text{C}$ -> C0 RMSE = $0.344023^\circ\text{C}$ (**+9.3986%**)
  - Año 2025: B0 RMSE = $0.333355^\circ\text{C}$ -> C0 RMSE = $0.318453^\circ\text{C}$ (**+4.4704%**)

---

## 12. Temporal Robustness (Stage D35)

- **Meses mejorados:** **18 de 24 meses** del bienio registraron $\Delta\text{RMSE} < 0$ (**75.0%** de estabilidad mensual).
  - Año 2024: **9 de 12 meses mejorados** (75.0%). Meses negativos: 2024-01, 2024-04, 2024-10.
  - Año 2025: **9 de 12 meses mejorados** (75.0%). Meses negativos: 2025-09, 2025-10, 2025-12.
- **Días mejorados:** **484 de 731 días** astronómicos evaluados arrojaron mejora (desempeño favorable en el **66.21%** de las fechas).
- **Inferencia estadística por bootstrap en bloques de 14 días (1,000 réplicas):**
  - Mediana de $\Delta\text{RMSE}$: $-0.025395^\circ\text{C}$
  - **Intervalo de confianza al 95% ($CI_{95}$):** **$[-0.042485, -0.009792]^\circ\text{C}$**
  - The 95% bootstrap interval remained entirely below zero, supporting the robustness of the aggregate RMSE improvement to short-range temporal dependence under the evaluated block length.

---

## 13. Spatial Robustness (Stage D35)

- **Amplitud espacial de la mejora:** **5,273 de las 5,275 celdas evaluadas** registraron reducción del RMSE cuadrático medio decenal (**99.96%** de cobertura espacial con beneficio).
- **Gradiente con profundidad batimétrica:**
  - 0–20 m (965 celdas): $\Delta\text{RMSE}$ medio = $-0.038311^\circ\text{C}$ (100.0% celdas mejoradas)
  - 20–50 m (417 celdas): $\Delta\text{RMSE}$ medio = $-0.032650^\circ\text{C}$ (100.0% celdas mejoradas)
  - 50–100 m (232 celdas): $\Delta\text{RMSE}$ medio = $-0.030423^\circ\text{C}$ (100.0% celdas mejoradas)
  - 100–500 m (1,363 celdas): $\Delta\text{RMSE}$ medio = $-0.030608^\circ\text{C}$ (100.0% celdas mejoradas)
  - >500 m (2,298 celdas): $\Delta\text{RMSE}$ medio = $-0.014452^\circ\text{C}$ (99.91% celdas mejoradas)
- **Correlaciones espaciales de Spearman:**
  $$\text{Spearman}(depth, \Delta\text{RMSE}) = +0.7376 \quad (p < 10^{-15})$$
  $$\text{Spearman}(distance\_coast\_km, \Delta\text{RMSE}) = +0.5336 \quad (p < 10^{-15})$$
  The positive rank associations indicate that $\Delta\text{RMSE}$ became less negative with increasing depth and offshore distance, corresponding to a smaller magnitude of improvement.

---

## 14. Residual-Regime Dependence (Stage D35)

Estratificación del desempeño en Final Test según los percentiles canónicos DEV de discrepancia $|R|$:
- **DEV-P0–P50** ($|R| < 0.2066^\circ\text{C}$, 48.16% de test): B0 = $0.1154^\circ\text{C}$, C0 = $0.1396^\circ\text{C}$ (**-20.99%** de degradación).
- **DEV-P50–P75** ($0.2066 \le |R| < 0.3604^\circ\text{C}$, 24.35% de test): B0 = $0.2808^\circ\text{C}$, C0 = $0.2710^\circ\text{C}$ (**+3.51%** de mejora).
- **DEV-P75–P90** ($0.3604 \le |R| < 0.5377^\circ\text{C}$, 15.68% de test): B0 = $0.4400^\circ\text{C}$, C0 = $0.4071^\circ\text{C}$ (**+7.48%** de mejora).
- **DEV-P90–P95** ($0.5377 \le |R| < 0.6652^\circ\text{C}$, 5.29% de test): B0 = $0.5961^\circ\text{C}$, C0 = $0.5412^\circ\text{C}$ (**+9.21%** de mejora).
- **DEV-P95–P99** ($0.6652 \le |R| < 0.9659^\circ\text{C}$, 4.71% de test): B0 = $0.7877^\circ\text{C}$, C0 = $0.6926^\circ\text{C}$ (**+12.07%** de mejora).
- **DEV-P99+** ($|R| \ge 0.9659^\circ\text{C}$, 1.80% de test): B0 = $1.1469^\circ\text{C}$, C0 = $1.0151^\circ\text{C}$ (**+11.49%** de mejora).
- **Exactitud de signo por régimen:** Crece monotónicamente con la magnitud del residual:
  $$56.4\% \, (\text{P0–P50}) \to 65.8\% \, (\text{P50–P75}) \to 70.9\% \, (\text{P75–P90}) \to 75.7\% \, (\text{P90–P95}) \to 82.5\% \, (\text{P95–P99}) \to 87.2\% \, (\text{P99+})$$

---

## 15. Final Scientific Result

1. El modelo parsimonioso `E3b-C0` provided a reproducible reduction in SST reconstruction error relative to bilinear interpolation over the Tulum–Cozumel marine corridor in an independent, previously withheld out-of-sample period (2024–2025).
2. Se confirma formalmente el dictamen: **D35-A: FINAL GENERALIZATION CONFIRMED**.
3. El conjunto de prueba queda registrado como **TEST CONSUMED** y el desarrollo metodológico de modelos de Machine Learning queda formalmente **CERRADO**.

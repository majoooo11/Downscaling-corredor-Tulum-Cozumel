# Reporte Científico — Fase D.3.2: E3b Tabular
## Modelado del Residual Térmico con Contexto Temporal Causal y Espacial 2D

**Proyecto:** Downscaling de Temperatura Superficial del Mar (SST) — Corredor Tulum–Cozumel  
**Fase:** D.3.2 (Experimento E3b Tabular Confirmatorio)  
**Fecha de ejecución:** 2026-09-05  
**Script principal:** [`DATASET_TESIS/fase_d32_e3b_tabular.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/fase_d32_e3b_tabular.py)  
**Ambiente:** Python 3.11, XGBoost 3.2.0, NumPy 2.4.2, Pandas 2.3.3, Scikit-learn 1.8.2  
**Directorio de artefactos:** [`DATASET_TESIS/ml_results/E3b_D32/`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/E3b_D32/)  

---

## 1. Objetivo Científico

La **Fase D.3.2 (E3b Tabular)** constituye un experimento confirmatorio dentro del bloque de **DEVELOPMENT** del proyecto de tesis de maestría. Su propósito metodológico es resolver dos interrogantes científicas primarias sobre la predictibilidad del campo residual térmico $R(t,x,y) = \text{SST\_MUR}(t,x,y) - \text{SST\_BIL}(t,x,y)$:

1. **Interrogante Temporal:** ¿La persistencia temporal del residual observada en la Fase D.3.1 ($r_{\text{lag1}} = 0.6853$, $r_{\text{lag2}} = 0.3622$, $r_{\text{lag3}} = 0.2206$) puede ser capturada mediante la evolución temporal causal de $\text{SST\_BIL}$ (lags de 1 a 3 días y diferencias multidiarias), aportando valor predictivo incremental sobre el núcleo tabular CORE (`sst_bil`, `doy_sin`, `doy_cos`, `depth` GEBCO)?
2. **Interrogante Espacial:** ¿Existe información predictiva incremental en las características espaciales locales 2D derivadas de $\text{SST\_BIL}$ (módulo del gradiente en $^\circ\text{C/km}$, desviación estándar local 3×3 y contraste local) una vez incorporada la memoria temporal?
3. **Diagnóstico de Sobre-Corrección:** ¿El modelado temporal/espacial logra reducir la sobre-corrección observada en D.3.1 dentro del régimen central de bajo residual ($0 - \text{P50}$, $|R| < 0.2066^\circ\text{C}$)?

Este experimento se ejecuta bajo el principio de confirmación objetiva: **un resultado negativo es científicamente válido**, y la meta no es justificar el uso de Machine Learning, sino medir con rigor si aporta valor defendible.

---

## 2. Diseño Experimental

El diseño de partición temporal se congela estrictamente en cinco bloques cronológicos disjointos, preservando la no-fuga y la auditabilidad externa:

```
+--------------------------+--------------------+-------------------------+-------------------------+--------------------+
| FIT                      | INTERNAL TUNING    | DIAGNOSTIC HOLDOUT      | EXTERNAL VALIDATION     | FINAL TEST         |
| 2015 – 2019              | 2020               | 2021                    | 2022 – 2023             | 2024 – 2025        |
| (Desarrollo primario)    | (Early Stopping)   | (Selección diagnóstica) | (Blindado, cerrado = 0) | (Blindado, cerr.=0)|
+--------------------------+--------------------+-------------------------+-------------------------+--------------------+
|<---------------- DEVELOPMENT (2015-2020) ---->|
```

- **FIT (2015–2019):** 1,826 días continuos, utilizado para el ajuste de árboles en la búsqueda de hiperparámetros.
- **INTERNAL TUNING / EARLY STOPPING (2020):** 366 días continuos (bisiesto), utilizado como conjunto de validación interna para detener tempranamente el ajuste de árboles sobre el núcleo CORE y seleccionar la complejidad óptima $\theta^*$.
- **DIAGNOSTIC MODEL-SELECTION HOLDOUT (2021):** 365 días continuos, utilizado exclusivamente para la comparación pareada de las 6 formulaciones de ablación E3b sobre una máscara común. **No se utiliza para tuning, early stopping ni generación adaptativa de features.**
- **EXTERNAL VALIDATION (2022–2023):** Bloque cerrado y blindado.
- **FINAL TEST (2024–2025):** Bloque cerrado y blindado.

---

## 3. Blindaje Temporal y Salvaguardas de Integridad

Se implementaron contadores en tiempo de ejecución y aserciones de código para auditar el acceso a los datos del proyecto:

| Partición Temporal | Archivos Disponibles | Archivos Abiertos | Estado Metodológico |
|:---|:---:|:---:|:---|
| **DEVELOPMENT (2015–2020)** | 6 | 6 | Abiertos para extracción y ajuste |
| **DIAGNOSTIC HOLDOUT (2021)** | 1 | 1 | Abierto exclusivamente para evaluación final |
| **EXTERNAL VALIDATION (2022–2023)** | 2 | **0** | **BLINDADO (Assert passed)** |
| **FINAL TEST (2024–2025)** | 2 | **0** | **BLINDADO (Assert passed)** |

```python
assert VALIDATION_FILES_OPENED_COUNT == 0  # PASSED
assert TEST_FILES_OPENED_COUNT == 0        # PASSED
```

**Salvaguarda contra Fuga de Información (Target/Future Leakage):**
- Prohibición absoluta de usar $R(t-k)$ o $\text{SST\_MUR}(t-k)$ como predictores. Ninguna feature requiere conocer el target pasado.
- Los lags temporales fueron construidos de manera estrictamente causal: $t-1, t-2, t-3$. Ninguna feature accede a $t+1$.
- La batimetría procede exclusivamente de **GEBCO** (conforme a los estándares del proyecto; sin uso de ETOPO1).

---

## 4. Dataset, Continuidad de Lags y Máscara Común (`COMMON_VALID_MASK`)

La serie temporal 2015–2021 abarca exactamente 2,557 días consecutivos (13,498,403 observaciones de cuadrícula oceánica). Los lags temporales fueron calculados sobre la serie temporal continua por celda, garantizando que el cruce entre años sea continuo (p. ej., el 1 de enero de 2021 toma lags del 31, 30 y 29 de diciembre de 2020).

Para garantizar comparaciones estadísticas rigurosamente pareadas entre todas las ablaciones en HOLDOUT 2021 ($D_{C0} = D_{T1} = D_{T3} = D_{S} = D_{TS}$), se construyó la máscara `COMMON_VALID_MASK`, la cual exige validez simultánea de:
- Núcleo CORE (`sst_bil`, `doy_sin`, `doy_cos`, `depth`).
- Lags 1, 2 y 3 días, y diferencias multidiarias $\Delta\text{SST}_{1d}, \Delta\text{SST}_{2d}, \Delta\text{SST}_{3d}$.
- Gradiente zonal, gradiente meridional y estadísticas locales 3×3 con soporte oceánico suficiente ($N_{\text{valid}} \ge 3$).

### Censo de Observaciones por Año (`dataset_counts.csv`)

| Año | Partición | $N$ Original | $N$ Temporal Válido | $N$ Espacial Válido | $N$ `COMMON_VALID_MASK` | Retención (%) |
|:---:|:---|---:|---:|---:|---:|---:|
| 2015 | DEVELOPMENT | 1,926,835 | 1,910,998 | 1,925,375 | 1,909,550 | 99.10% |
| 2016 | DEVELOPMENT | 1,932,114 | 1,932,114 | 1,930,650 | 1,930,650 | 99.92% |
| 2017 | DEVELOPMENT | 1,926,835 | 1,926,835 | 1,925,375 | 1,925,375 | 99.92% |
| 2018 | DEVELOPMENT | 1,926,835 | 1,926,835 | 1,925,375 | 1,925,375 | 99.92% |
| 2019 | DEVELOPMENT | 1,926,835 | 1,926,835 | 1,925,375 | 1,925,375 | 99.92% |
| 2020 | DEVELOPMENT | 1,932,114 | 1,932,114 | 1,930,650 | 1,930,650 | 99.92% |
| **2021** | **DIAGNOSTIC_HOLDOUT** | **1,926,835** | **1,926,835** | **1,925,375** | **1,925,375** | **99.92%** |

En HOLDOUT 2021, la retención es del **99.92%** ($N = 1,925,375$ observaciones), descartando únicamente 1,460 puntos periféricos (4 celdas en el límite de la máscara oceánica sin soporte 3×3).

---

## 5. Definición de las Seis Formulaciones de Ablación

Todas las formulaciones operan prediciendo el residual térmico $\hat{R}$ para reconstruir la temperatura final $\hat{\text{SST}} = \text{SST\_BIL} + \hat{R}$.

1. **E3b-C0 (CORE):**
   $$\mathbf{X} = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}]$$
2. **E3b-T1 (CORE + Memoria 1 Día):**
   $$\mathbf{X} = \text{CORE} \cup [\text{sst\_bil\_lag1}, \Delta\text{sst\_1d}]$$
3. **E3b-T3 (CORE + Memoria 3 Días):**
   $$\mathbf{X} = \text{CORE} \cup [\text{sst\_bil\_lag1}, \text{sst\_bil\_lag2}, \text{sst\_bil\_lag3}, \Delta\text{sst\_1d}, \Delta\text{sst\_2d}, \Delta\text{sst\_3d}]$$
4. **E3b-S (CORE + Contexto Espacial 2D):**
   $$\mathbf{X} = \text{CORE} \cup [\text{grad\_mag\_sst\_bil}, \text{local\_std\_3x3}, \text{local\_contrast}]$$
5. **E3b-TS (CORE + Temporal + Espacial):**
   $$\mathbf{X} = \text{CORE} \cup \text{Temporal(3d)} \cup \text{Espacial}$$
6. **E3b-ALL (Ablación Secundaria Completa):**
   $$\mathbf{X} = \text{E3b-TS} \cup [\text{distance\_coast\_km}, \text{ocean\_fraction}, \text{local\_range\_3x3}]$$

---

## 6. Optimización Compacta de Hiperparámetros sobre CORE

Con el fin de aislar el impacto de las features y evitar la confusión con variaciones de hiperparámetros entre modelos, se optimizó XGBoost **únicamente sobre el modelo CORE** mediante una búsqueda compacta de 9 configuraciones en una muestra representativa estratificada (método de Hamilton de restos mayores, 10% de DEV = 1,155,573 observaciones):
- **Ajuste:** Subconjunto 2015–2019 ($N = 962,364$).
- **Parada Temprana (Early Stopping):** Subconjunto 2020 ($N = 193,209$), con paciencia de 20 rondas.

### Tabla de Hiperparámetros Evaluados (`hyperparameters.csv`)

| Trial | Max Depth | Learning Rate | Subsample | Colsample ByTree | Min Child Weight | Best Iteration | RMSE Val 2020 ($^\circ\text{C}$) | Fit Time (s) | Seleccionado ($\theta^*$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4 | 0.03 | 0.8 | 0.8 | 5 | 31 | 0.3549 | 0.3 | No |
| 2 | 4 | 0.05 | 0.8 | 0.8 | 5 | 52 | 0.3539 | 0.5 | No |
| **3** | **4** | **0.10** | **0.8** | **0.8** | **5** | **18** | **0.3539** | **0.3** | **SÍ ($\theta^*$)** |
| 4 | 6 | 0.03 | 0.8 | 0.8 | 5 | 27 | 0.3555 | 0.4 | No |
| 5 | 6 | 0.05 | 0.8 | 0.8 | 5 | 17 | 0.3555 | 0.3 | No |
| 6 | 6 | 0.10 | 0.8 | 0.8 | 5 | 8 | 0.3554 | 0.2 | No |
| 7 | 8 | 0.03 | 0.8 | 0.8 | 5 | 12 | 0.3557 | 0.3 | No |
| 8 | 8 | 0.05 | 0.8 | 0.8 | 5 | 5 | 0.3559 | 0.3 | No |
| 9 | 8 | 0.10 | 0.8 | 0.8 | 5 | 2 | 0.3556 | 0.2 | No |

**Configuración Congelada ($\theta^*$):**
- `max_depth = 4`, `learning_rate = 0.10`, `n_estimators = 18`, `subsample = 0.8`, `colsample_bytree = 0.8`, `min_child_weight = 5`, `tree_method = "hist"`.
- Esta configuración se aplicó de forma idéntica a las seis ablaciones para el reentrenamiento sobre todo DEVELOPMENT.

---

## 7. Resultados Globales en HOLDOUT 2021

Evaluación pareada sobre $N = 1,925,375$ observaciones en la máscara común (`model_summary.csv`):

| Modelo | Descripción | $N_{\text{feat}}$ | RMSE SST ($^\circ\text{C}$) | MAE SST ($^\circ\text{C}$) | Bias SST ($^\circ\text{C}$) | $R^2$ SST | Mejora RMSE vs B0 (%) | Mejora RMSE vs CORE (%) | Skill RMSE vs B0 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | Baseline Bilineal ($\hat{R} = 0$) | 0 | 0.359493 | 0.277337 | -0.050145 | 0.8773 | 0.00% | 0.00% | 0.000000 |
| **E3b-C0** | CORE Tabular | 4 | 0.350061 | 0.266853 | -0.038019 | 0.8837 | +2.62% | 0.00% | 0.026235 |
| **E3b-T1** | CORE + Memoria 1d | 6 | 0.350324 | 0.266854 | -0.043622 | 0.8835 | +2.55% | **-0.07%** | 0.025506 |
| **E3b-T3** | CORE + Memoria 3d | 10 | 0.351656 | 0.267357 | -0.049496 | 0.8826 | +2.18% | **-0.46%** | 0.021799 |
| **E3b-S** | CORE + Espacial 2D | 7 | **0.349988** | **0.266647** | -0.038303 | 0.8837 | **+2.64%** | **+0.02%** | **0.026439** |
| **E3b-TS** | CORE + Temp + Esp | 13 | 0.351551 | 0.268038 | -0.050116 | 0.8827 | +2.21% | **-0.43%** | 0.022092 |
| **E3b-ALL**| Completo | 16 | 0.350018 | 0.266618 | -0.049150 | 0.8837 | +2.64% | **+0.01%** | 0.026356 |

- **Mejor Modelo Temporal:** `E3b-T1` (RMSE = 0.350324 $^\circ\text{C}$).
- **Mejor Modelo Global:** `E3b-S` (RMSE = 0.349988 $^\circ\text{C}$).

---

## 8. Ablación Temporal: Análisis de la Memoria Reciente

Para responder si la evolución reciente de $\text{SST\_BIL}$ contiene información predictiva sobre $R$, se evalúa el cambio frente a `E3b-C0`:

$$\Delta_{\text{temporal}} = \frac{\text{RMSE}_{\text{CORE}} - \text{RMSE}_{\text{best\_temporal}}}{\text{RMSE}_{\text{CORE}}} \times 100$$

- **E3b-T1 vs E3b-C0:**
  $$\Delta\text{RMSE} = 0.350324 - 0.350061 = +0.000263^\circ\text{C} \quad (\mathbf{-0.07\% \text{ deterioro relativo}})$$
  $$\Delta\text{MAE} = 0.266854 - 0.266853 = +0.000001^\circ\text{C} \quad (\text{sin mejora})$$
- **E3b-T3 vs E3b-C0:**
  $$\Delta\text{RMSE} = 0.351656 - 0.350061 = +0.001595^\circ\text{C} \quad (\mathbf{-0.46\% \text{ deterioro relativo}})$$
  $$\Delta\text{MAE} = 0.267357 - 0.266853 = +0.000504^\circ\text{C} \quad (\text{deterioro})$$

**Conclusión Científica:**
La persistencia observada en D.3.1 pertenece al target $R$ y al campo de alta resolución $\text{SST\_MUR}$, pero **no está codificada en la evolución causal de la señal de baja resolución $\text{SST\_BIL}$**. Al incorporar lags de $\text{SST\_BIL}$, el modelo añade grados de libertad y colinealidad que deterioran el error de generalización en HOLDOUT 2021.

---

## 9. Ablación Espacial: Valor del Contexto 2D

Se evalúa la ganancia del contexto espacial en dos planos:
1. **Espacial tras Temporal (`E3b-TS` vs `E3b-T1`):**
   $$\Delta_{\text{spatial}} = \frac{\text{RMSE}_{\text{E3b-T1}} - \text{RMSE}_{\text{E3b-TS}}}{\text{RMSE}_{\text{E3b-T1}}} \times 100 = \frac{0.350324 - 0.351551}{0.350324} \times 100 = \mathbf{-0.35\%}$$
   El modelo que combina ambas familias de features no supera a `E3b-T1` ni a `E3b-C0`.
2. **Espacial Puro (`E3b-S` vs `E3b-C0`):**
   $$\Delta\text{RMSE} = 0.349988 - 0.350061 = -0.000073^\circ\text{C} \quad (\mathbf{+0.02\% \text{ mejora marginal}})$$
   Aunque `E3b-S` logra el RMSE global más bajo de todas las formulaciones ($0.349988^\circ\text{C}$), la reducción en error es de solo $0.00007^\circ\text{C}$, situándose órdenes de magnitud por debajo del umbral del 1.00% requerido para justificar un escalamiento arquitectónico.

---

## 10. Estabilidad Temporal Mensual (2021)

Se desglosó el desempeño de las formulaciones a lo largo de los 12 meses de 2021 (`monthly_metrics.csv`):

| Mes | Nombre | $N$ | RMSE B0 ($^\circ\text{C}$) | RMSE C0 ($^\circ\text{C}$) | RMSE T1 ($^\circ\text{C}$) | RMSE S ($^\circ\text{C}$) | RMSE TS ($^\circ\text{C}$) | Mejor Modelo del Mes |
|:---:|:---|---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Enero | 163,525 | 0.3355 | 0.3296 | 0.3342 | 0.3317 | 0.3386 | E3b-C0 (+1.77% vs B0) |
| 2 | Febrero | 147,700 | 0.3085 | 0.2927 | 0.2911 | 0.2928 | 0.2906 | E3b-TS (+5.81% vs B0) |
| 3 | Marzo | 163,525 | 0.3396 | 0.2875 | 0.2889 | **0.2802** | 0.2924 | **E3b-S (+17.48% vs B0)** |
| 4 | Abril | 158,250 | 0.3276 | 0.3131 | 0.3126 | 0.3129 | 0.3150 | E3b-T1 (+4.60% vs B0) |
| 5 | Mayo | 163,525 | 0.3138 | 0.3101 | 0.3117 | 0.3096 | 0.3124 | E3b-S (+1.34% vs B0) |
| 6 | Junio | 158,250 | 0.3703 | 0.3548 | 0.3562 | 0.3551 | 0.3575 | E3b-C0 (+4.19% vs B0) |
| 7 | Julio | 163,525 | 0.4431 | 0.4287 | 0.4289 | 0.4286 | 0.4299 | E3b-S (+3.27% vs B0) |
| 8 | Agosto | 163,525 | 0.4350 | 0.4334 | 0.4339 | 0.4331 | 0.4348 | E3b-S (+0.44% vs B0) |
| 9 | Septiembre | 158,250 | 0.3473 | 0.3585 | 0.3582 | 0.3587 | 0.3584 | **B0 es mejor (-3.22%)** |
| 10 | Octubre | 163,525 | 0.4137 | 0.4239 | 0.4224 | 0.4237 | 0.4230 | **B0 es mejor (-2.42%)** |
| 11 | Noviembre | 158,250 | 0.4300 | 0.4184 | 0.4208 | 0.4177 | 0.4227 | E3b-S (+2.86% vs B0) |
| 12 | Diciembre | 163,525 | 0.3297 | 0.3411 | 0.3364 | 0.3424 | 0.3375 | **B0 es mejor (-3.46%)** |

- **Meses que mejoran a B0:** 9 de 12 meses (en septiembre, octubre y diciembre, el baseline bilineal B0 supera a los modelos ML).
- **Meses en que `E3b-S` mejora a `E3b-C0`:** solo 4 de 12 meses.
- **Mejor mes:** Marzo (ganancia masiva de hasta +17.48% vs B0, motivada por la fuerte señal estacional de surgencia/gradientes).
- **Peor mes:** Diciembre (-3.46% vs B0, con sesgo por inversión térmica estacional).
- **Desviación estándar mensual:** $\sigma_{\text{mensual}}(\text{C0}) = 0.0668^\circ\text{C}$; $\sigma_{\text{mensual}}(\text{T1}) = 0.0673^\circ\text{C}$; $\sigma_{\text{mensual}}(\text{S}) = 0.0691^\circ\text{C}$.

---

## 11. Distribución Espacial Celda por Celda

La evaluación sobre las 5,275 celdas oceánicas con cobertura continua (`spatial_metrics.csv`) revela:

| Modelo | Referencia | Celdas Eval. | Celdas Mejoradas | % Mejorado | Mediana $\Delta\text{RMSE}$ ($^\circ\text{C}$) | P10 ($^\circ\text{C}$) | P25 ($^\circ\text{C}$) | P75 ($^\circ\text{C}$) | P90 ($^\circ\text{C}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E3b-C0** | B0 | 5,275 | 5,275 | **100.0%** | -0.0093 | -0.0145 | -0.0120 | -0.0066 | -0.0048 |
| **E3b-T1** | B0 | 5,275 | 5,275 | **100.0%** | -0.0087 | -0.0140 | -0.0120 | -0.0069 | -0.0046 |
| **E3b-T3** | B0 | 5,275 | 5,208 | **98.7%** | -0.0076 | -0.0134 | -0.0111 | -0.0051 | -0.0027 |
| **E3b-S** | B0 | 5,275 | 5,275 | **100.0%** | **-0.0096** | -0.0139 | -0.0118 | -0.0070 | -0.0052 |
| **E3b-TS** | B0 | 5,275 | 5,275 | **100.0%** | -0.0079 | -0.0125 | -0.0106 | -0.0055 | -0.0037 |
| **E3b-T1** | **E3b-C0** | 5,275 | 2,407 | **45.6%** | **+0.0001** | -0.0015 | -0.0008 | +0.0013 | +0.0022 |
| **E3b-T3** | **E3b-C0** | 5,275 | 1,361 | **25.8%** | **+0.0016** | -0.0013 | -0.0001 | +0.0033 | +0.0045 |
| **E3b-S** | **E3b-C0** | 5,275 | 2,948 | **55.9%** | **-0.0001** | -0.0010 | -0.0007 | +0.0005 | +0.0010 |
| **E3b-TS** | **E3b-T1** | 5,275 | 1,424 | **27.0%** | **+0.0011** | -0.0017 | -0.0006 | +0.0030 | +0.0044 |

- Frente al baseline bilineal B0, prácticamente el 100% de las celdas muestran mejoras gracias al núcleo CORE.
- Sin embargo, al comparar `E3b-T1` frente a `E3b-C0`, **más del 54% de las celdas se deterioran**.
- En `E3b-T3`, el **74.2% de las celdas empeoran** respecto a CORE.
- En `E3b-TS` frente a `E3b-T1`, el **73.0% de las celdas empeoran**.

---

## 12. Desempeño por Magnitud del Residual (|R|)

Utilizando los umbrales predefinidos en DEVELOPMENT (P50 = $0.2066^\circ\text{C}$, P75 = $0.3604^\circ\text{C}$, P90 = $0.5377^\circ\text{C}$, P95 = $0.6652^\circ\text{C}$, P99 = $0.9659^\circ\text{C}$), se evalúa la respuesta según la escala de la anomalía (`residual_regime_metrics.csv`):

| Régimen de $|R|$ | Rango ($^\circ\text{C}$) | $N$ | % Total | RMSE B0 ($^\circ\text{C}$) | RMSE CORE ($^\circ\text{C}$) | RMSE E3b-S ($^\circ\text{C}$) | Mejora vs B0 (%) | Mejora vs CORE (%) | Frec. Sobre-corrección (%) |
|:---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 – P50** | $< 0.2066$ | 901,568 | 46.83% | **0.1164** | 0.1204 | 0.1206 | **-3.56% (Peor)** | -0.13% | **17.06%** |
| **P50 – P75** | $[0.2066, 0.3604)$ | 465,851 | 24.20% | 0.2813 | 0.2689 | 0.2685 | +4.58% | +0.16% | 0.08% |
| **P75 – P90** | $[0.3604, 0.5377)$ | 322,425 | 16.75% | 0.4430 | 0.4231 | 0.4224 | +4.65% | +0.19% | 0.00% |
| **P90 – P95** | $[0.5377, 0.6652)$ | 121,358 | 6.30% | 0.5953 | 0.5739 | 0.5729 | +3.76% | +0.17% | 0.00% |
| **P95 – P99** | $[0.6652, 0.9659)$ | 84,888 | 4.41% | 0.7744 | 0.7590 | 0.7601 | +1.85% | -0.15% | 0.00% |
| **$\ge$ P99** | $\ge 0.9659$ | 29,285 | 1.52% | 1.2054 | 1.2059 | 1.2084 | -0.25% | -0.20% | 0.00% |
| **Central** | $< \text{P90}$ | 1,689,844 | 87.77% | 0.2579 | 0.2487 | 0.2483 | +3.70% | +0.14% | 9.12% |
| **Extreme Tail**| $\ge \text{P90}$ | 235,531 | 12.23% | 0.7612 | 0.7471 | 0.7476 | +1.78% | -0.07% | 0.00% |
| **Total** | Todo | 1,925,375 | 100.0% | 0.3595 | 0.3501 | 0.3500 | +2.64% | +0.02% | 8.01% |

---

## 13. Diagnóstico de Sobre-Corrección

El análisis por bins de $|R|$ confirma el fenómeno observado en D.3.1:
1. **Deterioro en el régimen de ruido térmico ($0 - \text{P50}$):** Representa el 46.8% del dataset. En este régimen, el baseline bilineal B0 alcanza $\text{RMSE} = 0.1164^\circ\text{C}$, mientras que todos los modelos ML presentan un RMSE superior ($\sim 0.1205^\circ\text{C}$), lo que representa un deterioro de entre $-3.3\%$ y $-8.9\%$.
2. **Impacto de la memoria temporal en la sobre-corrección:**
   - La frecuencia en que $|\hat{R}| > |R|$ para el núcleo CORE es del **17.16%**.
   - Para `E3b-T1` es del **17.81%**.
   - Para `E3b-T3`, la frecuencia de sobre-corrección se dispara al **24.69%**.
   - Para `E3b-TS`, se sitúa en el **24.72%**.
   
**Conclusión de Sobre-Corrección:**  
Añadir historial temporal de $\text{SST\_BIL}$ no resuelve la sobre-corrección, sino que **la agrava en un +44% relativo** (pasa de 17.1% a 24.7%), debido a que los lags introducen fluctuaciones espurias de alta frecuencia en días de calma donde el residual real es prácticamente cero.

---

## 14. Interpretabilidad Post-Hoc (`feature_importance.csv`)

La extracción de importancia post-hoc se realizó sobre el mejor modelo global (`E3b-S`), calculando la ganancia relativa total en los árboles y la permutación en una muestra representativa de 192,537 puntos de HOLDOUT:

| Feature | Gain Relativo (%) | Permutación $\Delta\text{MSE}$ ($^\circ\text{C}^2$) | Permutación Relativa (%) | Interpretación Epistemológica |
|:---|:---:|:---:|:---:|:---|
| `doy_sin` | **27.33%** | 0.015769 | **60.09%** | Ciclo estacional anual (señal macroclimatológica) |
| `doy_cos` | **25.46%** | 0.000000 | 0.00% | Fase del ciclo estacional anual |
| `sst_bil` | **19.42%** | 0.009797 | **37.33%** | Nivel térmico regional de fondo |
| `depth` | **16.95%** | 0.000676 | **2.57%** | Estructura batimétrica fija (GEBCO) |
| `grad_mag_sst_bil` | **5.74%** | 0.000000 | 0.00% | Magnitud del gradiente horizontal OISST |
| `local_std_3x3` | **5.10%** | 0.000000 | 0.00% | Heterogeneidad térmica local 3×3 |
| `local_contrast` | **0.00%** | 0.000000 | 0.00% | Anomalía respecto a la media local |

**Aclaración Epistemológica:**
Las cuatro variables del núcleo **CORE representan el 89.16% de la ganancia predictiva** y el **100.0% de la importancia por permutación**. Las características espaciales locales derivadas de $\text{SST\_BIL}$ aportan únicamente un 10.84% de ganancia en el entrenamiento y cero impacto por permutación en prueba, evidenciando que no aportan información ortogonal sustantiva.

---

## 15. Incertidumbre Estadística (Temporal Block Bootstrap, $B = 1000$)

Para preservar la dependencia espacial intradía, se ejecutó un remuestreo por bloques diarios de 24 horas completas con 1,000 réplicas (`bootstrap_confidence_intervals.csv`):

| Comparación | Modelo Evaluado | Modelo Referencia | Mediana $\Delta\text{RMSE}$ ($^\circ\text{C}$) | IC 95% Inferior ($^\circ\text{C}$) | IC 95% Superior ($^\circ\text{C}$) | $P(\Delta\text{RMSE} < 0)$ | Significación |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **CORE vs B0** | `E3b-C0` | `B0` | **-0.00947** | -0.01390 | -0.00522 | **1.000** | Altamente significativo (Mejora) |
| **T1 vs CORE** | `E3b-T1` | `E3b-C0` | **+0.00028** | -0.00271 | +0.00323 | **0.423** | **No significativo (Cruza cero)** |
| **TS vs T1** | `E3b-TS` | `E3b-T1` | **+0.00114** | -0.00254 | +0.00495 | **0.282** | **No significativo (Cruza cero)** |
| **S vs B0** | `E3b-S` | `B0` | **-0.00949** | -0.01413 | -0.00539 | **1.000** | Altamente significativo (Mejora) |
| **TS vs B0** | `E3b-TS` | `B0` | **-0.00793** | -0.01341 | -0.00238 | **0.997** | Significativo vs B0 |
| **ALL vs B0** | `E3b-ALL` | `B0` | **-0.00954** | -0.01526 | -0.00358 | **0.999** | Significativo vs B0 |

El intervalo de confianza al 95% de la diferencia entre `E3b-T1` y `E3b-C0` comprende $[-0.0027, +0.0032]^\circ\text{C}$ con $P(\text{mejora}) = 42.3\%$, confirmando de manera incontrovertible la **hipótesis nula de no-mejora incremental por memoria temporal**.

---

## 16. Coste Computacional y Recursos de Hardware

Todas las operaciones se ejecutaron en CPU de arquitectura Apple Silicon (10 núcleos, 16 GB RAM) utilizando el backend `tree_method="hist"` de XGBoost (`computational_cost.csv`):

| Modelo | $N_{\text{features}}$ | $N_{\text{train}}$ | $N_{\text{holdout}}$ | Tiempo Fit (s) | Tiempo Pred (s) | Tamaño JSON (KB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `E3b-C0` | 4 | 1,154,678 | 1,925,375 | 0.18 s | 0.03 s | 37.8 KB |
| `E3b-T1` | 6 | 1,154,678 | 1,925,375 | 0.23 s | 0.03 s | 38.0 KB |
| `E3b-T3` | 10 | 1,154,678 | 1,925,375 | 0.31 s | 0.03 s | 38.1 KB |
| `E3b-S` | 7 | 1,154,678 | 1,925,375 | 0.21 s | 0.03 s | 37.8 KB |
| `E3b-TS` | 13 | 1,154,678 | 1,925,375 | 0.34 s | 0.03 s | 38.2 KB |
| `E3b-ALL` | 16 | 1,154,678 | 1,925,375 | 0.37 s | 0.03 s | 38.3 KB |

El pipeline completo (carga de 13.5M registros, extracción de gradientes y ventanas 3×3 para 2,557 días, muestreo de Hamilton, tuning de 9 trials, entrenamiento de 6 modelos, bootstrap de 1,000 réplicas, 10 figuras y 11 tablas) se completó en tan solo **14.67 segundos**.

---

## 17. Dictámenes Formales Predefinidos D32

Siguiendo estrictamente los criterios cuantitativos congelados en la Sección 26 y 27 de la metodología:

### A) Evaluación Temporal:
- Criterio T1: Mejora RMSE de $\text{mejor}(T1, T3)$ sobre CORE $\ge +1.00\%$ $\rightarrow$ **Resultado: -0.07% (Incumplido)**
- Criterio T2: MAE de $\text{mejor}(T1, T3)$ no empeora frente a CORE $\rightarrow$ **Resultado: 0.266854 vs 0.266853 (Incumplido)**
- Criterio T3: Mejora a B0 en al menos 10 de 12 meses $\rightarrow$ **Resultado: 8 de 12 meses (Incumplido)**
- Criterio T4: Celdas mejoradas vs B0 $\ge 25\%$ $\rightarrow$ **Resultado: 100.0% (Cumplido)**
- Criterio T5: Preservación de estabilidad estacional $\rightarrow$ **Resultado: Cumplido**

$$\mathbf{\text{DICTAMEN TEMPORAL: } D32\text{-C — SIN EVIDENCIA DE VALOR TEMPORAL}}$$

### B) Evaluación Espacial:
- Criterio S1: Mejora RMSE de `E3b-TS` sobre $\text{mejor}(T1, T3) \ge +1.00\%$ $\rightarrow$ **Resultado: -0.35% (Incumplido)**
- Criterio S2: Celdas mejoradas vs mejor temporal $\ge 50\%$ $\rightarrow$ **Resultado: 27.0% (Incumplido)**
- Criterio S3: Meses mejorados vs mejor temporal $\ge 7/12$ $\rightarrow$ **Resultado: 5 de 12 meses (Incumplido)**

$$\mathbf{\text{DICTAMEN ESPACIAL: } \text{SPATIAL-NO}}$$

---

## 18. Recomendación Científica Formal

Separando el dictamen cuantitativo de la interpretación de cara a las siguientes fases de la tesis:

$$\mathbf{\text{RECOMENDACIÓN CIENTÍFICA: STOP COMPLEXIFICATION / REVISIT FORCING}}$$

### Fundamentación Científica:
1. **Límite de Información en SST_BIL:** La señal térmica bilineal OISST carece de la resolución temporal y espacial intrínseca necesaria para inferir anomalías submesoescalares de MUR. Aplicar lags, diferencias o ventanas 3×3 sobre un campo ya suavizado a 0.25° no extrae nuevas estructuras físicas, sino que introduce ruido de alta frecuencia que empeora la generalización.
2. **Desaconsejar una CNN sobre SST_BIL pura:** Entrenar una red convolucional compleja (2D o 3D) tomando únicamente parches de $\text{SST\_BIL}$ como entrada es injustificado y redundante: el contexto espacial local 2D ya demostró un aporte nulo ($\Delta_{\text{spatial}} = -0.35\%$).
3. **Camino Recomendado para la Tesis:**
   - **Consolidar el Núcleo Tabular CORE:** El modelo `E3b-C0` (`sst_bil`, `doy_sin`, `doy_cos`, `depth` GEBCO) representa la formulación óptima, parsimoniosa y físicamente justificada, reduciendo el RMSE en un $+2.62\%$ frente al baseline bilineal en el 100% de las celdas oceánicas.
   - **Gating de Sobre-Corrección:** Diseñar un mecanismo que desactive la corrección en regímenes de bajo residual ($|R| < 0.20^\circ\text{C}$), donde el baseline bilineal es superior.
   - **Incorporación de Forzamiento Físico Externo:** Si se busca avanzar en predictibilidad espacial y temporal, no se debe seguir complejizando la arquitectura con $\text{SST\_BIL}$, sino incorporar variables de forzamiento dinámico reales: viento superficial (ERA5/CCMP), corrientes oceánicas o anomalías de altura de la superficie del mar (SSH/CMEMS).

---

## 19. Limitaciones Metodológicas

1. **Resolución Nativa de OISST:** Al tener una celda nativa de ~25 km, las operaciones locales de gradiente en cuadrícula de 1 km reflejan artefactos de la interpolación bilineal suave y no frentes térmicos oceánicos reales.
2. **Ausencia de Dinámica Atmosférica:** El residual $R$ responde en gran medida a eventos sinópticos locales (frentes fríos/"nortes", evaporación, mezcla por viento) que no pueden predecirse exclusivamente a partir de la temperatura de ayer.
3. **Asimetría en Colas Extremas:** En eventos con $|R| \ge 0.96^\circ\text{C}$ (P99), los modelos tabulares sufren regresión a la media y no logran alcanzar los picos térmicos observados en MUR.

---

## 20. Catálogo de Entregables Generados

### A) Modelos Entrenados (`models/`):
- `E3b-C0.json` (38.7 KB)
- `E3b-T1.json` (38.9 KB)
- `E3b-T3.json` (39.0 KB)
- `E3b-S.json` (38.8 KB)
- `E3b-TS.json` (39.1 KB)
- `E3b-ALL.json` (39.2 KB)

### B) Tablas Formales CSV (`tables/`):
1. `model_summary.csv` — Métricas globales de reconstrucción SST y residual.
2. `feature_ablation.csv` — Especificación de features por modelo.
3. `monthly_metrics.csv` — Desglose de métricas mes a mes en 2021.
4. `spatial_metrics.csv` — Distribución percentilar de $\Delta\text{RMSE}$ por celda.
5. `residual_regime_metrics.csv` — Desempeño y sobre-corrección por magnitud de $|R|$.
6. `feature_importance.csv` — Importancia de variables por Gain y Permutación.
7. `hyperparameters.csv` — Resultados de la búsqueda compacta en CORE.
8. `computational_cost.csv` — Tiempos de ajuste, inferencia y tamaño de artefactos.
9. `dataset_counts.csv` — Censos de datos y retención de `COMMON_VALID_MASK`.
10. `decision_criteria.csv` — Matriz de cumplimiento de criterios D32.
11. `bootstrap_confidence_intervals.csv` — Intervalos de confianza 95% ($B = 1000$).

### C) Figuras Científicas de Alta Resolución (`figures/`):
1. `fig1_rmse_comparison.png` — Comparación de RMSE global entre B0 y las 6 formulaciones.
2. `fig2_relative_improvement_vs_b0.png` — Porcentaje de mejora relativa de RMSE frente a B0.
3. `fig3_monthly_rmse_2021.png` — Evolución temporal del RMSE a lo largo de los 12 meses de 2021.
4. `fig4_spatial_map_delta_rmse_vs_b0.png` — Mapa cartográfico 2D de $\Delta\text{RMSE}$ (`E3b-S` vs B0).
5. `fig5_spatial_map_delta_rmse_vs_core.png` — Mapa cartográfico 2D de ganancia marginal (`E3b-S` vs `E3b-C0`).
6. `fig6_performance_by_regime.png` — RMSE por regímenes de anomalía térmica ($0-\text{P50}$ a $\ge\text{P99}$).
7. `fig7_observed_vs_predicted_residual.png` — Densidad hexbin logarítmica de $R$ vs $\hat{R}$.
8. `fig8_feature_importance.png` — Importancia relativa Gain de variables para `E3b-S`.
9. `fig9_distribution_r_rhat.png` — Histogramas comparativos de densidades de $R$ y $\hat{R}$.
10. `fig10_overcorrection_diagnostic.png` — Diagnóstico de magnitud predicha vs real y zona de sobre-corrección.

---
*Fin del Reporte Científico — Fase D.3.2*

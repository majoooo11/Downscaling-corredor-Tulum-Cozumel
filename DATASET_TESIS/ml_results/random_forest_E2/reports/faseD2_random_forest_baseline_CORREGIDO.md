# Reporte Científico — Fase D.2 / Experimento E2: Random Forest Residual Baseline

**Fecha de ejecución:** 2026-09-05 16:30:06  

**Script reproducible:** `DATASET_TESIS/fase_d2_random_forest_baseline.py`  

**Entorno de ejecución:** Python 3.11.16 | Sistema: macOS-26.6.2-arm64-arm-64bit  

**Modelo persistido:** `DATASET_TESIS/models/random_forest_E2_baseline.joblib`  

**Predicciones Parquet:** `DATASET_TESIS/ml_results/random_forest_E2/predictions/validation_predictions.parquet`  

---

## 1. Resumen Ejecutivo y Dictamen de Avance

- **Objetivo científico:** Evaluar si un Random Forest pixel-wise puede aprender una corrección residual no lineal:

  $$
  R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}},
  \qquad
  \text{SST}_{\text{RF}} = \text{SST}_{\text{BIL}} + \hat{R}
  $$

  que supere al interpolador bilineal OISST (baseline E0) durante el periodo de **VALIDATION 2022–2023**, compuesto por **3,853,670 observaciones**.

- **Dictamen metodológico final:** **C. RF NO MEJORA E0**

- **Resultado global de desempeño:**

  - **RMSE:** de **0.3357 °C** (E0) a **0.4138 °C** (RF), equivalente a un deterioro relativo de **23.27%** ($\Delta = +0.0781$ °C).

  - **MAE:** de **0.2636 °C** (E0) a **0.3232 °C** (RF), equivalente a un deterioro relativo de **22.59%** ($\Delta = +0.0596$ °C).

  - **Bias:** de **+0.0251 °C** a **-0.0448 °C**.

  - **$R^2$ de reconstrucción:** de **0.9002** a **0.8484**.

- **Desempeño diario:** RF reduce el RMSE en el **32.47% de los días** (237/730).

- **Desempeño espacial:** RF no reduce el RMSE agregado de VALIDATION en ninguna de las **5,279 celdas oceánicas**.

- **Salvaguarda TEST:** **TEST files opened = 0**. El periodo 2024–2025 permaneció completamente blindado.

---

## 2. Configuración Experimental e Hiperparámetros

- **Algoritmo:** `sklearn.ensemble.RandomForestRegressor`

- **Hiperparámetros:**

  - `n_estimators = 200`
  - `max_depth = 20`
  - `min_samples_leaf = 5`
  - `max_features = 1.0`
  - `bootstrap = True`
  - `random_state = 42`
  - `n_jobs = -1`

- **Variables predictoras ($X$):**

  1. `sst_bil` — SST interpolada bilinealmente (°C)
  2. `depth` — profundidad batimétrica GEBCO (m)
  3. `distance_coast_km` — distancia mínima a la costa (km)
  4. `ocean_fraction` — fracción oceánica sub-pixel [0, 1]
  5. `doy_sin` — componente sinusoidal anual:
     $$
     \sin\left(2\pi \frac{\text{doy}}{365.25}\right)
     $$
  6. `doy_cos` — componente cosenoidal anual:
     $$
     \cos\left(2\pi \frac{\text{doy}}{365.25}\right)
     $$

- **Target ($y$):**

  `residual`

  $$
  R = \text{sst\_mur} - \text{sst\_bil}
  $$

- **Variables excluidas de $X$:** `analysis_error` (solo control diagnóstico), `latitude`, `longitude`, `date`, `year`, `doy`, `split`, `cell_id`, `lat_idx`, `lon_idx`.

---

## 3. Muestreo Proporcional de TRAIN y Control de Representatividad

- **Población TRAIN (2015–2021):**

  $$
  N = 13,498,403
  $$

- **Muestra utilizada:**

  $$
  N = 1,349,840
  $$

  equivalente aproximadamente al **10.00%** de TRAIN.

- **Método de muestreo:** asignación proporcional con enteros mediante el método de Hamilton (*Largest Remainder*) sobre estratos conjuntos de:

  `year` × `month` × `depth_bin` × `residual_decile`

  La estrategia se utilizó para preservar la estructura temporal, batimétrica y distribucional del residual en la muestra de entrenamiento.

- **Tiempo de entrenamiento:** **82.11 s (1.37 min)**

- **Pico de memoria RAM:** **525.81 MB**

### Auditoría de representatividad numérica

- $|\Delta \text{Media Residual}| = 0.000007$ °C  
  Criterio: $< 0.005$ °C → **APROBADO**

- $|\Delta \text{Std Residual}| = 0.0768\%$  
  Criterio: $< 1.0\%$ → **APROBADO**

- Desviación máxima en percentiles P01–P99: **0.001178 °C**  
  Criterio: $< 0.010$ °C → **APROBADO**

- Desviación máxima por año: **0.0003 pp**  
  Criterio: $< 0.10$ pp → **APROBADO**

- Desviación máxima por mes: **0.0009 pp**  
  Criterio: $< 0.10$ pp → **APROBADO**

- Desviación máxima por profundidad: **0.0019 pp**  
  Criterio: $< 0.10$ pp → **APROBADO**

En conjunto, estos resultados indican que la muestra utilizada mantiene una representación muy cercana a la distribución del TRAIN completo en las dimensiones utilizadas para el muestreo.

---

## 4. Generalización TRAIN–VALIDATION

| Métrica | Muestra TRAIN (N = 1,349,840) | VALIDATION completa (N = 3,853,670) | Diferencia |
| :--- | :---: | :---: | :---: |
| **RMSE (°C)** | 0.1890 | 0.4138 | +0.2248 °C |
| **MAE (°C)** | 0.1337 | 0.3232 | +0.1895 °C |
| **RMSE residual (°C)** | 0.1890 | 0.4138 | +0.2248 °C |

La diferencia entre TRAIN y VALIDATION es considerable. El RMSE aumenta de **0.1890 °C** a **0.4138 °C**, lo que corresponde a una razón VALIDATION/TRAIN de aproximadamente:

$$
\frac{0.4138}{0.1890} \approx 2.19
$$

El MAE presenta una razón aproximada de:

$$
\frac{0.3232}{0.1337} \approx 2.42
$$

La métrica de TRAIN corresponde al mismo conjunto utilizado para ajustar el modelo y, por tanto, constituye una evaluación *in-sample*. La brecha observada es compatible con sobreajuste, con cambios temporales en la relación entre los predictores y el residual, o con una combinación de ambos mecanismos. D.2 por sí sola no permite distinguir causalmente entre estas explicaciones.

---

## 5. Comparativa Global frente al Baseline E0 en VALIDATION

| Métrica | Baseline Bilineal E0 | Random Forest Baseline (E2) | Diferencia | Mejora Relativa (%) |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (°C)** | **0.3357** | **0.4138** | **+0.0781 °C** | **-23.27%** |
| **MAE (°C)** | **0.2636** | **0.3232** | **+0.0596 °C** | **-22.59%** |
| **Bias (°C)** [$\text{Pred} - \text{MUR}$] | +0.0251 | -0.0448 | **-0.0699 °C** | — |
| **$|Bias|$ (°C)** | 0.0251 | 0.0448 | **+0.0197 °C** | — |
| **$R^2$ SST reconstruida** | 0.9002 | 0.8484 | **-0.0518** | — |
| **Pearson $r$ SST** | 0.9492 | 0.9329 | **-0.0163** | — |
| **$R^2$ residual ($R$)** | — | **-0.5282** | — | — |
| **Pearson $r$ residual** | — | **0.1425** | — | — |
| **Spearman $\rho$ residual** | — | **0.1498** | — | — |

**Interpretación:** El Random Forest E2 no supera al baseline bilineal E0 en VALIDATION. El RMSE aumenta en **0.0781 °C** y el MAE en **0.0596 °C**, equivalentes a deterioros relativos de **23.27%** y **22.59%**, respectivamente.

El Bias cambia de **+0.0251 °C** a **-0.0448 °C**, por lo que el cambio con signo es:

$$
-0.0448 - 0.0251 = -0.0699\ ^\circ\text{C}
$$

mientras que la magnitud absoluta del sesgo aumenta en:

$$
0.0448 - 0.0251 = 0.0197\ ^\circ\text{C}.
$$

La capacidad del modelo para predecir directamente el residual también es limitada, con un $R^2$ residual negativo y correlaciones de Pearson y Spearman bajas entre $R$ y $\hat{R}$.

---

## 6. Desempeño Temporal Diario (730 Días)

- **Días evaluados:** 730 fechas, correspondientes al periodo 2022-01-01 a 2023-12-31.

- **Días donde RF reduce el RMSE:** **237/730 = 32.47%**

- **Días donde RF reduce el MAE:** **246/730 = 33.70%**

- **Mediana de $\Delta\text{RMSE}$ diario:** **+0.0441 °C**

- **Percentil 05:** **-0.1461 °C**

- **Percentil 95:** **+0.4082 °C**

Aunque existen días individuales en los que RF supera a E0, el patrón dominante durante VALIDATION es de deterioro respecto al baseline bilineal.

---

## 7. Desempeño Espacial en Cuadrícula 86 × 96

Las **5,279 celdas oceánicas** presentan:

$$
\Delta RMSE =
RMSE_{RF} - RMSE_{E0} > 0
$$

durante la evaluación agregada de VALIDATION.

Por tanto:

- **Celdas donde RF mejora:** 0 / 5,279 (**0.0000%**)
- **Celdas donde RF empeora:** 5,279 / 5,279 (**100.0000%**)

Las diferencias espaciales observadas en el mapa deben interpretarse como variaciones en la **magnitud del deterioro**, no como zonas de mejora absoluta.

Algunas áreas costeras y sectores próximos al canal de Cozumel presentan incrementos de RMSE menores que otras regiones del dominio, pero E0 continúa siendo superior en todas las celdas oceánicas cuando el desempeño se agrega sobre todo el periodo de VALIDATION.

---

## 8. Estratificaciones Diagnósticas en VALIDATION

### A. Por Rango Batimétrico

| Estrato | N | RMSE_E0 | RMSE_RF | MAE_E0 | MAE_RF | Delta_RMSE | Impr_RMSE_pct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–20 m | 711750 | 0.3617 | 0.4315 | 0.2788 | 0.3392 | 0.0698 | -19.3016 |
| 20–100 m | 470120 | 0.3486 | 0.4192 | 0.2713 | 0.3298 | 0.0707 | -20.2736 |
| 100–500 m | 994990 | 0.3463 | 0.4171 | 0.2731 | 0.3281 | 0.0708 | -20.4405 |
| 500–1000 m | 632180 | 0.3297 | 0.4092 | 0.2623 | 0.3188 | 0.0795 | -24.1081 |
| >1000 m | 1044630 | 0.3031 | 0.3984 | 0.2416 | 0.3073 | 0.0953 | -31.4391 |

El deterioro se observa en todos los rangos batimétricos y aumenta hacia los estratos más profundos.

### B. Por Quintil de Distancia a la Costa

| Quintil | N | RMSE_E0 | RMSE_RF | MAE_E0 | MAE_RF | Delta_RMSE | Impr_RMSE_pct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 (Costa) | 770880 | 0.3447 | 0.4188 | 0.2658 | 0.3285 | 0.0740 | -21.4745 |
| Q2 | 770880 | 0.3469 | 0.4197 | 0.2715 | 0.3295 | 0.0728 | -20.9907 |
| Q3 | 770880 | 0.3442 | 0.4184 | 0.2716 | 0.3278 | 0.0742 | -21.5676 |
| Q4 | 770880 | 0.3292 | 0.4088 | 0.2620 | 0.3192 | 0.0795 | -24.1549 |
| Q5 (Mar adentro) | 770150 | 0.3120 | 0.4031 | 0.2471 | 0.3109 | 0.0911 | -29.1892 |

El RF presenta deterioro en todos los quintiles de distancia a costa, con una penalización relativamente mayor hacia las zonas más alejadas del litoral.

### C. Por Fracción Oceánica Sub-pixel

| Categoría | N | RMSE_E0 | RMSE_RF | MAE_E0 | MAE_RF | Delta_RMSE | Impr_RMSE_pct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocean_fraction == 1.0 (Mar abierto) | 3774830 | 0.3356 | 0.4138 | 0.2637 | 0.3232 | 0.0782 | -23.3086 |
| ocean_fraction < 1.0 (Borde costero) | 78840 | 0.3388 | 0.4120 | 0.2597 | 0.3232 | 0.0732 | -21.6106 |

No se observa una ventaja del modelo en las celdas costeras con fracción oceánica inferior a 1.

### D. Evaluación Retrospectiva Condicionada a Discrepancias Extremas

Los siguientes subconjuntos se definen utilizando el valor observado:

$$
|R| = |\text{SST}_{MUR} - \text{SST}_{BIL}|
$$

y umbrales obtenidos de TRAIN.

| Grupo | N | Pct_Total | RMSE_E0 | RMSE_RF | MAE_E0 | MAE_RF | Bias_E0 | Bias_RF | Delta_RMSE | Impr_RMSE_pct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Todos los datos | 3853670 | 100.0000 | 0.3357 | 0.4138 | 0.2636 | 0.3232 | 0.0251 | -0.0448 | 0.0781 | -23.2734 |
| \|Residual\| >= P90 (0.5435 °C) | 385697 | 10.0086 | 0.7288 | 0.6941 | 0.7114 | 0.6268 | 0.4097 | 0.3148 | -0.0347 | 4.7654 |
| \|Residual\| >= P95 (0.6698 °C) | 186796 | 4.8472 | 0.8446 | 0.7809 | 0.8311 | 0.7183 | 0.6486 | 0.5325 | -0.0637 | 7.5446 |
| \|Residual\| >= P99 (0.9810 °C) | 27667 | 0.7179 | 1.1265 | 1.0173 | 1.1193 | 0.9749 | 1.1149 | 0.9691 | -0.1092 | 9.6920 |

En los subconjuntos definidos retrospectivamente por discrepancias grandes entre MUR y OISST, RF presenta reducciones progresivas del RMSE respecto a E0, alcanzando aproximadamente **9.69%** en el subconjunto $|R|\ge P99$.

Este resultado es exclusivamente diagnóstico. El residual verdadero requiere conocer MUR y no está disponible durante la inferencia, por lo que estos subconjuntos no constituyen una regla operativa para identificar anticipadamente eventos en los que RF mejorará.

Tampoco debe interpretarse automáticamente que estas observaciones corresponden a frentes, eventos sinópticos o estructuras de submesoescala sin una identificación física independiente.

### E. Estratificación por `analysis_error` de MUR

| Estrato Analysis Error | N | Pct_Total | RMSE_E0 | RMSE_RF | MAE_E0 | MAE_RF | Delta_RMSE | Impr_RMSE_pct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| < P90 (0.40 °C) | 3048898 | 79.1167 | 0.3182 | 0.3939 | 0.2527 | 0.3098 | 0.0757 | -23.7832 |
| P90–P95 (0.40–0.41 °C) | 600653 | 15.5865 | 0.3777 | 0.4557 | 0.2876 | 0.3548 | 0.0780 | -20.6489 |
| >= P95 / P99 (0.41 °C; umbrales coincidentes por discretización del producto) | 204119 | 5.2967 | 0.4421 | 0.5521 | 0.3551 | 0.4296 | 0.1100 | -24.8882 |

`analysis_error` se utiliza exclusivamente como variable diagnóstica de control y no fue empleada como predictor, criterio de filtrado ni ponderación durante el entrenamiento.

Estos valores describen la incertidumbre de análisis proporcionada por el producto MUR y no deben interpretarse directamente como el error verdadero de la SST de MUR.

---

## 9. Importancia de Variables

| Feature | MDI_Importance | Permutation_Mean | Permutation_Std |
| :--- | ---: | ---: | ---: |
| `sst_bil` | 0.3557 | 3.2731 | 0.0134 |
| `doy_sin` | 0.2440 | 1.0106 | 0.0124 |
| `doy_cos` | 0.2105 | 0.1540 | 0.0058 |
| `depth` | 0.1076 | 0.0039 | 0.0020 |
| `distance_coast_km` | 0.0820 | -0.0072 | 0.0012 |
| `ocean_fraction` | 0.0002 | 0.0000 | 0.0000 |

Las importancias MDI se concentran principalmente en `sst_bil`, `doy_sin` y `doy_cos`, que en conjunto acumulan aproximadamente el **81% de la reducción de impureza atribuida internamente por el Random Forest**.

Esta proporción no debe interpretarse como porcentaje de varianza física explicada ni como evidencia causal.

La importancia por permutación en VALIDATION proporciona una perspectiva distinta. `sst_bil` y las componentes temporales muestran la mayor contribución predictiva, mientras que `depth`, `distance_coast_km` y `ocean_fraction` presentan contribuciones marginales o prácticamente nulas bajo este diagnóstico.

En particular, la importancia ligeramente negativa de `distance_coast_km` indica que su permutación no deterioró el desempeño en la muestra utilizada, por lo que no existe evidencia sólida de una contribución predictiva generalizable de esta variable en E2.

Estas importancias deben interpretarse exclusivamente como diagnósticos predictivos del modelo y no como medidas de importancia física o causal.

---

## 10. Catálogo de Figuras Científicas Generadas

Las siguientes 10 figuras fueron generadas en:

`DATASET_TESIS/ml_results/random_forest_E2/figures/`

1. `figura_D2_01_feature_importance.png`  
   Importancia de variables mediante MDI y permutación.

2. `figura_D2_02_scatter_residual_real_predicho.png`  
   Diagrama de densidad hexbin de $R_{\text{real}}$ frente a $\hat{R}$.

3. `figura_D2_03_rmse_diario_e0_vs_rf.png`  
   Serie temporal comparativa del RMSE diario en VALIDATION.

4. `figura_D2_04_delta_rmse_diario.png`  
   Serie temporal y distribución del $\Delta RMSE$ diario.

5. `figura_D2_05_mapa_rmse_e0.png`  
   Mapa espacial del RMSE del baseline bilineal E0.

6. `figura_D2_06_mapa_rmse_rf.png`  
   Mapa espacial del RMSE del Random Forest E2.

7. `figura_D2_07_mapa_delta_rmse.png`  
   Mapa espacial de:
   $$
   \Delta RMSE = RMSE_{RF} - RMSE_{E0}.
   $$

8. `figura_D2_08_error_por_profundidad.png`  
   Comparación de RMSE y MAE por rango batimétrico.

9. `figura_D2_09_error_por_distancia_costa.png`  
   Comparación de RMSE y MAE por distancia a la costa.

10. `figura_D2_10_error_extremos_residual.png`  
    Evaluación retrospectiva condicionada a umbrales de $|R|$ P90, P95 y P99.

---

## 11. Limitaciones, Interpretación y Próximos Pasos

### 11.1 Limitaciones del enfoque pixel-wise

El modelo E2 opera sobre observaciones individuales y no recibe información explícita sobre la vecindad espacial de cada celda.

Por esta razón, su conjunto de predictores no representa directamente gradientes horizontales, contrastes térmicos locales, frentes o patrones bidimensionales presentes en campos de SST.

Los resultados de E2 no permiten concluir que la ausencia de contexto espacial sea la causa principal de su bajo desempeño, pero constituye una limitación relevante del diseño experimental.

### 11.2 Capacidad predictiva residual limitada

La relación entre el residual observado y el residual predicho en VALIDATION es débil:

$$
R^2 = -0.5282
$$

$$
r_{\text{Pearson}} = 0.1425
$$

$$
\rho_{\text{Spearman}} = 0.1498
$$

Estos valores muestran que la configuración actual de Random Forest no generaliza adecuadamente la corrección residual utilizando únicamente las seis variables predictoras seleccionadas.

### 11.3 Hallazgo posterior de D.2.1

El diagnóstico posterior realizado en la Fase D.2.1 mostró que la predicción residual contiene una señal positiva pero débil.

Al introducir una amortiguación diagnóstica:

$$
\text{SST}_{\alpha}
=
\text{SST}_{BIL}
+
\alpha\hat{R}
$$

se obtuvo un mínimo de VALIDATION aproximadamente en:

$$
\alpha = 0.17
$$

con:

$$
RMSE_{\alpha=0.17} \approx 0.3316\ ^\circ\text{C}
$$

frente a:

$$
RMSE_{E0} = 0.3357\ ^\circ\text{C}.
$$

Esto representa una mejora diagnóstica aproximada de **1.22%** respecto a E0.

Este resultado no convierte a $\alpha=0.17$ en un modelo final, ya que el valor fue seleccionado utilizando VALIDATION. Su función es mostrar que $\hat{R}$ contiene cierta información predictiva aprovechable, aunque la corrección completa del RF ($\alpha=1$) resulta globalmente perjudicial.

### 11.4 Siguiente experimento

El siguiente paso será el **Experimento E3: XGBoost Residual Baseline**.

Para mantener comparabilidad con E2, el primer experimento XGBoost utilizará el mismo conjunto de seis variables:

$$
X =
[
\text{sst\_bil},
\text{depth},
\text{distance\_coast\_km},
\text{ocean\_fraction},
\text{doy\_sin},
\text{doy\_cos}
].
$$

La selección de hiperparámetros y el *early stopping* deberán realizarse exclusivamente dentro de TRAIN mediante una partición temporal interna, evitando utilizar repetidamente VALIDATION 2022–2023 para optimización.

Una estrategia apropiada será:

- **2015–2020:** ajuste interno
- **2021:** validación interna para selección y *early stopping*
- **2022–2023:** evaluación formal de E3
- **2024–2025:** TEST completamente blindado

### 11.5 Modelos espaciales posteriores

Las arquitecturas convolucionales 2D se evaluarán únicamente después de establecer el desempeño del benchmark tabular basado en boosting.

Esto permitirá determinar de manera más limpia si la incorporación explícita de contexto espacial proporciona una ganancia adicional respecto al mejor modelo tabular.

---

## 12. Conclusión de E2

El Random Forest residual E2 no supera al baseline bilineal E0 durante VALIDATION y, por tanto, queda descartado en su configuración original como candidato para evaluación final.

El RMSE aumenta de **0.3357 °C** a **0.4138 °C**, mientras que el MAE aumenta de **0.2636 °C** a **0.3232 °C**. Asimismo, ninguna de las 5,279 celdas oceánicas presenta una reducción del RMSE agregado.

Sin embargo, los diagnósticos posteriores muestran que la predicción residual contiene una señal débil pero aprovechable. Una fuerte amortiguación de la corrección reduce ligeramente el error respecto a E0, y el RF presenta un mejor comportamiento retrospectivo en los casos de mayor discrepancia entre MUR y OISST.

Estos resultados indican que el problema no se reduce simplemente a la ausencia total de señal predictiva, sino a una capacidad de generalización insuficiente y a una corrección residual mal calibrada en la configuración E2.

Por esta razón, el siguiente paso metodológico será evaluar un modelo **XGBoost residual (E3)** con regularización, *shrinkage* y *early stopping*, antes de avanzar hacia arquitecturas espaciales convolucionales.

---

**Blindaje final:** `TEST files opened = 0`
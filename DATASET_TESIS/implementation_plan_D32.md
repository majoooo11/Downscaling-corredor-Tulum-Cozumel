# Plan de Implementación — Fase D.3.2: E3b Tabular (Contexto Temporal y Espacial)

Este documento establece el diseño experimental, protocolo computacional y criterios formales de decisión para la **Fase D.3.2: E3b Tabular — Modelado del Residual con Contexto Temporal y Espacial**, correspondiente a la tesis de maestría sobre downscaling estadístico de SST en el corredor Tulum–Cozumel.

---

## 1. Contexto Científico y Motivación

En la Fase D.3.1 se determinó:
- **Baseline Bilineal B0:** $\text{RMSE} = 0.3595^\circ\text{C}$, $\text{MAE} = 0.2773^\circ\text{C}$.
- **Núcleo Tabular Óptimo (A4):** $\text{SST\_BIL} + \text{doy\_sin} + \text{doy\_cos} + \text{depth}$ logró una mejora de **$+4.94\%$** sobre B0 ($\text{RMSE} = 0.3417^\circ\text{C}$).
- **Contexto Espacial Local 2D (E-DIAG-SPATIAL):** aportó un incremento marginal de $+0.93\%$ sobre la base tabular ($+4.83\%$ vs B0), por debajo del umbral del $1.00\%$ requerido para escalar directamente a modelos espaciales 2D complejos (**Dictamen D31-B: Evidencia para E3b Tabular**).
- **Memoria Temporal Residual:** La autocorrelación espacial media del residual mostró una escala de persistencia significativa (Lag-1: $0.6853$, Lag-2: $0.3622$, Lag-3: $0.2206$).

### Restricción Causal Estricta
**PROHIBICIÓN ABSOLUTA:** No se permite usar como predictores $R(t-1)$, $R(t-2)$, etc., ni valores pasados de $\text{SST\_MUR}$. En un escenario operativo real de downscaling, la referencia de alta resolución MUR no está disponible en tiempo presente ni en latencia ultra-corta. Por tanto, toda feature temporal debe derivarse **exclusivamente de $\text{SST\_BIL}$**, el calendario astronómico o covariables geográficas estáticas.

### Preguntas Científicas Centrales
1. ¿Aporta la memoria temporal observable en $\text{SST\_BIL}$ (lags y tendencias multidiarias) información predictiva adicional sobre el residual más allá del núcleo tabular identificado en D.3.1?
2. ¿Aportan las features espaciales locales información adicional una vez incorporado el contexto temporal?
3. ¿Logra el modelo reducir el fenómeno de sobre-corrección en el régimen central ($|R| < \text{P50}$)?

---

## 2. Protocolo de Blindaje Temporal Estricto

1. **Partición DEVELOPMENT (2015–2020):**
   - Archivos: `train_2015.parquet` a `train_2020.parquet` ($N = 11,571,568$).
   - Muestreo representativo reproducible (Hamilton 10%, $N \approx 1,157,157$).
   - Subdivisión cronológica interna para tuning de hiperparámetros:
     - Ajuste: 2015–2019.
     - Early stopping / validación interna: 2020.
2. **Partición DIAGNOSTIC HOLDOUT (2021):**
   - Archivo: `train_2021.parquet` ($N = 1,926,835$).
   - Evaluado una sola vez como conjunto fuera de muestra para comparar configuraciones ya congeladas.
   - Prohibido su uso para selección de variables, tuning o early stopping.
3. **Bloqueo Absoluto:**
   - `VALIDATION (2022–2023)`: **0 archivos abiertos**.
   - `TEST (2024–2025)`: **0 archivos abiertos**.
   - Verificación explícita obligatoria en log y reportes:
     ```text
     VALIDATION files opened = 0
     TEST files opened = 0
     ```

---

## 3. Definición y Construcción de Features

### A. Núcleo Base (CORE)
- `sst_bil`: Temperatura interpolada bilinealmente (°C).
- `doy_sin`: $\sin(2\pi \cdot \text{DOY} / 365.25)$.
- `doy_cos`: $\cos(2\pi \cdot \text{DOY} / 365.25)$.
- `depth`: Batimetría ETOPO1 (profundidad en metros, signo negativo consistente).

### B. Features Temporales Causales (Derivadas de SST_BIL)
Calculadas ordenando cronológicamente los 2,557 días continuos (2015–2021) para cada una de las 5,279 celdas oceánicas:
- `sst_bil_lag1`: $\text{SST\_BIL}(t-1, x, y)$
- `sst_bil_lag2`: $\text{SST\_BIL}(t-2, x, y)$
- `sst_bil_lag3`: $\text{SST\_BIL}(t-3, x, y)$
- `delta_sst_1d`: $\text{SST\_BIL}(t) - \text{SST\_BIL}(t-1)$
- `delta_sst_2d`: $\text{SST\_BIL}(t) - \text{SST\_BIL}(t-2)$
- `delta_sst_3d`: $\text{SST\_BIL}(t) - \text{SST\_BIL}(t-3)$
- `sst_bil_mean_3d`: $(\text{SST\_BIL}(t) + \text{SST\_BIL}(t-1) + \text{SST\_BIL}(t-2)) / 3.0$
- `sst_bil_std_3d`: Desviación estándar causal de la ventana de 3 días.
*(Nota: Para los primeros 3 días de 2015, los lags no disponibles se descartan de forma documentada).*

### C. Features Espaciales 2D (Derivadas de SST_BIL del día t)
Calculadas celda por celda sobre la grilla regular C.2 ($86 \times 96$, $\Delta\text{lat}=0.01^\circ$, $\Delta\text{lon}=0.01^\circ$):
- `grad_mag_sst_bil`: Magnitud del gradiente horizontal térmico ($\sqrt{(\partial_x T)^2 + (\partial_y T)^2}$, en °C/km).
- `local_std_3x3`: Desviación estándar en ventana $3 \times 3$ con soporte oceánico ($N_{\text{valid}} \ge 3$).
- `local_contrast`: $\text{SST\_BIL} - \text{media\_local\_3x3}$.
- `local_range_3x3`: Rango térmico local ($\max - \min$).

### D. Máscara Común de Comparación (`COMMON_VALID_MASK`)
Para asegurar comparaciones estrictamente pareadas e idénticas:
- `COMMON_VALID_MASK` exige: features espaciales válidas ($N_{\text{valid}} \ge 3$, no-NaN) y features temporales válidas (lags 1–3 no-NaN).
- En HOLDOUT 2021: representa el **99.92%** de las observaciones ($N = 1,925,375$).
- Todas las comparaciones de ablación se evalúan sobre esta máscara común.

---

## 4. Conjunto Predefinido de Ablaciones E3b

Se entrenan y evalúan 6 configuraciones:

1. **`E3b-C0` (CORE):**  
   `[sst_bil, doy_sin, doy_cos, depth]`
2. **`E3b-T1` (CORE + Lag 1):**  
   $\text{CORE} + \text{[sst\_bil\_lag1, delta\_sst\_1d]}$
3. **`E3b-T3` (CORE + Memoria Temporal Corta 3 días):**  
   $\text{CORE} + \text{[sst\_bil\_lag1, sst\_bil\_lag2, sst\_bil\_lag3, delta\_sst\_1d, delta\_sst\_2d, delta\_sst\_3d]}$
4. **`E3b-S` (CORE + Espacial 2D):**  
   $\text{CORE} + \text{[grad\_mag\_sst_bil, local\_std_3x3, local\_contrast]}$
5. **`E3b-TS` (CORE + Temporal + Espacial):**  
   $\text{CORE} + \text{[lags 1..3, deltas 1..3, grad\_mag, local\_std, local\_contrast]}$
6. **`E3b-ALL` (Ablación Secundaria Completa):**  
   $\text{E3b-TS} + \text{[distance\_coast\_km, ocean\_fraction, local\_range\_3x3]}$

---

## 5. Algoritmo y Protocolo de Entrenamiento

- **Algoritmo:** XGBoost Regressor (`tree_method="hist"`).
- **Target:** $R = \text{SST\_MUR} - \text{SST\_BIL}$.
- **Reconstrucción:** $\hat{\text{SST}} = \text{SST\_BIL} + \hat{R}$.
- **Optimización de Hiperparámetros Interna (DEVELOPMENT):**
  - Muestra Hamilton representativa: 10% de DEV.
  - Conjunto de entrenamiento: 2015–2019.
  - Conjunto de validación interna: 2020.
  - Búsqueda compacta sobre CORE:
    - `max_depth` $\in [4, 6, 8]$
    - `learning_rate` $\in [0.03, 0.05, 0.10]$
    - `subsample` = $0.8$
    - `colsample_bytree` = $0.8$
    - `min_child_weight` = $5$
    - Early stopping (patience = 20) sobre el año 2020.
  - Una vez fijados los mejores hiperparámetros y `best_iteration`, se congelan y se entrenan los 6 modelos de ablación sobre el periodo completo de DEVELOPMENT (2015–2020).

---

## 6. Métricas y Diagnósticos Obligatorios

### A. Reconstrucción de SST (Métricas Principales en HOLDOUT 2021)
- $\text{RMSE}_{\text{SST}} = \sqrt{\frac{1}{N} \sum (\text{SST\_MUR} - \hat{\text{SST}})^2}$
- $\text{MAE}_{\text{SST}} = \frac{1}{N} \sum |\text{SST\_MUR} - \hat{\text{SST}}|$
- $\text{Bias}_{\text{SST}} = \frac{1}{N} \sum (\hat{\text{SST}} - \text{SST\_MUR})$
- $R^2_{\text{SST}}$ y $R^2_{\text{Residual}}$
- $\text{Improvement\_RMSE\_vs\_B0} = 100 \times \frac{\text{RMSE}_{\text{B0}} - \text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{B0}}}$
- $\text{Improvement\_RMSE\_vs\_CORE} = 100 \times \frac{\text{RMSE}_{\text{CORE}} - \text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{CORE}}}$

### B. Estabilidad Temporal Mensual
- Evaluación de RMSE por mes en 2021 para cada modelo.
- Reportar: meses con mejora vs B0, mejor/peor mes, desviación estándar mensual.

### C. Distribución Espacial Celda por Celda
- $\Delta\text{RMSE}_{\text{cell}} = \text{RMSE}_{\text{model}} - \text{RMSE}_{\text{B0}}$.
- Porcentaje de celdas mejoradas ($> 50\%$), mediana, P10 y P90.

### D. Diagnóstico por Régimen de Residual ($|R|$)
- Regímenes predefinidos con umbrales de DEV:
  - $0 - \text{P50}$ ($|R| < 0.2066^\circ\text{C}$)
  - $\text{P50} - \text{P75}$ ($0.2066 - 0.3604^\circ\text{C}$)
  - $\text{P75} - \text{P90}$ ($0.3604 - 0.5377^\circ\text{C}$)
  - $\text{P90} - \text{P95}$ ($0.5377 - 0.6652^\circ\text{C}$)
  - $\text{P95} - \text{P99}$ ($0.6652 - 0.9659^\circ\text{C}$)
  - $\ge \text{P99}$ ($|R| \ge 0.9659^\circ\text{C}$)
  - Agregados: Central ($< \text{P90}$), Cola Extrema ($\ge \text{P90}$), Cola P99 ($\ge \text{P99}$).

### E. Diagnóstico de Sobre-Corrección
- Análisis del error de corrección $e_R = \hat{R} - R$.
- Distribución comparada de $R$ vs $\hat{R}$.
- Relación entre magnitud real $|R|$ y corrección predicha $|\hat{R}|$.
- Verificación cuantitativa de si el contexto temporal o espacial atenúa el deterioro en el régimen de bajo residual ($0 - \text{P50}$).

### F. Importancia de Variables
- Gain importance relativa para el mejor modelo tabular.
- Permutation importance sobre muestra estratificada de HOLDOUT 2021.

---

## 7. Criterios de Decisión y Dictamen Formal

### Dictamen de Contexto Temporal
- **`D32-A — EVIDENCIA FUERTE PARA E3b TEMPORAL`**:
  Si el mejor modelo temporal (`E3b-T1` o `E3b-T3`):
  1. Mejora RMSE sobre `E3b-C0` en $\ge 1.00\%$ relativo;
  2. Mejora o no deteriora MAE vs `E3b-C0`;
  3. Mejora a B0 en $\ge 10$ de 12 meses;
  4. No concentra la mejora en $< 25\%$ de las celdas oceánicas;
  5. Muestra estabilidad estacional consistente.
- **`D32-B — GANANCIA TEMPORAL MODESTA`**:
  Existe mejora sobre CORE ($> 0\%$), pero la ganancia incremental es $< 1.00\%$ o la estabilidad temporal/espacial es insuficiente.
- **`D32-C — SIN EVIDENCIA DE VALOR TEMPORAL`**:
  Las features temporales no mejoran a CORE, deterioran consistentemente RMSE/MAE o producen inestabilidad fuera de muestra.

### Dictamen de Contexto Espacial
- **`SPATIAL-YES`**: Si `E3b-TS` mejora al mejor modelo temporal en $\ge 1.00\%$ relativo y es espacial y temporalmente estable.
- **`SPATIAL-MARGINAL`**: Si la mejora marginal espacial es $> 0\%$ pero $< 1.00\%$.
- **`SPATIAL-NO`**: Si no hay mejora o hay deterioro.

### Recomendación sobre Escalamiento a CNN
- Recomendar CNN únicamente si el contexto espacial muestra ganancias no explotables por modelos tabulares. De lo contrario, recomendar consolidar la formulación tabular óptima.

---

## 8. Catálogo de Entregables de la Fase D.3.2

### A. Estructura de Salida
Directorio base: `DATASET_TESIS/ml_results/E3b_D32/`
- `models/`: Modelos XGBoost JSON persistidos.
- `tables/`: 10 tablas CSV independientes.
- `figures/`: 10 figuras científicas en PNG de alta resolución.
- `reports/`: `faseD32_E3b_tabular.md`.
- `logs/`: Logs de ejecución.
- `WALKTHROUGH_D32.md`: Resumen técnico formal.

### B. Tablas CSV (10 archivos)
1. `tables/model_summary.csv`: Resumen de métricas globales de B0 y las 6 ablaciones.
2. `tables/feature_ablation.csv`: Comparativa detallada de ablaciones vs B0 y vs CORE.
3. `tables/monthly_metrics.csv`: Métricas mensuales (12 meses × modelos).
4. `tables/spatial_metrics.csv`: Resumen y cuantiles de $\Delta\text{RMSE}$ por celda.
5. `tables/residual_regime_metrics.csv`: Métricas por regímenes de discrepancia.
6. `tables/feature_importance.csv`: Importancias relativas Gain y Permutation.
7. `tables/hyperparameters.csv`: Hiperparámetros de búsqueda y configuración óptima congelada.
8. `tables/computational_cost.csv`: Tiempos de fit, predicción y uso de memoria.
9. `tables/dataset_counts.csv`: Censos de observaciones por año y por máscara.
10. `tables/decision_criteria.csv`: Evaluación binaria y cuantitativa de cada criterio.

### C. Figuras Científicas (10 archivos)
- `figures/fig1_rmse_comparison.png`: Barras comparativas de RMSE entre B0 y las 6 ablaciones.
- `figures/fig2_relative_improvement_vs_b0.png`: Mejora porcentual vs B0.
- `figures/fig3_monthly_rmse_2021.png`: Serie temporal mensual 2021 (B0 vs CORE vs Mejor E3b).
- `figures/fig4_spatial_map_delta_rmse_vs_b0.png`: Mapa cartográfico de $\Delta\text{RMSE}$ (Mejor vs B0).
- `figures/fig5_spatial_map_delta_rmse_vs_core.png`: Mapa cartográfico de $\Delta\text{RMSE}$ (Mejor vs CORE).
- `figures/fig6_performance_by_regime.png`: RMSE y ganancia por régimen de magnitud residual.
- `figures/fig7_observed_vs_predicted_residual.png`: Scatter/hexbin de $R$ vs $\hat{R}$.
- `figures/fig8_feature_importance.png`: Barras horizontales de Feature Importance (Gain).
- `figures/fig9_distribution_r_rhat.png`: Densidades KDE superpuestas de $R$ y $\hat{R}$.
- `figures/fig10_overcorrection_diagnostic.png`: Diagnóstico de sobre-corrección ($e_R$ vs $|R|$).

---

## 9. Plan de Verificación

### Pruebas Automatizadas
1. `py_compile` de `DATASET_TESIS/fase_d32_e3b_tabular.py`.
2. Verificación de blindaje: `assert validation_files_opened == 0` y `assert test_files_opened == 0`.
3. Verificación de consistencia causal: `assert (delta_sst_1d == sst_bil - sst_bil_lag1).all()`.
4. Verificación de dimensiones de grilla y máscaras: 5,279 celdas oceánicas auditadas.
5. Verificación de existencia de las 10 tablas CSV y 10 figuras PNG.
6. Verificación de emisión de dictámenes D32-A/B/C y SPATIAL-YES/MARGINAL/NO.

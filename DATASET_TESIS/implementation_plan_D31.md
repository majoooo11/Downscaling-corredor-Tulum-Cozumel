# Plan de Implementación — Fase D.3.1: Diagnóstico de Predictibilidad del Residual (Diseño Experimental Cerrado)

Este documento establece el diseño experimental, metodológico, matemático y algorítmico riguroso para la **Fase D.3.1: Diagnóstico de Predictibilidad del Residual**, que se ejecuta tras el cierre formal de E0, E2, D.2.1 y E3 (con su auditoría de Feature Importance corregida).

---

## 1. Contexto Científico y Preguntas de Investigación

### 1.1 Antecedentes Consolidados
En el experimento E3 (XGBoost Residual Baseline), evaluado formalmente sobre VALIDATION 2022–2023 ($N = 3,853,670$):
- **E0 (Bilineal OISST):** $\text{RMSE} = 0.3357\ ^\circ\text{C}$, $\text{MAE} = 0.2636\ ^\circ\text{C}$, $\text{Bias} = +0.0251\ ^\circ\text{C}$.
- **E2 (Random Forest):** $\text{RMSE} = 0.4138\ ^\circ\text{C}$ (deterioro de $-23.27\%$).
- **E3 (XGBoost):** $\text{RMSE} = 0.3365\ ^\circ\text{C}$ (diferencia relativa de $-0.25\%$ frente a E0; mejora de $+18.68\%$ frente a E2).
- **Diagnóstico Residual E3:** $R^2 = -0.0107$, Pearson $r = 0.2352$, Spearman $\rho = 0.3054$, $\text{std}(\hat{R})/\text{std}(R) = 0.4712$.
- **Comportamiento Dual:** E3 mejora a E0 en el **57.53% de los días** y reduce el error en eventos de gran discrepancia ($|R| \ge \text{P99}$, donde RMSE pasa de $1.1265$ a $0.9845\ ^\circ\text{C}$, $+12.61\%$). Sin embargo, en el régimen central ($|R| < \text{P90}$, 90% de los datos), E3 deteriora el error frente a E0 ($0.2806$ vs $0.2572\ ^\circ\text{C}$).
- **Feature Importance Auditada:** La temperatura bilineal (`sst_bil`) y los armónicos anuales (`doy_sin`, `doy_cos`) **concentran el 86.79% de la ganancia interna** asignada por XGBoost y el 98.7% del impacto por permutación. Las variables geomorfológicas locales (`depth`, `distance_coast_km`, `ocean_fraction`) concentran una fracción menor de ganancia (9.99%, 3.18% y 0.03%) y una degradación por permutación prácticamente nula ($+0.0040$, $+0.00003$ y $+0.00000\ ^\circ\text{C}$).

> [!NOTE]
> **Interpretación Metodológica de Feature Importance:**  
> Una Permutation Importance baja de `depth`, `distance_coast_km` u `ocean_fraction` significa estrictamente que:  
> *"Dichas variables aportan poca pérdida predictiva adicional cuando se permutan bajo la formulación pixel-wise de E3 y sobre la distribución espacial evaluada."*  
> Esto **NO** significa que la batimetría o la proximidad a la costa sean físicamente irrelevantes para los procesos hidrodinámicos marinos.

### 1.2 Preguntas Centrales de la Fase D.3.1
- **Q1:** ¿Cuánto del residual $R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$ corresponde a una señal climatológica/estacional sistemática frente a variabilidad estocástica local?
- **Q2:** ¿Las variables estáticas batimétricas y costeras aportan señal predictiva generalizable fuera del periodo de ajuste?
- **Q3:** ¿Las discrepancias térmicas entre MUR y OISST están correlacionadas con frentes, gradientes espaciales o contrastes locales derivados exclusivamente de $\text{SST}_{\text{BIL}}$?
- **Q4:** ¿Por qué los modelos predictivos mejoran las anomalías grandes pero deterioran las observaciones cercanas a cero ($R \approx 0$)?
- **Q5:** ¿La evidencia empírica interna en TRAIN justifica diseñar un modelo tabular refinado (E3b), avanzar hacia modelos convolucionales espaciales (CNN), o reconsiderar la formulación física del target residual?

---

## 2. Blindaje Temporal y Controles Anti-Leakage

> [!IMPORTANT]
> **BLINDAJE ABSOLUTO DE VALIDATION Y TEST:**
> Ninguna decisión metodológica, selección de features, calibración o ajuste en D.3.1 utilizará información de VALIDATION 2022–2023 ni abrirá TEST 2024–2025.
> - **VALIDATION 2022–2023 files opened = 0.** (Se cita únicamente como antecedente histórico ya cerrado de E3).
> - **TEST 2024–2025 files opened = 0.** (Estrictamente bloqueado).

### 2.1 Particiones Temporales en D.3.1
| Partición | Rango Temporal | Rol en Fase D.3.1 | Observaciones |
| :--- | :---: | :--- | :---: |
| **DEVELOPMENT** | 2015–2020 | Cálculo de climatologías, umbrales de percentiles, correlaciones espaciales, extracción de gradientes y ajuste de modelos diagnósticos. | 11,571,568 |
| **DIAGNOSTIC HOLDOUT** | 2021 | Evaluación comparativa estricta y selección diagnóstica de alternativas. | 1,926,835 |
| **VALIDATION OFICIAL** | 2022–2023 | **BLOQUEADO para selección.** Ningún modelo diagnóstico será evaluado en este bloque para decidir el rumbo del proyecto. | 0 archivos abiertos |
| **TEST OFICIAL** | 2024–2025 | **ESTRICTAMENTE BLOQUEADO.** No se abrirá ningún archivo de `ml_dataset/test/`. | 0 archivos abiertos |

---

## 3. Muestreo Estratificado Proporcional mediante Asignación Hamilton

Para el entrenamiento de las variantes diagnósticas en `DEVELOPMENT` (2015–2020), se utilizará una **muestra estratificada proporcional mediante asignación Hamilton** de tamaño $N = 1,157,157$ (equivalente aproximadamente al 10.00% de `TRAIN_SUB`).

- **Estratos:** `year` $\times$ `month` $\times$ `depth_bin` $\times$ `residual_decile` (deciles calculados exclusivamente sobre 2015–2020).
- **Objetivo metodológico:** Preservar fielmente la distribución natural de probabilidad de `TRAIN_SUB`, garantizando que ningún estrato sea balanceado artificialmente y asegurando tiempos de entrenamiento interactivos con representatividad auditada.

---

## 4. Geometría de la Cuadrícula y Operadores Espaciales Costeros

### 4.1 Cálculo Físico de Distancias y Orientación de Latitud
Los gradientes espaciales se calcularán utilizando las coordenadas reales de la cuadrícula extraídas dinámicamente del dataset C.2 (`faseC2_dataset_maestro_diario.nc`), sin recurrir a aproximaciones simplificadas ni constantes fijas no verificadas.

1. **Extracción, Monotonicidad y Orientación Física:**
   - Vector de latitud $\mathbf{lat}$ ($N_{\text{lat}} = 86$); vector de longitud $\mathbf{lon}$ ($N_{\text{lon}} = 96$).
   - Detección obligatoria de orientación:
     $$\text{lat\_ascending} = \text{all}(\Delta\text{lat} > 0), \qquad \text{lat\_descending} = \text{all}(\Delta\text{lat} < 0)$$
     *(El script abortará si ninguna condición se satisface).*
   - Se registrará explícitamente: `LAT ORIENTATION: ASCENDING / DESCENDING`.
   - Eje 0 = Dimensión de latitud ($Y$); Eje 1 = Dimensión de longitud ($X$).
2. **Métricas Físicas Locales:**
   $$dy_{\text{km}} = 111.32 \cdot |\Delta\text{lat}|$$
   $$dx_{\text{km}}(\text{lat}_i) = 111.32 \cdot \cos\left(\text{lat}_i \cdot \frac{\pi}{180}\right) \cdot |\Delta\text{lon}|$$
   - Comprobación aproximada para $\text{lat} \approx 20^\circ$ y $\Delta \approx 0.01^\circ$: $dy \approx 1.113\text{ km}$, $dx \approx 1.046\text{ km}$.
   - **Registro Previo:** Se registrará e imprimirá $dx_{\min}$, $dx_{\max}$ y $dy$ antes de calcular gradientes.

### 4.2 Definición Exacta de Operadores Espaciales Costeros
> [!CAUTION]
> **Tratamiento Riguroso de Tierra y Bordes Costeros:**
> - La máscara de tierra (`ocean_mask == 0`) no se rellena con valores térmicos artificiales para evitar gradientes espurios.
> - Todos los operadores tratan la tierra como celda ausente/inválida.

1. **Gradiente Zonal ($\text{grad\_x}$) y Meridional con Orientación Geográfica Real ($\text{grad\_y}$):**
   Para cada celda oceánica válida $(i, j)$:
   - **Gradiente Zonal $\partial T / \partial x$:**
     - Si ambos vecinos oeste $(i, j-1)$ y este $(i, j+1)$ son celdas oceánicas válidas:
       $$\text{grad\_x}(i, j) = \frac{\text{SST}_{\text{BIL}}(i, j+1) - \text{SST}_{\text{BIL}}(i, j-1)}{2 \cdot dx_{\text{km}}(\text{lat}_i)}$$
     - Si solo el vecino este $(i, j+1)$ es oceánico: diferencia unilateral hacia adelante:
       $$\text{grad\_x}(i, j) = \frac{\text{SST}_{\text{BIL}}(i, j+1) - \text{SST}_{\text{BIL}}(i, j)}{dx_{\text{km}}(\text{lat}_i)}$$
     - Si solo el vecino oeste $(i, j-1)$ es oceánico: diferencia unilateral hacia atrás:
       $$\text{grad\_x}(i, j) = \frac{\text{SST}_{\text{BIL}}(i, j) - \text{SST}_{\text{BIL}}(i, j-1)}{dx_{\text{km}}(\text{lat}_i)}$$
     - Si no existe ningún vecino oceánico zonal: `NaN`.
   - **Gradiente Meridional $\partial T / \partial y$ (Respetando Orientación Norte-Sur Real):**
     $$\text{grad\_y}(i, j) = \frac{T_{\text{north}} - T_{\text{south}}}{\text{distancia}_{\text{north-south}}}$$
     - Si $\mathbf{lat}$ es ascendente con el índice $i$, $T_{\text{north}} = \text{SST}_{\text{BIL}}(i+1, j)$ y $T_{\text{south}} = \text{SST}_{\text{BIL}}(i-1, j)$.
     - Si $\mathbf{lat}$ es descendente con el índice $i$, $T_{\text{north}} = \text{SST}_{\text{BIL}}(i-1, j)$ y $T_{\text{south}} = \text{SST}_{\text{BIL}}(i+1, j)$.
     - Si ambos vecinos meridionales existen: diferencia central con paso $2 \cdot dy_{\text{km}}$.
     - Si solo existe un vecino meridional: diferencia unilateral con paso $dy_{\text{km}}$ orientada formalmente hacia el norte físico.
     - Si no existe ningún vecino meridional oceánico: `NaN`.
   - **Módulo del Gradiente:**
     $$\text{grad\_mag\_sst\_bil}(i, j) = \sqrt{\text{grad\_x}(i, j)^2 + \text{grad\_y}(i, j)^2}$$
     (Válido solo si ambos componentes no son `NaN`; de lo contrario `NaN`).

2. **Estadísticas en Ventana Local $3 \times 3$ y Política Estricta de $N_{\text{valid}}$:**
   Para cada celda oceánica $(i, j)$:
   - Se consideran exclusivamente las celdas de la vecindad $3 \times 3$ pertenecientes al océano y con SST válida.
   - Conteo de soporte: $N_{\text{valid}} \in [1, 9]$.
   - **Regla Estricta de Soporte Costero:**
     $$\text{Si } N_{\text{valid}} < 3 \implies \text{local\_mean\_3x3} = \text{NaN}, \quad \text{local\_std\_3x3} = \text{NaN}, \quad \text{local\_range\_3x3} = \text{NaN}, \quad \text{local\_contrast} = \text{NaN}$$
   - **Si $N_{\text{valid}} \ge 3$:**
     - `local_mean_3x3`: Promedio aritmético de las $N_{\text{valid}}$ celdas, **incluyendo explícitamente la celda central**.
     - `local_std_3x3`: Desviación estándar insesgada (ddof=1) de las $N_{\text{valid}}$ celdas oceánicas.
     - `local_range_3x3`: $\max(T_{\text{valid}}) - \min(T_{\text{valid}})$.
     - `local_contrast`: $\text{SST}_{\text{BIL}}(i, j) - \text{local\_mean\_3x3}(i, j)$.

3. **Tratamiento del Operador Laplaciano:**
   - Se excluye justificadamente de los predictores diagnósticos para no fabricar valores numéricamente inestables en bordes costeros irregulares.

---

## 5. Plan Detallado de Análisis (A hasta H)

### Análisis A — Ablación Diagnóstica de Features en HOLDOUT 2021
Se entrenarán 5 configuraciones de XGBoost sobre `DEVELOPMENT` (2015–2020) con hiperparámetros congelados de E3 (`lr=0.02, depth=5, sub=0.8, col=0.8, l2=1.0, l1=0.0, mcw=5, n_estimators=352`), evaluándose en `DIAGNOSTIC HOLDOUT` 2021 ($N = 1,926,835$):
- **A1 — XGBoost residual SST-only:**  
  $$X = [\text{sst\_bil}]$$  
  *Aclaración conceptual obligatoria:* **A1 $\ne$ Baseline bilineal.** A1 es un modelo XGBoost no lineal que predice $\hat{R} = f(\text{sst\_bil})$ y reconstruye $\text{SST}_{\text{pred}} = \text{SST}_{\text{BIL}} + \hat{R}$. El verdadero baseline bilineal es B0 ($\hat{R} = 0$, $\text{SST}_{\text{pred}} = \text{SST}_{\text{BIL}}$).
- **A2 (Bilineal + Temporal):** $X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}]$
- **A3 (Bilineal + Geomorfológico):** $X = [\text{sst\_bil}, \text{depth}, \text{distance\_coast\_km}, \text{ocean\_fraction}]$
- **A4 (Bilineal + Temporal + Batimetría):** $X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}]$
- **A5 (E3 Completo):** $X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}, \text{distance\_coast\_km}, \text{ocean\_fraction}]$
- **Salida:** `tables/feature_ablation_2021.csv`.

### Análisis B — Baselines Climatológicos Residuales
Construidos exclusivamente con información de 2015–2020 y evaluados sobre 2021:
- **B0:** $\hat{R} = 0$, $\text{SST}_{\text{pred}} = \text{SST}_{\text{BIL}}$ (**Verdadero Baseline Bilineal Puro y Referencia Primaria**).
- **B1:** $\hat{R} = \bar{R}_{\text{train}}$ (Corrección constante del sesgo global de TRAIN).
- **B2:** $\hat{R}(m) = \bar{R}(m)_{\text{train}}$ (Corrección mensual, 12 valores escalonados).
- **B3:** $\hat{R}(t) = \beta_0 + \beta_1 \sin(2\pi t / 365.25) + \beta_2 \cos(2\pi t / 365.25)$ (Ajuste armónico estacional continuo).
- **B4:** $\hat{R}(c, m) = \bar{R}(c, m)_{\text{train}}$ (**Baseline diagnóstico de climatología espacial fija**).
  - *Fallback Predefinido:* Si una celda-mes no estuviera presente en 2015–2020, se aplicará el valor mensual global de B2. Se registrará y reportará el número exacto de fallbacks aplicados.
- **Salida:** `tables/climatological_baselines_2021.csv`.

### Análisis C — Estructura Condicional del Residual en 2015–2020
Evaluación probabilística de $E[R \mid X]$ y $E[|R| \mid X]$ en:
- $\text{SST}_{\text{BIL}}$ (bins de $0.5\ ^\circ\text{C}$ de $24\ ^\circ\text{C}$ a $31.5\ ^\circ\text{C}$).
- Mes de calendario (1 a 12).
- Profundidad batimétrica ($[0, 10), [10, 50), [50, 200), [200, 500), [500, 1000), \ge 1000\text{ m}$).
- Distancia a la costa ($[0, 2), [2, 5), [5, 10), [10, 20), [20, 50), \ge 50\text{ km}$).
- Para cada bin se reportan: $N$, media, mediana, desviación estándar, $\text{MAE}(R, 0)$ y percentiles P10, P25, P50, P75, P90, P95, P99.
- **Salidas:** `tables/residual_conditional_sst.csv`, `tables/residual_conditional_month.csv`, `tables/residual_conditional_depth.csv`, `tables/residual_conditional_distance.csv`.

### Análisis D — Contexto Espacial 2D Derivado de SST_BIL
- Derivación diaria de los campos: `grad_x`, `grad_y`, `grad_mag`, `local_mean_3x3`, `local_std_3x3`, `local_range_3x3` y `local_contrast` según la Sección 4.
- Correlaciones bivariadas Pearson y Spearman con $R$ y $|R|$ en 2015–2020.
- Estratificación por quintiles (Q1 a Q5) de $\text{grad\_mag}$, $\text{local\_std}$ y $\text{local\_range}$ reportando media, mediana, RMSE de E0 y percentiles de $|R|$.
- *Estatus Epistemológico:* Estas correlaciones proporcionan **evidencia mecanística/descriptiva** para interpretar la física de los errores, pero no constituyen umbrales binarios de decisión.
- **Salidas:** `tables/spatial_context_correlations.csv`, `tables/residual_by_gradient_quantile.csv`, `tables/residual_by_local_variability.csv`.

### Análisis E — Modelos Tabulares Diagnósticos con Contexto Espacial y Máscara de Comparación Idéntica
Se evalúan en `DIAGNOSTIC HOLDOUT` 2021 configuraciones predefinidas **estrictamente congeladas antes de observar 2021**:
- **`E-DIAG-BASE` (Modelo Tabular Base):**
  $$X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}]$$
- **`E-DIAG-SPATIAL` (Modelo Tabular con Features Espaciales Primario):**
  $$X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{grad\_mag\_sst\_bil}, \text{local\_std\_3x3}, \text{local\_contrast}]$$
- **`E-DIAG-SPATIAL-PLUS` (Modelo Tabular con Features Espaciales Secundario Predefinido):**
  $$X = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}, \text{grad\_mag\_sst\_bil}, \text{local\_std\_3x3}, \text{local\_range\_3x3}, \text{local\_contrast}]$$
  *(Predefinido para verificar si la combinación de batimetría y rango local añade señal complementaria, sin modificar el contraste primario BASE vs SPATIAL ni alterar el criterio D31-A).*

> [!IMPORTANT]
> **Definición Rigurosa de `SPATIAL_VALID_MASK`:**
> Para asegurar que la comparación formal `E-DIAG-BASE` vs `E-DIAG-SPATIAL` sea matemáticamente exacta y no atribuible a diferencias en la muestra:
> 1. `SPATIAL_VALID_MASK` identifica las observaciones de HOLDOUT 2021 donde todas las variables espaciales de `E-DIAG-SPATIAL` están disponibles y son finitas ($N_{\text{valid}} \ge 3$ y gradientes no-NaN).
> 2. La comparación formal entre `E-DIAG-BASE`, `E-DIAG-SPATIAL` y el baseline B0 se calcula sobre **exactamente las mismas filas de `SPATIAL_VALID_MASK`** (`N_base_comparison == N_spatial_comparison`).
> 3. Se reporta la descomposición muestral:
>    $$N_{\text{holdout\_total}}, \quad N_{\text{spatial\_valid}}, \quad N_{\text{spatial\_excluded}}, \quad \text{fraction\_spatial\_valid}$$
> 4. `E-DIAG-BASE` se reporta también sobre el total de 2021 como referencia histórica complementaria sin mezclarla con el contraste formal.

- **Salida:** `tables/spatial_feature_model_2021.csv`.

### Análisis F — Diagnóstico por Regímenes de Residual (Central vs Colas)
- Cuantiles de $|R|$ calculados **exclusivamente sobre 2015–2020**: P50, P75, P90, P95, P99.
- Aplicación de thresholds congelados a 2021 en bins mutuamente excluyentes: $[0, \text{P50})$, $[\text{P50}, \text{P75})$, $[\text{P75}, \text{P90})$, $[\text{P90}, \text{P95})$, $[\text{P95}, \text{P99})$, $[\ge \text{P99})$.
- Cuantificación para cada estrato de $N$, media de $R$, dispersión, error de predicción y balance entre ganancia en colas e inyección de error en el centro.
- **Salida:** `tables/residual_regime_diagnostic_2021.csv`.

### Análisis G — Discriminación Prospectiva de Grandes Discrepancias en HOLDOUT
Evaluación de la capacidad de alertar prospectivamente anomalías térmicas intensas sin conocer $R$ en inferencia:
- Eventos en 2021: $Y_{90} = \mathbb{I}(|R| \ge \text{P90}_{\text{train}})$, $Y_{95} = \mathbb{I}(|R| \ge \text{P95}_{\text{train}})$, $Y_{99} = \mathbb{I}(|R| \ge \text{P99}_{\text{train}})$.
- Scores proyectados: $|\hat{R}|$ de `E-DIAG-SPATIAL` y $\text{grad\_mag\_sst\_bil}$.
- **Métricas obligatorias:** AUROC, AUPRC obtenida, prevalencia basal del evento ($P$) y ratio de enriquecimiento ($\text{AUPRC} / P$).
- **Salida:** `tables/extreme_predictability_2021.csv`.

### Análisis H — Persistencia Temporal y Memoria del Residual
Distinción explícita entre diagnóstico físico y viabilidad operacional:
1. **Diagnóstico Científico de Memoria:**
   - Autocorrelación temporal celda por celda en 2015–2020: $\text{Corr}(R_t, R_{t-1})$, $\text{Corr}(R_t, R_{t-2})$, $\text{Corr}(R_t, R_{t-3})$, $\text{Corr}(R_t, R_{t-7})$.
   - Distribución espacial de la memoria: se reportará media, mediana, P10 y P90 de la autocorrelación por celda (evitando correlaciones globales que mezclen variabilidad espacial con temporal).
2. **Salvaguarda de Disponibilidad Operacional:**
   - Se establece formalmente que $R_{t-1} = \text{SST}_{\text{MUR}, t-1} - \text{SST}_{\text{BIL}, t-1}$ requiere la disponibilidad del producto de alta resolución MUR del día anterior.
   - En consecuencia, la autocorrelación se analiza como diagnóstico de persistencia física del fenómeno y no se introduce como feature operativa en inferencia basada puramente en OISST sin previo estudio de latencia satelital.
- **Salida:** `tables/residual_temporal_persistence.csv`.

---

## 6. Métricas Cuantitativas de Estabilidad Temporal y Distribución Espacial

### 6.1 Estabilidad Temporal Mensual con Máscara Idéntica (`tables/monthly_stability_2021.csv`)
Para cada mes $m \in \{1, \dots, 12\}$ de 2021, la comparación formal entre B0, `E-DIAG-BASE` y `E-DIAG-SPATIAL` se evalúa estrictamente sobre las filas de:
$$\text{MONTH\_SPATIAL\_MASK}(m) = (\text{year} == 2021) \ \& \ (\text{month} == m) \ \& \ \text{SPATIAL\_VALID\_MASK}$$
- Para cada modelo y cada mes se calcula: $\text{RMSE}_{\text{mensual}}$, $\text{MAE}_{\text{mensual}}$, $\Delta\text{RMSE}_{\text{mensual}} = \text{RMSE}_{\text{candidato}} - \text{RMSE}_{\text{B0}}$ y $N$ mensual evaluado.
- Conteo formal:
  $$\text{months\_improved\_vs\_B0} = \sum_{m=1}^{12} \mathbb{I}(\text{RMSE}_{\text{candidato}}(m) < \text{RMSE}_{\text{B0}}(m))$$
- **Criterio cuantitativo de estabilidad temporal:** $\text{months\_improved\_vs\_B0} \ge \mathbf{7}$ de 12 meses.

### 6.2 Distribución Espacial Celda por Celda (`tables/spatial_model_cellwise_2021.csv`)
Para la comparación sobre `SPATIAL_VALID_MASK` entre `E-DIAG-BASE` y `E-DIAG-SPATIAL`, se calcula para cada celda oceánica evaluable $c$:
- $\text{RMSE}_{\text{BASE}}(c)$
- $\text{RMSE}_{\text{SPATIAL}}(c)$
- $\Delta\text{RMSE}_{\text{cell}}(c) = \text{RMSE}_{\text{SPATIAL}}(c) - \text{RMSE}_{\text{BASE}}(c)$
  - Celda mejorada si $\Delta\text{RMSE}_{\text{cell}} < 0$.
  - Celda empeorada si $\Delta\text{RMSE}_{\text{cell}} > 0$.
- Métricas agregadas:
  - $\text{cells\_evaluable}$
  - $\text{cells\_improved}$
  - $\text{cells\_worsened}$
  - $\text{fraction\_cells\_improved} = \text{cells\_improved} / \text{cells\_evaluable}$
- **Criterio cuantitativo de distribución espacial:** $\text{fraction\_cells\_improved} > \mathbf{0.50}$.

---

## 7. Criterios Cuantitativos de Decisión Final (D31-A / B / C)

La decisión metodológica final se basa estrictamente en el desempeño cuantitativo fuera de muestra en 2021, tomando como referencia primaria el baseline bilineal B0/E0:
$$\text{Improvement\_RMSE\_2021} = 100 \cdot \frac{\text{RMSE}_{\text{B0}} - \text{RMSE}_{\text{candidato}}}{\text{RMSE}_{\text{B0}}}$$

### Criterio D31-A — Evidencia para Contexto Espacial / Modelo 2D
Se asignará si se satisfacen **simultáneamente las 5 condiciones numéricas obligatorias**:
1. `E-DIAG-SPATIAL` mejora el RMSE frente a `E-DIAG-BASE` en $\ge \mathbf{1.0\%}$, evaluados sobre exactamente `SPATIAL_VALID_MASK`.
2. `E-DIAG-SPATIAL` mejora el RMSE frente a B0 en $\ge \mathbf{1.0\%}$ sobre exactamente `SPATIAL_VALID_MASK`.
3. $\text{MAE}_{\text{SPATIAL}} \le \text{MAE}_{\text{B0}}$ sobre dicha máscara.
4. **Estabilidad temporal demostrada:** Mejora a B0 en $\ge \mathbf{7}$ de los 12 meses de 2021 ($\text{months\_improved\_vs\_B0} \ge 7$).
5. **Distribución espacial amplia demostrada:** Mejora a `E-DIAG-BASE` en $>\mathbf{50\%}$ de las celdas evaluables ($\text{fraction\_cells\_improved} > 0.50$).

*(Los resultados del Análisis D aportan evidencia mecanística/descriptiva en el reporte, pero no constituyen umbrales de decisión adicionales).*

*Conclusión formal permitida:*  
*"Existe evidencia empírica consistente dentro de TRAIN/2021 de que el contexto espacial local contiene información predictiva no representada por E3, por lo que está metodológicamente justificado evaluar posteriormente una arquitectura espacial (CNN o modelo 2D)."* (Sin afirmar a priori que una CNN necesariamente superará a E0).

### Criterio D31-B — Evidencia para E3b Tabular
Se asignará si:
1. Alguna formulación tabular (sea tabular base A1–A5/BASE o tabular con features espaciales SPATIAL/SPATIAL-PLUS) mejora a B0 en $\ge \mathbf{1.0\%}$ de RMSE en 2021.
2. El MAE no empeora frente a B0 ($\text{MAE} \le \text{MAE}_{\text{B0}}$).
3. Presenta estabilidad temporal con $\text{months\_improved\_vs\_B0} \ge 7$ de 12 meses.
4. `E-DIAG-SPATIAL` no obtiene $\ge 1.0\%$ adicional sobre la mejor formulación tabular base o no cumple los criterios de distribución espacial amplia de A ($\text{fraction\_cells\_improved} \le 0.50$).

*Conclusión formal permitida:*  
*"Se justifica diseñar un experimento intermedio E3b tabular formal con features mejoradas antes de escalar la complejidad arquitectónica hacia modelos convolucionales profundos."*

### Criterio D31-C — Evidencia Insuficiente con las Representaciones Evaluadas
Se asignará si ninguna formulación predefinida alcanza una mejora de RMSE $\ge \mathbf{1.0\%}$ respecto a B0/E0 en HOLDOUT 2021, o si las mejoras no satisfacen los criterios de estabilidad temporal o espacial.

*Redacción científica obligatoria:*  
*"Las representaciones y predictores evaluados en D.3.1 no muestran señal predictiva suficiente y generalizable para justificar por sí solos un aumento de complejidad arquitectónica. En ese escenario estaría justificado investigar predictores dinámicos adicionales físicamente plausibles —por ejemplo viento, corrientes o variables altimétricas— o reconsiderar la formulación del target, mediante un experimento posterior específicamente diseñado."*  
*(Queda prohibido afirmar que "no existe señal física" o que "OISST no contiene información" en sentido absoluto).*

---

## 8. Catálogo de Figuras Científicas (HOLDOUT 2021 / TRAIN)

Todas las figuras se generarán en `DATASET_TESIS/ml_results/diagnostics_D31/figures/`:
1. `D31_01_feature_ablation_2021.png`: Barras de RMSE y MAE para las 5 configuraciones de ablación (A1 a A5) en HOLDOUT 2021.
2. `D31_02_climatological_baselines_2021.png`: Comparativa de RMSE de los baselines B0 a B4 en 2021 indicando la cota de reducción del error estacional.
3. `D31_03_residual_vs_sst_bil.png`: Estructura del residual promedio y $|R|$ frente a los bines de temperatura $\text{SST}_{\text{BIL}}$ en 2015–2020.
4. `D31_04_residual_seasonality.png`: Ciclo anual medio del residual por mes de calendario (medias y bandas de dispersión P10–P90).
5. `D31_05_abs_residual_vs_gradient.png`: Relación entre la magnitud del gradiente de $\text{SST}_{\text{BIL}}$ y $|R|$ (media, mediana y P90 de discrepancia).
6. `D31_06_abs_residual_vs_local_std.png`: Dispersión residual frente a la variabilidad térmica local $3 \times 3$.
7. `D31_07_map_mean_abs_residual_train.png`: Mapa 2D de $|R|$ medio en 2015–2020 sobre la grilla $86 \times 96$ del corredor Tulum–Cozumel.
8. `D31_08_map_delta_rmse_spatial_2021.png`: Mapa de $\Delta\text{RMSE}$ por celda en 2021 ($\Delta\text{RMSE} = \text{RMSE}_{\text{E-DIAG-SPATIAL}} - \text{RMSE}_{\text{E-DIAG-BASE}}$), con escala divergente centrada en cero ($\Delta < 0$ mejora, azul; $\Delta > 0$ deterioro, rojo).
9. `D31_09_central_vs_tail_diagnostic_2021.png`: Dual panel del balance de error en el régimen central frente a las colas de discrepancia en 2021.
10. `D31_10_extreme_predictability_2021.png`: Curvas ROC y Precision-Recall para la **discriminación prospectiva de grandes discrepancias — HOLDOUT 2021** (eventos P90, P95 y P99).
11. `D31_11_temporal_persistence_residual.png`: Decaimiento temporal de la autocorrelación lag-1 a lag-7 en el dominio y distribución espacial de la persistencia lag-1.

---

## 9. Protocolo de Auditoría Espacial Obligatoria

Antes de iniciar cualquier operación bidimensional (Análisis D y E), se ejecutará la verificación cartográfica estricta:
1. Dimensión de la cuadrícula: $86 \times 96$.
2. Total de celdas oceánicas válidas: exactamente **5,279**.
3. Mapping unívoco `cell_id` $\leftrightarrow$ `(lat_idx, lon_idx)`.
4. Coherencia de vectores fisiográficos de C.2: `depth`, `distance_coast_km` y `ocean_fraction`.
5. Monotonicidad y orientación física de coordenadas (`lat_ascending` o `lat_descending`).

> [!WARNING]
> **Regla de Contingencia Cartográfica:**  
> Si la auditoría falla (`SPATIAL MAPPING AUDIT = FAIL`):
> - Se **abortarán inmediatamente** los análisis espaciales D y E y la generación de figuras espaciales.
> - Se mantendrán intactos y válidos los análisis no espaciales A, B, C, F y H.
> - Se emitirá el reporte formal [`reports/spatial_mapping_audit_D31_FAILED.md`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ml_results/diagnostics_D31/reports/spatial_mapping_audit_D31_FAILED.md).
> - Queda estrictamente prohibido reconstruir el mapping espacial mediante reordenamiento arbitrario de filas sin verificación formal previa.

---

## 10. Verification Plan y Aserciones Automáticas

El pipeline incluirá un bloque de validación riguroso con aserciones automatizadas antes y durante la ejecución:
- `assert A1_model_type == 'residual_model'`: A1 se procesa como modelo residual XGBoost ($X = [\text{sst\_bil}]$).
- `assert B0_residual_formula == 'R_hat = 0'`: B0 ejecuta analíticamente $\hat{R} = 0$.
- `assert N_base_comparison == N_spatial_comparison`: BASE y SPATIAL se evalúan sobre filas idénticas de `SPATIAL_VALID_MASK`.
- `assert (df_spatial.loc[df_spatial['N_valid'] < 3, 'local_std_3x3'].isna()).all()`: Toda celda con $N_{\text{valid}} < 3$ tiene NaN en estadísticas locales.
- `assert (df_spatial.loc[df_spatial['spatial_valid'], 'N_valid'] >= 3).all()`: Toda celda válida posee soporte $N_{\text{valid}} \ge 3$.
- **Validación de Estabilidad Mensual por Modelo:**
  ```python
  for model in expected_monthly_models:
      sub = monthly_stability_df[monthly_stability_df["model"] == model]
      assert len(sub) == 12, f"Modelo {model} no contiene exactamente 12 meses"
      assert set(sub["month"]) == set(range(1, 13)), f"Meses incompletos para {model}"
      assert sub["month"].is_unique, f"Meses duplicados para {model}"
  ```
- `assert fraction_cells_improved == cells_improved / cells_evaluable`: Porcentaje normalizado solo sobre celdas evaluables.
- `assert lat_ascending or lat_descending`: Latitud estrictamente monótona con orientación registrada.
- `assert VALIDATION_FILES_OPENED == 0` y `assert TEST_FILES_OPENED == 0`.
- `assert set(train_years) == set(range(2015, 2021))`: Climatologías y percentiles calculados exclusivamente con 2015–2020.
- `assert 'sst_mur' not in spatial_features`: Ninguna variable espacial o derivada utiliza $\text{SST}_{\text{MUR}}$.
- `assert spatial_audit_pass`: Mapping espacial validado antes de reconstrucción 2D.
- `assert dx_min > 0.9 and dx_max < 1.2 and dy > 1.0 and dy < 1.3`: Espaciamiento de coordenadas en kilómetros verificado.

---

## 11. Presupuesto Computacional y Métricas de Ejecución

*Estimaciones de referencia (no compromisos garantizados):*
- Carga de datos de TRAIN (2015–2021): ~1.2 GB en memoria (~10–15 s).
- Reconstrucción de grilla y operaciones 2D vectorizadas con tratamiento costero: ~30–45 s para 2,192 días de 2015–2020 y 365 días de 2021.
- Ajuste de modelos diagnósticos (`tree_method='hist'`): ~1.5 a 2.5 minutos.
- Autocorrelaciones celda por celda: ~20–30 s.
- **Tiempo total estimado de ejecución:** **~4.0 a 6.0 minutos**.
- **Pico de RAM estimado:** $\approx 2.5\text{ GB}$.

Durante la ejecución real, el script registrará y reportará formalmente:
- Tiempo transcurrido por módulo.
- Tiempo total de ejecución.
- Pico de memoria RAM consumido.

---

## 12. Estructura Completa de Archivos a Generar

```text
DATASET_TESIS/
├── fase_d31_diagnostico_predictibilidad_residual.py
├── implementation_plan_D31.md
└── ml_results/
    └── diagnostics_D31/
        ├── tables/
        │   ├── feature_ablation_2021.csv
        │   ├── climatological_baselines_2021.csv
        │   ├── residual_conditional_sst.csv
        │   ├── residual_conditional_month.csv
        │   ├── residual_conditional_depth.csv
        │   ├── residual_conditional_distance.csv
        │   ├── spatial_context_correlations.csv
        │   ├── residual_by_gradient_quantile.csv
        │   ├── residual_by_local_variability.csv
        │   ├── spatial_feature_model_2021.csv
        │   ├── residual_regime_diagnostic_2021.csv
        │   ├── extreme_predictability_2021.csv
        │   ├── residual_temporal_persistence.csv
        │   ├── monthly_stability_2021.csv
        │   └── spatial_model_cellwise_2021.csv
        ├── figures/
        │   ├── D31_01_feature_ablation_2021.png
        │   ├── ... (D31_02 a D31_11)
        └── reports/
            ├── faseD31_diagnostico_predictibilidad_residual.md
            └── WALKTHROUGH_D31.md
```

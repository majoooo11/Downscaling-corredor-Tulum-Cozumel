# Reporte Científico — Fase D.3 / Experimento E3: XGBoost Residual Baseline

**Fecha de ejecución:** 2026-09-05 17:43:39  
**Script reproducible:** `DATASET_TESIS/fase_d3_xgboost_baseline.py`  
**Entorno de ejecución:** Python 3.11.16 | XGBoost 3.2.0 | Sistema: macOS-26.6.2-arm64-arm-64bit  
**Modelo persistido:** `DATASET_TESIS/models/xgboost_E3_residual_baseline.json`  
**Predicciones Parquet:** `DATASET_TESIS/ml_results/xgboost_E3/predictions/validation_predictions_E3.parquet`  

---

## 1. Resumen Ejecutivo y Dictamen de Avance

- **Objetivo científico:** Evaluar si un modelo Gradient Boosting regularizado (`xgboost.XGBRegressor`) con árboles poco profundos, regularización $L_1/L_2$ y shrinkage puede aprender la corrección residual:
  $$R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}, \qquad \text{SST}_{\text{XGB}} = \text{SST}_{\text{BIL}} + \hat{R}_{\text{XGB}}$$
  superando al interpolador bilineal E0 y mitigando el sobreajuste del Random Forest E2 en **VALIDATION 2022–2023 (3,853,670 observaciones)**.
- **Dictamen metodológico formal:** **D**
- **Recomendación científica:** **R3** — Revisar target/formulación de features antes de continuar escalando complejidad.
- **Desempeño global en VALIDATION:**
  - **RMSE:** De **0.3357 °C** (E0) a **0.3365 °C** (E3) $\longrightarrow$ **Mejora relativa de -0.25%** ($\Delta = +0.0008\ ^\circ\text{C}$).
  - **MAE:** De **0.2636 °C** (E0) a **0.2646 °C** (E3) $\longrightarrow$ **Mejora relativa de -0.37%** ($\Delta = +0.0010\ ^\circ\text{C}$).
  - **Bias:** -0.0340 °C (vs +0.0251 °C en E0 y -0.0448 °C en E2).
- **Desempeño temporal:** E3 reduce el RMSE diario respecto a E0 en **420 de 730 días (57.53%)** y respecto a E2 en **540 de 730 días (73.97%)**.
- **Desempeño espacial:** E3 reduce el RMSE en 2174 de 5,279 celdas (41.18%).
- **Salvaguarda de blindaje:** **TEST files opened = 0** (Conjunto 2024–2025 completamente intacto).

---

## 2. Objetivo e Hipótesis Científica

**Hipótesis Principal (H_E3):** Un modelo de gradient boosting regularizado, mediante árboles secuenciales de baja profundidad, shrinkage, subsampling y regularización, puede extraer de forma más eficiente la señal residual débil identificada en D.2.1 y mejorar su generalización temporal respecto a Random Forest E2, evitando correcciones descalibradas.

---

## 3. Salvaguardas Experimentales y Blindaje de VALIDATION

1. **VALIDATION 2022–2023 congelado:** Ningún hiperparámetro, regularización ni criterio de parada temprana fue optimizado sobre 2022–2023.
2. **Partición temporal interna:** Ajuste en `TRAIN_SUB` (2015–2020) y validación en `INTERNAL_VALIDATION` (2021).
3. **Reentrenamiento con hiperparámetros congelados:** Modelo definitivo reentrenado sobre 2015–2021 completo con `n_estimators = best_iteration + 1`.
4. **Blindaje de TEST:** `TEST files opened = 0`.

---

## 4. Dataset y Variables Predictoras

Mismas 6 variables que en E2 para comparabilidad limpia:
- $X = [\text{sst\_bil}, \text{depth}, \text{distance\_coast\_km}, \text{ocean\_fraction}, \text{doy\_sin}, \text{doy\_cos}]$
- $y = \text{residual} = \text{sst\_mur} - \text{sst\_bil}$
- Excluidas de $X$: `latitude`, `longitude`, `cell_id`, `lat_idx`, `lon_idx`, `analysis_error`, `sst_mur`, `date`, `year`, `doy`, `split`.

---

## 5. Muestreo de Desarrollo TRAIN_SUB (Hamilton Proporcional)

- **Población TRAIN_SUB (2015–2020):** $N = 11,571,568$ observaciones.
- **Muestra extraída:** $N = 1,157,157$ observaciones, equivalente aproximadamente al **10.00%** de `TRAIN_SUB`.
- **Estratificación:** `year` $	imes$ `month` $	imes$ `depth_bin` $	imes$ `residual_decile` (deciles calculados exclusivamente sobre TRAIN_SUB).
- **Auditoría de Representatividad:**
  - $|\Delta 	ext{Media}| < 0.005\ ^\circ	ext{C}$ $	o$ **APROBADO** (ver detalles en `sampling_audit_E3.csv`).
  - Todos los criterios de percentiles y distribuciones marginales aprobados (ver `sampling_audit_E3.csv`).

---

## 6. Búsqueda Controlada de Hiperparámetros (12 Configuraciones)

Se evaluaron 12 configuraciones en `TRAIN_SUB sample` con *early stopping* (paciencia 100 rondas) sobre 2021 completo ($N = 1,926,835$):
- **Configuración Ganadora:** Config 2 (Shrinkage conservador)
  - `learning_rate`: 0.02
  - `max_depth`: 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `reg_lambda`: 1.0
  - `reg_alpha`: 0.0
  - `min_child_weight`: 5
  - `best_iteration`: 351
  - `RMSE 2021`: 0.3414 °C

---

## 7. Comparación Global Formal: E0 vs E2 vs E3

| Métrica | Baseline Bilineal (E0) | Random Forest (E2) | XGBoost Residual (E3) | Mejora E3 vs E0 (%) | Mejora E3 vs E2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE SST (°C)** | **0.3357** | 0.4138 | **0.3365** | **-0.25%** | **+18.68%** |
| **MAE SST (°C)** | **0.2636** | 0.3232 | **0.2646** | **-0.37%** | — |
| **Bias SST (°C)** | +0.0251 | -0.0448 | -0.0340 | — | — |
| **$|\text{Bias}|$ SST (°C)** | 0.0251 | 0.0448 | 0.0340 | — | — |
| **$R^2$ Reconstrucción** | 0.9002 | 0.8484 | 0.8997 | — | — |
| **Pearson $r$ SST** | 0.9612 | 0.9329 | 0.9552 | — | — |
| **$R^2$ Residual** | — | -0.5282 | -0.0107 | — | — |
| **Pearson $r$ Residual** | — | 0.1425 | 0.2352 | — | — |
| **Spearman $\rho$ Residual** | — | 0.1498 | 0.3054 | — | — |
| **Ratio $\text{std}(\hat{R})/\text{std}(R)$** | 0.0000 | 0.8708 | 0.4712 | — | — |

*Nota de referencia no independiente:* El modelo diagnóstico de RF amortiguado con $\alpha = 0.17$ produjo $\text{RMSE} \approx 0.3316\ ^\circ\text{C}$ sobre VALIDATION; sin embargo, al haber sido seleccionado sobre VALIDATION, es optimista y no independiente, por lo que no compite formalmente con E3.

---

## 8. Diagnóstico Residual: Amplitud, Alineación y Calibración

- **Ratio de dispersión:** $\text{std}(\hat{R}_{\text{XGB}}) / \text{std}(R) = \mathbf{0.4712}$.
- **Correlaciones:** Pearson $r = \mathbf{0.2352}$, Spearman $\rho = \mathbf{0.3054}$.
- **Calibración lineal diagnóstica:** $R = +0.0044 + 0.4991 \cdot \hat{R}_{\text{XGB}}$ ($R^2 = 0.0553$).
  - *Interpretación:* La pendiente $b = 0.4991$ se reporta como diagnóstico de calibración y no se utiliza para modificar la predicción de E3.

---

## 9. Desempeño en Régimen Central vs Discrepancias Extremas

| Régimen | N Observaciones | % Total | RMSE E0 (°C) | RMSE E2 (°C) | RMSE E3 (°C) | Mejora E3 vs E0 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Régimen Central ($|R| < \text{P90}$)** | 3,467,973 | 89.99% | **0.2572** | 0.3697 | **0.2806** | **-9.10%** |
| **Cola Extrema ($|R| \ge \text{P99}$)** | 27,660 | 0.72% | 1.1265 | 1.0173 | **0.9845** | **+12.61%** |

- *Hallazgo:* Se evalúa si XGBoost reduce el deterioro observado con RF en el régimen central $|R| < \text{P90}$, evitando correcciones residuales que incrementen el error respecto a E0.
- Los análisis por estratos de $|R|$ son retrospectivos condicionales al residual observado y no una regla operativa de inferencia.

---

## 10. Desempeño Temporal Diario (730 Días)

- **Días con mejora de E3 vs E0:** **420 / 730 (57.53%)**.
- **Días con mejora de E3 vs E2:** **540 / 730 (73.97%)**.
- Mediana de $\Delta\text{RMSE}$ diario: ver `daily_metrics_validation_E3.csv`.

---

## 11. Desempeño Espacial (5,279 Celdas Oceánicas)

- **Estado de Auditoría Espacial:** **PASS**.
- Celdas donde E3 supera a E0: **2174 / 5,279 (41.18%)**.
- Celdas donde E3 supera a E2: **5279 / 5,279 (100.00%)**.

---

## 12. Feature Importance (Gain vs Permutación)

La extracción de importancias del modelo persistido (`xgboost_E3_residual_baseline.json`) fue auditada para resolver el mapeo interno de variables (`f0..f5`), obteniendo concordancia exacta con la estructura del Booster:

| Feature | Gain Raw | Gain Relativo (%) | Permutation Mean ($\Delta\text{RMSE}$, °C) | Permutation Std (°C) |
| :--- | :---: | :---: | :---: | :---: |
| `sst_bil` | 854.71 | **32.86%** | **+0.3638** | 0.0003 |
| `doy_sin` | 707.44 | **27.20%** | **+0.1563** | 0.0010 |
| `doy_cos` | 695.31 | **26.73%** | **+0.0049** | 0.0002 |
| `depth` | 259.80 | **9.99%** | **+0.0040** | 0.0001 |
| `distance_coast_km` | 82.79 | **3.18%** | **+0.00003** | 0.00002 |
| `ocean_fraction` | 0.88 | **0.03%** | **+0.00000015** | 0.00000003 |

### Interpretación Metodológica Obligatoria:
1. **Diferencia conceptual entre métricas:**
   - **Gain Importance:** Es una métrica puramente interna del algoritmo, calculada como la mejora promedio (o acumulada) en la función objetivo de pérdida (MSE cuadrático) atribuible a los splits donde interviene cada variable durante el ajuste sobre TRAIN.
   - **Permutation Importance:** Evalúa el impacto empírico en la generalización predictiva al permutar aleatoriamente cada feature sobre una muestra independiente de VALIDATION ($N = 100,000$, 5 repeticiones), midiendo el incremento directo en RMSE.
2. **Ausencia de interpretación causal:**
   - Ninguna de estas métricas constituye una atribución causal ni debe interpretarse como que una variable "explica un porcentaje determinado de la física oceanográfica". Ambas son diagnósticos puramente estadísticos y de sensibilidad algorítmica.
3. **Discrepancia entre rankings internos y de generalización:**
   - Mientras que `doy_cos` recibe una fracción de ganancia interna muy cercana a `doy_sin` en los splits del árbol (~26.7% vs ~27.2%), su destrucción por permutación en VALIDATION tiene un impacto en RMSE sensiblemente menor (+0.0049 °C frente a +0.1563 °C de `doy_sin`). Esta diferencia ilustra la brecha entre la frecuencia de uso en las particiones de los árboles (Gain) y la contribución real a la generalización en datos no vistos (Permutation), donde `sst_bil` y `doy_sin` dominan la capacidad predictiva del modelo.
   - Las variables geomorfológicas estáticas (`depth`, `distance_coast_km`, `ocean_fraction`) aportan una fracción menor de ganancia y su permutación degrada marginalmente el error, coherente con su naturaleza invariante en el tiempo.

---

## 13. Limitaciones del Experimento E3

1. El modelo opera celda por celda sin información explícita de contexto bidimensional (parches 2D), lo que limita su capacidad para representar frentes térmicos y estructuras de submesoescala.
2. La ausencia de contexto espacial es una hipótesis plausible a evaluar posteriormente mediante arquitecturas convolucionales.

---

## 14. Dictamen Final y Recomendación

- **Dictamen E3:** **D**
- **Recomendación:** **R3** (Revisar target/formulación de features antes de continuar escalando complejidad.)

---

**Blindaje final:** `TEST files opened = 0`

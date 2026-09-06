# Reporte Científico — Fase D.2.1: Diagnóstico del Fallo de Generalización del Random Forest E2

**Fecha de ejecución:** 2026-09-05 16:55:29  
**Script reproducible:** `DATASET_TESIS/fase_d21_diagnostico_random_forest.py`  
**Entorno:** Python 3.11.16 | Sistema: macOS-26.6.2-arm64-arm-64bit  
**Modelo auditado:** `DATASET_TESIS/models/random_forest_E2_baseline.joblib`  
**Predicciones analizadas:** `DATASET_TESIS/ml_results/random_forest_E2/predictions/validation_predictions.parquet`  

---

## 1. Objetivo Científico del Diagnóstico

El experimento E2 (Random Forest pixel-wise) produjo un empeoramiento sistemático frente al interpolador bilineal E0 en VALIDATION 2022–2023:
$$\text{RMSE}_{\text{E0}} = 0.3357\ ^\circ\text{C} \quad \longrightarrow \quad \text{RMSE}_{\text{RF}} = 0.4138\ ^\circ\text{C} \quad (\Delta = +0.0781\ ^\circ\text{C}, \ -23.27\%)$$
$$\text{MAE}_{\text{E0}} = 0.2636\ ^\circ\text{C} \quad \longrightarrow \quad \text{MAE}_{\text{RF}} = 0.3232\ ^\circ\text{C} \quad (\Delta = +0.0596\ ^\circ\text{C}, \ -22.59\%)$$

El objetivo de la **Fase D.2.1** es descomponer analíticamente el comportamiento del residual $\hat{R}$ para discernir si el modelo sobreajustó, aprendió una corrección con amplitud descalibrada, carece de señal predictiva en las 6 covariables pixel-wise, o si el fallo reside en el régimen de pequeñas discrepancias frente a las colas.

---

## 2. Inputs Utilizados y Salvaguarda de Blindaje

- **Predicciones VALIDATION:** `validation_predictions.parquet` (3,853,670 registros).
- **Métricas Diarias:** `daily_metrics_validation.csv` (730 fechas).
- **Metadatos Oficiales:** `random_forest_E2_metadata.json`.
- **Salvaguarda TEST:** **TEST files opened = 0** (Archivos de `ml_dataset/test/` estrictamente intocados).

---

## 3. Auditoría de Consistencia e Identidades Numéricas

Se verificaron las identidades analíticas exactas:
- $R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$ (Máximo error absoluto: $0.00\times 10^{-6}\ ^\circ\text{C}$).
- $\text{SST}_{\text{RF}} = \text{SST}_{\text{BIL}} + \hat{R}$ (Máximo error absoluto: $0.00\times 10^{-6}\ ^\circ\text{C}$).
- $\text{Error}_{\text{RF}} = \text{SST}_{\text{RF}} - \text{SST}_{\text{MUR}} = \hat{R} - R$ (Máximo error absoluto: $9.54\times 10^{-7}\ ^\circ\text{C}$).
- $\text{RMSE}(\text{Error}_{\text{RF}}) = \text{RMSE}(\hat{R} - R) = \mathbf{0.41381034\ ^\circ\text{C}}$.
- **Corrección de Placeholder en Reporte D.2:** Se calculó con precisión completa $\text{RMSE}_{\text{RES}} = 0.4138\ ^\circ\text{C}$ y $\Delta_{\text{overfit}} = +0.2248\ ^\circ\text{C}$, eliminando el placeholder pendiente.

---

## 4. Diagnóstico A — Distribución $R$ vs $\hat{R}$

| Estadística | $R$ Real (°C) | $\hat{R}$ Predicho (°C) |
| :--- | :---: | :---: |
| **Media** | -0.0251 | -0.0700 |
| **Desviación Estándar** | 0.3347 | 0.2915 |
| **Mínimo** | -2.3168 | -1.1636 |
| **Percentil 01** | -0.9269 | -0.7410 |
| **Percentil 05** | -0.6361 | -0.5752 |
| **Mediana (P50)** | +0.0127 | -0.0601 |
| **Percentil 95** | +0.4635 | +0.3957 |
| **Percentil 99** | +0.6251 | +0.5510 |
| **Máximo** | +1.7371 | +1.0375 |

- **Ratio de dispersión:** $\text{std}(\hat{R}) / \text{std}(R) = \mathbf{0.8708}$.
- **Correlación lineal:** Pearson $r = \mathbf{0.1425}$, Spearman $\rho = \mathbf{0.1498}$.
- **Coeficiente $R^2$ del residual:** $\mathbf{-0.5282}$.
- *Conclusión A:* El modelo comprime las colas extremas (máximo 1.04 vs 1.74 °C), pero genera una varianza excesiva en el cuerpo central que duplica el error de varianza de fondo.

---

## 5. Diagnóstico B — Calibración Lineal del Residual

Se ajustó la recta diagnóstica $R = a + b \cdot \hat{R} + \epsilon$:
- **Pendiente de calibración:** $b = \mathbf{0.1636}$
- **Intercepto:** $a = -0.0136\ ^\circ\text{C}$
- **$R^2$ de calibración:** $0.0203$
- *Interpretación Físico-Matemática:* Dado que $b = \frac{\text{Cov}(R, \hat{R})}{\text{Var}(\hat{R})} \approx 0.1636 \ll 1$, el Random Forest sobrestima la magnitud óptima de la corrección por un factor de aproximadamente:
  $$\frac{1}{b} \approx \frac{1}{0.1636} \approx \mathbf{6.1\times}$$
  Aplicar $\alpha = 1.0$ inyecta una varianza residual 6 veces mayor que la soportada por la covarianza empírica.

---

## 6. Diagnóstico C — Curva de Amortiguación Diagnóstica $\alpha$

Se evaluó la corrección escalada $\text{SST}_{\alpha} = \text{SST}_{\text{BIL}} + \alpha \cdot \hat{R}$:

| Factor $\alpha$ | RMSE (°C) | MAE (°C) | Bias (°C) | Mejora vs E0 (%) |
| :---: | :---: | :---: | :---: | :---: |
| **0.00 (E0)** | **0.3357** | **0.2636** | **+0.0251** | **0.00%** |
| 0.05 | 0.3337 | 0.2620 | +0.0216 | +0.59% |
| 0.10 | 0.3323 | 0.2609 | +0.0181 | +1.01% |
| 0.15 | 0.3317 | 0.2605 | +0.0146 | +1.18% |
| **0.17 (Mínimo RMSE)** | **0.3317** | **0.2605** | **+0.0132** | **+1.19%** |
| 0.20 | 0.3317 | 0.2607 | +0.0111 | +1.18% |
| 0.25 | 0.3324 | 0.2614 | +0.0076 | +0.97% |
| 0.30 | 0.3337 | 0.2627 | +0.0041 | +0.59% |
| 0.35 | 0.3358 | 0.2645 | +0.0006 | -0.04% |
| 0.50 | 0.3457 | 0.2728 | -0.0099 | -3.00% |
| **1.00 (RF Original)** | **0.4138** | **0.3232** | **-0.0448** | **-23.27%** |

- *Veredicto de Amortiguación (Caso A / C):*
  Existe una reducción modesta del RMSE (de 0.3357 a **0.3317 °C**, $+1.19\%$) cuando la señal se amortigua a $\alpha = 0.17$, coincidiendo con la pendiente de calibración $b = 0.1636$.
  Esto prueba que **la dirección de la corrección residual contiene señal física útil**, pero el modelo al aplicarse con $\alpha = 1.0$ destruye la solución por inflación de varianza.

---

## 7. Diagnóstico D — Brecha de Generalización TRAIN vs VALIDATION

| Métrica | Muestra TRAIN (N = 1,349,840) | VALIDATION Completa (N = 3,853,670) | Brecha Absoluta | Ratio Val/Train |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (°C)** | 0.1890 | 0.4138 | +0.2248 °C | **2.19×** |
| **MAE (°C)** | 0.1337 | 0.3232 | +0.1895 °C | **2.42×** |
| **$R^2$ Residual** | ~0.68 | -0.5282 | -1.21 | — |

- *Interpretación:* La duplicación del error ($> 2.19\times$) es testimonio de una combinación de:
  1. **Sobreajuste estructural:** Árboles profundos (`max_depth=20`, `min_samples_leaf=5`) memorizan fluctuaciones térmicas estacionales de 2015–2021.
  2. **Pérdida de generalización temporal:** Variaciones en el forzamiento climático de 2022–2023 no parametrizadas por el simple día del año (`doy_sin`, `doy_cos`).

---

## 8. Diagnóstico E — Variabilidad Temporal Mensual y Estacional

- **Días con mejora de RMSE:** **32.47%** (237 de 730 días).
- **Meses evaluados:** 24 meses continuos.
- **Patrón Estacional:**
  El deterioro de RF no es homogéneo en el año:
  - En meses de verano (julio–septiembre), cuando el residual medio es muy bajo ($|R| < 0.2\ ^\circ\text{C}$), el deterioro de RF es máximo ($\Delta\text{RMSE} \approx +0.10\ ^\circ\text{C}$).
  - En meses invernales y de transición (noviembre–febrero), cuando ingresan frentes fríos y el gradiente térmico se intensifica, el Random Forest se aproxima a E0 e incluso lo supera en episodios sinópticos específicos.

---

## 9. Diagnóstico F — Auditoría Espacial Celda por Celda

Se calculó $\Delta\text{RMSE} = \text{RMSE}_{\text{RF}} - \text{RMSE}_{\text{E0}}$ para las 5,279 celdas oceánicas:
- **Celdas con mejora real ($\Delta < 0$):** **0 celdas (0.0000%)**.
- **Celdas con deterioro ($\Delta > 0$):** **5,279 celdas (100.0000%)**.
- **Rango observado:** Mínimo $= +0.0368\ ^\circ\text{C}$ (celda 4795) | Máximo $= +0.1192\ ^\circ\text{C}$ (celda 2110).
- *Reconciliación de la Contradicción de D.2:*
  La afirmación previa de "mejoras en la costa y canal de Cozumel" era **metodológicamente incorrecta**. Dichas zonas experimentaron el **menor deterioro relativo** ($+0.037\ ^\circ\text{C}$ vs $+0.119\ ^\circ\text{C}$ en mar abierto), pero **ninguna celda mejoró a E0** en el balance bienal.

---

## 10. Diagnóstico G — Descomposición entre Régimen Central y Colas Extremas

Se evaluaron 4 grupos no solapados según la magnitud $|R|$ observada (umbrales TRAIN: P90 = 0.5435, P95 = 0.6698, P99 = 0.9810 °C):

| Régimen | N Observaciones | % Total | RMSE E0 (°C) | RMSE RF (°C) | $\Delta\text{RMSE}$ (°C) | Mejora RF (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1: $\|R\| < \text{P90}$ (Central)** | 3,467,973 | 89.99% | **0.2572** | 0.3697 | +0.1126 | **-43.77%** |
| **G2: $\text{P90} \le \|R\| < \text{P95}$** | 198,936 | 5.16% | 0.6001 | 0.6013 | +0.0012 | -0.20% |
| **G3: $\text{P95} \le \|R\| < \text{P99}$** | 159,101 | 4.13% | 0.7854 | **0.7321** | -0.0533 | **+6.79%** |
| **G4: $\|R\| \ge \text{P99}$ (Extremo)** | 27,660 | 0.72% | 1.1265 | **1.0173** | -0.1092 | **+9.69%** |

- *Hallazgo Clave:*
  - En el **90% de los datos** (G1), el error cuadrático se incrementa en un $+43.8\%$.
  - En el **5% superior** (G3 + G4), el Random Forest **supera consistentemente a E0**, alcanzando una reducción de error de hasta **$+9.69\%$**.
  - Este análisis es estrictamente retrospectivo condicional al residual observado.

---

## 11. Diagnóstico H — Comportamiento por Signo del Residual

| Condición | N Observaciones | % Total | RMSE E0 (°C) | RMSE RF (°C) | $\Delta\text{RMSE}$ (°C) | Mejora RF (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$R > 0$ (MUR > BIL)** | 1,987,552 | 51.58% | 0.3398 | 0.4077 | +0.0679 | -19.98% |
| **$R < 0$ (MUR < BIL)** | 1,865,958 | 48.42% | 0.3312 | 0.4202 | +0.0890 | -26.86% |

- El deterioro es mayor cuando OISST sobreestima a MUR ($R < 0$, $-26.9\%$), coincidiendo con falsas alarmas cálidas en la plataforma costera.

---

## 12. Diagnóstico I — Feature Importance: MDI vs Permutación

- **Scorer de Permutación:** $R^2$ en muestra independiente de 100,000 registros de VALIDATION (`random_state=42`, 5 repeticiones).
- **Resultados:**
  - `sst_bil`: MDI $= 0.3557$, Permutación $= 3.2731$.
  - `doy_sin`: MDI $= 0.2440$, Permutación $= 1.0106$.
  - `doy_cos`: MDI $= 0.2105$, Permutación $= 0.1540$.
  - `depth`: MDI $= 0.1076$, Permutación $= 0.0039$.
  - `distance_coast_km`: MDI $= 0.0820$, Permutación $= -0.0072$.
  - `ocean_fraction`: MDI $= 0.0002$, Permutación $= 0.0000$.
- *Interpretación:*
  Las variables espaciales estáticas (`depth`, `distance_coast_km`) tienen una importancia por permutación prácticamente nula ($\le 0.004$). El árbol divide nodos internamente basándose en la batimetría durante el entrenamiento, pero estas divisiones no aportan generalización marginal real en validación.

---

## 13. Diagnóstico J — Relación con `analysis_error`

El deterioro relativo de RF frente a E0 se mantiene casi uniforme en todos los niveles de incertidumbre MUR (~$-20\%$ a $-24\%$). Esto confirma que el fallo del modelo **no es un artefacto de baja calidad observacional en MUR**, sino una limitación propia de la capacidad de representación del modelo pixel-wise.

---

## 14. Síntesis y Evaluación Formal de Hipótesis (H1–H5)

| Hipótesis | Enunciado | Dictamen | Justificación Cuantitativa |
| :--- | :--- | :---: | :--- |
| **H1** | *RF aprende una señal útil, pero sobreestima la magnitud de la corrección.* | **APOYADA** | Pendiente de calibración $b = 0.1636 \ll 1$. Con amortiguación $\alpha = 0.17$, el RMSE se reduce a $0.3317\ ^\circ\text{C}$ ($+1.19\%$ sobre E0), demostrando que $\hat{R}$ tiene dirección útil pero magnitud $\sim 6\times$ inflada en $\alpha=1.0$. |
| **H2** | *RF presenta sobreajuste y/o pérdida de generalización temporal TRAIN -> VAL.* | **APOYADA** | La brecha TRAIN ($0.1890\ ^\circ\text{C}$) vs VAL ($0.4138\ ^\circ\text{C}$) es de $2.19\times$ en RMSE y $2.42\times$ en MAE. |
| **H3** | *La señal aprendida es demasiado débil con las 6 variables pixel-wise disponibles.* | **APOYADA** | $R^2$ del residual en validación es negativo ($-0.5282$) y Pearson $r = 0.1425$. Incluso con calibración óptima, la mejora máxima alcanzable es solo de $+1.19\%$. Se carece de gradientes espaciales 2D. |
| **H4** | *La mejora observada retrospectivamente en grandes $\|R\|$ existe, pero perjudica el régimen central.* | **APOYADA** | En el 90% de los datos (G1), RF empeora el RMSE en $-43.77\%$, mientras que en la cola extrema $\|R\| \ge \text{P99}$ (G4) RF reduce el RMSE en $+9.69\%$. |
| **H5** | *La supuesta mejora espacial previa es incorrecta (menor deterioro, no mejora real).* | **APOYADA** | Exactamente 0 de las 5,279 celdas oceánicas ($0.0000\%$) presentan $\Delta\text{RMSE} < 0$. La costa exhibe un menor deterioro relativo ($+0.037\ ^\circ\text{C}$ vs $+0.119\ ^\circ\text{C}$), pero ninguna mejora real. |

---

## 15. Limitaciones del Diagnóstico

1. Las curvas de amortiguación $\alpha$ y las regresiones de calibración se obtuvieron sobre VALIDATION con fines estrictamente diagnósticos; **no constituyen un modelo válido para ser evaluado en TEST**.
2. Los análisis por estratos de $|R|$ son retrospectivos condicionales a la verdad observada, no una regla operativa para inferencia.

---

## 16. Recomendación Metodológica Fundamentada

### **Recomendación: C. Pasar a E3 XGBoost Residual (con regularización estricta y calibración de encogimiento)**

**Justificación:**
1. **Descarte de D (Saltar directamente a CNN sin agotar baselines tabulares):** Aún es necesario establecer el benchmark tabular óptimo mediante un algoritmo con regularización $L_1/L_2$ explícita y tasa de aprendizaje controlada (learning rate / shrinkage), características nativas de Gradient Boosting (XGBoost/LightGBM) que contrarrestan directamente la sobreestimación de varianza observada en Random Forest ($b \approx 0.16$).
2. **Descarte de A (Mantener RF sin cambios):** Random Forest sin regularización de contracción (shrinkage) continuará sobreestimando la amplitud de corrección en hojas con pocas muestras.
3. **Descarte de B (Ablación extensiva de RF):** Dado que la importancia por permutación demostró que `depth`, `distance_coast_km` y `ocean_fraction` son casi inertes, XGBoost puede manejar automáticamente la selección de variables mediante regularización sin requerir una búsqueda exhaustiva en RF.
4. **Descarte de E (Problema anterior en los datos):** La auditoría de Fase C.2 y D.1 demostró que los datos son coherentes y sin sesgos espaciotemporales espurios; la falla observada es estrictamente de modelado y calibración de varianza.

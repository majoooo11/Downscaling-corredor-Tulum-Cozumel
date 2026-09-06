# Reporte Científico — Fase D.3.4
## Post-Validation Diagnostic Audit of Frozen E3b-C0
### Versión Final Corregida

---

## 1. Objetivo

Auditar de manera no adaptativa y descriptiva el comportamiento del modelo congelado `E3b-C0` en el periodo de validación temporal **2022–2023**, investigando las características estadísticas asociadas con la variabilidad mensual del *skill* observada en la Fase D.3.3 (**16/24 meses con mejora**, dictamen formal **D33-B**).

La Fase D.3.4 no tiene como finalidad mejorar el modelo ni utilizar los resultados de VALIDATION para modificar su formulación. Su propósito es caracterizar el comportamiento del estimador ya congelado y verificar la robustez de las conclusiones obtenidas en D.3.3 antes de preparar el protocolo de FINAL TEST.

---

## 2. Estado Heredado de D.3.3

- **Dictamen D33 Formal e Inmutable:** `D33-B — PARTIAL / MIXED GENERALIZATION`.
- **Interpretación canónica:** *Positive external generalization with insufficient month-level stability to satisfy the predeclared D33-A criterion.*
- **Modelo congelado:** `E3b-C0`.
- **Features congeladas:** `['sst_bil', 'doy_sin', 'doy_cos', 'depth']`.
- **Hiperparámetros congelados:** `max_depth=4`, `learning_rate=0.10`, `n_estimators=19`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `tree_method='hist'`, `objective='reg:squarederror'`.
- **Hashes SHA256 inmutables:**
  - Modelo: `fb151cafd466613d0bda2cadfee61e9ebc1f8e0f889d9a646cc010ba6436ef5d`
  - Celdas congeladas: `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb`

---

## 3. Principio de No Adaptación

La Fase D.3.4 tiene carácter **estrictamente diagnóstico y observacional**.

No se realizó:

- reentrenamiento mediante `fit()` o `train()`;
- ajuste de hiperparámetros;
- modificación de umbrales;
- adición o eliminación de variables;
- evaluación de formulaciones alternativas;
- modificación de la máscara espacial;
- exclusión de celdas a partir del comportamiento observado durante VALIDATION.

La pregunta rectora de esta fase fue exclusivamente:

> **¿Qué características estadísticas explican el comportamiento observado del modelo ya congelado en VALIDATION 2022–2023?**

La partición FINAL TEST correspondiente a **2024–2025** permaneció completamente cerrada durante todo el análisis.

---

## 4. Integridad del Modelo y Celdas Congeladas

- `MODEL HASH VERIFIED`: **YES**
- `FROZEN CELLS HASH VERIFIED`: **YES**
- Número de celdas congeladas: **5,275**
- `MODEL MODIFIED`: **NO**
- `MODEL RETRAINED`: **NO**
- `TEST_FILES_OPENED_COUNT`: **0**

Los hashes SHA256 coincidieron exactamente con los artefactos congelados durante D.3.3, confirmando que D.3.4 utilizó el mismo modelo y el mismo dominio espacial.

---

## 5. Reproducción Exacta de Métricas D33

El pipeline reprodujo, sin redondeos intermedios relevantes, los resultados centrales de VALIDATION 2022–2023:

- **Baseline B0 RMSE:** 0.335666 °C
- **Modelo C0 RMSE:** 0.323838 °C
- **Mejora global de RMSE:** **+3.5237%**
- **Meses con mejora:** **16 / 24** (66.7%)
- **Celdas con mejora:** **4,755 / 5,275** (90.14%)
- `BUG DETECTED`: **NO**

La reproducción de D.3.3 confirma que los análisis diagnósticos posteriores no parten de cambios en el modelo ni en el procesamiento.

---

## 6. Identidad Algebraica SST–Residual

El modelo utiliza:

\[
R = SST_{MUR} - SST_{BIL}
\]

y reconstruye la SST mediante:

\[
\widehat{SST} = SST_{BIL} + \hat{R}.
\]

Por tanto:

\[
\widehat{SST} - SST_{MUR}
=
(SST_{BIL}+\hat{R}) - SST_{MUR}
=
\hat{R} - R.
\]

En consecuencia:

\[
RMSE(\widehat{SST}, SST_{MUR})
\equiv
RMSE(\hat{R},R),
\]

y análogamente:

\[
MAE(\widehat{SST}, SST_{MUR})
\equiv
MAE(\hat{R},R).
\]

Esto implica que RMSE y MAE calculados en el espacio residual no constituyen métricas independientes de las correspondientes a la SST reconstruida.

Las métricas adicionales de interés son principalmente:

- \(R^2\) residual;
- correlación entre \(R\) y \(\hat{R}\);
- precisión de signo;
- calibración de amplitud;
- relación entre \(\mathrm{std}(\hat{R})\) y \(\mathrm{std}(R)\).

---

## 7. Métricas Directas del Residual

| Periodo | RMSE Residual (°C) | MAE Residual (°C) | Bias Residual (°C) | \(R^2\) Residual | Pearson \(r(R,\hat{R})\) | Spearman \(\rho\) | \(\mathrm{std}(\hat{R})/\mathrm{std}(R)\) | Pendiente \(b\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **2022** | 0.314512 | 0.241482 | -0.005682 | **0.043928** | 0.2140 | 0.2655 | 0.1744 | 0.0373 |
| **2023** | 0.332904 | 0.267432 | +0.004742 | **0.066461** | 0.2627 | 0.3680 | 0.3116 | 0.0819 |
| **2022–2023** | 0.323838 | 0.254457 | -0.000470 | **0.064029** | 0.2534 | 0.3410 | 0.2672 | 0.0677 |

El \(R^2\) residual combinado es bajo:

\[
R^2_R = 0.0640,
\]

mientras que Pearson:

\[
r = 0.2534
\]

y Spearman:

\[
\rho = 0.3410
\]

muestran una asociación positiva modesta entre el residual observado \(R\) y el residual predicho \(\hat{R}\).

Este resultado es compatible con una mejora global de la reconstrucción de SST del **3.52%**, dado que el baseline bilineal ya presenta un \(R^2\) de SST cercano a 0.90 y reproduce gran parte de la variabilidad de gran escala antes de introducir la corrección residual.

### Compresión de amplitud

El cociente:

\[
\frac{\mathrm{std}(\hat{R})}{\mathrm{std}(R)}
=
0.2672
\]

y la pendiente descriptiva:

\[
b = 0.0677
\]

muestran una fuerte compresión de la amplitud predicha.

Este comportamiento es consistente con:

> *a pattern of regression toward the conditional mean in the MSE-trained and regularized model.*

No implica que MSE sea por sí solo la causa del *shrinkage*, ya que este comportamiento también depende de la regularización, la capacidad del modelo y el contenido informativo de los predictores disponibles.

---

## 8. Bias por Año

- **2022:** Bias B0 = -0.005902 °C; Bias C0 = **-0.005682 °C**
- **2023:** Bias B0 = +0.055959 °C; Bias C0 = **+0.004742 °C**
- **2022–2023 combinado:** Bias C0 = **-0.000470 °C**

El sesgo combinado cercano a cero refleja parcialmente compensación entre sesgos anuales pequeños de signo opuesto.

En 2022, el bias permaneció prácticamente sin cambios:

\[
-0.005902
\rightarrow
-0.005682^\circ C,
\]

mientras que en 2023 se produjo una reducción considerable:

\[
+0.055959
\rightarrow
+0.004742^\circ C.
\]

Por tanto, la mejora del bias no fue homogénea entre ambos años.

---

## 9. Auditoría de los 8 Meses Negativos

Los ocho meses con degradación relativa fueron:

- `2022-06`: -3.17%
- `2022-09`: -1.12%
- `2022-10`: -11.78%
- `2022-12`: -4.07%
- `2023-01`: -2.81%
- `2023-06`: -23.55%
- `2023-10`: -1.02%
- `2023-11`: -0.64%

El patrón estadístico más consistente fue una menor concordancia entre el signo del residual predicho y el signo del residual observado MUR–BIL.

> *Negative-skill months were characterized by reduced agreement between the sign of the predicted residual and the sign of the observed MUR–BIL residual.*

La precisión media de signo durante los meses negativos fue aproximadamente **47.6%**, frente a valores superiores al **65%** en los meses con mejora.

Este resultado describe una asociación diagnóstica y no atribuye el deterioro a una feature específica ni a un mecanismo oceanográfico particular.

---

## 10. Diagnóstico Mayo–Agosto 2023

Se auditó diariamente el periodo mayo–agosto de 2023 para investigar el contraste entre:

\[
2023\text{-}06 = -23.55\%
\]

y:

\[
2023\text{-}07 = +18.58\%.
\]

La auditoría confirmó:

- 123 días cronológicos continuos;
- 5,275 celdas por día;
- cero duplicados;
- cero NaN inesperados;
- continuidad correcta del DOY;
- continuidad de `doy_sin` y `doy_cos`;
- profundidad GEBCO constante para cada celda;
- mismo modelo;
- mismo preprocessing.

No se identificó ninguna discontinuidad computacional o de procesamiento.

> *June 2023 was characterized by a residual distribution for which the frozen model exhibited poor sign agreement and negative skill, whereas July showed larger MUR–BIL discrepancies with substantially better residual-sign agreement and positive skill.*

El contraste se interpreta como **variabilidad temporal del desempeño del modelo**, sin atribución causal física.

---

## 11. Población por Regímenes Definidos en DEVELOPMENT

| Régimen | Umbral \(|R|\) | N | % VALIDATION | Media \(|R|\) | Mediana \(|R|\) | P90 \(|R|\) |
|---|---:|---:|---:|---:|---:|---:|
| **DEV-P0-P50** | <0.2066 °C | 1,827,764 | 47.47% | 0.0994 | 0.0976 | 0.1833 |
| **DEV-P50-P75** | 0.2066–0.3604 °C | 994,741 | 25.83% | 0.2781 | 0.2754 | 0.3414 |
| **DEV-P75-P90** | 0.3604–0.5377 °C | 630,094 | 16.36% | 0.4374 | 0.4319 | 0.5117 |
| **DEV-P90-P95** | 0.5377–0.6652 °C | 206,608 | 5.37% | 0.5938 | 0.5900 | 0.6474 |
| **DEV-P95-P99** | 0.6652–0.9659 °C | 161,143 | 4.18% | 0.7742 | 0.7579 | 0.8985 |
| **DEV-P99+** | ≥0.9659 °C | 30,400 | 0.79% | 1.1060 | 1.0673 | 1.2931 |

La distribución de magnitudes residuales durante VALIDATION **se asemeja estrechamente** a la partición percentilar definida previamente en DEVELOPMENT.

En particular, el régimen DEV-P0-P50 contiene el **47.47%** de las observaciones, cercano al 50% asociado con el umbral de DEVELOPMENT.

Esta semejanza es descriptiva y no constituye una prueba formal de igualdad de distribuciones.

---

## 12. Precisión de Signo del Residual

La precisión de signo aumenta conforme crece la magnitud de la discrepancia MUR–BIL:

- DEV-P0-P50: **55.81%**
- DEV-P50-P75: **62.37%**
- DEV-P75-P90: **66.45%**
- DEV-P90-P95: **73.75%**
- DEV-P95-P99: **80.45%**
- DEV-P99+: **88.87%**

El régimen DEV-P0-P50 presenta, por tanto, una **baja precisión de signo residual (55.81%)**.

No se interpreta este valor como equivalente al azar, dado que esa comparación requeriría considerar explícitamente la prevalencia de residuales positivos y negativos y establecer un baseline de clasificación apropiado.

---

## 13. Sobre-Corrección y Sub-Corrección

En DEV-P0-P50:

- Sobre-corrección:

\[
23.80\%
\]

- Sub-corrección:

\[
76.20\%
\]

- Igualdad de magnitud:

\[
0.0002\%
\]

correspondiente a 3 observaciones.

Además:

\[
D_{mag}=|\hat{R}|-|R|
\]

presentó una media de:

\[
-0.0475^\circ C.
\]

Esto confirma que el deterioro del régimen de bajo residual **no se debe principalmente a una sobre-corrección generalizada de magnitud**.

La principal limitación es la discriminación del signo residual, combinada con una contribución menor de sobre-corrección.

---

## 14. Descomposición del Régimen DEV-P0-P50

| Categoría | N | % | RMSE B0 | RMSE C0 | \(\Delta RMSE\) | Media \(|R|\) | Media \(|\hat{R}|\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A. Signo correcto + sub-corrección** | 783,240 | 42.85% | 0.1283 | 0.1018 | -0.0264 | 0.1164 | 0.0295 |
| **B. Signo correcto + sobre-corrección** | 236,821 | 12.96% | 0.0784 | 0.1153 | +0.0369 | 0.0579 | 0.1336 |
| **C. Signo incorrecto + magnitud pequeña** | 761,763 | 41.68% | 0.1121 | 0.1481 | +0.0360 | 0.0952 | 0.0352 |
| **D. Signo incorrecto + magnitud grande** | 45,778 | 2.50% | 0.1112 | 0.3932 | +0.2820 | 0.0952 | 0.2901 |
| **E. Signo cero / igualdad** | 162 | 0.01% | 0.0001 | 0.1139 | +0.1138 | 0.0000 | 0.0636 |

Las categorías C y D representan conjuntamente:

\[
41.68 + 2.50
=
\boxed{44.18\%}
\]

de las observaciones del régimen, es decir, casos en los que el residual fue predicho con signo incorrecto.

En la categoría C, que representa el **41.68%** de los datos, la corrección predicha es pequeña:

\[
E(|\hat{R}|)=0.0352^\circ C,
\]

pero su dirección incorrecta hace que el error de reconstrucción sea mayor que el de B0.

Por tanto, el mecanismo estadístico dominante del deterioro en DEV-P0-P50 es la **imprecisión de signo cuando la discrepancia MUR–BIL es pequeña**, y no una sobre-corrección masiva de amplitud.

---

## 15. Diagnóstico de Grandes Discrepancias MUR–BIL

La mejora relativa aumenta conforme se consideran regímenes de mayor \(|R|\):

- DEV-P50-P75: **+1.93%**
- DEV-P75-P90: **+4.65%**
- DEV-P90-P95: **+5.86%**
- DEV-P95-P99: **+6.77%**
- DEV-P99+: **+7.45%**

La proporción de días con mejora también aumenta:

\[
63.2\%,\ 67.2\%,\ 72.1\%,\ 78.0\%,\ 82.1\%.
\]

El régimen DEV-P99+ contiene **30,400 observaciones**, distribuidas en **623 días** y en las **5,275 celdas espaciales**.

> *The DEV-P99+ result has broad temporal and spatial support despite representing only 0.79% of validation observations.*

Por tanto, la ganancia observada para grandes discrepancias MUR–BIL no depende únicamente de unos pocos días o celdas aisladas.

---

## 16. Sensibilidad Bootstrap: 1, 7 y 14 días

| Bootstrap | \(L\) | Mediana \(\Delta RMSE\) | CI95 inferior | CI95 superior | \(P(\Delta RMSE<0)\) | Tail fraction bilateral |
|---|---:|---:|---:|---:|---:|---:|
| **1-Day Cluster** | 1 | -0.011902 | -0.016446 | **-0.007224** | 1.000 | 0.001998 |
| **7-Day Moving Block** | 7 | -0.011774 | -0.021985 | **-0.002782** | 0.989 | 0.023976 |
| **14-Day Moving Block** | 14 | -0.011778 | -0.023610 | **-0.001051** | 0.978 | 0.045954 |

Al aumentar la longitud de los bloques se amplía el intervalo de incertidumbre, pero el límite superior permanece negativo tanto para 7 como para 14 días.

Por tanto:

> **The global RMSE improvement is robust to short-range temporal dependence under the evaluated bootstrap block lengths.**

Más específicamente:

> *The results support the robustness of the global validation improvement to short-range temporal dependence up to the evaluated 14-day block length.*

Esta conclusión no se extrapola a escalas temporales superiores a 14 días.

La `Tail fraction bilateral` se interpreta como una medida bootstrap descriptiva y no como un valor \(p\) inferencial exacto bajo una hipótesis nula formal.

---

## 17. Diagnóstico Espacial Descriptivo

### 17.1 Estratificación por profundidad

| Profundidad | N celdas | % celdas mejoradas | Mediana \(\Delta RMSE\) |
|---|---:|---:|---:|
| **0–20 m** | 965 | 96.27% | -0.0184 °C |
| **20–50 m** | 417 | 95.68% | -0.0116 °C |
| **50–100 m** | 232 | 99.14% | -0.0103 °C |
| **100–500 m** | 1,363 | 99.78% | -0.0132 °C |
| **>500 m** | 2,298 | 79.94% | -0.0039 °C |

La definición utilizada es:

\[
\Delta RMSE=
RMSE_{C0}-RMSE_{B0},
\]

por lo que valores negativos representan una reducción del error.

Las aguas comprendidas entre 0 y 500 m presentan mejoras medianas relativamente elevadas y más del 95% de sus celdas mejoran.

En cambio, el dominio superior a 500 m conserva una mejora mediana negativa, pero de menor magnitud absoluta.

### 17.2 Profundidad y skill

La correlación global fue:

\[
\rho(\text{water depth},\Delta RMSE)
=
+0.6312.
\]

Esto representa una **asociación positiva de rango considerable**: conforme aumenta la profundidad, \(\Delta RMSE\) tiende a hacerse más positivo, lo que equivale a una menor ganancia relativa de C0.

Sin embargo, esta asociación global está fuertemente influida por el dominio profundo, especialmente por las celdas con profundidades superiores a 500 m.

Restringiendo el análisis al intervalo:

\[
0-500\text{ m},
\]

se obtiene:

\[
\rho=+0.1050,
\]

que corresponde a una asociación débil.

Además, el patrón observado entre los distintos estratos de 0–500 m no es estrictamente monotónico.

Por tanto:

> **Across the full spatial domain, water depth shows a substantial positive rank association with \(\Delta RMSE\), indicating progressively smaller gains toward the deep basin. This relationship is largely driven by cells deeper than 500 m; within the 0–500 m platform-and-slope domain, the association is weak and the depth-stratified pattern is non-monotonic. No causal bathymetric interpretation is inferred.**

### 17.3 Distancia a costa

Se obtuvo:

\[
\rho(\text{distance\_coast\_km},\Delta RMSE)
=
+0.3992.
\]

Esto corresponde a una **asociación positiva moderada** en el dominio completo: conforme aumenta la distancia a costa, la ganancia relativa de C0 tiende a disminuir.

Sin embargo, dentro de la franja costera:

\[
distance<20\text{ km},
\]

la correlación es:

\[
\rho=+0.0556,
\]

es decir, prácticamente nula.

Por tanto:

> **Across the full domain, distance from the coast shows a moderate positive rank association with \(\Delta RMSE\), whereas within the coastal band (<20 km) the association is negligible.**

Profundidad y distancia a costa se interpretan de manera independiente y no se asume que representen el mismo gradiente espacial ni que tengan efectos causales sobre el desempeño.

---

## 18. Correcciones de Interpretación Científica

La auditoría establece las siguientes reglas terminológicas para la interpretación de los resultados:

1. Utilizar **low-residual regime** en lugar de *low-gradient regime*.
2. Utilizar **small MUR–BIL discrepancies** en lugar de *weak submesoscale anomalies*.
3. Interpretar el *shrinkage* como:

   > *a pattern consistent with regression toward the conditional mean in the MSE-trained and regularized model.*

4. Utilizar **spatially heterogeneous performance** o **spatially heterogeneous degradation** sin atribuir causalidad a la batimetría.
5. Describir 2022–2023 como:

   > *out-of-development temporal validation under the frozen D33 protocol.*

6. No interpretar una precisión de signo de 55.81% como equivalente al azar sin disponer de un baseline basado en la prevalencia de clases.
7. No utilizar la magnitud del residual MUR–BIL como equivalente automático de un evento térmico extremo, una ola de calor marina o una estructura submesoescala.

---

## 19. Dictamen D33 Heredado

El dictamen formal de D.3.3 permanece:

\[
\boxed{\text{D33-B — UNCHANGED}}
\]

> *Positive external generalization with insufficient month-level stability to satisfy the predeclared D33-A criterion.*

D.3.4 no modifica ni reclasifica este resultado.

---

## 20. Recomendación respecto a FINAL TEST

La regla predeclarada para recomendar la preparación del FINAL TEST requería:

1. `MODEL HASH VERIFIED = YES`
2. `FROZEN CELLS HASH VERIFIED = YES`
3. reproducción de D33;
4. ausencia de bugs de datos o preprocessing;
5. ausencia de discontinuidades computacionales junio–julio;
6. \(CI95_{upper}<0\) para bootstrap de 7 días;
7. \(CI95_{upper}<0\) para bootstrap de 14 días.

Todos los criterios fueron satisfechos.

Para 7 días:

\[
CI95_{7d}
=
[-0.021985,-0.002782]^\circ C.
\]

Para 14 días:

\[
CI95_{14d}
=
[-0.023610,-0.001051]^\circ C.
\]

Por tanto:

\[
\boxed{\textbf{RECOMENDACIÓN FORMAL: PREPARE FINAL TEST}}
\]

Esta recomendación **no constituye autorización automática para abrir 2024–2025**.

La apertura deberá realizarse mediante un protocolo final independiente y congelado antes de inspeccionar cualquier resultado del periodo TEST.

---

## 21. Limitaciones

### 21.1 Month-level skill variability

Aunque el modelo presenta skill positivo global, anual y espacial, su desempeño mensual varía considerablemente.

Los meses negativos se caracterizan por una menor concordancia entre el signo de \(\hat{R}\) y el signo del residual MUR–BIL observado.

### 21.2 Low-residual regime

Cuando:

\[
|R|<0.2066^\circ C,
\]

B0 ya se encuentra próximo a MUR y la utilidad de la corrección ML disminuye.

En este régimen, el **44.18%** de las observaciones presenta error de signo residual, lo que puede incrementar el error incluso cuando la magnitud de \(\hat{R}\) es pequeña.

### 21.3 Short-range temporal dependence

El moving-block bootstrap muestra intervalos más amplios al pasar de bloques de 1 día a 7 y 14 días.

Esto refleja mayor incertidumbre cuando se preserva dependencia temporal de corto alcance, aunque el intervalo completo continúa favoreciendo a C0 hasta la escala de 14 días evaluada.

### 21.4 Limitada explicación de la varianza residual

El modelo presenta:

\[
R^2_R=0.0640,
\]

lo que indica que explica únicamente una fracción limitada de la varianza del residual MUR–BIL, pese a producir una reducción reproducible del error de reconstrucción de SST.

### 21.5 Heterogeneidad espacial del skill

La mejora no es espacialmente uniforme.

Aunque el **90.14%** de las celdas mejora, la magnitud del beneficio tiende a reducirse en la cuenca profunda y, en el dominio completo, también presenta asociación con una mayor distancia a costa.

Estas relaciones son descriptivas y no implican causalidad batimétrica u oceanográfica.

---

## 22. Microauditoría Final de Interpretación

### A. Convención de \(\Delta RMSE\)

Se verificó:

\[
\Delta RMSE=
RMSE_{C0}-RMSE_{B0}.
\]

Por tanto:

\[
\Delta RMSE<0
\]

indica mejora de C0.

Todas las tablas y figuras utilizan esta misma convención.

### B. Profundidad GEBCO

La variable `depth` está codificada como profundidad positiva bajo el nivel del mar:

\[
1.0\le depth\le1435.8\text{ m}.
\]

La mediana es:

\[
410.5\text{ m}.
\]

La variable diagnóstica `water_depth_m` es idéntica a `depth`.

### C. Reconciliación espacial

La correlación global:

\[
\rho=+0.6312
\]

muestra una asociación de rango considerable entre mayor profundidad y menor beneficio de C0.

Sin embargo, restringiendo el dominio a profundidades inferiores a 500 m:

\[
\rho=+0.1050,
\]

por lo que la relación dentro de plataforma y talud es débil.

Los resultados por bins y la correlación global son compatibles: la principal diferencia espacial se observa entre el dominio 0–500 m y la cuenca profunda >500 m.

### D. Régimen DEV-P0-P50

Se verificó:

\[
41.68\%+2.50\%
=
44.18\%
\]

de observaciones con signo residual incorrecto.

La categoría C presenta:

\[
E(|\hat{R}|)=0.0352^\circ C.
\]

La principal limitación del modelo en este régimen es la discriminación de signo y no una sobre-corrección generalizada.

### E. Bias

El bias combinado cercano a cero se beneficia parcialmente de sesgos anuales de signo opuesto.

La corrección fue prácticamente nula en 2022 y considerable en 2023.

### F. Meses negativos y junio–julio

No se identificaron discontinuidades computacionales ni de preprocessing.

Las diferencias corresponden a variabilidad temporal del skill asociada con cambios en la distribución del residual y en la precisión de signo.

### G. Bootstrap

La mejora global permanece respaldada por intervalos completamente negativos para bloques de 7 y 14 días.

La conclusión se limita explícitamente a la dependencia temporal evaluada hasta 14 días.

### H. Dictamen Final de Integridad

- Hash del modelo: **VERIFICADO**
- Hash de celdas: **VERIFICADO**
- Métricas D33: **REPRODUCIDAS**
- Bugs de cálculo: **NO DETECTADOS**
- Modelo modificado: **NO**
- Modelo reentrenado: **NO**
- TEST 2024–2025 abierto: **NO**

Estado D34:

\[
\boxed{\textbf{INTERPRETATIONALLY CLOSED}}
\]

Recomendación:

\[
\boxed{\textbf{PREPARE FINAL TEST}}
\]

---

## 23. Catálogo de Entregables

Directorio:

`DATASET_TESIS/ml_results/E3b_D34_postvalidation_diagnostics/`

### Tablas

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
14. `tables/spatial_depth_microaudit.csv`
15. `tables/d34_microaudit_changes.csv`

### Figuras

1. `figures/fig_d34_1_monthly_improvement_negative_months.png`
2. `figures/fig_d34_2_daily_delta_rmse_rolling7d.png`
3. `figures/fig_d34_3_may_august_2023_daily_diagnostics.png`
4. `figures/fig_d34_4_sign_accuracy_by_regime.png`
5. `figures/fig_d34_5_over_under_correction_by_regime.png`
6. `figures/fig_d34_6_improvement_by_regime_with_counts.png`
7. `figures/fig_d34_7_bootstrap_sensitivity_comparison.png`
8. `figures/fig_d34_8_spatial_delta_rmse_vs_depth.png`
9. `figures/fig_d34_8b_depth_vs_delta_rmse_scatter_lowess.png`

---

# Estado Final de la Fase

\[
\boxed{\text{D.3.4 — INTERPRETATIONALLY CLOSED}}
\]

\[
\boxed{\text{D33-B — UNCHANGED}}
\]

\[
\boxed{\text{PREPARE FINAL TEST}}
\]

\[
\boxed{\text{FINAL TEST 2024–2025 — STILL CLOSED}}
\]
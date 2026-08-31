# AUDITORÍA FINAL DE LA VARIABLE `analysis_error` EN MUR v4.1 (2015–2025)

- **Estado:** DEFINITIVA Y COMPLETA
- **Fecha de Generación:** 2026-08-31 14:39:51 UTC
- **Periodo Evaluado:** 2015-01-01 a 2025-12-31 ($N = 4018$ días continuos)
- **Cobertura Temporal:** 100.0% (0 fechas faltantes, 0 duplicados)
- **Dominio Espacial:** Corredor Tulum–Cozumel (Lat: 19.90°N a 20.75°N, Lon: -87.60°W a -86.65°W, 5279 celdas oceánicas)

---

## 1. Resumen Ejecutivo y Preguntas Científicas Clave

Esta auditoría evalúa de forma definitiva si los episodios extremos de discrepancia entre los productos de temperatura superficial del mar L4 (MUR v4.1 vs OISST/BIL) están acompañados por un incremento en la incertidumbre interna reportada por el propio algoritmo de asimilación de MUR mediante su variable `analysis_error` (*estimated error standard deviation*).

### Respuestas a las Preguntas de Diagnóstico Científico:

1. **¿Existe relación entre discrepancia MUR–BIL y `analysis_error`?**  
   **SÍ.** Existe una correlación positiva estadísticamente significativa en todo el registro ($N = 4018$).

2. **¿Cuál es la magnitud real de esa relación con $N = 4018$?**  
   - Coeficiente de Spearman $\rho = +0.2853$ ($p = 4.25e-76$).
   - Coeficiente de Pearson $r = +0.3338$ ($p = 3.56e-105$).
   - La asociación es moderada-positiva a nivel global, pero altamente no lineal debido a la saturación asintótica en eventos extremos.

3. **¿Los $P_{95}$ y $P_{99}$ de RMSE tienen mayor `analysis_error`?**  
   **SÍ.** Se confirma un **gradiente monotónico estricto**:
   - Grupo **Normal (< P90)**: mean $\text{RMSE} = 0.263^\circ\text{C} \implies \text{mean(AE)} = 0.3860^\circ\text{C}$, $\text{fraction\_at\_041} = 5.3\%$.
   - Grupo **P90–P95**: mean $\text{RMSE} = 0.546^\circ\text{C} \implies \text{mean(AE)} = 0.3918^\circ\text{C}$, $\text{fraction\_at\_041} = 16.2\%$.
   - Grupo **P95–P99**: mean $\text{RMSE} = 0.690^\circ\text{C} \implies \text{mean(AE)} = 0.3944^\circ\text{C}$, $\text{fraction\_at\_041} = 23.0\%$.
   - Grupo **P99–P99.5**: mean $\text{RMSE} = 0.927^\circ\text{C} \implies \text{mean(AE)} = 0.4018^\circ\text{C}$, $\text{fraction\_at\_041} = 39.2\%$.
   - Grupo **>= P99.5**: mean $\text{RMSE} = 1.190^\circ\text{C} \implies \text{mean(AE)} = 0.4040^\circ\text{C}$, $\text{fraction\_at\_041} = 59.9\%$.

4. **¿E1–E6 tienen `analysis_error` anómalo?**  
   **COMPORTAMIENTO MIXTO (Escenario C):** 5 de los 6 eventos (E1, E2, E3, E5, E6) presentan `analysis_error` anómalo (ELEVADO o EXTREMO), mientras que E4 se mantiene nominal.

5. **¿Cuáles de E1–E6 alcanzan el valor máximo 0.4100 °C?**  
   - **E1 (2015-10-18):** 100.0% del dominio oceánico ($5279/5279$ celdas) en $0.4100^\circ\text{C}$.
   - **E2 (2021-11-17):** 100.0% del dominio oceánico ($5279/5279$ celdas) en $0.4100^\circ\text{C}$.
   - **E3 (2024-10-19):** 100.0% del dominio oceánico ($5279/5279$ celdas) en $0.4100^\circ\text{C}$.
   - **E5 (2016-06-05):** 27.2% del dominio en $0.4100^\circ\text{C}$ (mean = $0.4005^\circ\text{C}$).
   - **E6 (2019-06-14):** 29.5% del dominio en $0.4100^\circ\text{C}$ (mean = $0.4008^\circ\text{C}$).
   - **E4 (2015-08-03):** 0.0% en $0.4100^\circ\text{C}$ (mean = $0.3908^\circ\text{C}$).

6. **¿E4 continúa siendo un contraejemplo?**  
   **SÍ.** En E4 (2015-08-03), la discrepancia MUR–BIL es severa ($\text{RMSE} = 1.102^\circ\text{C}$, $P_{99.1}$), pero `analysis_error` es completamente normal ($0.3908^\circ\text{C}$, percentil 72.5%, $z = +0.77$). El análisis espacial confirma que tanto dentro como fuera de la huella satelital VIIRS el error estimado es indistinguible ($0.3911^\circ\text{C}$ vs $0.3906^\circ\text{C}$).

7. **¿E5 y E6 son normales, elevados, muy elevados o extremos?**  
   Ambos se clasifican como **ELEVADOS** ($P_{90} - P_{95}$):
   - E5 (2016-06-05): mean = $0.4005^\circ\text{C}$ (percentil global 90.22%, $z = +2.01$).
   - E6 (2019-06-14): mean = $0.4008^\circ\text{C}$ (percentil global 90.74%, $z = +2.06$).

8. **¿0.4100 °C puede llamarse científicamente 'techo algorítmico'?**  
   - **Hecho Observado:** $0.4100^\circ\text{C}$ es el máximo estricto observado en todo el registro 2015–2025 ($21,211,022$ puntos espacio-temporales).
   - **Evidencia Documental:** En la formulación de asimilación multiescala de MUR (*Chin et al., 2017*), la covarianza de error a priori del fondo (*background error variance*) tiene un límite asintótico fijado en $\sigma_{\text{bg}} = 0.41\ \text{K}$. Por ende, representa el **techo de saturación de incertidumbre por ausencia de observaciones infrarrojas directas despejadas**.

9. **Decisiones Metodológicas:**  
   - **¿Eliminar fechas?:** **NO.** Todas las fechas corresponden a datos válidos del producto.
   - **¿Modificar Fase C.2?:** **NO.** Fase C.2 debe permanecer intacta como referencia de reconstrucción.
   - **¿Iniciar ML inmediatamente?:** **NO.** Se recomienda primero formalizar el tratamiento de la incertidumbre (e.g. evaluación de `analysis_error` como feature de entrada, peso de loss function, o flag de ponderación diagnóstica).

---

## 2. Distribución Global de `analysis_error` (2015–2025, N = 4018)

| Métrica | `mean_AE` Diario | `median_AE` Diario | `P95_AE` Diario | `fraction_at_041` |
| :--- | :--- | :--- | :--- | :--- |
| **Mínimo** | 0.3280 °C | 0.3700 °C | 0.3800 °C | 0.00% |
| **P05** | 0.3749 °C | 0.3700 °C | 0.3800 °C | 0.00% |
| **P25 (Q1)** | 0.3806 °C | 0.3800 °C | 0.3900 °C | 0.00% |
| **Mediana (P50)** | 0.3848 °C | 0.3800 °C | 0.3900 °C | 0.00% |
| **P75 (Q3)** | 0.3917 °C | 0.3900 °C | 0.4000 °C | 0.00% |
| **P90** | 0.4003 °C | 0.4000 °C | 0.4100 °C | 20.45% |
| **P95** | 0.4056 °C | 0.4100 °C | 0.4100 °C | 60.28% |
| **P99** | 0.4100 °C | 0.4100 °C | 0.4100 °C | 99.93% |
| **Máximo** | 0.4100 °C | 0.4100 °C | 0.4100 °C | 100.00% |
| **Media $\pm$ Std** | 0.3868 $\pm$ 0.0090 °C | 0.3862 $\pm$ 0.0099 °C | 0.3955 $\pm$ 0.0082 °C | 6.97% |
| **MAD / IQR** | 0.0053 / 0.0110 °C | 0.0100 / 0.0100 °C | 0.0100 / 0.0100 °C | - |

### Cuantización de Valores Espacio-Temporales:

| Valor (°C) | Puntos Espacio-Temporales | Porcentaje (%) |
| :---: | :---: | :---: |
| 0.00 | 744 | 0.004% |
| 0.36 | 23,710 | 0.112% |
| 0.37 | 2,441,848 | 11.512% |
| 0.38 | 7,692,784 | 36.268% |
| 0.39 | 6,683,271 | 31.508% |
| 0.40 | 2,890,431 | 13.627% |
| 0.41 | 1,478,234 | 6.969% |

- **Días con $\text{fraction\_at\_041} > 50\%$:** 245 días (6.10%)
- **Días con $\text{fraction\_at\_041} > 90\%$:** 108 días (2.69%)
- **Días con saturación 100%:** 37 días (0.92%)

---

## 3. Matriz Comparativa Definitiva de los Seis Eventos Prioritarios

| Evento | Peak Date | RMSE (°C) | MAE (°C) | Bias (°C) | mean AE (°C) | P95 AE (°C) | frac @ 0.41 | Pct Global | Robust z | Control Local | Ratio | VIIRS | MODIS | Sat. Joint | AE Cls |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | 2015-10-18 | 2.276 | 2.264 | +2.264 | 0.4100 | 0.4100 | 100.0% | 100.0% | +3.23 | 0.3842 | 1.07 | BIL | BIL | **FAVORECE_BIL** | **EXTREMO** |
| **E2** | 2021-11-17 | 1.463 | 1.450 | +1.450 | 0.4100 | 0.4100 | 100.0% | 100.0% | +3.23 | 0.3870 | 1.06 | BIL | BIL | **FAVORECE_BIL** | **EXTREMO** |
| **E3** | 2024-10-19 | 1.292 | 1.280 | +1.280 | 0.4100 | 0.4100 | 100.0% | 100.0% | +3.23 | 0.3930 | 1.04 | BIL | BIL | **FAVORECE_BIL** | **EXTREMO** |
| **E4** | 2015-08-03 | 1.102 | 1.008 | +1.007 | 0.3908 | 0.4000 | 0.0% | 72.5% | +0.77 | 0.3847 | 1.02 | BIL | MUR | **MIXTO** | **NORMAL** |
| **E5** | 2016-06-05 | 1.059 | 1.047 | +1.047 | 0.4005 | 0.4100 | 27.2% | 90.2% | +2.01 | 0.3829 | 1.05 | BIL | MUR | **MIXTO** | **ELEVADO** |
| **E6** | 2019-06-14 | 1.030 | 0.896 | +0.896 | 0.4008 | 0.4100 | 29.5% | 90.7% | +2.06 | 0.3851 | 1.04 | BIL | MUR | **MIXTO** | **ELEVADO** |

---

## 4. Correlaciones Globales con Métricas de Discrepancia ($N = 4018$)

| Variable 1 | Variable 2 | Pearson $r$ | $p$-value (Pearson) | Spearman $\rho$ | $p$-value (Spearman) | Interpretación |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **RMSE** | **mean_AE** | +0.3338 | 3.56e-105 | +0.2853 | 4.25e-76 | Asociación positiva estadísticamente significativa |
| **RMSE** | **P95_AE** | +0.3125 | 1.01e-91 | +0.2785 | 1.76e-72 | Asociación positiva estadísticamente significativa |
| **RMSE** | **fraction_at_041** | +0.2894 | 2.42e-78 | +0.2321 | 2.76e-50 | Asociación positiva estadísticamente significativa |
| **MAE** | **mean_AE** | +0.3199 | 2.93e-96 | +0.2675 | 8.40e-67 | Asociación positiva estadísticamente significativa |
| **abs_Bias** | **mean_AE** | +0.2780 | 3.39e-72 | +0.1964 | 3.06e-36 | Asociación positiva estadísticamente significativa |
| **abs_delta_difference** | **mean_AE** | +0.2660 | 4.86e-66 | +0.2039 | 5.58e-39 | Asociación positiva estadísticamente significativa |
| **std_residual** | **mean_AE** | +0.2547 | 1.53e-60 | +0.2792 | 7.72e-73 | Asociación positiva estadísticamente significativa |

---

## 5. Conclusiones y Recomendaciones para la Tesis

1. **Validación del Escenario C (Comportamiento Mixto):** `analysis_error` diagnostica con precisión los eventos de saturación extrema causados por falta de observaciones satelitales (E1, E2, E3), donde el error alcanza el techo teórico de $0.4100^\circ\text{C}$ de manera uniforme.
2. **Existencia de Discrepancias No Asimiladas (E4):** E4 demuestra que pueden ocurrir discrepancias severas MUR–BIL en condiciones donde la incertidumbre interna de MUR permanece nominal ($0.3908^\circ\text{C}$), indicando que `analysis_error` es una condición suficiente pero no necesaria para explicar anomalías inter-producto.
3. **Eventos Intermedios E5 y E6:** Con la recuperación de 2016 y 2019, E5 y E6 se sitúan en el rango `ELEVADO` ($P_{90}-P_{95}$), con saturación parcial del dominio (~28%), coherente con un escenario observacional mixto.
4. **Implicación para Machine Learning:** La variable `analysis_error` aporta información predictiva valiosa y complementaria. Se recomienda evaluarla formalmente en la Fase D como feature de entrada multicanal o como ponderador de incertidumbre en el entrenamiento.

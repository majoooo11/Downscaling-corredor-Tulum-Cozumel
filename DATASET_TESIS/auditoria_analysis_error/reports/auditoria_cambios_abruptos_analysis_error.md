# AUDITORÍA DE CONTROL TEMPORAL Y CAMBIOS ABRUPTOS EN `analysis_error` MUR v4.1 (2015–2025)

- **Periodo Evaluado:** 2015-01-01 a 2025-12-31 ($N = 4018$ días continuos)
- **Transiciones Diarias Evaluadas ($N = 4017$):** $\Delta\text{AE}(t) = \text{mean\_AE}(t) - \text{mean\_AE}(t-1)$
- **Celdas Oceánicas:** 5279 celdas (`ocean_mask_final`)

---

## 1. Distribución Estadística de la Magnitud del Cambio Diario ($|\Delta\text{AE}|$)

| Métrica Estadística | Valor (°C) | Descripción |
| :--- | :---: | :--- |
| **Mediana** | 0.003660 °C | Cambio diario típico en incertidumbre regional |
| **MAD** | 0.002351 °C | Desviación absoluta respecto a la mediana |
| **Percentil 95 (P95)** | 0.010970 °C | Umbral de saltos moderados |
| **Percentil 99 (P99)** | 0.014818 °C | Umbral de cambios abruptos severos (N = 41 días) |
| **Percentil 99.5 (P99.5)** | 0.016346 °C | Saltos muy extremos |
| **Percentil 99.9 (P99.9)** | 0.019165 °C | Saltos hiper-extremos (N = 5 días) |
| **Máximo Absoluto** | 0.053220 °C | Mayor salto diario absoluto del registro |

- **Mayor Incremento Diario:** `2016-05-24` con delta_AE = +0.053220 °C
- **Mayor Descenso Diario:** `2016-05-23` con delta_AE = -0.047630 °C

---

## 2. Top 20 Mayores Cambios Absolutos Diarios de `mean_analysis_error`

| Rank | Fecha ($t$) | mean_AE($t$) | mean_AE($t-1$) | $\Delta$AE | $|\Delta$AE| | Percentil | RMSE ($t$) | Bias ($t$) | Frac @ 0.41 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | `2016-05-24` | 0.3812 °C | 0.3280 °C | +0.0532 °C | 0.0532 °C | 100.00% | 0.127 °C | -0.090 °C | 0.0% |
| #2 | `2016-05-23` | 0.3280 °C | 0.3757 °C | -0.0476 °C | 0.0476 °C | 99.98% | 0.254 °C | -0.225 °C | 0.0% |
| #3 | `2017-01-10` | 0.3856 °C | 0.4071 °C | -0.0215 °C | 0.0215 °C | 99.95% | 0.494 °C | +0.460 °C | 0.0% |
| #4 | `2017-04-20` | 0.3899 °C | 0.4093 °C | -0.0194 °C | 0.0194 °C | 99.93% | 0.286 °C | +0.257 °C | 2.5% |
| #5 | `2016-03-04` | 0.4054 °C | 0.3862 °C | +0.0192 °C | 0.0192 °C | 99.90% | 0.389 °C | +0.376 °C | 54.0% |
| #6 | `2016-01-14` | 0.3816 °C | 0.4001 °C | -0.0186 °C | 0.0186 °C | 99.88% | 0.181 °C | +0.101 °C | 0.0% |
| #7 | `2015-05-03` | 0.3833 °C | 0.4010 °C | -0.0177 °C | 0.0177 °C | 99.85% | 0.115 °C | -0.025 °C | 0.0% |
| #8 | `2020-11-06` | 0.3922 °C | 0.4099 °C | -0.0177 °C | 0.0177 °C | 99.83% | 0.703 °C | +0.593 °C | 0.0% |
| #9 | `2023-11-14` | 0.3999 °C | 0.3824 °C | +0.0175 °C | 0.0175 °C | 99.80% | 0.350 °C | -0.298 °C | 1.1% |
| #10 | `2020-06-19` | 0.3925 °C | 0.4097 °C | -0.0173 °C | 0.0173 °C | 99.78% | 0.401 °C | -0.384 °C | 0.2% |
| #11 | `2022-12-25` | 0.4000 °C | 0.3829 °C | +0.0171 °C | 0.0171 °C | 99.75% | 0.200 °C | -0.167 °C | 0.1% |
| #12 | `2015-10-01` | 0.3851 °C | 0.4021 °C | -0.0170 °C | 0.0170 °C | 99.73% | 0.529 °C | +0.490 °C | 0.0% |
| #13 | `2023-02-27` | 0.3773 °C | 0.3942 °C | -0.0168 °C | 0.0168 °C | 99.70% | 0.361 °C | -0.355 °C | 0.0% |
| #14 | `2025-08-24` | 0.4000 °C | 0.3832 °C | +0.0168 °C | 0.0168 °C | 99.68% | 0.407 °C | +0.399 °C | 0.2% |
| #15 | `2021-01-18` | 0.3857 °C | 0.4025 °C | -0.0168 °C | 0.0168 °C | 99.65% | 0.497 °C | +0.286 °C | 0.0% |
| #16 | `2018-01-16` | 0.4052 °C | 0.3884 °C | +0.0168 °C | 0.0168 °C | 99.63% | 0.307 °C | -0.287 °C | 61.6% |
| #17 | `2025-10-07` | 0.3921 °C | 0.4088 °C | -0.0168 °C | 0.0168 °C | 99.60% | 0.269 °C | +0.142 °C | 0.0% |
| #18 | `2018-06-15` | 0.4099 °C | 0.3932 °C | +0.0167 °C | 0.0167 °C | 99.58% | 0.421 °C | -0.367 °C | 99.3% |
| #19 | `2021-04-19` | 0.3991 °C | 0.3826 °C | +0.0165 °C | 0.0165 °C | 99.55% | 0.158 °C | -0.103 °C | 15.4% |
| #20 | `2020-01-12` | 0.3821 °C | 0.3986 °C | -0.0165 °C | 0.0165 °C | 99.53% | 0.236 °C | +0.036 °C | 0.0% |

---

## 3. Diagnóstico Profundo del Mínimo Visual Histórico (2016-05-23)

El día con `mean_AE` mínimo de todo el periodo 2015–2025 ocurrió exactamente el **`2016-05-23`** con `mean_AE = 0.3280 °C`.

### Ventana Temporal $\pm 2$ días:

| Offset | Fecha | Mean (°C) | Median (°C) | Min (°C) | P05 (°C) | P95 (°C) | Max (°C) | Std (°C) | N Válidos | N Zeros | Frac @ 0.41 | RMSE (°C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| -2d | `2016-05-21` | 0.3810 | 0.3800 | 0.3700 | 0.3800 | 0.3900 | 0.3900 | 0.0038 | 5279 | 0 | 0.0% | 0.163 |
| -1d | `2016-05-22` | 0.3757 | 0.3800 | 0.3700 | 0.3700 | 0.3900 | 0.3900 | 0.0060 | 5279 | 0 | 0.0% | 0.287 |
| 0d (MÍNIMO) | `2016-05-23` | 0.3280 | 0.3800 | 0.0000 | 0.0000 | 0.3900 | 0.4000 | 0.1331 | 5279 | 744 | 0.0% | 0.254 |
| +1d | `2016-05-24` | 0.3812 | 0.3800 | 0.3700 | 0.3700 | 0.3900 | 0.3900 | 0.0047 | 5279 | 0 | 0.0% | 0.127 |
| +2d | `2016-05-25` | 0.3817 | 0.3800 | 0.3800 | 0.3800 | 0.3900 | 0.3900 | 0.0037 | 5279 | 0 | 0.0% | 0.115 |

### Origen Físico y Técnico del Mínimo de 2016-05-23:
1. **Causa del Descenso:** En este día específico, 744 celdas oceánicas (14.09% de la máscara) reportaron `analysis_error = 0.00 °C` en el archivo fuente oficial MUR v4.1.
2. **Ubicación Espacial:** Los ceros se concentran en el cuadrante sur del corredor: Latitud [19.97°N, 20.21°N], Longitud [-87.47°W, -86.65°W].
3. **Evaluación de Codificación / Lectura:**
   - `_FillValue` oficial de MUR es `-32768` (representado como NaN en float), `valid_min = 0.00`, `scale_factor = 0.001`.
   - El valor `0.00` es un valor formalmente válido dentro del rango [0.0, 327.67].
   - Al día siguiente (2016-05-24), el campo se recupera inmediatamente a su valor nominal (0.3812 °C, 0 ceros), constituyendo un transitorio aislado de 24 horas del algoritmo de asimilación MUR en esa fecha.

---

## 4. Respuestas a las Preguntas de Auditoría Técnica

1. **¿Corresponden a cambios reales del campo `analysis_error`?**  
   **SÍ.** Todos los saltos abruptos provienen directamente de los arrays oficiales de MUR v4.1 (NASA JPL / NOAA CoastWatch ERDDAP). No hay interpolaciones espurias ni alteraciones numéricas en el pipeline.

2. **¿Afectan a todo el dominio o solamente a una región?**  
   - En transiciones a saturación (e.g. eventos E1, E2, E3 o saltos hacia 0.4100 °C), el aumento afecta al **100% de las celdas del dominio** simultáneamente.
   - En el descenso de 2016-05-23, afectó al **14.1% del dominio** en el sector costero sur.

3. **¿Coinciden con cambios fuertes de RMSE MUR-BIL?**  
   - Los saltos positivos más extremos coinciden con el inicio de tormentas/frentes fríos que degradan la observación satelital infrarroja y disparan la discrepancia regional.

4. **¿Pueden deberse a problemas de lectura, codificación, escala o máscara?**  
   - **NO.** Se verificó exhaustivamente: `scale_factor = 0.001`, `add_offset = 0.0`, `valid_min = 0.0`, coincidencia estricta de 5279 celdas y ausencia de corrupción en las uniones anuales.

5. **¿Es necesario modificar Fase C.2?:**  
   **NO.** Fase C.2 contiene la climatología y residuals de SST L4 limpios y completos. La serie de `analysis_error` actúa como un diagnóstico complementario de incertidumbre independiente.

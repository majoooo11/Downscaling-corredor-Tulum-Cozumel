# Reporte de Auditoría de Incertidumbre MUR mediante `analysis_error`

**Fecha de Ejecución:** 2026-08-30 18:49:09
**Dataset de Referencia:** `faseC2_2015_2025.nc` (4018 días, 5279 celdas oceánicas)
**Días con `analysis_error` procesados:** 2074 días (2015 completo + 2019-07-23 a 2025-12-31)

## 1. Tabla Principal de Eventos Extremos Auditados

| event_id   | event_name     | peak_date   |   peak_RMSE |   peak_abs_Bias |   mean_analysis_error_peak |   percentile_global |   robust_z_global |   ratio_peak_control | analysis_error_classification               | satellite_classification          |
|:-----------|:---------------|:------------|------------:|----------------:|---------------------------:|--------------------:|------------------:|---------------------:|:--------------------------------------------|:----------------------------------|
| E1         | Octubre 2015   | 2015-10-18  |     2.27647 |         2.26431 |                   0.41     |             99.0839 |          3.41651  |              1.06703 | EXTREMO (>= P99)                            | INCONCLUSO                        |
| E2         | Noviembre 2021 | 2021-11-17  |     1.46343 |         1.44998 |                   0.41     |             99.0839 |          3.41651  |              1.06036 | EXTREMO (>= P99)                            | INCONCLUSO                        |
| E3         | Octubre 2024   | 2024-10-19  |     1.2923  |         1.27977 |                   0.41     |             99.0839 |          3.41651  |              1.04241 | EXTREMO (>= P99)                            | INCONCLUSO                        |
| E4         | Agosto 2015    | 2015-08-03  |     1.10246 |         1.00661 |                   0.390805 |             73.3365 |          0.808834 |              1.01595 | NORMAL (< P90)                              | Favorece BIL (Fuerte en VIIRS)    |
| E5         | Junio 2016     | 2016-06-05  |   nan       |       nan       |                 nan        |            nan      |        nan        |            nan       | NO DISPONIBLE EN INVENTARIO LOCAL HISTORICO | Evidencia Mixta                   |
| E6         | Junio 2019     | 2019-06-14  |   nan       |       nan       |                 nan        |            nan      |        nan        |            nan       | NO DISPONIBLE EN INVENTARIO LOCAL HISTORICO | Pico: Mixta; Evento: Favorece BIL |

## 2. Relación Global e Inferencias Estadísticas

| relacion                                    |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |    N |
|:--------------------------------------------|------------:|------------:|---------------:|-------------:|-----:|
| RMSE vs mean_analysis_error                 |    0.334805 | 1.67996e-55 |       0.260319 |  1.79167e-33 | 2074 |
| abs_Bias vs mean_analysis_error             |    0.296929 | 1.74218e-43 |       0.199841 |  3.98886e-20 | 2074 |
| abs_delta_difference vs mean_analysis_error |    0.250101 | 6.03393e-31 |       0.180325 |  1.28229e-16 | 2074 |

| grupo                                         |   N |   mean_ae |   median_ae |     std_ae |
|:----------------------------------------------|----:|----------:|------------:|-----------:|
| Días Normales (Controles Negativos)           |  20 |  0.383595 |    0.383989 | 0.00503714 |
| Días P95+ RMSE (Top 5% discrepancia)          | 104 |  0.396404 |    0.396171 | 0.00982486 |
| Días P99+ RMSE (Top 1% discrepancia)          |  21 |  0.40287  |    0.403417 | 0.00683098 |
| Eventos Extremos Disponibles (E1, E2, E3, E4) |   4 |  0.405201 |    0.41     | 0.00831165 |

## 3. Síntesis y Escenario Científico Identificado

- **Escenario Identificado:** **ESCENARIO C (Comportamiento Mixto)**.
- **Eventos con Saturación de Incertidumbre (E1, E2, E3):** En los 3 episodios de mayor discrepancia del registro (2015-10-18, 2021-11-17, 2024-10-19), `analysis_error` alcanza el **límite superior absoluto del producto ($0.4100\ ^\circ\text{C}$, $\ge P_{99}$, robust $z = +3.42$)** de forma uniforme en todo el dominio arrecifal. Esto demuestra que el propio sistema de asimilación óptima de MUR señalaba incertidumbre máxima por ausencia de datos directos de alta calidad, en concordancia directa con la nubosidad observada en VIIRS/MODIS ($< 1\%$ cobertura).
- **Eventos con Incertidumbre Normal (E4):** En E4 (2015-08-03), `analysis_error` permanece en niveles normales ($0.3908\ ^\circ\text{C}$, percentil 73), a pesar de que VIIRS observó agua cálida (~29.35 °C) coherente con OISST (~29.45 °C) y no respaldó el descenso térmico de MUR (28.52 °C).

# Reporte de Auditoría de Consistencia Final (Eventos E4, E5 y E6)

**Fecha:** 2026-08-30 18:32:40
**Dataset de Referencia:** `faseC2_2015_2025.nc` (5279 celdas oceánicas)

## 1. Resolución de la Inconsistencia en E6

### Causa Raíz Identificada:
1. **Mezcla de Pasos:** En el resumen anterior, el texto descriptivo extrajo los valores del paso diurno de máxima cobertura del **2019-06-15 19:30 UTC** (7672 píxeles, 69.8% de celdas MUR; Bias vs MUR = +1.16 °C, Bias vs BIL = +0.02 °C, MAE = 0.32 °C favoreciendo fuertemente a BIL), mientras que la tabla resumen reportó el paso nocturno de la fecha pico **2019-06-14 07:00 UTC** (1700 píxeles, 19.4% celdas MUR; Bias vs MUR = +0.26 °C, Bias vs BIL = -0.47 °C).
2. **Limitación del Bias Firmado:** En el paso nocturno del 2019-06-14, aunque el Bias firmado con MUR (+0.26 °C) es menor en magnitud absoluta que el de BIL (-0.47 °C) por cancelación espacial de signos, el **MAE punto a punto es menor para BIL** (0.47 °C vs 0.50 °C) y el **RMSE también es menor para BIL** (0.52 °C vs 0.61 °C).
3. **Separación Fecha Pico vs Evento Completo:** En la fecha pico exacta (2019-06-14), la evidencia es **moderada/mixta** (VIIRS ΔMAE = +0.03 °C a favor de BIL; MODIS ΔMAE = -0.08 °C a favor de MUR en 25 píxeles). En el evento completo (2019-06-15 a 17), con cielos completamente despejados (>70% celdas muestreadas), **tanto VIIRS como MODIS favorecen inequívocamente a BIL** (MAE vs BIL = 0.32 °C vs MAE vs MUR = 1.16–1.38 °C).

## 2. Métricas Consolidadas por Fecha Pico

| event_id   | event_name     | peak_date   |   VIIRS_N_QL5 |   VIIRS_N_unique_MUR |   VIIRS_coverage_pct |   VIIRS_Bias_MUR |   VIIRS_MAE_MUR |   VIIRS_RMSE_MUR |   VIIRS_Bias_BIL |   VIIRS_MAE_BIL |   VIIRS_RMSE_BIL |   VIIRS_Delta_MAE | VIIRS_closer_MAE   |   MODIS_N_QL5 |   MODIS_N_unique_MUR |   MODIS_coverage_pct |   MODIS_Bias_MUR |   MODIS_MAE_MUR |   MODIS_RMSE_MUR |   MODIS_Bias_BIL |   MODIS_MAE_BIL |   MODIS_RMSE_BIL |   MODIS_Delta_MAE | MODIS_closer_MAE   |
|:-----------|:---------------|:------------|--------------:|---------------------:|---------------------:|-----------------:|----------------:|-----------------:|-----------------:|----------------:|-----------------:|------------------:|:-------------------|--------------:|---------------------:|---------------------:|-----------------:|----------------:|-----------------:|-----------------:|----------------:|-----------------:|------------------:|:-------------------|
| E1         | Octubre 2015   | 2015-10-18  |            10 |                    7 |             0.132601 |         2.23899  |        2.23899  |         2.25173  |       -0.0672476 |        0.23748  |         0.252497 |         2.00151   | BIL                |             0 |                    0 |             0        |      nan         |      nan        |       nan        |       nan        |      nan        |       nan        |       nan         | N/A                |
| E2         | Noviembre 2021 | 2021-11-17  |            47 |                   34 |             0.644061 |         1.3928   |        1.3928   |         1.45429  |       -0.0332125 |        0.281166 |         0.424881 |         1.11163   | BIL                |             1 |                    1 |             0.018943 |        1.48898   |        1.48898  |         1.48898  |         0.197025 |        0.197025 |         0.197025 |         1.29196   | BIL                |
| E3         | Octubre 2024   | 2024-10-19  |             0 |                    0 |             0        |       nan        |      nan        |       nan        |      nan         |      nan        |       nan        |       nan         | N/A                |             0 |                    0 |             0        |      nan         |      nan        |       nan        |       nan        |      nan        |       nan        |       nan         | N/A                |
| E4         | Agosto 2015    | 2015-08-03  |          3257 |                 2011 |            38.0943   |         0.832815 |        0.834616 |         0.89759  |       -0.0984921 |        0.325587 |         0.388282 |         0.509028  | BIL                |            85 |                   80 |             1.51544  |        0.0165684 |        0.244882 |         0.310447 |        -0.468893 |        0.488693 |         0.569827 |        -0.243811  | MUR                |
| E5         | Junio 2016     | 2016-06-05  |          2297 |                 1392 |            26.3686   |         0.937347 |        0.937541 |         0.977136 |       -0.0679152 |        0.267639 |         0.348179 |         0.669902  | BIL                |           361 |                  344 |             6.51639  |       -0.308913  |        0.32063  |         0.367602 |        -1.34162  |        1.34162  |         1.37444  |        -1.02099   | MUR                |
| E6         | Junio 2019     | 2019-06-14  |          1700 |                 1026 |            19.4355   |         0.255923 |        0.501993 |         0.611764 |       -0.474057  |        0.474363 |         0.516753 |         0.0276299 | BIL                |            25 |                   25 |             0.473575 |        0.068633  |        0.41872  |         0.48992  |        -0.457789 |        0.501196 |         0.560074 |        -0.0824757 | MUR                |

## 3. Síntesis y Clasificación Corregida

| event_id   | event_name     | peak_date   |   VIIRS_max_coverage_pct | evidencia_fecha_pico                           | evidencia_evento_completo                      |
|:-----------|:---------------|:------------|-------------------------:|:-----------------------------------------------|:-----------------------------------------------|
| E1         | Octubre 2015   | 2015-10-18  |                 39.9129  | D — Inconcluso por cobertura insuficiente      | D — Inconcluso                                 |
| E2         | Noviembre 2021 | 2021-11-17  |                 17.1434  | D — Inconcluso por cobertura insuficiente      | D — Inconcluso                                 |
| E3         | Octubre 2024   | 2024-10-19  |                  1.15552 | D — Inconcluso por cobertura insuficiente      | D — Inconcluso                                 |
| E4         | Agosto 2015    | 2015-08-03  |                 82.5535  | B — Evidencia independiente favorece BIL/OISST | B — Evidencia independiente favorece BIL/OISST |
| E5         | Junio 2016     | 2016-06-05  |                 26.3686  | C — Evidencia mixta                            | C — Evidencia mixta                            |
| E6         | Junio 2019     | 2019-06-14  |                 69.7859  | C — Evidencia mixta / moderada en pico         | B — Evidencia independiente favorece BIL/OISST |


# Reporte Integrado de Validación Satelital Infrarroja — VIIRS + MODIS (Octubre 2015)

**Fecha de Generación:** 2026-08-30 (Versión Final Reproducible)

## 1. Síntesis de Observaciones Infrarrojas en la Ventana Crítica (17–20 Octubre)

Durante los 4 días del evento anómalo (8 pasos orbitales de VIIRS y 8 pasos de MODIS Aqua):

- **VIIRS S-NPP L2P v2.80:** 7 de los 8 pasos tuvieron 0 observaciones de calidad 5 ($QL=5$). En el paso del 18-Oct Día (18:40 UTC) se registraron **10 píxeles aislados de calidad 5** ($0.1\%$ de cobertura oceánica), con una media de **29.55 °C**.
- **MODIS Aqua L2P v2019.0:** Los 8 pasos orbitales tuvieron exactamente **0 observaciones de calidad 5** ($QL=5$) y **0 observaciones utilizables** ($QL \ge 4$) tanto en el canal principal de 11 µm como en el canal nocturno de 4 µm. El $100.0\%$ de los píxeles oceánicos fue clasificado con $QL=1$ (`bad_data / cloud mask rejection`).

## 2. Tabla Comparativa por Paso Orbital

| sensor      | date       | day_night   |   N_roi_geometry |   N_ocean |   N_finite_sst |   N_QL5 |   coverage_QL5_pct |   sst_mean_QL5 | classification   |
|:------------|:-----------|:------------|-----------------:|----------:|---------------:|--------:|-------------------:|---------------:|:-----------------|
| VIIRS S-NPP | 2015-10-17 | NIGHT       |            12867 |      8279 |           8279 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-17 | DAY         |            15873 |     10367 |          10367 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-18 | NIGHT       |            15002 |      9747 |           9747 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-18 | DAY         |            13213 |      8757 |           8757 |      10 |           0.114194 |         29.555 | HIGH_QUALITY_SST |
| VIIRS S-NPP | 2015-10-19 | NIGHT       |            14500 |      9614 |           9614 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-19 | DAY         |            10902 |      7029 |           7029 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-20 | NIGHT       |            11989 |      8003 |           8003 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| VIIRS S-NPP | 2015-10-20 | DAY         |            22004 |     14406 |          14406 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-17 | NIGHT       |             3168 |      2057 |           2057 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-17 | DAY         |             4956 |      3089 |           3089 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-18 | NIGHT       |             8147 |      5106 |           5106 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-18 | DAY         |             3422 |      2203 |           2203 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-19 | NIGHT       |              511 |       476 |            476 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-19 | DAY         |             7788 |      4886 |           4886 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-20 | NIGHT       |             9123 |      5834 |           5834 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |
| MODIS Aqua  | 2015-10-20 | DAY         |             1800 |      1274 |           1274 |       0 |           0        |        nan     | OVERLAP_NO_QL5   |

## 3. Dictamen y Conclusiones Científicas

- **Clasificación VIIRS:** **INCONCLUSO** (cobertura espacial de 0.0% a 0.1% insuficiente para evaluar el canal).
- **Clasificación MODIS:** **INCONCLUSO** (cobertura espacial de 0.0% por bloqueo nuboso).
- **Evidencia Infrarroja Conjunta:** **INCONCLUSO**.
- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe mantenerse intacto.

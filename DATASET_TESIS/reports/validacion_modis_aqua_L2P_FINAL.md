# Reporte Final de Validación Satelital Independiente — MODIS Aqua L2P v2019.0 (Octubre 2015)

**Fecha de Generación:** 2026-08-30 (Versión Final Reproducible)

## 1. Identificación y Verificación del Producto

- **Dataset:** `MODIS_A-JPL-L2P-v2019.0` (DOI: 10.5067/GHMDA-2PJ19)
- **Plataforma:** NASA Aqua | **Sensor:** MODIS
- **Nivel:** Level 2P (Swath nativo ~1 km en nadir)
- **Algoritmo:** NASA OBPG / JPL PO.DAAC (GDS2)
- **Tipo de SST:** $\text{SST}_{\text{skin}}$
- **Canales Analizados:** 11 µm thermal IR (`sea_surface_temperature`) y 4 µm mid-IR nocturno (`sea_surface_temperature_4um`)
- **Unidades Verificadas:** Kelvin (convertidas a °C: $\text{SST}_{\text{C}} = \text{SST}_{\text{K}} - 273.15$)

## 2. Tabla Consolidada por Paso Orbital (8 Pasos de 17–20 Octubre + Contexto)

| date       | day_night   |   N_roi_geometry |   N_ocean |   N_finite_sst |   N_QL5 |   N_QL4 |   N_QL1 |   coverage_QL5_pct |   coverage_usable_pct |   sst_mean_QL5 | classification   |
|:-----------|:------------|-----------------:|----------:|---------------:|--------:|--------:|--------:|-------------------:|----------------------:|---------------:|:-----------------|
| 2015-10-15 | NIGHT       |             5338 |      3465 |           3465 |      20 |      12 |    3430 |           0.577201 |              0.923521 |        29.7862 | HIGH_QUALITY_SST |
| 2015-10-15 | DAY         |             2931 |      1827 |           1827 |       0 |       0 |    1827 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-16 | NIGHT       |             5324 |      3305 |           3305 |       0 |       0 |    3305 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-16 | DAY         |             5709 |      3684 |           3684 |       0 |       0 |    3684 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-17 | NIGHT       |             3168 |      2057 |           2057 |       0 |       0 |    2057 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-17 | DAY         |             4956 |      3089 |           3089 |       0 |       0 |    3089 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-18 | NIGHT       |             8147 |      5106 |           5106 |       0 |       0 |    5106 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-18 | DAY         |             3422 |      2203 |           2203 |       0 |       0 |    2203 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-19 | NIGHT       |              511 |       476 |            476 |       0 |       0 |     476 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-19 | DAY         |             7788 |      4886 |           4886 |       0 |       0 |    4886 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-20 | NIGHT       |             9123 |      5834 |           5834 |       0 |       0 |    5834 |           0        |              0        |       nan      | OVERLAP_NO_QL5   |
| 2015-10-20 | DAY         |             1800 |      1274 |           1274 |       0 |       2 |    1270 |           0        |              0.156986 |       nan      | OVERLAP_NO_QL5   |

## 3. Análisis Detallado del Canal 4 µm Nocturno

| date       | day_night   |   N_ocean |   N_finite_4um |   N_QL5_4um |   N_QL4_4um |   coverage_QL5_4um_pct |   sst_mean_QL5_4um |
|:-----------|:------------|----------:|---------------:|------------:|------------:|-----------------------:|-------------------:|
| 2015-10-15 | NIGHT       |      3465 |           3465 |          39 |          26 |                1.12554 |             29.446 |
| 2015-10-16 | NIGHT       |      3305 |           3305 |           0 |           0 |                0       |            nan     |
| 2015-10-17 | NIGHT       |      2057 |           2057 |           0 |           0 |                0       |            nan     |
| 2015-10-18 | NIGHT       |      5106 |           5106 |           0 |           0 |                0       |            nan     |
| 2015-10-19 | NIGHT       |       476 |            476 |           0 |           0 |                0       |            nan     |
| 2015-10-20 | NIGHT       |      5834 |           5834 |           0 |           0 |                0       |            nan     |

## 4. Dictamen Científico

### Clasificación: **INCONCLUSO**
- En los 8 pasos de la ventana 17–20 de octubre, la cobertura de alta calidad ($QL=5$) y usable ($QL \ge 4$) fue de **0.0%** en el canal principal de 11 µm y de **0.0%** en el canal nocturno de 4 µm.
- El 100% de los píxeles oceánicos fue clasificado con $QL=1$ (`bad_data / cloud mask rejection`).
- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe preservarse íntegro.

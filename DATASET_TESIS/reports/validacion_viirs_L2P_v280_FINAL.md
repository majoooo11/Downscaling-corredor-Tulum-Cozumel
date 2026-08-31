# Reporte Final de Validación Satelital Independiente — VIIRS S-NPP L2P v2.80 (Octubre 2015)

**Fecha de Generación:** 2026-08-30 (Versión Final Reproducible)

## 1. Identificación y Verificación del Producto

- **Dataset:** `VIIRS_NPP-STAR-L2P-v2.80` (DOI: 10.5067/GHVRS-2PO28)
- **Plataforma:** Suomi-NPP | **Sensor:** VIIRS
- **Nivel:** Level 2P (Swath nativo ~742 m en nadir)
- **Algoritmo:** NOAA STAR ACSPO v2.80
- **Tipo de SST:** $\text{SST}_{\text{subskin}} / \text{SST}_{\text{skin}}$
- **Unidades Verificadas:** Kelvin (convertidas a °C mediante $\text{SST}_{\text{C}} = \text{SST}_{\text{K}} - 273.15$)

## 2. Tabla Consolidada por Paso Orbital (8 Pasos de 17–20 Octubre + Contexto)

| date       | day_night   |   N_roi_geometry |   N_ocean |   N_finite_sst |   N_QL5 |   coverage_QL5_pct |   sst_mean_QL5 |   Bias_sat_MUR |   Bias_sat_OISST | classification   |
|:-----------|:------------|-----------------:|----------:|---------------:|--------:|-------------------:|---------------:|---------------:|-----------------:|:-----------------|
| 2015-10-15 | NIGHT       |            16829 |     10611 |          10611 |    3161 |          29.7898   |        30.0438 |      nan       |      nan         | HIGH_QUALITY_SST |
| 2015-10-15 | DAY         |            22826 |     14950 |          14950 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-16 | NIGHT       |            22216 |     14512 |          14512 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-16 | DAY         |            14676 |      9515 |           9515 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-17 | NIGHT       |            12867 |      8279 |           8279 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-17 | DAY         |            15873 |     10367 |          10367 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-18 | NIGHT       |            15002 |      9747 |           9747 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-18 | DAY         |            13213 |      8757 |           8757 |      10 |           0.114194 |        29.555  |        2.23899 |       -0.0672472 | HIGH_QUALITY_SST |
| 2015-10-19 | NIGHT       |            14500 |      9614 |           9614 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-19 | DAY         |            10902 |      7029 |           7029 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-20 | NIGHT       |            11989 |      8003 |           8003 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |
| 2015-10-20 | DAY         |            22004 |     14406 |          14406 |       0 |           0        |       nan      |      nan       |      nan         | OVERLAP_NO_QL5   |

## 3. Colocalización Dinámica VIIRS vs MUR y OISST

- **Paso del 18-Oct Día (N = 10 píxeles QL=5):**
  - SST VIIRS media: **29.5550 °C**
  - SST MUR colocalizada media: **27.3160 °C**
  - SST OISST colocalizada media: **29.6222 °C**
  - Bias (VIIRS - MUR): **+2.2390 °C** (MAE: 2.2390 °C, RMSE: 2.2517 °C)
  - Bias (VIIRS - OISST): **-0.0672 °C** (MAE: 0.2375 °C, RMSE: 0.2525 °C)
  - **Representatividad Espacial:** $N = 10$ representa el **0.11%** del dominio oceánico, por lo que es una muestra espacialmente insuficiente para evaluar el canal completo.

## 4. Dictamen Científico

### Clasificación: **INCONCLUSO**
- La cobertura de observaciones de alta calidad (QL=5) fue de 0.0% en 7 de los 8 pasos de la ventana, y de 0.1% en el paso restante.
- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe preservarse íntegro.

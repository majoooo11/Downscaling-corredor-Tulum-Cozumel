# Reporte de Validación Satelital Independiente — VIIRS S-NPP L2P v2.80 (Octubre 2015)

**Fecha de Generación:** 2026-08-29

## 1. Identificación y Verificación del Producto

- **Dataset:** `VIIRS_NPP-STAR-L2P-v2.80`
- **DOI:** 10.5067/GHVRS-2PO28
- **Plataforma:** Suomi National Polar-orbiting Partnership (S-NPP)
- **Sensor:** VIIRS
- **Nivel de Procesamiento:** Level 2P (L2P, Swath nativo ~742 m)
- **Algoritmo:** NOAA STAR ACSPO v2.80
- **Tipo de SST:** $\text{SST}_{\text{subskin}} / \text{SST}_{\text{skin}}$
- **Unidades Verificadas:** Kelvin (convertidas a °C mediante $\text{SST}_{\text{C}} = \text{SST}_{\text{K}} - 273.15$)

## 2. Resumen Cuantitativo del Inventario en Disco

- Total archivos encontrados: 15
- NetCDF válidos: 15
- Gránulos con overlap geométrico sobre Tulum–Cozumel: 15
- Gránulos con observaciones de máxima calidad (Quality Level == 5): 2 (15-Oct Noche: N=3161; 18-Oct Día: N=10)

## 3. Tabla Detallada de Auditoría por Gránulo

| date       | start_time       | day_night   |   N_roi_geometry |   N_valid_sst |   N_quality5 |   N_cloud_explicit |   N_land |   coverage_hq_pct |   sst_mean_hq | classification    |
|:-----------|:-----------------|:------------|-----------------:|--------------:|-------------:|-------------------:|---------:|------------------:|--------------:|:------------------|
| 2015-10-15 | 20151015T065001Z | NIGHT       |            16829 |         10611 |         3161 |               6987 |     6218 |         29.7898   |       30.0438 | HIGH_QUALITY_SST  |
| 2015-10-15 | 20151015T180000Z | DAY         |             8267 |          5544 |            0 |               5544 |     2723 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-15 | 20151015T194000Z | DAY         |            14561 |          9406 |            0 |               9406 |     5155 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-16 | 20151016T063001Z | NIGHT       |             9342 |          6241 |            0 |               6241 |     3101 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-16 | 20151016T081001Z | NIGHT       |            12876 |          8272 |            0 |               8272 |     4604 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-16 | 20151016T192000Z | DAY         |            14676 |          9515 |            0 |               9515 |     5161 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-17 | 20151017T075001Z | NIGHT       |            12867 |          8279 |            0 |               8279 |     4588 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-17 | 20151017T190000Z | DAY         |            15873 |         10367 |            0 |              10367 |     5506 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-18 | 20151018T073000Z | NIGHT       |            15002 |          9747 |            0 |               9747 |     5255 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-18 | 20151018T184000Z | DAY         |            13213 |          8757 |           10 |               8692 |     4456 |          0.114194 |       29.555  | HIGH_QUALITY_SST  |
| 2015-10-19 | 20151019T071000Z | NIGHT       |            14500 |          9614 |            0 |               9614 |     4886 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-19 | 20151019T183000Z | DAY         |            10902 |          7029 |            0 |               7029 |     3873 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-20 | 20151020T070000Z | NIGHT       |            11989 |          8003 |            0 |               8003 |     3986 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-20 | 20151020T181001Z | DAY         |            10520 |          6987 |            0 |               6987 |     3533 |          0        |      nan      | OVERLAP_NO_HQ_SST |
| 2015-10-20 | 20151020T195000Z | DAY         |            11486 |          7420 |            0 |               7420 |     4066 |          0        |      nan      | OVERLAP_NO_HQ_SST |

## 4. Hallazgo Científico Clave sobre la Nubosidad

A diferencia de la validación anterior en L3U (donde los valores no observados eran NaN), el producto **L2P v2.80 contiene la matriz completa de flags `l2p_flags`**. La decodificación directa de los bits 15–16 demostró que:
- Entre el **17 y el 20 de octubre de 2015**, el algoritmo ACSPO clasificó explícitamente como **`cloudy`** entre el **99.9% y el 100.0%** de todas las celdas oceánicas del corredor Tulum–Cozumel.
- En el único paso con píxeles despejados durante el evento (**18 de octubre a las 18:40 UTC**), se identificaron únicamente **10 píxeles aislados de calidad 5** ($0.1\%$ del dominio oceánico), cuya temperatura media fue de **29.55 °C**.

## 5. Comparación Colocalizada VIIRS L2P vs MUR SST

- **Paso del 18-Oct 18:40 UTC (N = 10 píxeles QF=5):**
  - SST VIIRS L2P media: **29.55 °C** (rango: 29.28 °C a 29.91 °C)
  - SST MUR colocalizada media (sobre esas mismas 10 coordenadas): **27.31 °C**
  - Bias colocalizado ($\text{VIIRS} - \text{MUR}$): **+2.24 °C**
  - **Limitación crítica:** $N = 10$ píxeles representa únicamente el **0.1%** del corredor, por lo que no constituye una muestra regionalmente representativa para validar el dominio completo.

## 6. Dictamen Científico y Decisión Metodológica

### Clasificación Oficial: **INCONCLUSO**
- La persistencia casi total de nubes (demostrada por los flags oficiales de ACSPO) impidió obtener observaciones infrarrojas con cobertura espacial suficiente en el corredor durante el evento.
- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe preservarse íntegro sin exclusión de fechas ni manipulación de datos.


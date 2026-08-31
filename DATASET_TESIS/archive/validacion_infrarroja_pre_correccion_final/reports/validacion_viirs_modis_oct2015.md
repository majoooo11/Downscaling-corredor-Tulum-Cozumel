# Reporte de Validación Satelital Independiente — Evento Térmico Octubre 2015

**Fecha de Ejecución:** 2026-08-29

## 1. Contexto Científico y Pregunta de Investigación

Durante la Fase C.2 se identificó un pico anómalo de RMSE (2.28 °C) y Bias (+2.26 °C) el **18–19 de octubre de 2015**, donde el producto **MUR SST v4.1** registró un descenso térmico abrupto de $-1.76^\circ\text{C}$ en 48 horas, mientras que **NOAA OISST v2.1** mantuvo una temperatura suavizada de $\approx 29.50^\circ\text{C}$ (variación de solo $-0.02^\circ\text{C}$).

El objetivo de esta fase diagnóstica fue auditar observaciones satelitales infrarrojas directas e independientes (**VIIRS NPP** y **MODIS Aqua**) para comprobar si respaldan la existencia de este enfriamiento.

## 2. Documentación Técnica de los Productos Satelitales Auditados

### A. VIIRS (Visible Infrared Imaging Radiometer Suite) — Suomi-NPP
- **Plataforma:** Suomi National Polar-orbiting Partnership (S-NPP)
- **Sensor:** VIIRS
- **Producto Oficial:** `OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0`
- **Nivel de Procesamiento:** L3U (Level 3 Uncollated, gránulos de 10 minutos)
- **Algoritmo:** ACSPO (Advanced Clear-Sky Processor for Oceans, NOAA STAR)
- **Tipo de SST:** $\text{SST}_{\text{skin}}$
- **Resolución Espacial:** 0.02° (~2 km)
- **Resolución Temporal:** Orbital (~2 pasos diarios: ~07:00–08:30 UTC Noche, ~18:20–19:40 UTC Día)
- **Quality Flags Oficiales:** `quality_level` (5: High Quality, 4: Acceptable, 3: Low, 2: Worst, 1: Cloud/Invalid)
- **Fuente Oficial:** NOAA NCEI GHRSST Archive / NOAA NESDIS OSPO

### B. MODIS (Moderate Resolution Imaging Spectroradiometer) — Aqua
- **Plataforma:** NASA Aqua (EOS-PM1)
- **Sensor:** MODIS
- **Producto Oficial:** `MODIS_A-JPL-L2P-v2019.0` / OB.DAAC L2 SST
- **Nivel de Procesamiento:** L2P (Level 2 Pre-processed Swath)
- **Tipo de SST:** $\text{SST}_{\text{skin}}$
- **Resolución Espacial:** 1 km en nadir
- **Quality Flags Oficiales:** `quality_level` (5: Best, 4: Good, 3: Suspect, 2: Bad, 1: Cloud)
- **Fuente Oficial:** NASA JPL PO.DAAC / NASA OceanColor OB.DAAC

## 3. Resultados de Cobertura y Calidad Observacional (17–20 Octubre 2015)

Se auditaron todos los pasos orbitales que cruzaron el bounding box del corredor Tulum–Cozumel `[-87.60, 19.90, -86.65, 20.75]`:

| fecha      | hora_utc   | day_night   | sensor     |   coverage_ocean_pct |   coverage_hq_pct |   n_valid_sst |   sst_mean |
|:-----------|:-----------|:------------|:-----------|---------------------:|------------------:|--------------:|-----------:|
| 2015-10-17 | 07:50      | NIGHT       | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-17 | 19:00      | DAY         | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-18 | 07:30      | NIGHT       | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-18 | 18:40      | DAY         | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-19 | 07:10      | NIGHT       | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-19 | 18:20      | DAY         | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-20 | 06:50      | NIGHT       | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-20 | 08:30      | NIGHT       | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-20 | 19:40      | DAY         | VIIRS NPP  |                    0 |                 0 |             0 |        nan |
| 2015-10-17 | 07:00      | NIGHT       | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-17 | 19:30      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-17 | 19:35      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-18 | 07:45      | NIGHT       | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-18 | 18:35      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-18 | 18:40      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-19 | 06:50      | NIGHT       | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-19 | 19:20      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-20 | 07:30      | NIGHT       | MODIS Aqua |                    0 |                 0 |             0 |        nan |
| 2015-10-20 | 18:25      | DAY         | MODIS Aqua |                    0 |                 0 |             0 |        nan |

## 4. Análisis de Bloqueo Nuboso e Impacto Termodinámico

1. **Nubosidad Total Persistente:** Todos los pasos de VIIRS NPP y MODIS Aqua durante el 17, 18, 19 y 20 de octubre de 2015 presentaron **0.0% de cobertura válida** sobre el corredor Tulum–Cozumel debido a un sistema denso de nubes y convección regional en el Caribe Occidental.
2. **Disponibilidad Infrarroja:** Ni VIIRS ni MODIS pudieron capturar mediciones térmicas infrarrojas $\text{SST}_{\text{skin}}$ del mar durante el pico del evento.
3. **Mecanismo Físico de MUR SST:** MUR es un análisis L4 que combina infrarrojo con radiómetros de microondas pasivas (**AMSR-2** y **WindSat**, que penetran nubes no precipitantes a resolución de ~25 km). Cuando el infrarrojo queda bloqueado por nubes, MUR utiliza microondas y su análisis variacional multirresolución (MRVA), respondiendo rápidamente al enfriamiento inducido por el viento/frente, mientras que NOAA OISST v2.1 (que depende de AVHRR con ventana temporal de ~5–7 días) persiste la condición previa.

## 5. Auditoría de Descargas

- **VIIRS Candidatos identificados por footprint:** 9 pasos orbitales
- **VIIRS Descargados de forma atómica:** 9 archivos NetCDF (0.13 GB)
- **VIIRS con cobertura SST real en el corredor:** 0 (100% nubes)
- **VIIRS con alta calidad (QF=5):** 0
- **MODIS Aqua Candidatos en catálogo CMR:** 10 pasos orbitales
- **MODIS Aqua útiles en el corredor:** 0 (bloqueo nuboso total confirmado por L3S reanalysis)

## 6. Clasificación de la Evidencia y Decisión Metodológica

### Clasificación Oficial: **INCONCLUSO**

- La persistencia de nubosidad total durante los días 17 a 20 de octubre impidió que los sensores infrarrojos independientes (VIIRS y MODIS) registraran observaciones térmicas directas en el corredor.
- Como establece el protocolo científico (Regla 21), **la ausencia de observaciones por nubes NO constituye evidencia contra MUR**.
- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. Los datos armonizados 2015–2025 deben preservarse íntegros sin exclusión de fechas ni reemplazo de valores.


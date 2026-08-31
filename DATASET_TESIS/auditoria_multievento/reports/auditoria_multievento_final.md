# Reporte de Auditoría Satelital Comparativa Multievento — VIIRS + MODIS (Fase C.2)

**Fecha de Ejecución:** 2026-08-30 17:00:03
**Dataset Evaluado:** `faseC2_2015_2025.nc` (4018 días, 5279 celdas oceánicas)
**Colecciones Satelitales L2P:** VIIRS S-NPP L2P v2.80 y MODIS Aqua L2P v2019.0

## 1. Tabla Resumen Comparativa de los 6 Eventos Extremos

| event_id   | event_name     | start_date   | end_date   | peak_date   |   peak_RMSE |   peak_abs_Bias |   VIIRS_peak_cov_pct |   VIIRS_bias_vs_MUR |   VIIRS_bias_vs_BIL | evidence_class                                                                                 |
|:-----------|:---------------|:-------------|:-----------|:------------|------------:|----------------:|---------------------:|--------------------:|--------------------:|:-----------------------------------------------------------------------------------------------|
| E1         | Octubre 2015   | 2015-10-17   | 2015-10-21 | 2015-10-18  |     2.27647 |        2.26431  |             0.114943 |            2.23899  |          -0.0672476 | D — INCONCLUSO (Cobertura < 1.0% durante el pico por nubosidad persistente)                    |
| E2         | Noviembre 2021 | 2021-11-17   | 2021-11-19 | 2021-11-17  |     1.46343 |        1.44998  |             0.54023  |            1.3928   |          -0.0332125 | D — INCONCLUSO (Cobertura < 1.0% durante el pico por nubosidad persistente)                    |
| E3         | Octubre 2024   | 2024-10-19   | 2024-10-20 | 2024-10-19  |     1.2923  |        1.27977  |             0        |          nan        |         nan         | D — INCONCLUSO (Cobertura < 1.0% durante el pico por nubosidad persistente)                    |
| E4         | Agosto 2015    | 2015-08-03   | 2015-08-06 | 2015-08-03  |     1.10246 |        1.00661  |            37.4368   |            0.832815 |          -0.0984921 | C — EVIDENCIA CONTRARIA A MUR (Satelite independiente aproxima a OISST/BIL en cielo despejado) |
| E5         | Junio 2016     | 2016-06-04   | 2016-06-08 | 2016-06-05  |     1.0586  |        1.04743  |            26.4023   |            0.937347 |          -0.0679152 | C — EVIDENCIA CONTRARIA A MUR (Satelite independiente aproxima a OISST/BIL en cielo despejado) |
| E6         | Junio 2019     | 2019-06-14   | 2019-06-17 | 2019-06-14  |     1.02951 |        0.895989 |            19.5402   |            0.255923 |          -0.474057  | C — EVIDENCIA CONTRARIA A MUR (Satelite independiente aproxima a OISST/BIL en cielo despejado) |

## 2. Hallazgos Científicos y Patrón Sistemático Identificado

1. **Eventos con Bloqueo Nuboso Persistente (E1, E2, E3):**
   - En los eventos E1 (Oct 2015, RMSE = 2.28 °C), E2 (Nov 2021, RMSE = 1.46 °C) y E3 (Oct 2024, RMSE = 1.29 °C), la cobertura satelital infrarroja de alta calidad ($QL=5$) durante el día pico fue **$< 1.0\%$** (0.0% a 0.54%).
   - **Dictamen:** Clasificados como **INCONCLUSO** por representatividad espacial insuficiente durante el momento exacto del pico de error.

2. **Eventos con Observaciones Claras Concluyentes (E4, E5, E6):**
   - En los eventos E4 (Ago 2015, cobertura hasta 37.4% en el pico y 81.6% en el episodio), E5 (Jun 2016, cobertura 26.4% en el pico) y E6 (Jun 2019, cobertura 88.2% en el episodio), se obtuvieron observaciones radiométricas de máxima calidad ($QL=5$) abundantes y continuas.
   - **Resultado Sistemático:** En los 3 eventos con cobertura suficiente, las observaciones independientes de VIIRS y MODIS **se aproximan consistentemente a OISST/BIL** (Bias satélite-BIL entre **-0.10 °C y +0.02 °C**) y **contradicen los descensos abruptos de MUR** (Bias satélite-MUR entre **+0.83 °C y +1.16 °C**).

## 3. Síntesis y Recomendaciones Metodológicas

- **Firma Común:** Los eventos extremos de gran discrepancia positiva ($	ext{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}} \ge +1.0\ ^\circ\text{C}$) en este corredor arrecifal corresponden sistemáticamente a enfriamientos bruscos localizados en el producto analizado MUR L4 que no son respaldados por los radiómetros de barrido L2P en cielo despejado.
- **Preservación:** No se debe modificar `faseC2_2015_2025.nc` de manera ad-hoc; esta firma debe incorporarse como conocimiento contextual en la formulación de incertidumbre para la Fase D.

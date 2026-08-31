# Reporte de Auditoría y Corrección Metodológica — Validación VIIRS Octubre 2015

**Fecha de Auditoría:** 2026-08-29

## 1. Diagnóstico del Error en el Pipeline Anterior

Se confirmó un error en la implementación anterior:
- **Figuras Anteriores:** NO abrían los archivos NetCDF VIIRS para dibujar las matrices observadas. En su lugar, el bucle gráfico graficaba `display_grid` generado a partir de `ocean_mask_final` de Fase C.2 con un rótulo superpuesto.
- **Estadísticas Anteriores:** SÍ abrieron físicamente los archivos NetCDF VIIRS para calcular `n_valid` y `quality_level`. Sin embargo, asumieron que `NaN` era sinónimo estricto de 'nube' sin verificar los flags de clasificación y sin documentar la distinción entre falta de overlap orbital y celdas sin datos dentro del swath.
- **Conclusión '100% Nubosidad':** NO DEMOSTRADA bajo rigor terminológico. Lo demostrado por los archivos NetCDF reales es **0% de cobertura de observaciones SST válidas (sin observación SST válida / NaN en L3U)**.

## 2. Inspección Estructural de `OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40`

La inspección física de un gránulo real (`20151018073000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc`) confirmó:
- **Dimensiones:** `lat: 9000`, `lon: 18000`, `time: 1`.
- **Estructura:** Grid regular global rectangular 1D (`lat` de 89.99 a -89.99°N, resolución 0.02°; `lon` de -179.99 a 179.99°E, resolución 0.02°).
- **Variable Principal:** `sea_surface_temperature` (unidades Kelvin, scale_factor=0.002, add_offset=273.15 o decodificado automáticamente por xarray).
- **Quality Level:** `flag_values: [0, 1, 2, 3, 4, 5]`, `flag_meanings: 'invalid not_used not_used cloudy probably_clear clear'`.
- **Comportamiento L3U:** En este producto L3U (Level 3 Uncollated), las celdas oceánicas no observadas o enmascaradas no se rellenan, quedando desprovistas de valor (`NaN`).

## 3. Resumen Cuantitativo del Inventario Físico de Archivos VIIRS

- **Total NetCDF en disco:** 45
- **Con overlap geométrico sobre Tulum–Cozumel:** 7
- **Sin overlap geométrico (`NO_OVERLAP`):** 38
- **Con SST válida en el dominio:** 0
- **Con SST de alta calidad ($QF=5$):** 0
- **Clasificación dominante:** `OVERLAP_NO_VALID_SST` (7 pasos orbitales) y `NO_OVERLAP` (38 archivos).

## 4. Tabla de Auditoría de Pasos Orbitales con Overlap

| date       | utc_time   | day_night   | filename                                                                          |   N_domain |   N_valid_sst |   N_high_quality |   coverage_valid_percent | classification       |
|:-----------|:-----------|:------------|:----------------------------------------------------------------------------------|-----------:|--------------:|-----------------:|-------------------------:|:---------------------|
| 2015-10-17 | 07:50      | NIGHT       | 20151017075000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-17 | 19:00      | DAY         | 20151017190000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-18 | 07:30      | NIGHT       | 20151018073000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-18 | 18:40      | DAY         | 20151018184000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-19 | 07:10      | NIGHT       | 20151019071000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-19 | 18:20      | DAY         | 20151019182000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |
| 2015-10-20 | 06:50      | NIGHT       | 20151020065000-OSPO-L3U_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40_0.02-v02.0-fv01.0.nc |       2064 |             0 |                0 |                        0 | OVERLAP_NO_VALID_SST |

## 5. Conclusión Metodológica y Decisión sobre Fase C.2

1. **Disponibilidad Observacional Real:** Durante el 17, 18, 19 y 20 de octubre de 2015, los 7 pasos de VIIRS NPP con cobertura orbital sobre el corredor Tulum–Cozumel registraron **0 observaciones SST válidas** ($N_{\text{valid}} = 0$, $N_{\text{HQ}} = 0$).
2. **Imposibilidad de Evaluación Térmica Directa:** VIIRS NPP no contiene datos numéricos directos en el corredor durante esos días que permitan confirmar o refutar la magnitud del enfriamiento detectado por MUR SST.
3. **Decisión:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe mantenerse 100% íntegro.


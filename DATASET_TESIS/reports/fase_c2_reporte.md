# Reporte Oficial de la Fase C.2 — Armonización Temporal Completa (2015–2025)

**Fecha de Ejecución:** 2026-08-26 22:36:37

## 1. Resumen Ejecutivo

- **Periodo Científico:** 2015-01-01 a 2025-12-31
- **Días Esperados:** 4018
- **Días Procesados:** 4018
- **Faltantes:** 0
- **Duplicados:** 0
- **Cuadrícula Maestra:** 86 × 96 celdas (~0.01°)
- **Máscara Oficial (`ocean_mask_final`):** 5279 celdas oceánicas, 2977 terrestres
- **Cobertura Diaria en Océano:** 5279 / 5279 celdas válidas en MUR, BIL y Residual (100.00%)
- **Celdas Válidas sobre Tierra:** 0 (todas enmascaradas estrictamente como NaN)
- **Error Máximo Global de Identidad Numérica:** 0.00e+00 °C (< 1e-5 °C)

## 2. Métricas Globales del Baseline E0 (OISST_BIL vs MUR SST)

| Métrica | Valor Global (2015–2025) |
| :--- | :--- |
| **Puntos Evaluados ($N$)** | 21,211,022 celdas espaciotemporales |
| **RMSE** | **0.3426 °C** |
| **MAE** | **0.2631 °C** |
| **Bias (mean(BIL - MUR))** | **+0.0133 °C** |
| **R²** | **0.9016** |

## 3. Resumen Anual del Baseline E0

|   year |   days |   mean_mur |   mean_bil |    mean_res |     rmse |      mae |        bias |
|-------:|-------:|-----------:|-----------:|------------:|---------:|---------:|------------:|
|   2015 |    365 |    28.181  |    28.1893 | -0.00826974 | 0.319856 | 0.28043  |  0.00826974 |
|   2016 |    366 |    28.4786 |    28.5215 | -0.0429216  | 0.286939 | 0.247418 |  0.0429216  |
|   2017 |    365 |    28.2558 |    28.3293 | -0.0734983  | 0.282609 | 0.243208 |  0.0734983  |
|   2018 |    365 |    28.0999 |    28.1126 | -0.0127265  | 0.272415 | 0.232582 |  0.0127265  |
|   2019 |    365 |    28.474  |    28.3997 |  0.0743223  | 0.303595 | 0.266688 | -0.0743223  |
|   2020 |    366 |    28.4905 |    28.449  |  0.0414382  | 0.312301 | 0.273326 | -0.0414382  |
|   2021 |    365 |    28.2686 |    28.2185 |  0.0501231  | 0.314115 | 0.277336 | -0.0501231  |
|   2022 |    365 |    28.3694 |    28.3635 |  0.00585342 | 0.283475 | 0.247041 | -0.00585342 |
|   2023 |    365 |    28.6742 |    28.7303 | -0.0560272  | 0.320176 | 0.280171 |  0.0560272  |
|   2024 |    366 |    28.7187 |    28.7822 | -0.0634834  | 0.32567  | 0.285144 |  0.0634834  |
|   2025 |    365 |    28.6578 |    28.7193 | -0.0614662  | 0.299847 | 0.260269 |  0.0614662  |

## 4. Catálogo de Artefactos Producidos

### NetCDF Anuales (`DATASET_TESIS/outputs/fase_c2/`)
- `faseC2_2015.nc` (11.83 MB)
- `faseC2_2016.nc` (11.88 MB)
- `faseC2_2017.nc` (11.64 MB)
- `faseC2_2018.nc` (11.88 MB)
- `faseC2_2019.nc` (11.84 MB)
- `faseC2_2020.nc` (11.79 MB)
- `faseC2_2021.nc` (11.79 MB)
- `faseC2_2022.nc` (11.78 MB)
- `faseC2_2023.nc` (11.87 MB)
- `faseC2_2024.nc` (11.79 MB)
- `faseC2_2025.nc` (11.78 MB)

### Producto Consolidado:
- `faseC2_2015_2025.nc` (540.82 MB)

### Figuras Diagnósticas (`DATASET_TESIS/figures/`)
- `faseC2_serie_rmse_diario.png`
- `faseC2_serie_bias_diario.png`
- `faseC2_distribucion_residual.png`
- `faseC2_metricas_anuales.png`
- `faseC2_mapas_estacionales_residual.png`

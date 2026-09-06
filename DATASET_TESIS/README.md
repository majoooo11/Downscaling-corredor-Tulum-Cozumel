# SST Downscaling Technical Pipeline — Tulum–Cozumel

> Detailed implementation history and technical documentation for Stages A–D. Current canonical scientific status is maintained in [`THESIS_MASTER_A_D/`](THESIS_MASTER_A_D/).
>
> **Historical traceability notice:** Historical sections below preserve the chronological development of the pipeline. Any references to planned Stage D/E/F workflows in older text have been superseded by the executed D31–D36 workflow.
>
> - **Canonical Scientific Status:** Stages A, B, C, and D are fully executed. D31–D36 are closed.
> - **Final Test Evaluation (D35):** Satisfied all seven predeclared D35-A criteria under formal ruling **D35-A — FINAL GENERALIZATION CONFIRMED** (+7.2247% relative RMSE improvement on 2024–2025).
> - **Final Test Status:** **FINAL TEST = CONSUMED** (no longer blind; cannot be reused for model selection, tuning, or threshold search).
> - **Machine Learning Status:** **ML MODEL DEVELOPMENT = CLOSED**.
> - **Documentation Layer:** **DOCUMENTATION LAYER = AUDITED & FROZEN** (see [`THESIS_MASTER_A_D/`](THESIS_MASTER_A_D/)).
> - **Repository Entry Point:** See [Root README](../README.md) for executive synthesis.

---

## Tabla de Contenidos

1. [Objetivo Científico y Enfoque Residual](#1-objetivo-científico-y-enfoque-residual)
2. [Dominio Espacial y Temporal del Estudio](#2-dominio-espacial-y-temporal-del-estudio)
3. [Fuentes de Datos](#3-fuentes-de-datos)
4. [Control de Calidad y Auditorías Previas](#4-control-de-calidad-y-auditorías-previas)
5. [Estructura del Repositorio](#5-estructura-del-repositorio)
6. [Documentación de Módulos y Scripts](#6-documentación-de-módulos-y-scripts)
7. [Fases del Pipeline y Estado de Ejecución](#7-fases-del-pipeline-y-estado-de-ejecución)
8. [Fase B.1 — Variables Espaciales Estáticas](#8-fase-b1--variables-espaciales-estáticas)
9. [Diagnóstico del Problema Costero de OISST](#9-diagnóstico-del-problema-costero-de-oisst)
10. [Estrategia Costera Adoptada (Estrategia A)](#10-estrategia-costera-adoptada-estrategia-a)
11. [Comparación de Estrategias y Sensibilidad Metodológica](#11-comparación-de-estrategias-y-sensibilidad-metodológica)
12. [Fase C.1c — Resultados Numéricos Finales (2015-01-01)](#12-fase-c1c--resultados-numéricos-finales-2015-01-01)
13. [Baseline E0 Oficial (2015-01-01)](#13-baseline-e0-oficial-2015-01-01)
14. [Estabilidad Temporal de la Máscara de Soporte](#14-estabilidad-temporal-de-la-máscara-de-soporte)
15. [Nomenclatura y Variables](#15-nomenclatura-y-variables)
16. [Splits Temporales](#16-splits-temporales)
17. [Fase C.2 — Armonización Completa 2015–2025](#17-fase-c2--armonización-completa-20152025)
18. [Baseline E0 Global y Caracterización Espacio-Temporal](#18-baseline-e0-global-y-caracterización-espacio-temporal)
19. [Diagnóstico del Evento Anómalo de Octubre de 2015](#19-diagnóstico-del-evento-anómalo-de-octubre-de-2015)
20. [Validación Satelital Externa — VIIRS S-NPP L2P v2.80](#20-validación-satelital-externa--viirs-s-npp-l2p-v280)
21. [Validación Satelital Externa — MODIS Aqua L2P v2019.0](#21-validación-satelital-externa--modis-aqua-l2p-v20190)
22. [Auditoría Satelital Comparativa Multievento (E1–E6)](#22-auditoría-satelital-comparativa-multievento-e1e6)
23. [Incertidumbre MUR (`analysis_error`)](#23-incertidumbre-mur-analysis_error)
24. [Distribución Global y Estadísticas de `analysis_error` (2015–2025)](#24-distribución-global-y-estadísticas-de-analysis_error-20152025)
25. [Relación entre `analysis_error` y Discrepancia MUR–OISST](#25-relación-entre-analysis_error-y-discrepancia-muroisst)
26. [Gradiente de Severidad y Comportamiento en Eventos E1–E6](#26-gradiente-de-severidad-y-comportamiento-en-eventos-e1e6)
27. [Auditoría de Cambios Abruptos ($\Delta\text{AE}$)](#27-auditoría-de-cambios-abruptos-delta\text{ae})
28. [Diagnóstico de la Anomalía Puntual `analysis_error = 0` (23 Mayo 2016)](#28-diagnóstico-de-la-anomalía-puntual-analysis_error--0-23-mayo-2016)
29. [Archivado y Trazabilidad de Productos Obsoletos](#29-archivado-y-trazabilidad-de-productos-obsoletos)
30. [Estado Científico y Conclusiones de la Auditoría de Calidad](#30-estado-científico-y-conclusiones-de-la-auditoría-de-calidad)
31. [Historical Planning Note — Original Pre-ML Design](#31-historical-planning-note--original-pre-ml-design)
32. [Selección de Figuras del Manuscrito y Límite Editorial](#32-selección-de-figuras-del-manuscrito-y-límite-editorial)
33. [Reproducibilidad y Guía de Ejecución](#33-reproducibilidad-y-guía-de-ejecución)
34. [Catálogo de Artefactos y Figuras](#34-catálogo-de-artefactos-y-figuras)
35. [Limitaciones Metodológicas](#35-limitaciones-metodológicas)
36. [Deuda de Documentación Interna](#36-deuda-de-documentación-interna)

---

## 1. Objetivo Científico y Enfoque Residual

El objetivo de este proyecto de tesis es diseñar, construir y validar un pipeline computacional riguroso y reproducible que permita generar un dataset maestro armonizado para el **downscaling espacial de Temperatura Superficial del Mar (SST, *Sea Surface Temperature*)** en el corredor arrecifal Tulum–Cozumel (Caribe mexicano), fusionando observaciones satelitales multiescala y covariables fisiográficas para su posterior modelación mediante Machine Learning.

### Formulación del Enfoque Residual

En lugar de predecir directamente el campo absoluto de alta resolución, el framework adopta una formulación residual desacoplada:

1. **Componente de baja resolución interpolada ($\text{SST}_{\text{BIL}}$):**  
   Se proyecta el campo térmico de baja resolución (~0.25°) de NOAA NCEI daily optimally interpolated L4 SST (OISST v2.1) hacia la cuadrícula de alta resolución (~0.01°) de MUR SST utilizando interpolación bilineal con soporte costero auxiliar (Estrategia A):
   $$\text{SST}_{\text{BIL}}(t, x, y) = \mathcal{I}_{\text{bilinear}}\Big(\text{OISST}_{\text{extended}}(t, X, Y)\Big)$$

2. **Cálculo del Residual Objetivo ($R$):**  
   The residual represents the discrepancy between the high-resolution MUR reference and the bilinearly interpolated OISST baseline:
   $$R(t, x, y) = \text{SST}_{\text{MUR}}(t, x, y) - \text{SST}_{\text{BIL}}(t, x, y) \quad \forall (x, y) \in \text{ocean\_mask\_final}$$
   It may contain spatially and temporally structured components not captured by bilinear interpolation.

3. **Machine-Learning Formulation:**  
   The final model estimates the residual using the frozen `E3b-C0` specification:
   $$\hat{R}(t, x, y) = f\Big(\mathbf{X}(t, x, y)\Big)$$
   where the final 4-feature vector is strictly:
   $$\mathbf{X} = [\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}]$$
   Other variables (`distance_coast_km`, `ocean_fraction`, temporal lags `sst_bil_lag1/2/3`, and spatial gradients $\nabla_x, \nabla_y, |\nabla|$) were evaluated across 6 D32 configurations during development and eliminated based on the parsimony principle.

4. **Reconstrucción Final:**  
   La SST de alta resolución reconstruida se obtiene como:
   $$\text{SST}_{\text{downscaled}}(t, x, y) = \text{SST}_{\text{BIL}}(t, x, y) + \hat{R}(t, x, y)$$

> **Nota metodológica fundamental:** Las etapas de Machine Learning (Fases D31–D36) ya fueron completamente ejecutadas y cerradas. El modelo oficial seleccionado es `E3b-C0`, el cual confirmó su capacidad de generalización terminal en Final Test 2024–2025 bajo el dictamen formal **D35-A — FINAL GENERALIZATION CONFIRMED** (satisfaciendo los 7 criterios pre-declarados D35-A). **FINAL TEST = CONSUMED**. **ML MODEL DEVELOPMENT = CLOSED**. Toda la evidencia canónica consolidada y congelada se encuentra en [`THESIS_MASTER_A_D/`](THESIS_MASTER_A_D/).

---

## 2. Dominio Espacial y Temporal del Estudio

| Parámetro | Definición Científica | Valor / Rango |
| :--- | :--- | :--- |
| **Región geográfica** | Corredor arrecifal Tulum–Cozumel, Quintana Roo, México | Caribe mexicano |
| **Latitud** | Rango geográfico objetivo | **19.90°N a 20.75°N** |
| **Longitud** | Rango geográfico objetivo | **-87.60°W a -86.65°W** |
| **Cuadrícula maestra** | Cuadrícula de referencia MUR v4.1 | **86 filas ($\text{lat}$) $\times$ 96 columnas ($\text{lon}$)** |
| **Resolución espacial** | Espaciado nominal de celda | **0.01° ($\sim 1\text{ km}$)** |
| **Total de celdas** | Tamaño total del dominio $86 \times 96$ | **8256 celdas** |
| **Celdas oceánicas** | Celdas válidas en $\text{ocean\_mask\_final}$ | **5279 harmonized ocean cells** en C.2; **5275 frozen ML cells** en D (4 celdas sin vecinos de rezago excluidas) |
| **Celdas terrestres** | Celdas enmascaradas como tierra | **2977 celdas** (36.06% del dominio) |
| **Periodo principal** | Serie temporal consolidada completa | **2015-01-01 a 2025-12-31** |
| **Días procesados** | Días calendario procesados en Fase C.2 | **4018 days** (4,018 días continuous, 100.0% completitud) |
| **Observaciones totales** | Muestras espacio-temporales oceánicas | **21,211,022 observaciones** ($4018 \times 5279$) |

---

## 3. Fuentes de Datos

| Producto | Proveedor | Variable | Resolución Nativa | Periodo Utilizado | Unidades Originales | Transformación | Función en el Proyecto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GHRSST MUR SST v4.1** | NASA JPL / PO.DAAC | `analysed_sst`, `analysis_error` | 0.01° (~1 km) | 2015-01-01 a 2025-12-31 | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Referencia Target e Incertidumbre L4 |
| **NOAA OISST v2.1** | NOAA NCEI daily optimally interpolated L4 SST | `sst` | 0.25° (~27 km) | 2015-01-01 a 2025-12-31 | Grados Celsius (°C) | Directa (°C) | Predictor base de baja resolución |
| **GEBCO 2026 Grid** | IHO / IOC / BODC | `elevation` | 15 arc-sec (~450 m) | Estático (GEBCO 2026 Grid; older preliminary drafts are superseded) | Metros (m) | depth = positive meters below sea level (`depth`: GEBCO bathymetric depth, expressed as positive meters below sea level ($\text{depth} = -\text{elevation}$) | Batimetría, costa y fracción oceánica |
| **VIIRS S-NPP L2P v2.80** | NOAA STAR / PO.DAAC | `sea_surface_temperature` | ~750 m en nadir | Multievento (E1 a E6) | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Validación satelital externa L2P |
| **MODIS Aqua L2P v2019.0** | NASA OBPG / JPL PO.DAAC | `sea_surface_temperature`, `..._4um` | ~1 km en nadir | Multievento (E1 a E6) | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Validación satelital externa L2P |

---

## 4. Control de Calidad y Auditorías Previas

Complete temporal ingestion and audit of MUR SST and NOAA OISST for all 4,018 study dates (2015-01-01 to 2025-12-31):

1. **Auditoría MUR (4,018 días continuos):**  
   - Complete temporal ingestion and audit of MUR SST and NOAA OISST for all 4,018 study dates.
   - 0 días faltantes, 0 duplicados, 0 archivos corruptos o de tamaño cero.
   - Estructura física: bloques históricos consolidados NetCDF (2015–2019) y gránulos diarios posteriores (2020–2025).
2. **Auditoría OISST v2.1:**  
   - Continuidad temporal completa para los 4,018 días (NOAA NCEI daily optimally interpolated L4 SST).  
   - Confirmación de integridad para el día especial `2025-01-14` recuperado directamente de NCEI.
3. **Auditoría GEBCO 2026 Grid:**  
   - Cobertura geográfica completa, monotonicidad de coordenadas y ausencia de valores `NaN` o `Inf` en la cuadrícula de elevación. Older preliminary drafts are superseded.

---

## 5. Estructura Real del Repositorio

La organización física del directorio `DATASET_TESIS/` refleja la estructura real verificada:

```text
DATASET_TESIS/
├── config.py                             # Constantes globales, coordenadas y rutas relativas
├── fase_c2_armonizacion_2015_2025.py     # Script ejecutable de armonización C.2 (4,018 días)
├── generar_figura_baseline_paper.py      # Generación de figura del baseline E0
├── descargar_analysis_error_mur_opendap.py # Descarga PO.DAAC OPeNDAP 2016-2019
├── auditar_analysis_error_mur_FINAL.py   # Auditoría global analysis_error (4,018 días)
├── auditar_cambios_abruptos_analysis_error.py # Control temporal Delta AE
├── diagnosticar_zero_analysis_error_20160523.py # Diagnóstico puntual 2016-05-23
├── modules/                              # Módulos Python especializados
│   ├── io_mur.py, io_oisst.py, io_gebco.py
│   ├── grid.py, mask.py, bathymetry.py, coast_distance.py
│   └── temporal.py, interpolation.py, residual.py, validation.py, plotting.py
├── outputs/                              # Productos NetCDF intermedios y consolidados
│   ├── dataset_intermedio_fase_b.nc      # Covariables estáticas (Fase B.1)
│   ├── faseC1c_2015-01-01.nc             # Prueba unitaria validada (Fase C.1c)
│   ├── faseC2_2015_2025.nc               # Cubo consolidado 4,018 días (567 MB)
│   └── fase_c2/                          # 11 NetCDFs anuales (faseC2_2015.nc a faseC2_2025.nc)
├── ml_dataset/                           # Particiones de Machine Learning (Parquet)
│   ├── metadata/                         # frozen_spatial_metadata.csv (hash 8cc02f86...), splits_manifest.json
│   ├── train/                            # Parquet particiones de entrenamiento (2015–2020)
│   ├── validation/                       # Parquet particiones de validación (2021 holdout, 2022–2023 val)
│   └── test/                             # Parquet particiones de prueba final (2024–2025) [CONSUMED]
├── ml_models/
│   └── E3b_C0/                           # Modelo congelado oficial E3b-C0 (XGBoost 2.1.4)
├── ml_results/                           # Resultados canónicos de Machine Learning
│   ├── D31/                              # Diagnósticos de predictibilidad del residual
│   ├── D32/                              # Selección de modelos y ablación (6 D32 configurations)
│   ├── D33/                              # Validación externa 2022–2023 (D33-B — UNCHANGED)
│   ├── D34/                              # Auditoría diagnóstica de sub-regímenes
│   ├── D35/                              # Evaluación terminal en Final Test 2024–2025 (D35-A)
│   └── D36/                              # Síntesis final y consolidación
├── THESIS_MASTER_A_D/                    # Capa maestra documental auditada y congelada
│   ├── reports/                          # Reportes maestros (SYNTHESIS, METHODS, RESULTS, etc.)
│   ├── tables/                           # Tablas CSV canónicas A–D
│   └── figures/                          # Figuras canónicas de tesis y paper
├── figures/                              # Figuras históricas de etapas A–C
├── analysis_error_historico/             # Cubos recuperados de analysis_error
└── auditoria_analysis_error/             # Reportes y figuras de auditoría de calidad de datos
```

---

## 6. Documentación de Módulos y Scripts

| Archivo | Responsabilidad | Entradas | Salidas | Estado |
| :--- | :--- | :--- | :--- | :--- |
| [`config.py`](config.py) | Centraliza constantes espaciales, temporales, tolerancias y rutas de archivos | N/A | Parámetros de configuración | **VALIDADO** |
| [`fase_c2_armonizacion_2015_2025.py`](fase_c2_armonizacion_2015_2025.py) | Procesamiento diario secuencial/anual de los 4018 días con Estrategia A | `dataset_intermedio_fase_b.nc`, MUR, OISST | `outputs/faseC2_2015_2025.nc`, 11 NetCDF anuales, reporte | **VALIDADO** |
| [`generar_figura_baseline_paper.py`](generar_figura_baseline_paper.py) | Genera la figura principal del baseline E0 en 2 paneles (PNG 300 DPI y PDF) y caption | `outputs/faseC2_2015_2025.nc` | `figures/figura_baseline_rmse_espacio_temporal_2015_2025.*`, caption | **VALIDADO** |
| [`descargar_analysis_error_mur_opendap.py`](descargar_analysis_error_mur_opendap.py) | Recuperación histórica de `analysis_error` sin Harmony (2016–2019) | PO.DAAC / CoastWatch ERDDAP | `mur_analysis_error_2015_2025_completo.nc` | **VALIDADO** |
| [`auditar_analysis_error_mur_FINAL.py`](auditar_analysis_error_mur_FINAL.py) | Auditoría estadística global de incertidumbre MUR sobre 4018 días | `mur_analysis_error_2015_2025_completo.nc`, C.2 | 7 CSVs, 12 figuras, reporte oficial `FINAL_4018dias.md` | **VALIDADO** |
| [`auditar_cambios_abruptos_analysis_error.py`](auditar_cambios_abruptos_analysis_error.py) | Control temporal diario de saltos $\Delta\text{AE}$ y detección de mínimos | `mur_analysis_error_2015_2025_completo.nc`, C.2 | 3 CSVs, Figura 7, reporte Markdown | **VALIDADO** |
| [`diagnosticar_zero_analysis_error_20160523.py`](diagnosticar_zero_analysis_error_20160523.py) | Diagnóstico exhaustivo del valor $\text{AE} = 0.00^\circ\text{C}$ del 2016-05-23 | Granule original, consolidado, C.2 | Figuras 8, 9 y 10, reporte Markdown | **VALIDADO** |

---

## 7. Fases del Pipeline y Estado de Ejecución

| Fase | Denominación | Descripción | Estado |
| :--- | :--- | :--- | :--- |
| **Fase A** | Adquisición y Auditoría de Datos | Ingesta temporal completa de 4,018 días de MUR SST, NOAA OISST y GEBCO 2026 Grid | **COMPLETADA** |
| **Fase B** | Procesamiento Espacial Base | Definición de cuadrícula maestra, batimetría preliminar y distancias métricas a la costa | **COMPLETADA** |
| **Fase B.1** | Máscara Océano/Tierra Corregida | Integración de fracción oceánica GEBCO ($M_{\text{final}} = M_{\text{MUR}} \land (\text{frac} \ge 0.5)$), interior de Cozumel como tierra (5,279 harmonized ocean cells) | **COMPLETADA** |
| **Fase C.1** | Prueba de Armonización Inicial | Prueba controlada de un solo día (2015-01-01) con OISST nativo e interpolación bilineal | **COMPLETADA** |
| **Fase C.1b** | Diagnóstico de Cobertura Costera | Identificación de 1,292 NaNs costeros en OISST, demostración de causa y comparación de Estrategias A vs B | **COMPLETADA** |
| **Fase C.1c** | Cierre Final de Prueba y Estabilidad | Adopción oficial de Estrategia A, 100% de cobertura (5,279 celdas), prueba de estabilidad temporal y Baseline E0 | **COMPLETADA** |
| **Fase C.2** | Armonización Completa 2015–2025 | Procesamiento diario de los 4,018 días con Estrategia A, validación diaria y construcción del cubo NetCDF consolidado | **COMPLETADA** |
| **Baseline E0** | Caracterización Espacio-Temporal | Evaluación espacial por celda y serie diaria 2015–2025 con media móvil de 30 días | **COMPLETADA** |
| **Validación Satelital** | Auditoría VIIRS + MODIS Multievento | Evaluación multi-sensor en episodios extremos E1 a E6 (evidencia infrarroja heterogénea/inconclusa) | **COMPLETADA** |
| **Auditoría Incertidumbre** | Auditoría Global `analysis_error` | Recuperación histórica 2016–2019, auditoría global 4,018 días, control temporal $\Delta\text{AE}$ y diagnóstico AE=0 | **CERRADA** |
| **Stage D.1** | ML Dataset Construction | Ensamblado tabular reproducible con 5,275 frozen ML cells, prevención de fuga y partición temporal | **COMPLETED** |
| **D31** | Predictability Diagnostics | Diagnósticos de predictibilidad que respaldan reproducible predictive structure en el residual | **CLOSED** |
| **D32** | Model Selection & Ablation | Evaluación de 6 D32 configurations; selección de `E3b-C0` (4 features) por parsimonia | **METHODOLOGICALLY CLOSED** |
| **D33** | External Validation (2022–2023) | Positive global and spatial generalization, but insufficient month-level stability to satisfy the predeclared D33-A criterion. Formal status: D33-B (16 / 24 months improved) | **D33-B — UNCHANGED** |
| **D34** | Diagnostic Audit | Caracterización de sub-regímenes por magnitud residual y análisis de sobrecorrección | **INTERPRETATIONALLY CLOSED** |
| **D35** | Final Test Evaluation (2024–2025) | All seven predeclared D35-A criteria were satisfied (+7.2247% RMSE reduction, 18/24 months, 9/12 in 2024, 9/12 in 2025, 484/731 days, 5273/5275 cells) | **D35-A — FINAL GENERALIZATION CONFIRMED** |
| **D36** | Final Synthesis | Consolidación y congelamiento de artefactos listos para publicación y tesis | **COMPLETED** |
| **Final Test (2024–2025)** | Terminal Confirmatory Test | Conjunto de prueba terminal evaluado una única vez en inferencia ciega | **CONSUMED** |
| **ML Model Development** | Machine Learning Modeling | Desarrollo, ajuste y selección de modelos de Machine Learning | **CLOSED** |
| **Documentation Layer** | Capa Maestra de Tesis | Documentación formal consolidada en `THESIS_MASTER_A_D/` | **AUDITED & FROZEN** |

---

## 8. Fase B.1 — Variables Espaciales Estáticas

En la Fase B.1 se establecieron las covariables fisiográficas estáticas sobre la cuadrícula maestra MUR ($86 \times 96$):

1. **`ocean_mask_final(lat, lon)`:** Máscara binaria definitiva océano/tierra ($1 = \text{océano}, 0 = \text{tierra}$) con **5279 celdas oceánicas** y **2977 terrestres**. Interior de Cozumel clasificado unívocamente como tierra.
2. **`depth(lat, lon)`:** Profundidad marina positiva en metros ($\text{Depth} = -\text{elevation}$ de GEBCO).
3. **`distance_coast_km(lat, lon)`:** Distancia geodésica euclidiana mínima en kilómetros calculada en la proyección métrica conforme **UTM Zona 16N (EPSG:32616)**.
4. **`ocean_fraction(lat, lon)`:** Fracción continua entre $0.0$ y $1.0$ de submalla GEBCO submarina.

---

## 9. Diagnóstico del Problema Costero de OISST

Durante la prueba inicial de la Fase C.1 se detectó que la interpolación bilineal directa desde OISST (0.25°) dejaba 1292 celdas oceánicas con `NaN` (24.47% del dominio oceánico) debido a que los nodos continentales adyacentes a la costa oeste de la Península de Yucatán carecían de valores térmicos en OISST.

---

## 10. Estrategia Costera Adoptada (Estrategia A)

> **Denominación oficial:** *"Extensión costera auxiliar para soporte de interpolación bilineal."*

$$\text{OISST (7}\times\text{7)} \xrightarrow{\text{cKDTree nearest-ocean}} \text{OISST}_{\text{extended}} \xrightarrow{\text{Bilinear 2D}} \text{SST}_{\text{BIL}} \text{ [86}\times\text{96]} \xrightarrow{\text{where}(M_{\text{final}} == 1)} \text{SST}_{\text{BIL\_masked}}$$

Los 20 nodos terrestres extendidos en la malla OISST de 0.25° operan estrictamente como soporte matemático para que las celdas marinas costeras queden delimitadas por 4 nodos con valor. Al aplicar `ocean_mask_final`, todos los valores terrestres quedan enmascarados como `NaN`.

---

## 11. Comparación de Estrategias y Sensibilidad Metodológica

En la Fase C.1b se comparó la Estrategia A frente a la Triangulación 2D Delaunay (Estrategia B):
- **Celdas oceánicas válidas:** 5279 / 5279 (100.0%) en ambas estrategias.
- **Discrepancia MAE entre A y B:** **0.0038 °C** (P95: 0.0214 °C, max: 0.0626 °C).
- **Conclusión de sensibilidad:** La reconstrucción costera es numéricamente insensible a la elección del método de soporte; se adoptó la Estrategia A por estabilidad en cuadrícula regular y eficiencia computacional.

---

## 12. Fase C.1c — Resultados Numéricos Finales (2015-01-01)

- **Celdas oceánicas procesadas:** 5279 / 5279 (100.0% cobertura).
- **Estadísticas térmicas:**
  - $\text{SST}_{\text{MUR}}$: Media = 26.9485 °C, Mediana = 26.9460 °C, Desv. Est. = 0.1481 °C.
  - $\text{SST}_{\text{BIL}}$: Media = 27.1357 °C, Mediana = 27.1460 °C, Desv. Est. = 0.0381 °C.
  - $\text{Residual } R$: Media = -0.1872 °C, Mediana = -0.1956 °C, Desv. Est. = 0.1448 °C.
- **Identidad numérica:** Error máximo global $|R - (\text{MUR} - \text{BIL})| = 9.46 \times 10^{-7}\ ^\circ\text{C}$.

---

## 13. Baseline E0 Oficial (2015-01-01)

- **$\text{RMSE}$:** 0.2367 °C
- **$\text{MAE}$:** 0.2004 °C
- **$\text{Bias}$:** +0.1872 °C ($\text{mean}(\text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}})$)
- **$R^2$:** -1.6878 (negativo por baja varianza espacial en una fecha aislada).

---

## 14. Estabilidad Temporal de la Máscara de Soporte

La máscara de soporte costero `oisst_coastal_support_mask` (29 nodos marinos, 20 nodos de soporte auxiliar) se verificó idéntica en pruebas estacionales (invierno, primavera, verano, otoño) y a lo largo de los 4018 días de la Fase C.2.

---

## 15. Nomenclatura y Variables

| Símbolo / Variable | Nombre Completo | Unidades / Tipo | Descripción en el Framework |
| :--- | :--- | :--- | :--- |
| **$\text{SST}$** | *Sea Surface Temperature* | °C | Temperatura superficial del mar |
| **$\text{OISST}$** | *Optimum Interpolation SST v2.1* | °C | Producto satelital de baja resolución (~0.25°) de NOAA |
| **$\text{MUR}$** | *Multi-scale Ultra-high Resolution SST* | °C | Producto satelital L4 de alta resolución (~0.01°) de NASA JPL |
| **$\text{BIL}$ / $\text{SST}_{\text{BIL}}$** | *Bilinear Interpolated SST* | °C | OISST interpolado bilinealmente a la cuadrícula MUR con soporte costero |
| **$R$** | *Residual SST* | °C | Diferencia térmica objetivo: $R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$ |
| **$\hat{R}$** | *Predicted Residual* | °C | Residual estimado por el modelo de Machine Learning |
| **$\text{SST}_{\text{downscaled}}$** | *Downscaled SST* | °C | Reconstrucción final: $\text{SST}_{\text{BIL}} + \hat{R}$ |
| **`analysis_error`** | *MUR Analysis Error* | °C | Desviación estándar de error estimada por el producto MUR v4.1 |
| **$\text{DOY}$** | *Day Of Year* | Entero $[1, 366]$ | Día del año calendario |
| **$\text{depth}$** | *Bathymetric Depth* | Metros ($\text{m}$) | depth = positive meters below sea level (`depth`: GEBCO bathymetric depth, expressed as positive meters below sea level ($\text{depth} = -\text{elevation}$) |
| **$\text{distance\_coast\_km}$** | *Distance to Coast* | Kilómetros ($\text{km}$) | Distancia geodésica mínima a la costa en UTM 16N |
| **$\text{ocean\_fraction}$** | *Ocean Fraction* | Proporción $[0, 1]$ | Fracción de submalla GEBCO con elevación submarina |
| **$\text{ocean\_mask\_final}$** | *Final Ocean Mask* | Binaria $\{0, 1\}$ | Máscara definitiva océano/tierra (5279 celdas oceánicas) |

---

## 16. Splits Temporales y Diseño Anti-Fuga

Para prevenir de forma estricta la fuga de información temporal (*data leakage*), el proyecto adoptó una partición temporal secuencial hacia adelante (*strictly forward temporal splitting*), sin traslapes ni aleatorización:

```text
2015-01-01             2020-12-31 2021-01-01  2021-12-31 2022-01-01          2023-12-31 2024-01-01          2025-12-31
[------- DEVELOPMENT (TRAIN) -------][-- HOLDOUT (D32) --][-- VALIDATION (D33) --][------ FINAL TEST (D35) ------]
        6 años (2,192 días)              1 año (365 días)         2 años (730 días)            2 años (731 días)
      11,562,800 obs nominales          1,925,375 obs             3,850,750 obs               3,856,025 obs
                                                                                           [CONSUMED — D35-A]
```

- **Development Partition (2015–2020):** 2,192 días (11,562,800 filas nominales sobre 5,275 celdas ML).  
  *Nota sobre recuento de filas de desarrollo:* Nominal period counts are shown. The D32 common ablation comparison used 11,546,975 development rows after applying the common temporal-validity mask required by the three-day lag configuration.
- **Diagnostic Holdout Partition (2021):** 365 días (1,925,375 observaciones). Utilizada en D32 para selección de modelo bajo parsimonia.
- **External Validation Partition (2022–2023):** 730 días (3,850,750 observaciones). Utilizada en D33 para validación fuera de muestra.
- **Final Test Partition (2024–2025):** 731 días (3,856,025 observaciones). Evaluada exclusivamente en D35. **FINAL TEST = CONSUMED**.
- **Ventanas de reentrenamiento (*Refit training windows*):**
  - **D33 Validation Refit:** 2015–2021 (7 años, 2,557 días, 13,488,175 observaciones nominales).
  - **D35 Final Refit:** 2015–2023 (9 años, 3,287 días, 17,338,925 observaciones).

---

## 17. Fase C.2 — Armonización Completa 2015–2025

La **Fase C.2** ejecutó la armonización temporal y espacial completa de los 11 años del estudio mediante el script [`fase_c2_armonizacion_2015_2025.py`](fase_c2_armonizacion_2015_2025.py):

- **Periodo:** 2015-01-01 a 2025-12-31.
- **Total de días procesados:** **4018 / 4018 días (100.0% completitud, 0 días faltantes)**.
- **Dimensiones del cubo:** `(time: 4018, lat: 86, lon: 96)`.
- **Celdas oceánicas por día:** **5279 celdas**.
- **Observaciones espacio-temporales totales:** **21,211,022 observaciones**.
- **Identidad numérica diaria:** Verificada estrictamente en todos los días ($\max |R - (\text{MUR} - \text{BIL})| < 10^{-5}\ ^\circ\text{C}$).
- **Archivos generados:** 11 NetCDFs anuales en `DATASET_TESIS/outputs/fase_c2/` y el archivo consolidado [`DATASET_TESIS/outputs/faseC2_2015_2025.nc`](outputs/faseC2_2015_2025.nc) (567 MB).

---

## 18. Baseline E0 Global y Caracterización Espacio-Temporal

### Métricas Globales Agregadas (2015–2025)

Evaluación global del predictor base OISST interpolado ($\text{SST}_{\text{BIL}}$) frente a MUR SST ($\text{SST}_{\text{MUR}}$) sobre las 21,211,022 observaciones del periodo 2015–2025:

- **Número total de observaciones ($N$):** **21,211,022**
- **Pooled spatiotemporal RMSE:** **0.342596 °C** (Pooled spatiotemporal aggregation over all 21,211,022 spacetime cells)
- **Mean daily RMSE:** **0.301914 °C** (Mean of daily spatial RMSEs across 4,018 days)
- *Clarification:* These are distinct aggregation statistics.
- **$\text{MAE}$:** **0.2631 °C**
- **$\text{Bias}$ ($\text{mean}(\text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}})$):** **+0.0133 °C**
- **Coeficiente de determinación ($R^2$):** **0.9016**

> **Convención de Signo:** $\text{Bias} = \text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}}$. Un sesgo positivo indica que el predictor base OISST se encuentra en promedio ligeramente más cálido que MUR sobre el dominio.

### Caracterización Espacio-Temporal del Baseline

Además de las métricas globales agregadas, se evaluó la distribución espacial y temporal del error del baseline durante los 4018 días del periodo 2015–2025:

1. **Distribución Espacial del Error (Panel a de la Figura Principal):**
   El RMSE calculado individualmente para cada una de las 5279 celdas oceánicas a lo largo de los 4018 días:
   $$\text{RMSE}(x, y) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left[\text{SST}_{\text{BIL}}(t, x, y) - \text{SST}_{\text{MUR}}(t, x, y)\right]^2}$$
   - **RMSE espacial mínimo:** **0.317538 °C** (en mar abierto profundo hacia el oriente).
   - **RMSE espacial máximo:** **0.401785 °C** (en la franja costera continental occidental).
   - **Media espacial:** **0.342240 °C** (mediana espacial = 0.338157 °C).
   - *Interpretación científica:* El RMSE presenta una clara estructura espacial dentro del dominio Tulum–Cozumel, evidenciando que el desempeño de la interpolación bilineal no es espacialmente uniforme. No se afirma causalidad directa sobre los patrones espaciales observados ni se atribuyen automáticamente a profundidad, distancia a costa, batimetría o corrientes sin análisis experimental específico.

2. **Evolución Temporal del Error Diario (Panel b de la Figura Principal):**
   El RMSE diario calculado espacialmente sobre las 5279 celdas oceánicas para cada día:
   $$\text{RMSE}_t = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left[\text{SST}_{\text{BIL}}(t, i) - \text{SST}_{\text{MUR}}(t, i)\right]^2}$$
   - **Media del RMSE diario:** **0.301914 °C**
   - **Mediana del RMSE diario:** **0.263512 °C**
   - **Percentil 95 ($P_{95}$):** **0.603460 °C**
   - **Percentil 99 ($P_{99}$):** **0.858581 °C**
   - **Máximo diario observado:** **2.276469 °C** (ocurrido el **18 de octubre de 2015**, correspondiente al evento extremo E1 previamente diagnosticado).
   - *Interpretación temporal:* El error del baseline presenta variabilidad temporal considerable. Aunque la mayor parte de los días exhibe valores moderados de RMSE ($\text{mediana} = 0.2635\ ^\circ\text{C}$), existen episodios puntuales con discrepancias sustancialmente superiores que no deben interpretarse automáticamente como errores de MUR u OISST.

3. **Implicación para el Downscaling mediante Machine Learning:**
   Estos resultados demuestran que el buen desempeño agregado del baseline ($R^2 = 0.9016$) no implica que las discrepancias sean uniformes en el espacio o en el tiempo. La existencia demostrada de estructura espacial y variabilidad temporal en el residual proporciona la base y justificación científica para evaluar posteriormente si los modelos de Machine Learning pueden reducir dichas discrepancias respecto al baseline E0.

---

## 19. Diagnóstico del Evento Anómalo de Octubre de 2015

Durante la inspección de las series temporales de error de la Fase C.2 se detectó el pico máximo de discrepancia diaria en el año 2015:

- **Fecha crítica:** **2015-10-18**
  - **$\text{RMSE}$:** **2.2765 °C**
  - **$\text{MAE}$:** **2.2643 °C**
  - **$\text{Bias}$ ($\text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}}$):** **+2.2643 °C**
  - **Ratio $|\text{Bias}| / \text{RMSE}$:** **0.9947** (desplazamiento térmico casi uniforme en el dominio).
- **Fecha adyacente (2015-10-19):** $\text{RMSE} = 2.1525\ ^\circ\text{C}$, $\text{Bias} = +2.1456\ ^\circ\text{C}$.

### Evolución Temporal de Medias Regionales (Corredor Tulum–Cozumel):

| Fecha | $\text{SST}_{\text{MUR}}$ Media (°C) | $\text{SST}_{\text{OISST/BIL}}$ Media (°C) | $\Delta \text{MUR}$ Diario (°C) | $\Delta \text{OISST}$ Diario (°C) |
| :--- | :--- | :--- | :--- | :--- |
| **2015-10-15** | 30.0047 | 29.7172 | — | — |
| **2015-10-16** | 29.8854 | 29.5894 | -0.1193 | -0.1278 |
| **2015-10-17** | 29.0611 | 29.5929 | -0.8243 | +0.0035 |
| **2015-10-18** | **27.3015** | **29.5658** | **-1.7596** | **-0.0271** |
| **2015-10-19** | **27.4260** | **29.5716** | +0.1245 | +0.0058 |
| **2015-10-20** | 28.3724 | 29.1724 | +0.9464 | -0.3992 |
| **2015-10-21** | 28.4880 | 28.6253 | +0.1156 | -0.5471 |

---

## 20. Validación Satelital Externa — VIIRS S-NPP L2P v2.80

Se realizó una validación satelital externa con el producto infrarrojo de Nivel 2P **VIIRS S-NPP L2P v2.80** (DOI: 10.5067/GHVRS-2PO28, resolución ~750 m):
- **Archivos analizados:** 15 NetCDF crudos en la ventana del 15 al 20 de octubre de 2015.
- **Resultados en la ventana crítica:** En 7 de 8 pasos, la cobertura de calidad $QL=5$ fue $0.0\%$. En el paso del 18-Oct 18:40 UTC se observaron 10 píxeles aislados ($0.11\%$ de cobertura) con media de $29.55\ ^\circ\text{C}$ (cercano a OISST $29.62\ ^\circ\text{C}$ y superior a MUR $27.32\ ^\circ\text{C}$).
- **Clasificación:** **INCONCLUSO** por cobertura muestral insuficiente para caracterizar el canal regional.

---

## 21. Validación Satelital Externa — MODIS Aqua L2P v2019.0

Se procesaron los 14 archivos NetCDF de **MODIS Aqua L2P v2019.0** (DOI: 10.5067/GHMDA-2PJ19, resolución ~1 km):
- **Canales evaluados:** Térmico de 11 µm y nocturno de 4 µm.
- **Resultados en la ventana crítica:** Durante la ventana principal del evento no se obtuvieron observaciones de alta calidad utilizables bajo los criterios establecidos para la validación ($N_{\text{QL5}} = 0$ y $N_{\text{usable}} = 0$ para $QL \geq 4$).
- **Interpretación:** La ausencia de observaciones utilizables impide realizar una comparación externa directa suficientemente robusta entre MODIS, MUR y OISST durante la ventana crítica. Los niveles de calidad observados no permiten atribuir por sí solos la ausencia de datos utilizables exclusivamente a nubosidad.
- **Clasificación:** **INCONCLUSO**, debido a disponibilidad insuficiente de observaciones SST de alta calidad durante la ventana analizada.

---

## 22. Auditoría Satelital Comparativa Multievento (E1–E6)

Se evaluaron los seis principales eventos de discrepancia extrema detectados en la serie 2015–2025 mediante observaciones satelitales externas VIIRS S-NPP y MODIS Aqua. La disponibilidad de observaciones infrarrojas de alta calidad fue heterogénea entre eventos, por lo que la interpretación se realizó considerando tanto la cobertura disponible como las métricas de comparación colocalizada:

| Evento | Periodo | $\text{RMSE}_{\max}$ (°C) | Evidencia VIIRS S-NPP | Evidencia MODIS Aqua | Dictamen |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **E1** | 2015-10-17 → 2015-10-21 | **2.276** | Cobertura QL=5 extremadamente limitada (0.11%); los 10 píxeles disponibles fueron localmente más próximos a OISST/BIL | Sin observaciones de alta calidad utilizables durante la ventana principal | **INCONCLUSO** |
| **E2** | 2021-11-17 → 2021-11-19 | **1.463** | Cobertura QL=5 insuficiente para una evaluación regional robusta | Cobertura de alta calidad insuficiente | **INCONCLUSO** |
| **E3** | 2024-10-19 → 2024-10-20 | **1.292** | Sin cobertura de alta calidad suficiente para una comparación regional | Sin cobertura de alta calidad suficiente para una comparación regional | **INCONCLUSO** |
| **E4** | 2015-08-03 → 2015-08-06 | **1.102** | La comparación colocalizada favoreció OISST/BIL sobre una fracción considerable de celdas MUR muestreadas | La muestra disponible, considerablemente menor, favoreció MUR | **HETEROGÉNEO; evidencia VIIRS favorable a BIL** |
| **E5** | 2016-06-04 → 2016-06-08 | **1.059** | Comparación favorable a OISST/BIL en la fecha pico | Comparación favorable a MUR en la fecha pico | **MIXTO / INCONCLUSO** |
| **E6** | 2019-06-14 → 2019-06-17 | **1.030** | Diferencias pequeñas en la fecha pico; considerando el evento completo, la comparación favorece OISST/BIL | Muestra reducida y comportamiento mixto en la fecha pico; el evento completo presenta mayor proximidad a BIL | **HETEROGÉNEO; tendencia hacia BIL en el evento completo** |

> **Nota metodológica:** Los porcentajes de cobertura empleados en esta auditoría representan la fracción de celdas oceánicas de la cuadrícula MUR que contienen al menos una observación satelital externa válida y no deben interpretarse como una equivalencia exacta de área geográfica observada.

> **Conclusión de la Validación Satelital:** La evidencia infrarroja obtenida es **HETEROGÉNEA Y, EN VARIOS EVENTOS, INCONCLUSA**. Los eventos E1–E3 carecen de cobertura externa suficiente para confirmar o refutar regionalmente las discrepancias MUR–OISST. En E4 existe evidencia VIIRS sobre una fracción considerable del dominio observado que favorece OISST/BIL, mientras que E5 presenta resultados contradictorios entre sensores. E6 muestra comportamiento mixto en la fecha pico, aunque el análisis del evento completo tiende a favorecer OISST/BIL. En consecuencia, no existe evidencia suficiente para clasificar de manera general los eventos extremos como errores de MUR ni para modificar el dataset armonizado de la Fase C.2.

---

## 23. Incertidumbre MUR (`analysis_error`)

Inicialmente, la serie local de `analysis_error` presentaba un hueco temporal entre `2016-01-01` y `2019-07-22` debido a la descarga histórica previa de sólo SST. Dicho hueco fue recuperado de manera completa y directa desde NASA PO.DAAC / NOAA CoastWatch ERDDAP sin utilizar intermediarios Harmony:

- **Producto:** `MUR-JPL-L4-GLOB-v4.1` (DOI: 10.5067/GHGMR-4FJ04).
- **Periodo Final Disponible:** **2015-01-01 a 2025-12-31** (4018 / 4018 días).
- **Faltantes / Duplicados:** **0 faltantes, 0 duplicados**.
- **Archivo Consolidado:** [`DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc`](analysis_error_historico/mur_analysis_error_2015_2025_completo.nc) ($4018 \times 86 \times 96$, 133 MB).

### Verificación Numérica de Recuperación (Prueba Piloto 2016-06-05):
Se verificó la variable `analysed_sst` recuperada frente al NetCDF de MUR utilizado en la Fase C.2:
- $\text{MAE} \approx 0.000000\ ^\circ\text{C}$
- $\text{RMSE} \approx 0.000001\ ^\circ\text{C}$
- $\text{Max Abs Difference} \approx 0.000001\ ^\circ\text{C}$
- **Conclusión:** La recuperación histórica corresponde con exactitud de máquina al mismo producto MUR v4.1 empleado en Fase C.2.

---

## 24. Distribución Global y Estadísticas de `analysis_error` (2015–2025)

Sobre la totalidad de los 4018 días calendario ($N = 21,211,022$ observaciones espacio-temporales):

| Métrica Estadística | Valor (°C) | Descripción |
| :--- | :---: | :--- |
| **Mediana Global ($\text{mean\_AE}$)** | **0.3848 °C** | Nivel de incertidumbre típico del análisis multiescala |
| **Percentil 90 ($P_{90}$)** | **0.4003 °C** | Umbral de incertidumbre moderadamente elevada |
| **Percentil 95 ($P_{95}$)** | **0.4056 °C** | Umbral de incertidumbre alta |
| **Percentil 99 ($P_{99}$)** | **0.4100 °C** | Valor alcanzado por el extremo superior de la distribución diaria |
| **Máximo Observado** | **0.4100 °C** | Máximo observado de `analysis_error` en el conjunto analizado |
| **MAD (Median Absolute Deviation)** | **0.0053 °C** | Dispersión robusta de la incertidumbre media diaria |
| **IQR (Rango Intercuartílico)** | **0.0110 °C** | Dispersión intercuartil ($P_{75} - P_{25}$) |

### Frecuencia del Valor Máximo Observado ($0.4100^\circ\text{C}$):
- **Frecuencia espacio-temporal global:** **6.97%** (1,477,419 de 21,211,022 celdas oceánicas).
- **Días con más del 50% de las celdas oceánicas en el valor máximo observado (`fraction_at_041 > 0.50`):** **245 días** (6.10% del registro).
- **Días con más del 90% de las celdas oceánicas en el valor máximo observado (`fraction_at_041 > 0.90`):** **108 días** (2.69% del registro).

---

## 25. Relación entre `analysis_error` y Discrepancia MUR–OISST

Se evaluó la correlación entre la incertidumbre reportada por MUR y las métricas de discrepancia frente a OISST sobre los 4018 días continuos:

| Par de Variables Evaluado | Pearson $r$ | Spearman $\rho$ | $p$-value | Interpretación Estadística |
| :--- | :---: | :---: | :---: | :--- |
| **$\text{RMSE}$ vs $\text{mean\_AE}$** | **+0.3338** | **+0.2853** | $< 10^{-70}$ | Asociación positiva moderada |
| **$|\text{Bias}|$ vs $\text{mean\_AE}$** | **+0.2780** | **+0.1964** | $< 10^{-40}$ | Asociación positiva leve a moderada |
| **$|\Delta\text{MUR} - \Delta\text{BIL}|$ vs $\text{mean\_AE}$** | **+0.2660** | **+0.2039** | $< 10^{-44}$ | Asociación positiva con saltos térmicos diferenciales |

> **Interpretación:** Existe una asociación positiva moderada y estadísticamente significativa entre la discrepancia regional MUR–OISST y la incertidumbre interna de MUR, lo que indica que en episodios de gran discrepancia la incertidumbre tiende a incrementarse. Sin embargo, **la relación no es determinista** ($r \approx 0.33$), por lo que una alta discrepancia no implica unívocamente un fallo en MUR ni viceversa.

---

## 26. Gradiente de Severidad y Comportamiento en Eventos E1–E6

### Gradiente de Severidad Monotónico Promedio:

| Rango de Severidad $\text{RMSE}$ | Días ($N$) | $\text{mean\_AE}$ Promedio (°C) | Fracción en el Máximo Observado ($0.4100^\circ\text{C}$) |
| :--- | :---: | :---: | :---: |
| **Normal ($< P_{90}$, $\text{RMSE} < 0.44^\circ\text{C}$)** | 3616 | **0.3860 °C** | 5.26% |
| **Moderado ($P_{90} \le \text{RMSE} < P_{95}$)** | 201 | **0.3918 °C** | 16.21% |
| **Alto ($P_{95} \le \text{RMSE} < P_{99}$)** | 161 | **0.3944 °C** | 22.97% |
| **Muy Alto ($P_{99} \le \text{RMSE} < P_{99.5}$)** | 20 | **0.4018 °C** | 39.19% |
| **Extremo ($\ge P_{99.5}$, $\text{RMSE} \ge 0.90^\circ\text{C}$)** | 20 | **0.4040 °C** | **59.92%** |

### Comportamiento Específico en los Eventos E1 a E6:

| Evento | Fecha Pico | $\text{RMSE}$ (°C) | $\text{mean\_AE}$ (°C) | Fracción @ máximo observado ($0.4100^\circ\text{C}$) | Percentil $\text{mean\_AE}$ | Clasificación de Incertidumbre |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1** | `2015-10-18` | **2.276 °C** | **0.4100 °C** | **100.0%** | **100.0%** | **EXTREMO (100% del dominio en 0.4100 °C)** |
| **E2** | `2021-11-17` | **1.463 °C** | **0.4100 °C** | **100.0%** | **100.0%** | **EXTREMO (100% del dominio en 0.4100 °C)** |
| **E3** | `2024-10-19` | **1.292 °C** | **0.4100 °C** | **100.0%** | **100.0%** | **EXTREMO (100% del dominio en 0.4100 °C)** |
| **E4** | `2015-08-03` | **1.102 °C** | **0.3908 °C** | **0.0%** | **72.5%** | **NORMAL (Contraejemplo claro)** |
| **E5** | `2016-06-05` | **1.059 °C** | **0.4005 °C** | **27.2%** | **90.2%** | **ELEVADO** |
| **E6** | `2019-06-14` | **1.030 °C** | **0.4008 °C** | **29.5%** | **90.7%** | **ELEVADO** |

> **Evaluación Científica:**
> - **Eventos E1–E3:** Presentan discrepancias extremas MUR–OISST coincidentes con el valor superior de saturación de `analysis_error` ($0.4100^\circ\text{C}$) en la totalidad del dominio. No obstante, la evidencia externa disponible no permite confirmar por sí sola que estos eventos sean artefactos instrumentales de MUR.
> - **Evento E4:** Constituye un contraejemplo contundente donde ocurrió una discrepancia extrema ($\text{RMSE} = 1.102^\circ\text{C}$) con un nivel de incertidumbre estrictamente normal ($\text{mean\_AE} = 0.3908^\circ\text{C}$, percentil 72.5%, 0% de saturación).
> - **Conclusión:** La hipótesis de una firma universal de `analysis_error` en todos los eventos extremos se cumple de manera **PARCIAL**.

---

## 27. Auditoría de Cambios Abruptos ($\Delta\text{AE}$)

Se evaluaron las $4017$ transiciones diarias $\Delta\text{AE}(t) = \text{mean\_AE}(t) - \text{mean\_AE}(t-1)$ en el periodo 2015–2025:

- **Mediana de $|\Delta\text{AE}|$:** **0.003660 °C**
- **MAD:** **0.002351 °C**
- **Percentil 95 ($P_{95}$):** **0.010970 °C**
- **Percentil 99 ($P_{99}$):** **0.014818 °C** ($N = 41$ días)
- **Percentil 99.9 ($P_{99.9}$):** **0.019165 °C** ($N = 5$ días)
- **Mayor Incremento Diario:** `2016-05-24` ($\Delta\text{AE} = +0.053220^\circ\text{C}$)
- **Mayor Descenso Diario:** `2016-05-23` ($\Delta\text{AE} = -0.047630^\circ\text{C}$)

---

## 28. Diagnóstico de la Anomalía Puntual `analysis_error = 0` (23 Mayo 2016)

Durante el control temporal se identificó que el mínimo absoluto del registro ocurrió el **`2016-05-23`** con $\text{mean\_AE} = 0.3280^\circ\text{C}$, debido a que 744 de las 5279 celdas oceánicas reportaron $\text{analysis\_error} = 0.000^\circ\text{C}$.

### Resultados de la Investigación Diagnóstica:
1. **Presencia en Fuente Remota:** Se confirmó en el NetCDF original de NASA JPL / NOAA CoastWatch que las 744 celdas contienen el entero `RAW = 0` de forma nativa ($\text{max diff} = 0.000000^\circ\text{C}$, $\text{_FillValue} = -32768$). No es un fallo de descarga, decodificación CF ni de manejo de NaNs.
2. **Estructura Espacial en Franjas:** Las 744 celdas forman franjas horizontales continuas a lo largo de 10 filas de latitud en el sector sur ($19.97^\circ\text{N}$ a $20.21^\circ\text{N}$).
3. **Ubicación Geográfica:** Se ubican en mar abierto profundo (mediana de profundidad GEBCO: $748.42\text{ m}$, distancia a costa: $19.09\text{ km}$, $\text{ocean\_fraction} = 1.000$), descartando artefactos costeros someros.
4. **Continuidad de SST:** La temperatura superficial del mar (`analysed_sst`) en esas 744 celdas es físicamente continua y nominal ($\text{mean} = 29.0909^\circ\text{C}$, $\text{RMSE} = 0.2812^\circ\text{C}$).
5. **Unicidad Temporal:** Los 744 valores cero ocurren exclusivamente durante 24 horas el 23 de mayo de 2016. En todo el registro 2015–2025 ($N = 4018$ días), **el 100% de los ceros pertenece a este único día**.

> **Interpretación Oficial:**  
> *"Los valores `analysis_error = 0` están presentes en el producto MUR v4.1 original y no resultan de errores de descarga, decodificación, máscara o concatenación. Su aparición exclusivamente durante un día, su estructura espacial en franjas y la ausencia de una anomalía correspondiente en `analysed_sst` indican un comportamiento excepcional del campo de incertidumbre. La documentación consultada no proporciona una explicación suficiente para establecer su origen algorítmico."*  
> **Clasificación:** Posible anomalía del campo `analysis_error`, **NO** de `analysed_sst`.  
> **Decisión:** Se conserva el dato original sin modificar Fase C.2, sin eliminar la fecha y sin alterar las conclusiones de E1–E6.

---

## 29. Archivado y Trazabilidad de Productos Obsoletos

Para garantizar la reproducibilidad y mantener un historial auditable:
- `DATASET_TESIS/archive/analysis_error_auditoria_2074dias_obsoleta/`: Contiene la auditoría preliminar basada en $N = 2074$ días (superada por la versión consolidada $N = 4018$).
- `DATASET_TESIS/archive/figure_versions/`: Contiene versiones gráficas intermedias de figuras (e.g. `figura1_serie_temporal_rmse_analysis_error_legend_old.png`, `figura7_delta_analysis_error_2015_2025_legend_old.png`).
- `DATASET_TESIS/archive/validacion_infrarroja_pre_correccion_final/`: Manifiesto y respaldos de validaciones satelitales preliminares.

---

## 30. Estado Científico y Conclusiones de la Auditoría de Calidad

```text
============================================================
ESTADO DE LAS AUDITORÍAS DE CONTROL DE CALIDAD: CERRADAS
============================================================
```

1. **`analysis_error` aporta información diagnóstica complementaria** sobre determinados episodios de discrepancia entre MUR y OISST. A escala diaria existe una asociación positiva entre la incertidumbre media reportada por MUR y la magnitud de la discrepancia MUR–OISST; sin embargo, dicha asociación es moderada y no determinista.

2. **`analysis_error` no constituye una medida directa del error verdadero de MUR** y, por tanto, no puede utilizarse de forma aislada para clasificar observaciones como correctas o incorrectas, identificar automáticamente artefactos ni justificar la eliminación de fechas del dataset.

3. **Los eventos E1–E3 presentan simultáneamente discrepancias MUR–OISST extremas y `analysis_error = 0.4100 °C` en el 100% del dominio**, mientras que E4 demuestra que una discrepancia extrema también puede producirse con niveles ordinarios de `analysis_error`. Por ello, no existe una firma universal de incertidumbre asociada a todos los eventos extremos.

4. **El episodio del 23 de mayo de 2016 constituye una anomalía específica del campo `analysis_error`**, presente directamente en el producto MUR v4.1 original. Los 744 valores iguales a 0.000 °C no corresponden a `_FillValue`, errores de decodificación, máscara o concatenación y no están acompañados por una discontinuidad equivalente en `analysed_sst`. Su origen algorítmico específico no puede determinarse con la evidencia actualmente disponible.

5. **No existe evidencia suficiente para eliminar o corregir manualmente observaciones de MUR.** Los eventos investigados se conservan en su forma original para mantener la integridad y trazabilidad del producto utilizado.

6. **El producto consolidado `faseC2_2015_2025.nc` permanece íntegro y sin modificaciones ad hoc.** Ninguno de los resultados de las auditorías justifica modificar la Fase C.2.

7. **Las figuras de auditoría de incertidumbre y validación satelital se conservan como material de control de calidad o suplementario.** No sustituyen a las figuras principales del manuscrito orientadas a la evaluación del downscaling.

---

## 31. Historical Planning Note — Original Pre-ML Design

> **Historical traceability notice:** This section is preserved for historical traceability only. The planned workflow described below was subsequently executed and superseded by Stages D.1 and D31–D36.
>
> **Actual Executed Workflow Summary:**
> - **Stage D.1:** Constructed the unified tabular ML dataset with 5,275 frozen ML cells across all 4,018 days (21,194,950 total rows) with zero missing dates and zero duplicate timestamps. spatial metadata hash = 8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365.
> - **Stage D31:** Predictability diagnostics supporting reproducible predictive structure in the residual.
> - **Stage D32:** Evaluated 6 D32 configurations across tree-based algorithms; selected the 4-feature specification `E3b-C0` (`sst_bil`, `doy_sin`, `doy_cos`, `depth`) based on the parsimony principle.
> - **Stage D33:** D33: Positive global and spatial generalization, but insufficient month-level stability to satisfy the predeclared D33-A criterion. Formal status: D33-B (16 / 24 months improved).
> - **Stage D34:** Completed diagnostic audit characterizing performance across residual regimes.
> - **Stage D35:** Terminal confirmatory evaluation on Final Test 2024–2025: All seven predeclared D35-A criteria were satisfied. Formal status: **D35-A — FINAL GENERALIZATION CONFIRMED** (+7.2247% relative RMSE improvement, 18/24 months, 9/12 in 2024, 9/12 in 2025, 484/731 days, 5273/5275 cells, moving-block bootstrap $CI_{95} = [-0.0425, -0.0098]^\circ	ext{C}$).
> - **Terminal Status:** **FINAL TEST = CONSUMED**. **ML MODEL DEVELOPMENT = CLOSED**. **DOCUMENTATION LAYER = AUDITED & FROZEN**.

### Original Pre-ML Design Concept (Archived)

1. **Variable objetivo original:** Se mantuvo la formulación residual $R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$.
2. **Predictores candidatos evaluados:** En el diseño preliminar se listaron `sst_bil`, `depth`, `distance_coast_km`, `ocean_fraction`, `doy_sin`, `doy_cos`, lags temporales y gradientes térmicos. Tras la ablación rigurosa de D32, únicamente los 4 predictores esenciales fueron retenidos en `E3b-C0`.
3. **Tratamiento de `analysis_error`:** Conforme a lo previsto, no se utilizó como predictor directo por provenir de la propia referencia L4, preservando la independencia metodológica.
4. **Prevención de fuga:** Implementada rigurosamente mediante el particionamiento temporal hacia adelante.

---

## 32. Selección de Figuras del Manuscrito y Catálogo Gráfico

### Final ML Paper Figures (ICITS'27 Core Selection)

Las figuras principales seleccionadas para el manuscrito científico de Machine Learning corresponden a los resultados consolidados de D35–D36:

1. **Figure 1: Final Test Performance (`fig1_final_performance.png`):**  
   Panel dual: comparación de densidad de error residual y dispersión observada vs predicha en Final Test 2024–2025 ($B_0$ vs $C_0$).
2. **Figure 2: Temporal Robustness (`fig2_temporal_robustness.png`):**  
   Serie temporal de $\Delta\text{RMSE}$ diario con media móvil de 30 días y desglose mensual de mejora a lo largo de los 731 días de prueba (18/24 months, 9/12 in 2024, 9/12 in 2025, 484/731 days).
3. **Figure 3: Spatial Skill (`fig3_spatial_skill.png`):**  
   Distribución espacial del $\Delta\text{RMSE}$ por celda oceánica (5273/5275 cells mejoradas, 99.96% cobertura espacial). Larger improvements were observed in shallow cells, and improvement magnitude also decreased with increasing offshore distance.
4. **Figure 4: Residual-Regime Dependence (`fig4_residual_regimes.png`):**  
   Desempeño estratificado por percentiles pre-congelados de discrepancia MUR–BIL ($|R|$). Muestra una ganancia notable en discrepancias elevadas y a 20.99% relative degradation within the low-discrepancy regime (DEV-P0–P50 with DEV-P50 = 0.2066, spanning to DEV-P99 = 0.9659), donde los errores base ya eran pequeños dejando poco margen para corrección beneficiosa.
5. **Figure 5: Historical Descriptive Skill (`fig5_historical_skill.png`):**  
   Evaluación retrospectiva descriptiva a lo largo de la serie decadal completa 2015–2025 (candidata para tesis y suplemento).

### Historical A–C Figures (Preserved for Methods and Appendices)

Las siguientes figuras corresponden a las fases iniciales de adquisición y armonización (Etapas A–C) y se preservan como candidatos para metodología de tesis y apéndices suplementarios:

- **Figura A-1 (Dominio Fisiográfico):** Caracterización de máscara oceánica, batimetría y distancia a costa.
- **Figura A-2 (Armonización Espacial OISST):** Soporte costero auxiliar de 20 nodos terrestres e interpolación bilineal.
- **Figura A-3 (Baseline E0 Espacio-Temporal):** [`figura_baseline_rmse_espacio_temporal_2015_2025.png`](figures/figura_baseline_rmse_espacio_temporal_2015_2025.png) con mapas de RMSE por celda y serie decadal diaria.
- **Figuras de Auditoría Satelital e Incertidumbre:** Figuras de validación L2P (VIIRS/MODIS) y auditoría de `analysis_error` en `auditoria_analysis_error/figures/`.

---

## 33. Reproducibilidad y Guía de Ejecución

### Entorno de Ejecución
- **Sistema Operativo:** macOS Darwin (`arm64`) / Linux x86_64
- **Python:** 3.11.16
- **Core ML Runtime:** `xgboost == 2.1.4` (exact canonical environment; avoid speculative runtime compatibility)

### Guía de Comandos Reproducibles

1. **Ejecución de Fase C.2 (Armonización completa 4018 días):**
   ```bash
   python DATASET_TESIS/fase_c2_armonizacion_2015_2025.py
   ```
2. **Generación de la Figura Principal del Baseline E0 para el Paper (PNG/PDF):**
   ```bash
   python DATASET_TESIS/generar_figura_baseline_paper.py
   ```
3. **Recuperación histórica de `analysis_error` (PO.DAAC OPeNDAP):**
   ```bash
   python DATASET_TESIS/descargar_analysis_error_mur_opendap.py
   ```
4. **Auditoría final consolidada de `analysis_error` (N = 4018 días):**
   ```bash
   python DATASET_TESIS/auditar_analysis_error_mur_FINAL.py
   ```
5. **Control temporal de saltos diarios ($\Delta\text{AE}$):**
   ```bash
   python DATASET_TESIS/auditar_cambios_abruptos_analysis_error.py
   ```
6. **Diagnóstico puntual de `analysis_error = 0` (2016-05-23):**
   ```bash
   python DATASET_TESIS/diagnosticar_zero_analysis_error_20160523.py
   ```

---

## 34. Catálogo de Artefactos y Figuras

### Productos NetCDF Principales
- [`DATASET_TESIS/outputs/faseC2_2015_2025.nc`](outputs/faseC2_2015_2025.nc): Cubo consolidado 4018 días (567 MB).
- [`DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc`](analysis_error_historico/mur_analysis_error_2015_2025_completo.nc): Serie continua de `analysis_error` (133 MB).

### Figura Principal del Paper (`DATASET_TESIS/figures/`)
- **[Figura Baseline E0 (PNG 300 DPI): `figura_baseline_rmse_espacio_temporal_2015_2025.png`](figures/figura_baseline_rmse_espacio_temporal_2015_2025.png)**
- **[Figura Baseline E0 (PDF Vectorial): `figura_baseline_rmse_espacio_temporal_2015_2025.pdf`](figures/figura_baseline_rmse_espacio_temporal_2015_2025.pdf)**
  - **Descripción:** Panel (a) RMSE temporal espacial por celda oceánica durante 2015–2025; Panel (b) Evolución temporal del RMSE diario y media móvil de 30 días.
  - **Propósito:** Caracterizar el desempeño espacio-temporal del baseline $\text{SST}_{\text{BIL}}$ antes de aplicar Machine Learning.
  - **Clasificación:** **Figura principal del paper.**
  - **Caption oficial:** [`caption_figura_baseline_rmse.md`](figures/caption_figura_baseline_rmse.md).

### Figuras Oficiales de Auditoría de Calidad / Material Suplementario (`DATASET_TESIS/auditoria_analysis_error/figures/`)
- **[Figura 1 (Auditoría): `figura1_serie_temporal_rmse_analysis_error.png`](auditoria_analysis_error/figures/figura1_serie_temporal_rmse_analysis_error.png):** Serie temporal completa 2015–2025 de RMSE vs `analysis_error` (leyendas optimizadas: superior derecha / inferior derecha).
- **[Figura 2 (Auditoría): `figura2_scatter_rmse_analysis_error.png`](auditoria_analysis_error/figures/figura2_scatter_rmse_analysis_error.png):** Dispersión entre RMSE y `mean_AE` ($r = +0.3338$).
- **[Figura 3 (Auditoría): `figura3_scatter_absbias_analysis_error.png`](auditoria_analysis_error/figures/figura3_scatter_absbias_analysis_error.png):** Dispersión entre $|Bias|$ y `mean_AE` ($r = +0.2780$).
- **[Figura 4 (Auditoría): `figura4_boxplot_analysis_error_grupos.png`](auditoria_analysis_error/figures/figura4_boxplot_analysis_error_grupos.png):** Distribución de `analysis_error` por grupos de severidad de RMSE.
- **[Figura 5 (Auditoría): `figura5_distribucion_fraction_at_041.png`](auditoria_analysis_error/figures/figura5_distribucion_fraction_at_041.png):** Frecuencia y dispersión de la fracción en saturación ($0.4100^\circ\text{C}$).
- **[Figura 6 (Auditoría): `figura6_comparacion_eventos_E1_E6.png`](auditoria_analysis_error/figures/figura6_comparacion_eventos_E1_E6.png):** Comparación cuantitativa de los eventos extremos E1 a E6.
- **[Figura 7 (Auditoría): `figura7_delta_analysis_error_2015_2025.png`](auditoria_analysis_error/figures/figura7_delta_analysis_error_2015_2025.png):** Serie temporal de cambios abruptos $\Delta\text{AE}$ con Top 10 transiciones marcadas.
- **[Figuras 8, 9 y 10 (Auditoría Episodio 2016-05-23):** Paneles espaciales, máscara sobre batimetría y serie temporal del mínimo de incertidumbre.

### Reportes Oficiales (`DATASET_TESIS/auditoria_analysis_error/reports/`)
- [`auditoria_analysis_error_FINAL_4018dias.md`](auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md): Informe técnico de la auditoría final de 4018 días.
- [`auditoria_cambios_abruptos_analysis_error.md`](auditoria_analysis_error/reports/auditoria_cambios_abruptos_analysis_error.md): Informe de control temporal de saltos diarios.
- [`diagnostico_analysis_error_zero_20160523.md`](auditoria_analysis_error/reports/diagnostico_analysis_error_zero_20160523.md): Diagnóstico exhaustivo del valor cero del 23 de mayo de 2016.

---

## 35. Limitaciones Metodológicas

Conforme al marco epistemológico establecido en `THESIS_LIMITATIONS_A_D.md`:

1. **Naturaleza de la Referencia:** MUR SST v4.1 opera como una referencia analizada L4 multiescala de alta resolución, no como verdad absoluta (*ground truth* in situ). En periodos con nubosidad prolongada, su resolución dinámica efectiva se reduce.
2. **Dependencia de Régimen Discrepante:** El modelo `E3b-C0` demuestra una fuerte dependencia del régimen de discrepancia: reduce el error marcadamente en discrepancias grandes (DEV-P75–P90, DEV-P90–P99), pero exhibe a 20.99% relative degradation within the low-discrepancy regime (anchored at DEV-P50 = 0.2066 up to DEV-P99 = 0.9659). Baseline absolute errors in this regime were already small, leaving less margin for beneficial correction.
3. **Estructura Predictiva y Compresión de Amplitud:** Las predicciones muestran compresión de amplitud respecto al residual observado ($\text{Var}(\hat{R}) < \text{Var}(R)$), una propiedad característica de la minimización del error cuadrático medio. Los diagnósticos respaldan estructura predictiva reproducible en el residual sin inferir determinismo físico estricto.
4. **Asociación Espacial Descriptiva:** Larger improvements were observed in shallow cells, and improvement magnitude also decreased with increasing offshore distance. Se reportan como patrones descriptivos empíricos manteniendo separadas `depth` y `distance_coast_km` sin inferir causalidad batimétrica directa ni reconstrucción submesoescala.
5. **No Extrapolación a Olas de Calor Marinas (MHW):** El framework de downscaling no fue diseñado específicamente para la detección o resolución de eventos extremos térmicos marinos agudos.
6. **Dependencia Temporal de Corto Alcance:** El remanente residual conserva autocorrelación temporal de corto rango (~3–5 días), verificada mediante bootstrap por bloques de 14 días.

---

## 36. Deuda de Documentación Interna

1. [`DATASET_TESIS/config.py`](config.py): Especificar unidades físicas y sistemas de referencia geodésica a nivel de constante.
2. [`DATASET_TESIS/modules/bathymetry.py`](modules/bathymetry.py): Documentar explícitamente la cobertura exhaustiva en bordes costeros.
3. [`DATASET_TESIS/modules/residual.py`](modules/residual.py): Registrar nota de deprecación interna señalando a `interpolate_strategy_a_coastal_support` como función canónica.

---

> **Última actualización:** 2026-09-05  
> **Estado:** FASES A–D = COMPLETADAS | D31–D36 = CERRADAS | FINAL TEST = CONSUMED | ML MODEL DEVELOPMENT = CLOSED | DOCUMENTATION LAYER = AUDITED & FROZEN

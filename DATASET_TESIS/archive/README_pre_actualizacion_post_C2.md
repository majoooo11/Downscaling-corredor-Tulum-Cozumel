# SST Downscaling Framework for the Tulum–Cozumel Reef Corridor

> **Subtítulo:** Pipeline reproducible para la armonización espacial y temporal de MUR SST, NOAA OISST v2.1 y GEBCO orientado al downscaling de temperatura superficial del mar mediante Machine Learning en el Caribe mexicano.  
> **Estado del proyecto:** **EN DESARROLLO** (Fases B.1 y C.1 completadas y validadas; Fase C.2 y modelación Machine Learning pendientes).

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
13. [Baseline E0 Oficial](#13-baseline-e0-oficial)
14. [Estabilidad Temporal de la Máscara de Soporte](#14-estabilidad-temporal-de-la-máscara-de-soporte)
15. [Nomenclatura y Variables](#15-nomenclatura-y-variables)
16. [Splits Temporales](#16-splits-temporales)
17. [Reproducibilidad y Guía de Ejecución](#17-reproducibilidad-y-guía-de-ejecución)
18. [Catálogo de Artefactos Generados](#18-catálogo-de-artefactos-generados)
19. [Limitaciones Metodológicas](#19-limitaciones-metodológicas)
20. [Próxima Fase: Fase C.2](#20-próxima-fase-fase-c2)
21. [Deuda de Documentación Interna](#21-deuda-de-documentación-interna)

---

## 1. Objetivo Científico y Enfoque Residual

El objetivo de este proyecto de tesis es diseñar, construir y validar un pipeline computacional riguroso y reproducible que permita generar un dataset maestro armonizado para el **downscaling espacial de Temperatura Superficial del Mar (SST, *Sea Surface Temperature*)** en el corredor arrecifal Tulum–Cozumel (Caribe mexicano), fusionando observaciones satelitales multiescala y covariables fisiográficas para su posterior modelación mediante Machine Learning.

### Formulación del Enfoque Residual

En lugar de predecir directamente el campo absoluto de alta resolución, el framework adopta una formulación residual desacoplada:

1. **Componente de baja resolución interpolada ($\text{SST}_{\text{BIL}}$):**  
   Se proyecta el campo térmico de baja resolución (~0.25°) de NOAA OISST v2.1 hacia la cuadrícula de alta resolución (~0.01°) de MUR SST utilizando interpolación bilineal con soporte costero:
   $$\text{SST}_{\text{BIL}}(t, x, y) = \mathcal{I}_{\text{bilinear}}\Big(\text{OISST}_{\text{extended}}(t, X, Y)\Big)$$

2. **Cálculo del Residual Objetivo ($R$):**  
   El residual representa la señal térmica sub-malla, gradientes locales costeros y variabilidad de mesoescala no capturada por OISST:
   $$R(t, x, y) = \text{SST}_{\text{MUR}}(t, x, y) - \text{SST}_{\text{BIL}}(t, x, y) \quad \forall (x, y) \in \text{ocean\_mask\_final}$$

3. **Modelación Futura (Machine Learning):**  
   Posteriormente, los modelos de aprendizaje supervisado (árboles de decisión potenciados, redes neuronales convolucionales, etc.) aprenderán a estimar dicho residual $\hat{R}$ a partir de covariables espaciales y temporales:
   $$\hat{R}(t, x, y) = f\Big(\mathbf{X}(t, x, y)\Big)$$
   donde $\mathbf{X} = [\text{SST}_{\text{BIL}}, \text{depth}, \text{distance\_coast\_km}, \text{ocean\_fraction}, \text{DOY\_sin}, \text{DOY\_cos}, \dots]$.

4. **Reconstrucción Final:**  
   La SST de alta resolución reconstruida se obtendrá como:
   $$\text{SST}_{\text{downscaled}}(t, x, y) = \text{SST}_{\text{BIL}}(t, x, y) + \hat{R}(t, x, y)$$

> **Nota metodológica fundamental:** En el estado actual del repositorio **NO se ha entrenado todavía ningún modelo de Machine Learning**. Las fases actuales están dedicadas a la armonización espacial, corrección de máscara, solución de cobertura costera y validación controlada de la señal residual.

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
| **Periodo principal** | Serie temporal de análisis científico | **2015-01-01 a 2025-12-31** |
| **Días esperados** | Días calendario en el periodo de 11 años | **4018 días** |

---

## 3. Fuentes de Datos

| Producto | Proveedor | Variable | Resolución Nativa | Periodo Utilizado | Unidades Originales | Transformación | Función en el Proyecto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GHRSST MUR SST v4.1** | NASA JPL / PO.DAAC | `analysed_sst` | 0.01° (~1 km) | 2015-01-01 a 2025-12-31 | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Referencia de alta resolución (Target) |
| **NOAA OISST v2.1** | NOAA / NCEI / CoastWatch | `sst` | 0.25° (~27 km) | 2015-01-01 a 2025-12-31 | Grados Celsius (°C) | Directa (°C) | Predictor base de baja resolución |
| **GEBCO Grid** | IHO / IOC / BODC | `elevation` | 15 arc-sec (~450 m) | Estático (versión 2024/2026) | Metros (m) | $\text{Depth} = -\text{elevation}$ | Batimetría, costa y fracción oceánica |

### Detalle de Implementación de Fuentes:

- **MUR SST v4.1:**  
  - *Bloque histórico (2015-01-01 a 2019-07-22):* Descargado en recortes netCDF anuales desde AWS S3 Zarr (`s3://mur-sst/zarr`) y almacenado localmente en `MUR_ZARR/MUR_HISTORICO_2015_2019/`.  
  - *Bloque diario (2019-07-23 a 2025-12-31):* Granules netCDF4 diarios individuales almacenados en `MUR_ZARR/MUR-JPL-L4-GLOB-v4.1_4.1-20260814_211604/`.
- **NOAA OISST v2.1:**  
  - Obtenido con tolerancia a fallas mediante múltiples proveedores oficiales de NOAA: NOAA PSL THREDDS (`psl.noaa.gov`), NOAA NCEI Direct HTTP (`ncei.noaa.gov`) y NOAA CoastWatch ERDDAP (`coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180`).
  - *Tratamiento especial 2025-01-14:* Se utiliza el archivo local recuperado `OISST/OISST_2025-01-14_Tulum_Cozumel.nc` debido a que dicha fecha estaba ausente en el catálogo de Google Earth Engine (GEE), pero validada y descargada desde NCEI.
- **GEBCO:**  
  - Archivo NetCDF local `GEBCO/gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc` que cubre exhaustivamente el dominio con resolución de 15 segundos de arco.

---

## 4. Control de Calidad y Auditorías Previas

Antes de la integración en el pipeline de downscaling, se ejecutaron scripts de control de calidad y auditoría sobre las colecciones crudas:

1. **Auditoría MUR (4018 días esperados):**  
   - 4018 días encontrados en la unión del bloque histórico Zarr (2015–2019) y los granules diarios (2019–2025).  
   - 0 días faltantes, 0 duplicados, 0 archivos corruptos o de tamaño cero.
2. **Auditoría OISST v2.1:**  
   - Verificación de continuidad temporal completa para los 4018 días.  
   - Confirmación de integridad para el día especial `2025-01-14`.
3. **Auditoría GEBCO:**  
   - Verificación de cobertura geográfica completa, monotonicidad de coordenadas y ausencia de valores `NaN` o `Inf` en la cuadrícula de elevación.

---

## 5. Estructura del Repositorio

El árbol real del proyecto y los directorios de datos relacionados se organiza como sigue:

```text
/Users/mariajosenande/Documents/Lole/
├── auditar_datos_tesis.py                    # Script de auditoría global previa
├── .venv/                                    # Entorno virtual Python 3.11.16 (arm64)
├── GEBCO/                                    # Datos y QA de batimetría GEBCO
│   ├── gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc
│   ├── revisar_gebco.py                      # QA de batimetría
│   └── revision_gebco.log
├── MUR_ZARR/                                 # Datos y scripts de adquisición MUR
│   ├── MUR_HISTORICO_2015_2019/              # Bloque histórico 2015-01-01 a 2019-07-22 (.nc)
│   ├── MUR-JPL-L4-GLOB-v4.1_4.1-20260814_211604/ # Bloque diario 2019-07-23 a 2026 (.nc4)
│   ├── descargar_mur_historico_zarr.py       # Script de adquisición histórica AWS S3
│   ├── recuperar_granules_faltantes.py       # Script de recuperación PO.DAAC
│   └── revisar_mur.py                        # Script de inspección rápida
├── OISST/                                    # Datos y QA de NOAA OISST
│   ├── OISST_2025-01-14_Tulum_Cozumel.nc     # Granule local recuperado
│   ├── revisar_oisst.py                      # Auditoría temporal OISST
│   └── verificar_dia_faltante.py             # Diagnóstico del día 2025-01-14
└── DATASET_TESIS/                            # Pipeline principal de armonización
    ├── README.md                             # Documentación formal del framework
    ├── requirements.txt                      # Dependencias exactas del entorno virtual
    ├── config.py                             # Configuración global y parámetros científicos
    ├── armonizar_datos_tesis.py              # Orquestador general (Inspección y Fase B.1)
    ├── ejecutar_fase_c1.py                   # Runner de prueba inicial Fase C.1
    ├── ejecutar_fase_c1b.py                  # Runner de diagnóstico y comparación A vs B (C.1b)
    ├── ejecutar_fase_c1c.py                  # Runner de cierre final y estabilidad temporal (C.1c)
    ├── modules/                              # Módulos científicos especializados
    │   ├── __init__.py
    │   ├── logging_utils.py                  # Configuración de logs y metadatos
    │   ├── io_mur.py                         # Lectura y conversión de MUR SST
    │   ├── io_oisst.py                       # Lectura multi-fuente de NOAA OISST
    │   ├── io_gebco.py                       # Lectura de GEBCO
    │   ├── grid.py                           # Definición y validación de la cuadrícula maestra
    │   ├── mask.py                           # Construcción de ocean_mask_final (Fase B.1)
    │   ├── bathymetry.py                     # Agregación batimétrica (Depth = -elev)
    │   ├── coast_distance.py                 # Distancia a la costa (UTM 16N / EPSG:32616)
    │   ├── temporal.py                       # Codificación armónica DOY y splits
    │   ├── interpolation.py                  # Estrategia A de soporte costero y bilineal
    │   ├── residual.py                       # Cálculo de residuales R = MUR - BIL y estadísticas
    │   ├── validation.py                     # Validación de identidad y métricas Baseline E0
    │   └── plotting.py                       # Generación de figuras cartográficas de control
    ├── outputs/                              # Productos NetCDF intermedios generados
    │   ├── dataset_intermedio_fase_b.nc
    │   ├── faseC1_2015-01-01.nc
    │   ├── faseC1b_diagnostico_2015-01-01.nc
    │   └── faseC1c_2015-01-01.nc
    ├── figures/                              # Figuras cartográficas PNG en alta resolución
    ├── logs/                                 # Logs de auditoría y ejecución de cada fase
    └── reports/                              # Reportes oficiales en texto plano
```

> **Nota sobre scripts externos:** Los scripts localizados en `MUR_ZARR/`, `OISST/`, `GEBCO/` y en la raíz (`auditar_datos_tesis.py`) corresponden a etapas previas de adquisición, recuperación de datos históricos y control de calidad; no forman parte del pipeline modular activo de `DATASET_TESIS/`.

---

## 6. Documentación de Módulos y Scripts

| Archivo | Responsabilidad | Funciones Principales | Entradas | Salidas |
| :--- | :--- | :--- | :--- | :--- |
| [`config.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/config.py) | Centraliza constantes espaciales, temporales, tolerancias y rutas de archivos | Constantes globales (`LAT_MIN`, `LAT_MAX`, `MUR_HIST_DIR`, etc.) | N/A | Parámetros de configuración |
| [`armonizar_datos_tesis.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/armonizar_datos_tesis.py) | Orquestador de inspección previa y ejecución de Fase B.1 | `run_inspection()`, `run_phase_b()`, `main()` | Datasets de entrada crudos | `dataset_intermedio_fase_b.nc`, reportes, figuras B.1 |
| [`ejecutar_fase_c1c.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/ejecutar_fase_c1c.py) | Runner de validación final de la Fase C.1c y prueba de estabilidad temporal | `run_fase_c1c()` | `dataset_intermedio_fase_b.nc`, MUR, OISST | `faseC1c_2015-01-01.nc`, 6 figuras oficiales, reporte C.1c |
| [`modules/logging_utils.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/logging_utils.py) | Registro unificado en consola y archivo con metadatos del entorno | `setup_logging()`, `log_environment_metadata()` | `Path` de archivo de log | Objeto `logging.Logger` |
| [`modules/io_mur.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/io_mur.py) | Lectura de MUR histórico (Zarr/NetCDF) y diario, conversión K $\rightarrow$ °C | `inspect_mur_sources()`, `load_mur_reference()`, `load_mur_single_day()` | Archivos MUR locales | `xr.Dataset`, `xr.DataArray` en °C [86, 96] |
| [`modules/io_oisst.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/io_oisst.py) | Consulta multi-proveedor de NOAA OISST (PSL, NCEI, ERDDAP, local) | `inspect_oisst_sources()`, `load_oisst_day()` | `date_str`, URLs / archivos OISST | `xr.DataArray` de OISST en °C con/sin halo |
| [`modules/io_gebco.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/io_gebco.py) | Lectura e inspección del modelo batimétrico GEBCO | `inspect_gebco_sources()`, `load_gebco_dataset()` | `GEBCO_FILE` | `xr.Dataset` GEBCO |
| [`modules/grid.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/grid.py) | Definición y verificación de uniformidad de la cuadrícula maestra | `define_target_grid()`, `check_grid_uniformity()` | `ds_mur` | `xr.Dataset` con grilla [86, 96], dict de estadísticas |
| [`modules/mask.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/mask.py) | Generación de la máscara oceánica final mediante combinación con GEBCO | `create_ocean_mask()` | `ds_mur`, `ds_gebco` | `xr.Dataset` (`ocean_mask_final`, `ocean_fraction`) |
| [`modules/bathymetry.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/bathymetry.py) | Agregación de batimetría GEBCO a MUR ($\text{Depth} = -\text{elevation}$) | `aggregate_gebco_to_mur_grid()` | `ds_gebco`, `ds_mur`, `ds_mask` | `xr.Dataset` (`depth` en metros, NaN en tierra) |
| [`modules/coast_distance.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/coast_distance.py) | Cálculo de distancia mínima a la costa en UTM 16N (EPSG:32616) | `calculate_distance_to_coast()` | `ds_gebco`, `ds_mur`, `ds_mask` | `xr.Dataset` (`distance_coast_km`, NaN en tierra) |
| [`modules/temporal.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/temporal.py) | Codificación armónica del día del año y asignación de splits | `compute_doy()`, `compute_doy_sin_cos()`, `get_temporal_features()`, `get_split_name()` | Fechas / strings | DOY, $\sin/\cos(\text{DOY})$, nombre del split |
| [`modules/interpolation.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/interpolation.py) | Implementación de Estrategia A (soporte costero), Estrategia B y verificación | `select_oisst_with_halo()`, `interpolate_strategy_a_coastal_support()`, `interpolate_strategy_b_triangulation()`, `verify_coordinates_alignment()` | OISST halo, coordenadas MUR | `da_bil_a`, `da_support_mask`, dict de verificación |
| [`modules/residual.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/residual.py) | Cálculo del residual $R = \text{MUR} - \text{BIL}$ y estadísticas descriptivas | `compute_residual()`, `compute_residual_stats()` | `sst_mur`, `sst_bil`, `ocean_mask_final` | `da_residual`, dict de percentiles y momentos |
| [`modules/validation.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/validation.py) | Verificación de identidad numérica y métricas de evaluación Baseline E0 | `verify_point_residual()`, `compute_baseline_metrics()` | `sst_mur`, `sst_bil`, `residual`, `ocean_mask_final` | `pd.DataFrame` de identidad, dict de métricas E0 |
| [`modules/plotting.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/plotting.py) | Generación cartográfica de figuras de control (Fase B.1 y C.1c) | `plot_all_control_figures()`, `plot_phase_c1c_all_figures()` | Datasets procesados, DataArrays de SST | Figuras PNG en `DATASET_TESIS/figures/` |

---

## 7. Fases del Pipeline y Estado de Ejecución

| Fase | Denominación | Descripción | Estado |
| :--- | :--- | :--- | :--- |
| **Fase A** | Adquisición y Auditoría de Datos | Descarga, verificación temporal de 4018 días e inspección de integridad de MUR, OISST y GEBCO | **COMPLETADA** |
| **Fase B** | Procesamiento Espacial Base | Definición de cuadrícula maestra, batimetría preliminar y distancias métricas a la costa | **COMPLETADA** |
| **Fase B.1** | Máscara Océano/Tierra Corregida | Integración de fracción oceánica GEBCO ($M_{\text{final}} = M_{\text{MUR}} \land (\text{frac} \ge 0.5)$), interior de Cozumel como tierra (5279 celdas oceánicas) | **COMPLETADA** |
| **Fase C.1** | Prueba de Armonización Inicial | Prueba controlada de un solo día (2015-01-01) con OISST nativo e interpolación bilineal | **COMPLETADA** |
| **Fase C.1b** | Diagnóstico de Cobertura Costera | Identificación de 1292 NaNs costeros en OISST, demostración de causa y comparación de Estrategias A vs B | **COMPLETADA** |
| **Fase C.1c** | Cierre Final de Prueba y Estabilidad | Adopción oficial de Estrategia A, 100% de cobertura (5279 celdas), prueba de estabilidad temporal estacional y Baseline E0 | **COMPLETADA** |
| **Fase C.2** | Armonización Completa 2015–2025 | Procesamiento diario de los 4018 días con Estrategia A, validación diaria y construcción del cubo NetCDF | **PENDIENTE** |
| **Fase D** | Construcción del Dataset ML | Ensamblado tabular/tensorial con covariables estáticas y dinámicas, y partición Train/Val/Test | **PENDIENTE** |
| **Fase E** | Entrenamiento de Modelos ML | Entrenamiento de modelos de regresión y downscaling sobre el residual $\hat{R} = f(\mathbf{X})$ | **PENDIENTE** |
| **Fase F** | Evaluación Independiente | Validación cruzada, evaluación en Test (2024–2025) y comparación con productos independientes | **PENDIENTE** |

---

## 8. Fase B.1 — Variables Espaciales Estáticas

En la Fase B.1 se establecieron las covariables fisiográficas estáticas del corredor Tulum–Cozumel sobre la cuadrícula maestra MUR ($86 \times 96$):

1. **`ocean_mask_final(lat, lon)`:**  
   Máscara binaria definitiva océano/tierra ($1 = \text{océano}, 0 = \text{tierra}$).  
   - **Regla metodológica:** $M_{\text{final}} = M_{\text{MUR}} \land (\text{ocean\_fraction} \ge 0.5)$.  
   - **Ajuste de borde sur:** La celda $(0, 17)$ en $\text{Lat} = 19.9000^\circ\text{N}, \text{Lon} = -87.4300^\circ\text{W}$ (costa sur de Sian Ka'an/Punta Allen) con $\text{ocean\_fraction} = 0.5000$ exacto fue clasificada unívocamente como tierra ($0$) para concordancia estricta con la auditoría.  
   - **Conteo final:** **5279 celdas oceánicas**, **2977 celdas terrestres** (8256 totales).  
   - **Celdas modificadas respecto a MUR:** 383 celdas reclasificadas a tierra por no cumplir $\text{ocean\_fraction} \ge 0.5$.  
   - **Cozumel:** Interior clasificado correctamente como **TIERRA** (muestreado en Lat 20.43°N, Lon -86.92°W).
2. **`depth(lat, lon)`:**  
   Profundidad batimétrica marina positiva en metros ($\text{Depth} = -\text{elevation}$ de GEBCO). Las celdas terrestres se enmascaran estrictamente como `NaN`.
3. **`distance_coast_km(lat, lon)`:**  
   Distancia geodésica euclidiana mínima en kilómetros calculada en la proyección métrica conforme **UTM Zona 16N (EPSG:32616)** mediante `scipy.spatial.cKDTree`. Las celdas terrestres son `NaN`.
4. **`ocean_fraction(lat, lon)`:**  
   Fracción continua entre $0.0$ y $1.0$ de submalla GEBCO submarina ($\text{elevation} < 0$). Se preserva como covariable espacial densa.

---

## 9. Diagnóstico del Problema Costero de OISST

Durante la prueba inicial de la Fase C.1 se detectó un problema crítico de cobertura:
- **Celdas oceánicas esperadas:** 5279
- **Celdas válidas en $\text{SST}_{\text{BIL}}$ estándar:** 3987
- **Celdas oceánicas con `NaN`:** 1292 (**24.47% del dominio oceánico perdido**)

### Causa Matemática Demostrada:
NOAA OISST v2.1 posee una resolución gruesa de 0.25° (~27 km). En el halo de $7 \times 7$ píxeles que rodea el dominio, los 20 píxeles situados al oeste corresponden a la masa continental de la Península de Yucatán y están enmascarados como `NaN` en OISST.  

La interpolación bilineal 2D regular exige que los **4 nodos esquina** del cuadrilátero envolvente posean valores numéricos válidos. Si incluso 1 solo de los 4 nodos es `NaN`, la fórmula bilineal se evalúa como `NaN`. Como consecuencia, 1292 celdas oceánicas MUR (~0.01°) situadas a menos de ~27 km de la costa continental perdían su valor de interpolación a pesar de ser agua abierta en la cuadrícula de alta resolución.

---

## 10. Estrategia Costera Adoptada (Estrategia A)

Para resolver la pérdida costera sin degradar el método bilineal, se adoptó oficialmente la **ESTRATEGIA A**:

> **Denominación oficial:** *"Extensión costera auxiliar para soporte de interpolación bilineal."*

### Flujo Metodológico:
$$\text{OISST (7}\times\text{7)} \xrightarrow{\text{cKDTree nearest-ocean}} \text{OISST}_{\text{extended}} \xrightarrow{\text{Bilinear 2D}} \text{SST}_{\text{BIL}} \text{ [86}\times\text{96]} \xrightarrow{\text{where}(M_{\text{final}} == 1)} \text{SST}_{\text{BIL\_masked}}$$

### Principios Metodológicos Fundamentales:
1. **Función estrictamente matemática:** Los valores extendidos hacia nodos terrestres existen única y exclusivamente como soporte de interpolación para definir la superficie continua de aproximación en la franja marina costera.
2. **No representan observaciones reales:** En ningún caso se interpretan como observaciones físicas de SST sobre tierra ni como "temperatura terrestre".
3. **Enmascaramiento final estricto:** Al aplicar `ocean_mask_final`, todos los valores sobre tierra desaparecen y quedan fijados como `NaN`.
4. **Trazabilidad explícita:** Se genera y preserva la variable diagnóstica `oisst_coastal_support_mask`, que registra exactamente qué nodos OISST de 0.25° fueron extendidos como soporte auxiliar.

---

## 11. Comparación de Estrategias y Sensibilidad Metodológica

En la Fase C.1b se comparó la Estrategia A frente a una alternativa no estructurada (Estrategia B: Triangulación 2D Delaunay sobre puntos oceánicos válidos con fallback nearest fuera del *convex hull*):

| Métrica / Parámetro | Bilineal Original (Sin Soporte) | Estrategia A (Soporte Costero en Malla 0.25°) | Estrategia B (Triangulación 2D Delaunay) |
| :--- | :--- | :--- | :--- |
| **Celdas Oceánicas Válidas** | 3987 / 5279 (75.53%) | **5279 / 5279 (100.00%)** | **5279 / 5279 (100.00%)** |
| **Celdas NaN Oceánicas** | 1292 (24.47%) | **0 (0.00%)** | **0 (0.00%)** |
| **RMSE vs MUR (2015-01-01)** | 0.2348 °C | **0.2367 °C** | **0.2363 °C** |
| **MAE vs MUR (2015-01-01)** | 0.1942 °C | **0.2004 °C** | **0.1996 °C** |
| **Bias ($\text{BIL} - \text{MUR}$)** | +0.1773 °C | **+0.1872 °C** | **+0.1862 °C** |
| **$R^2$** | -1.3322 | **-1.6878** | **-1.6804** |

### Discrepancia Numérica Directa entre Estrategia A y Estrategia B:
- **MAE de discrepancia:** **0.0038 °C**
- **Percentil 95 (P95):** **0.0214 °C**
- **Discrepancia máxima:** **0.0626 °C**

> **Declaración de Sensibilidad Metodológica:**  
> Las pequeñas diferencias observadas entre las estrategias A y B indican que, para la fecha evaluada, la reconstrucción de la superficie costera es numéricamente poco sensible al método de interpolación considerado. Se adoptó la Estrategia A por preservar la grilla cartesiana regular, su eficiencia computacional en memoria y su trazabilidad estricta.

---

## 12. Fase C.1c — Resultados Numéricos Finales (2015-01-01)

Resumen de la ejecución validada de cierre de la Fase C.1c para la fecha de prueba `2015-01-01`:

- **Celdas válidas en $\text{ocean\_mask\_final}$:** 5279
- **Celdas válidas en $\text{SST}_{\text{MUR}}$:** 5279
- **Celdas válidas en $\text{SST}_{\text{BIL}}$:** 5279
- **Celdas válidas en $\text{Residual}$:** 5279 (**100.00% de cobertura oceánica**)

### Estadísticas Térmicas y de Residual:

| Variable | Mínimo | Máximo | Media | Mediana | Desv. Est. | P1 | P5 | P95 | P99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{SST}_{\text{MUR}}$ (°C)** | 26.4960 | 27.3900 | 26.9485 | 26.9460 | 0.1481 | 26.6340 | 26.7190 | 27.2020 | 27.2910 |
| **$\text{SST}_{\text{BIL}}$ (°C)** | 27.0403 | 27.1943 | 27.1357 | 27.1460 | 0.0381 | 27.0494 | 27.0673 | 27.1852 | 27.1923 |
| **$\text{Residual } R$ (°C)** | -0.6286 | +0.2376 | -0.1872 | -0.1956 | 0.1448 | -0.5696 | -0.4064 | +0.0483 | +0.1411 |

### Verificación de Identidad Numérica:
$$\text{error\_identidad} = R - (\text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}})$$
- **Error numérico máximo global:** **$9.46 \times 10^{-7}\ ^\circ\text{C}$** (estrictamente $< 10^{-5}\ ^\circ\text{C}$, dentro de la precisión flotante de 32 bits).

---

## 13. Baseline E0 Oficial

El **Baseline E0** evalúa la discrepancia directa entre la entrada de baja resolución interpolada ($\text{SST}_{\text{BIL}}$) y la referencia de alta resolución ($\text{SST}_{\text{MUR}}$) sobre las 5279 celdas oceánicas antes de cualquier proceso de aprendizaje automático.

### Métricas Oficiales E0 (2015-01-01):
- **$\text{RMSE}$:** **0.2367 °C**
- **$\text{MAE}$:** **0.2004 °C**
- **$\text{Bias}$:** **+0.1872 °C** ($\text{mean}(\text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}})$)
- **$R^2$:** **-1.6878**
- **$N$:** **5279 celdas**

> **Interpretación del $R^2$ en E0:**  
> Un coeficiente $R^2$ negativo en una fecha individual no indica un fallo del futuro modelo de Machine Learning. Refleja que un campo liso interpolado de 0.25° con sesgo medio (+0.19°C en esa fecha) tiene un error cuadrático mayor que la varianza espacial interna de MUR en ese día específico. Esto demuestra la necesidad de modelar el residual sub-malla.

---

## 14. Estabilidad Temporal de la Máscara de Soporte

Se evaluó la invariancia espacial de la máscara de soporte costero en 4 fechas estacionales distribuidas a lo largo del año 2015:

| Fecha Evaluada | Estación | Nodos OISST Totales | Nodos Válidos (Océano) | Nodos NaN (Soporte Auxiliar) | Discrepancia con Referencia |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2015-01-01** | Invierno | 49 (7×7) | 29 | 20 | **Referencia base** |
| **2015-04-01** | Primavera | 49 (7×7) | 29 | 20 | **Idéntica (0 nodos modificados)** |
| **2015-07-01** | Verano | 49 (7×7) | 29 | 20 | **Idéntica (0 nodos modificados)** |
| **2015-10-01** | Otoño | 49 (7×7) | 29 | 20 | **Idéntica (0 nodos modificados)** |

> **Alcance metodológico:** Esta prueba demuestra estabilidad estacional completa para las fechas evaluadas. Durante la Fase C.2 se incorporará una verificación automatizada diaria para comprobar que la máscara permanezca invariante a lo largo de los 4018 días (2015–2025).

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
| **$\text{DOY}$** | *Day Of Year* | Entero $[1, 366]$ | Día del año calendario |
| **$\text{DOY\_sin}$** | *Sine Harmonic Day of Year* | Adimensional $[-1, 1]$ | $\sin(2\pi \cdot \text{DOY} / 365.25)$ |
| **$\text{DOY\_cos}$** | *Cosine Harmonic Day of Year* | Adimensional $[-1, 1]$ | $\cos(2\pi \cdot \text{DOY} / 365.25)$ |
| **$\text{depth}$** | *Bathymetric Depth* | Metros ($\text{m}$) | Profundidad marina positiva ($-\text{elevation}$ GEBCO, NaN en tierra) |
| **$\text{distance\_coast\_km}$** | *Distance to Coast* | Kilómetros ($\text{km}$) | Distancia geodésica mínima a la costa en UTM 16N (NaN en tierra) |
| **$\text{ocean\_fraction}$** | *Ocean Fraction* | Proporción $[0, 1]$ | Fracción de submalla GEBCO con elevación submarina |
| **$\text{ocean\_mask\_final}$** | *Final Ocean Mask* | Binaria $\{0, 1\}$ | Máscara definitiva océano/tierra (5279 celdas oceánicas) |
| **$\text{oisst\_coastal\_support\_mask}$** | *OISST Support Mask* | Booleana | Identifica nodos OISST de 0.25° extendidos como soporte matemático |
| **$\mathbf{X}$** | *Feature Vector* | Tensor / Matriz | Matriz de variables predictoras para Machine Learning |
| **$t, x, y$** | *Time, Latitude, Longitude* | Tiempo y coordenadas | Índices espaciotemporales del dataset |

---

## 16. Splits Temporales

Para garantizar una evaluación rigurosa y evitar la fuga de información temporal (*data leakage*), la partición de datos se realiza de forma estrictamente cronológica:

```text
2015-01-01                             2021-12-31 2022-01-01          2023-12-31 2024-01-01          2025-12-31
[------------------ ENTRENAMIENTO (TRAIN) ------------------][-- VALIDACIÓN (VAL) --][----- PRUEBA (TEST) -----]
                     7 años (2557 días)                       2 años (730 días)         2 años (731 días)
```

- **Entrenamiento (TRAIN):** 2015-01-01 a 2021-12-31 (7 años, 2557 días).
- **Validación (VAL):** 2022-01-01 a 2023-12-31 (2 años, 730 días) para ajuste de hiperparámetros y selección de modelos.
- **Prueba independiente (TEST):** 2024-01-01 a 2025-12-31 (2 años, 731 días) para evaluación final no sesgada.

---

## 17. Reproducibilidad y Guía de Ejecución

### Requisitos del Sistema
- **Sistema Operativo:** macOS Darwin (probado en Apple Silicon `arm64`) / Linux x86_64
- **Python:** 3.11.16
- **Gestor de entorno:** `uv` o `venv` estándar

### Instalación del Entorno
```bash
# Navegar al directorio raíz del proyecto
cd /Users/mariajosenande/Documents/Lole

# Activar el entorno virtual existente
source .venv/bin/activate

# Opcional: Instalar dependencias exactas si se recrea el entorno
uv pip install -r DATASET_TESIS/requirements.txt
```

### Ejecución de Fases Disponibles

1. **Inspección de fuentes de datos y verificación de rutas:**
   ```bash
   python DATASET_TESIS/armonizar_datos_tesis.py --inspect
   ```
2. **Ejecución de la Fase B.1 (Máscara, batimetría y distancias a la costa):**
   ```bash
   python DATASET_TESIS/armonizar_datos_tesis.py --run-phase-b
   ```
3. **Ejecución de la Fase C.1c (Cierre oficial de prueba controlada 2015-01-01 y estabilidad):**
   ```bash
   python DATASET_TESIS/ejecutar_fase_c1c.py
   ```

---

## 18. Catálogo de Artefactos Generados

### Productos NetCDF (`DATASET_TESIS/outputs/`)
- `dataset_intermedio_fase_b.nc`: Contiene `ocean_mask_final`, `ocean_fraction`, `depth`, `distance_coast_km` y máscaras base de MUR.
- `faseC1c_2015-01-01.nc`: Dataset NetCDF de la Fase C.1c con `sst_mur`, `sst_bil`, `residual`, `ocean_mask_final`, `oisst_coastal_support_mask`, `depth` y `distance_coast_km`.

### Figuras Cartográficas Oficiales (`DATASET_TESIS/figures/`)
- `mapa_ocean_mask.png`: Mapa de la máscara oceánica final ($M_{\text{final}}$).
- `mapa_ocean_fraction.png`: Fracción oceánica continua derivada de GEBCO.
- `mapa_depth_mur.png`: Batimetría marina agregada en metros positivos.
- `mapa_distance_coast.png`: Distancia mínima a la costa en kilómetros (UTM 16N).
- `faseC1c_oisst_original.png`: Píxeles nativos OISST (~0.25°) sin suavizado.
- `faseC1c_oisst_coastal_support.png`: Malla OISST 7×7 con extensión costera auxiliar y marco delimitador MUR.
- `faseC1c_coastal_support_mask.png`: Mapa diagnóstico que etiqueta los 29 nodos oceánicos y los 20 nodos de soporte auxiliar con sus valores térmicos exactos.
- `faseC1c_sst_bil.png`: Campo $\text{SST}_{\text{BIL}}$ interpolado a 0.01° con 100% de cobertura oceánica.
- `faseC1c_sst_mur.png`: Campo $\text{SST}_{\text{MUR}}$ de alta resolución de NASA JPL.
- `faseC1c_residual.png`: Residual $R = \text{MUR} - \text{BIL}$ con colormap divergente centrado en 0.

### Logs y Reportes Oficiales
- `DATASET_TESIS/logs/faseC1c_2015-01-01.log`: Registro cronológico completo de ejecución.
- `DATASET_TESIS/reports/reporte_faseC1c_2015-01-01.txt`: Reporte oficial validado de la Fase C.1c.
- `DATASET_TESIS/reports/reporte_fase_b.txt`: Reporte oficial de la Fase B.1.

---

## 19. Limitaciones Metodológicas

1. **Naturaleza del producto MUR:** MUR SST v4.1 es un producto analizado multiescala de Nivel 4 (L4) obtenido mediante interpolación óptima que combina sensores infrarrojos (AVHRR, MODIS), microondas (AMSR-E, WindSat) e *in situ*. No constituye una observación satelital directa sin procesar.
2. **Resolución nominal vs. efectiva:** Una resolución de malla de ~1 km (0.01°) no implica necesariamente una resolución dinámica efectiva equivalente en zonas con nubosidad persistente, donde MUR depende de la escala de correlación espacial y sensores de microondas.
3. **Disparidad de escalas:** NOAA OISST v2.1 (~27 km) es aproximadamente 25 veces más grueso que MUR (~1 km).
4. **Naturaleza de la extensión costera:** La extensión nearest-ocean sobre los 20 nodos terrestres de OISST es una construcción matemática de soporte para la interpolación bilineal y no debe interpretarse como medición oceanográfica en tierra.
5. **Dependencia del target:** El residual $R$ depende de la precisión de MUR. Las métricas de evaluación respecto a MUR miden la capacidad de reproducir dicha referencia satelital L4.
6. **Validación independiente futura:** Para evaluar la exactitud física en arrecifes someros será indispensable la validación posterior con mediciones *in situ* independientes (estaciones costeras, termógrafos subsuperficiales) o productos alternativos (ej. ESA SST CCI).

---

## 20. Próxima Fase: Fase C.2

```text
============================================================
ESTADO ACTUAL: FASE C.1c APROBADA DEFINITIVAMENTE
PRÓXIMO PASO:  FASE C.2 — ARMONIZACIÓN COMPLETA 2015–2025
============================================================
```

### Objetivos y Tareas de la Fase C.2:
1. **Procesamiento de los 4018 días (2015-01-01 a 2025-12-31):**  
   - Carga diaria secuencial/paralela de $\text{SST}_{\text{MUR}}$ (bloque histórico 2015–2019 y bloque diario 2019–2025).
   - Extracción diaria de OISST con halo y aplicación de la Estrategia A (soporte costero e interpolación bilineal).
   - Enmascaramiento estricto con `ocean_mask_final` (5279 celdas oceánicas por día).
   - Cálculo del cubo tridimensional del residual: $R(t, x, y) = \text{SST}_{\text{MUR}}(t, x, y) - \text{SST}_{\text{BIL}}(t, x, y)$.
2. **Validación y Control Diario Automatizado:**  
   - Confirmación de 5279 celdas válidas en MUR, BIL y Residual para cada uno de los 4018 días.
   - Verificación de la identidad numérica diaria ($\max |R - (\text{MUR} - \text{BIL})| < 10^{-5}\ ^\circ\text{C}$).
   - Verificación de estabilidad diaria de `oisst_coastal_support_mask`.
   - Cálculo del Baseline E0 global y series temporales de RMSE, MAE, Bias y $R^2$.
3. **Construcción del Cubo NetCDF Maestro:**  
   - Generación de `dataset_maestro_downscaling_2015_2025.nc` con dimensiones `(time: 4018, lat: 86, lon: 96)`.

> **Control:** No se iniciará la Fase C.2 hasta recibir la confirmación explícita del usuario.

---

## 21. Deuda de Documentación Interna

Resultado de la auditoría de documentación interna del código existente:

1. [`DATASET_TESIS/config.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/config.py): Convendría añadir docstrings a nivel de constante especificando unidades físicas (°C, m, km, grados decimales) y los sistemas de referencia geodésica.
2. [`DATASET_TESIS/modules/bathymetry.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/bathymetry.py): En las líneas 78–80 existe un bloque condicional de contingencia con `pass`; convendría documentar explícitamente si se requiere una política formal de interpolación para bordes extremos o si la cobertura actual es exhaustiva.
3. [`DATASET_TESIS/modules/residual.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/residual.py): La función `interpolate_oisst_to_mur` fue la versión preliminar antes de la creación del módulo especializado [`modules/interpolation.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/interpolation.py); convendría agregar una nota de deprecación interna indicando que la función canónica es `interpolate_strategy_a_coastal_support`.
4. [`DATASET_TESIS/modules/io_mur.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/io_mur.py): Convendría tipar explícitamente los argumentos de entrada y salida con anotaciones de tipo (`typing.Dict`, `typing.Optional`, `xarray.DataArray`).

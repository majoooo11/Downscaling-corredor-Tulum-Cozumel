# SST Downscaling Framework for the Tulum–Cozumel Reef Corridor

> **Subtítulo:** Pipeline reproducible para la armonización espacial y temporal de MUR SST, NOAA OISST v2.1 y GEBCO orientado al downscaling de temperatura superficial del mar mediante Machine Learning en el Caribe mexicano.  
> **Estado del proyecto:** **FASE C.2, VALIDACIÓN SATELITAL Y AUDITORÍA DE INCERTIDUMBRE COMPLETADAS**
> - **Fase C.2:** **COMPLETADA** (Dataset armonizado 2015–2025 generado y validado con 4018 días continuos, 5279 celdas oceánicas).
> - **Validación Satelital VIIRS/MODIS:** **COMPLETADA** (Evidencia infrarroja clasificada como **INCONCLUSA / HETEROGÉNEA SEGÚN EVENTO**).
> - **Recuperación MUR `analysis_error`:** **COMPLETADA** (4018 / 4018 días disponibles, 0 faltantes, 0 duplicados).
> - **Auditoría Final `analysis_error` (2015–2025, N = 4018):** **COMPLETADA Y CERRADA**.
> - **Auditoría de Cambios Abruptos ($\Delta\text{AE}$):** **COMPLETADA**.
> - **Diagnóstico `analysis_error = 0` (2016-05-23):** **COMPLETADO**.
> - **Modelación Machine Learning:** **NO INICIADO** (Dataset de entrenamiento pendiente de ensamblado).

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
18. [Baseline E0 Global 2015–2025](#18-baseline-e0-global-20152025)
19. [Diagnóstico del Evento Anómalo de Octubre de 2015](#19-diagnóstico-del-evento-anómalo-de-octubre-de-2015)
20. [Validación Independiente — VIIRS S-NPP L2P v2.80](#20-validación-independiente--viirs-s-npp-l2p-v280)
21. [Validación Independiente — MODIS Aqua L2P v2019.0](#21-validación-independiente--modis-aqua-l2p-v20190)
22. [Auditoría Satelital Comparativa Multievento (E1–E6)](#22-auditoría-satelital-comparativa-multievento-e1e6)
23. [Incertidumbre MUR (`analysis_error`)](#23-incertidumbre-mur-analysis_error)
24. [Distribución Global y Estadísticas de `analysis_error` (2015–2025)](#24-distribución-global-y-estadísticas-de-analysis_error-20152025)
25. [Relación entre `analysis_error` y Discrepancia MUR–OISST](#25-relación-entre-analysis_error-y-discrepancia-muroisst)
26. [Gradiente de Severidad y Comportamiento en Eventos E1–E6](#26-gradiente-de-severidad-y-comportamiento-en-eventos-e1e6)
27. [Auditoría de Cambios Abruptos ($\Delta\text{AE}$)](#27-auditoría-de-cambios-abruptos-delta\text{ae})
28. [Diagnóstico de la Anomalía Puntual `analysis_error = 0` (23 Mayo 2016)](#28-diagnóstico-de-la-anomalía-puntual-analysis_error--0-23-mayo-2016)
29. [Archivado y Trazabilidad de Productos Obsoletos](#29-archivado-y-trazabilidad-de-productos-obsoletos)
30. [Estado Científico y Conclusiones de la Auditoría](#30-estado-científico-y-conclusiones-de-la-auditoría)
31. [Próxima Fase: Tratamiento de Incertidumbre y Fase D (ML)](#31-próxima-fase-tratamiento-de-incertidumbre-y-fase-d-ml)
32. [Reproducibilidad y Guía de Ejecución](#32-reproducibilidad-y-guía-de-ejecución)
33. [Catálogo de Artefactos Generados](#33-catálogo-de-artefactos-generados)
34. [Limitaciones Metodológicas](#34-limitaciones-metodológicas)
35. [Deuda de Documentación Interna](#35-deuda-de-documentación-interna)

---

## 1. Objetivo Científico y Enfoque Residual

El objetivo de este proyecto de tesis es diseñar, construir y validar un pipeline computacional riguroso y reproducible que permita generar un dataset maestro armonizado para el **downscaling espacial de Temperatura Superficial del Mar (SST, *Sea Surface Temperature*)** en el corredor arrecifal Tulum–Cozumel (Caribe mexicano), fusionando observaciones satelitales multiescala y covariables fisiográficas para su posterior modelación mediante Machine Learning.

### Formulación del Enfoque Residual

En lugar de predecir directamente el campo absoluto de alta resolución, el framework adopta una formulación residual desacoplada:

1. **Componente de baja resolución interpolada ($\text{SST}_{\text{BIL}}$):**  
   Se proyecta el campo térmico de baja resolución (~0.25°) de NOAA OISST v2.1 hacia la cuadrícula de alta resolución (~0.01°) de MUR SST utilizando interpolación bilineal con soporte costero auxiliar (Estrategia A):
   $$\text{SST}_{\text{BIL}}(t, x, y) = \mathcal{I}_{\text{bilinear}}\Big(\text{OISST}_{\text{extended}}(t, X, Y)\Big)$$

2. **Cálculo del Residual Objetivo ($R$):**  
   El residual representa la señal térmica sub-malla, gradientes locales costeros y variabilidad de mesoescala no capturada por OISST:
   $$R(t, x, y) = \text{SST}_{\text{MUR}}(t, x, y) - \text{SST}_{\text{BIL}}(t, x, y) \quad \forall (x, y) \in \text{ocean\_mask\_final}$$

3. **Modelación Futura (Machine Learning):**  
   Posteriormente, los modelos de aprendizaje supervisado aprenderán a estimar dicho residual $\hat{R}$ a partir de covariables espaciales y temporales:
   $$\hat{R}(t, x, y) = f\Big(\mathbf{X}(t, x, y)\Big)$$
   donde $\mathbf{X} = [\text{SST}_{\text{BIL}}, \text{depth}, \text{distance\_coast\_km}, \text{ocean\_fraction}, \text{DOY\_sin}, \text{DOY\_cos}, \dots]$.

4. **Reconstrucción Final:**  
   La SST de alta resolución reconstruida se obtendrá como:
   $$\text{SST}_{\text{downscaled}}(t, x, y) = \text{SST}_{\text{BIL}}(t, x, y) + \hat{R}(t, x, y)$$

> **Nota metodológica fundamental:** En el estado actual del repositorio **NO se ha entrenado todavía ningún modelo de Machine Learning**. Las fases completadas corresponden a la armonización espacial, corrección de máscara, solución de cobertura costera, generación del cubo consolidado 2015–2025, auditorías satelitales multievento, recuperación y auditoría completa de `analysis_error` y diagnóstico temporal.

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
| **Celdas oceánicas** | Celdas válidas en $\text{ocean\_mask\_final}$ | **5279 celdas** (63.94% del dominio) |
| **Celdas terrestres** | Celdas enmascaradas como tierra | **2977 celdas** (36.06% del dominio) |
| **Periodo principal** | Serie temporal consolidada completa | **2015-01-01 a 2025-12-31** |
| **Días procesados** | Días calendario procesados en Fase C.2 | **4018 días (100.0% completitud)** |
| **Observaciones totales** | Muestras espacio-temporales oceánicas | **21,211,022 observaciones** ($4018 \times 5279$) |

---

## 3. Fuentes de Datos

| Producto | Proveedor | Variable | Resolución Nativa | Periodo Utilizado | Unidades Originales | Transformación | Función en el Proyecto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GHRSST MUR SST v4.1** | NASA JPL / PO.DAAC | `analysed_sst`, `analysis_error` | 0.01° (~1 km) | 2015-01-01 a 2025-12-31 | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Referencia Target e Incertidumbre L4 |
| **NOAA OISST v2.1** | NOAA / NCEI / CoastWatch | `sst` | 0.25° (~27 km) | 2015-01-01 a 2025-12-31 | Grados Celsius (°C) | Directa (°C) | Predictor base de baja resolución |
| **GEBCO Grid** | IHO / IOC / BODC | `elevation` | 15 arc-sec (~450 m) | Estático (versión 2024/2026) | Metros (m) | $\text{Depth} = -\text{elevation}$ | Batimetría, costa y fracción oceánica |
| **VIIRS S-NPP L2P v2.80** | NOAA STAR / PO.DAAC | `sea_surface_temperature` | ~750 m en nadir | Multievento (E1 a E6) | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Validación satelital independiente |
| **MODIS Aqua L2P v2019.0** | NASA OBPG / JPL PO.DAAC | `sea_surface_temperature`, `..._4um` | ~1 km en nadir | Multievento (E1 a E6) | Kelvin (K) | $\text{SST}_{^\circ\text{C}} = \text{SST}_{\text{K}} - 273.15$ | Validación satelital independiente |

---

## 4. Control de Calidad y Auditorías Previas

Antes de la integración en el pipeline de downscaling, se ejecutaron scripts de control de calidad y auditoría sobre las colecciones crudas:

1. **Auditoría MUR (4018 días esperados):**  
   - 4018 días encontrados en la serie temporal continua (2015–2025).  
   - 0 días faltantes, 0 duplicados, 0 archivos corruptos o de tamaño cero.
2. **Auditoría OISST v2.1:**  
   - Continuidad temporal completa para los 4018 días.  
   - Confirmación de integridad para el día especial `2025-01-14` recuperado directamente de NCEI.
3. **Auditoría GEBCO:**  
   - Cobertura geográfica completa, monotonicidad de coordenadas y ausencia de valores `NaN` o `Inf` en la cuadrícula de elevación.

---

## 5. Estructura del Repositorio

```text
/Users/mariajosenande/Documents/Lole/
├── auditar_datos_tesis.py                    # Script de auditoría global previa
├── .venv/                                    # Entorno virtual Python 3.11.16 (arm64)
├── GEBCO/                                    # Datos y QA de batimetría GEBCO
│   ├── gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc
│   └── revisar_gebco.py
├── MUR_ZARR/                                 # Datos y scripts de adquisición MUR SST
│   ├── MUR_HISTORICO_2015_2019/              # Bloque histórico 2015-01-01 a 2019-07-22 (.nc)
│   └── MUR-JPL-L4-GLOB-v4.1_4.1-20260814_211604/ # Bloque diario 2019-07-23 a 2026 (.nc4)
├── OISST/                                    # Granules locales complementarios
│   └── OISST_2025-01-14_Tulum_Cozumel.nc     # Granule local recuperado
├── VALIDACION_SATELITAL/                     # Granules L2P para validación independiente
│   ├── AUDITORIA/                            # Subconjuntos multievento E1 a E6
│   ├── VIIRS/                                # NetCDF de VIIRS S-NPP L2P v2.80
│   └── MODIS/                                # NetCDF de MODIS Aqua L2P v2019.0
└── DATASET_TESIS/                            # Pipeline principal de armonización y downscaling
    ├── README.md                             # Documentación formal del framework (este archivo)
    ├── requirements.txt                      # Dependencias exactas del entorno virtual
    ├── config.py                             # Configuración global y parámetros científicos
    ├── armonizar_datos_tesis.py              # Orquestador general (Inspección y Fase B.1)
    ├── fase_c2_armonizacion_2015_2025.py     # Runner de ejecución completa Fase C.2 (4018 días)
    ├── descargar_analysis_error_mur_opendap.py # Descarga directa PO.DAAC sin Harmony (2016-2019)
    ├── auditar_analysis_error_mur_FINAL.py   # Auditoría completa de analysis_error (N = 4018 días)
    ├── auditar_cambios_abruptos_analysis_error.py # Control temporal de saltos diarios (Delta AE)
    ├── diagnosticar_zero_analysis_error_20160523.py # Diagnóstico puntual del mínimo de 2016-05-23
    ├── modules/                              # Módulos científicos especializados
    │   ├── io_mur.py, io_oisst.py, io_gebco.py
    │   ├── grid.py, mask.py, bathymetry.py, coast_distance.py
    │   ├── temporal.py, interpolation.py, residual.py, validation.py, plotting.py
    ├── outputs/                              # Productos NetCDF intermedios y consolidados
    │   ├── dataset_intermedio_fase_b.nc      # Covariables estáticas (Fase B.1)
    │   ├── faseC1c_2015-01-01.nc             # Prueba unitaria validada (Fase C.1c)
    │   ├── faseC2_2015_2025.nc               # Cubo consolidado 4018 días (567 MB)
    │   └── fase_c2/                          # 11 NetCDFs anuales (faseC2_2015.nc a faseC2_2025.nc)
    ├── analysis_error_historico/             # Recuperación consolidada de incertidumbre MUR
    │   ├── mur_analysis_error_2015_2025_completo.nc # 4018 días completos de analysis_error
    │   └── mur_analysis_error_2016_2019_recuperado.nc
    ├── auditoria_analysis_error/             # Artefactos oficiales de auditoría de incertidumbre
    │   ├── csv/                              # 10 CSVs con estadísticas, deltas, severidad y eventos
    │   ├── figures/                          # 12 Figuras oficiales (Figuras 1 a 10 + mapas E1-E6)
    │   ├── reports/                          # Reportes Markdown oficiales (FINAL 4018d, deltas, zero)
    │   └── logs/                             # Registros de ejecución
    └── archive/                              # Archivo y trazabilidad de productos derivados obsoletos
        ├── figure_versions/                  # Versiones gráficas previas (e.g. figura7_legend_old)
        ├── analysis_error_auditoria_2074dias_obsoleta/ # Auditoría parcial preliminar archivada
        └── validacion_infrarroja_pre_correccion_final/ # Manifiesto de validación satelital previa
```

---

## 6. Documentación de Módulos y Scripts

| Archivo | Responsabilidad | Entradas | Salidas | Estado |
| :--- | :--- | :--- | :--- | :--- |
| [`config.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/config.py) | Centraliza constantes espaciales, temporales, tolerancias y rutas de archivos | N/A | Parámetros de configuración | **VALIDADO** |
| [`fase_c2_armonizacion_2015_2025.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/fase_c2_armonizacion_2015_2025.py) | Procesamiento diario secuencial/anual de los 4018 días con Estrategia A | `dataset_intermedio_fase_b.nc`, MUR, OISST | `outputs/faseC2_2015_2025.nc`, 11 NetCDF anuales, reporte | **VALIDADO** |
| [`descargar_analysis_error_mur_opendap.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/descargar_analysis_error_mur_opendap.py) | Recuperación histórica de `analysis_error` sin Harmony (2016–2019) | PO.DAAC / CoastWatch ERDDAP | `mur_analysis_error_2015_2025_completo.nc` | **VALIDADO** |
| [`auditar_analysis_error_mur_FINAL.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditar_analysis_error_mur_FINAL.py) | Auditoría estadística global de incertidumbre MUR sobre 4018 días | `mur_analysis_error_2015_2025_completo.nc`, C.2 | 7 CSVs, 12 figuras, reporte oficial `FINAL_4018dias.md` | **VALIDADO** |
| [`auditar_cambios_abruptos_analysis_error.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditar_cambios_abruptos_analysis_error.py) | Control temporal diario de saltos $\Delta\text{AE}$ y detección de mínimos | `mur_analysis_error_2015_2025_completo.nc`, C.2 | 3 CSVs, Figura 7, reporte Markdown | **VALIDADO** |
| [`diagnosticar_zero_analysis_error_20160523.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/diagnosticar_zero_analysis_error_20160523.py) | Diagnóstico exhaustivo del valor $\text{AE} = 0.00^\circ\text{C}$ del 2016-05-23 | Granule original, consolidado, C.2 | Figuras 8, 9 y 10, reporte Markdown | **VALIDADO** |

---

## 7. Fases del Pipeline y Estado de Ejecución

| Fase | Denominación | Descripción | Estado |
| :--- | :--- | :--- | :--- |
| **Fase A** | Adquisición y Auditoría de Datos | Descarga, verificación temporal de 4018 días e inspección de integridad de MUR, OISST y GEBCO | **COMPLETADA** |
| **Fase B** | Procesamiento Espacial Base | Definición de cuadrícula maestra, batimetría preliminar y distancias métricas a la costa | **COMPLETADA** |
| **Fase B.1** | Máscara Océano/Tierra Corregida | Integración de fracción oceánica GEBCO ($M_{\text{final}} = M_{\text{MUR}} \land (\text{frac} \ge 0.5)$), interior de Cozumel como tierra (5279 celdas oceánicas) | **COMPLETADA** |
| **Fase C.1** | Prueba de Armonización Inicial | Prueba controlada de un solo día (2015-01-01) con OISST nativo e interpolación bilineal | **COMPLETADA** |
| **Fase C.1b** | Diagnóstico de Cobertura Costera | Identificación de 1292 NaNs costeros en OISST, demostración de causa y comparación de Estrategias A vs B | **COMPLETADA** |
| **Fase C.1c** | Cierre Final de Prueba y Estabilidad | Adopción oficial de Estrategia A, 100% de cobertura (5279 celdas), prueba de estabilidad temporal y Baseline E0 | **COMPLETADA** |
| **Fase C.2** | Armonización Completa 2015–2025 | Procesamiento diario de los 4018 días con Estrategia A, validación diaria y construcción del cubo NetCDF consolidado | **COMPLETADA** |
| **Validación Satelital** | Auditoría VIIRS + MODIS Multievento | Evaluación multi-sensor en episodios extremos E1 a E6 (evidencia infrarroja heterogénea/inconclusa) | **COMPLETADA** |
| **Auditoría Incertidumbre** | Auditoría Global `analysis_error` | Recuperación histórica 2016–2019, auditoría global 4018 días, control temporal $\Delta\text{AE}$ y diagnóstico AE=0 | **CERRADA** |
| **Fase D** | Construcción del Dataset ML | Ensamblado tabular/tensorial con covariables estáticas/dinámicas, tratamiento de incertidumbre y splits | **PENDIENTE** |
| **Fase E** | Entrenamiento de Modelos ML | Entrenamiento de modelos supervisados sobre el residual $\hat{R} = f(\mathbf{X})$ | **PENDIENTE** |
| **Fase F** | Evaluación Independiente | Validación cruzada, evaluación en Test (2024–2025) y comparación final de downscaling | **PENDIENTE** |

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
| **$\text{depth}$** | *Bathymetric Depth* | Metros ($\text{m}$) | Profundidad marina positiva ($-\text{elevation}$ GEBCO) |
| **$\text{distance\_coast\_km}$** | *Distance to Coast* | Kilómetros ($\text{km}$) | Distancia geodésica mínima a la costa en UTM 16N |
| **$\text{ocean\_fraction}$** | *Ocean Fraction* | Proporción $[0, 1]$ | Fracción de submalla GEBCO con elevación submarina |
| **$\text{ocean\_mask\_final}$** | *Final Ocean Mask* | Binaria $\{0, 1\}$ | Máscara definitiva océano/tierra (5279 celdas oceánicas) |

---

## 16. Splits Temporales

```text
2015-01-01                             2021-12-31 2022-01-01          2023-12-31 2024-01-01          2025-12-31
[------------------ ENTRENAMIENTO (TRAIN) ------------------][-- VALIDACIÓN (VAL) --][----- PRUEBA (TEST) -----]
                     7 años (2557 días)                       2 años (730 días)         2 años (731 días)
```
- **Entrenamiento (TRAIN):** 2015-01-01 a 2021-12-31 (7 años, 2557 días).
- **Validación (VAL):** 2022-01-01 a 2023-12-31 (2 años, 730 días).
- **Prueba independiente (TEST):** 2024-01-01 a 2025-12-31 (2 años, 731 días).

---

## 17. Fase C.2 — Armonización Completa 2015–2025

La **Fase C.2** ejecutó la armonización temporal y espacial completa de los 11 años del estudio mediante el script [`fase_c2_armonizacion_2015_2025.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/fase_c2_armonizacion_2015_2025.py):

- **Periodo:** 2015-01-01 a 2025-12-31.
- **Total de días procesados:** **4018 / 4018 días (100.0% completitud, 0 días faltantes)**.
- **Dimensiones del cubo:** `(time: 4018, lat: 86, lon: 96)`.
- **Celdas oceánicas por día:** **5279 celdas**.
- **Observaciones espacio-temporales totales:** **21,211,022 observaciones**.
- **Identidad numérica diaria:** Verificada estrictamente en todos los días ($\max |R - (\text{MUR} - \text{BIL})| < 10^{-5}\ ^\circ\text{C}$).
- **Archivos generados:** 11 NetCDFs anuales en `DATASET_TESIS/outputs/fase_c2/` y el archivo consolidado [`DATASET_TESIS/outputs/faseC2_2015_2025.nc`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/outputs/faseC2_2015_2025.nc) (567 MB).

---

## 18. Baseline E0 Global 2015–2025

Evaluación global del predictor base OISST interpolado ($\text{SST}_{\text{BIL}}$) frente a MUR SST ($\text{SST}_{\text{MUR}}$) sobre las 21,211,022 observaciones del periodo 2015–2025:

- **Número total de observaciones ($N$):** **21,211,022**
- **$\text{RMSE}$:** **0.3426 °C**
- **$\text{MAE}$:** **0.2631 °C**
- **$\text{Bias}$ ($\text{mean}(\text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}})$):** **+0.0133 °C**
- **Coeficiente de determinación ($R^2$):** **0.9016**

> **Convención de Signo:** $\text{Bias} = \text{SST}_{\text{BIL}} - \text{SST}_{\text{MUR}}$. Un sesgo positivo indica que el predictor base OISST se encuentra en promedio ligeramente más cálido que MUR sobre el dominio.

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

## 20. Validación Independiente — VIIRS S-NPP L2P v2.80

Se realizó una validación independiente con el producto infrarrojo de Nivel 2P **VIIRS S-NPP L2P v2.80** (DOI: 10.5067/GHVRS-2PO28, resolución ~750 m):
- **Archivos analizados:** 15 NetCDF crudos en la ventana del 15 al 20 de octubre de 2015.
- **Resultados en la ventana crítica:** En 7 de 8 pasos, la cobertura de calidad $QL=5$ fue $0.0\%$. En el paso del 18-Oct 18:40 UTC se observaron 10 píxeles aislados ($0.11\%$ de cobertura) con media de $29.55\ ^\circ\text{C}$ (cercano a OISST $29.62\ ^\circ\text{C}$ y superior a MUR $27.32\ ^\circ\text{C}$).
- **Clasificación:** **INCONCLUSO** por cobertura muestral insuficiente para caracterizar el canal regional.

---

## 21. Validación Independiente — MODIS Aqua L2P v2019.0

Se procesaron los 14 archivos NetCDF de **MODIS Aqua L2P v2019.0** (DOI: 10.5067/GHMDA-2PJ19, resolución ~1 km):

- **Canales evaluados:** Térmico de 11 µm y nocturno de 4 µm.
- **Resultados en la ventana crítica:** Durante la ventana principal del evento no se obtuvieron observaciones de alta calidad utilizables bajo los criterios establecidos para la validación ($N_{\text{QL5}} = 0$ y $N_{\text{usable}} = 0$ para $QL \geq 4$).
- **Interpretación:** La ausencia de observaciones utilizables impide realizar una comparación independiente suficientemente robusta entre MODIS, MUR y OISST durante la ventana crítica. Los niveles de calidad observados no permiten atribuir por sí solos la ausencia de datos utilizables exclusivamente a nubosidad.
- **Clasificación:** **INCONCLUSO**, debido a disponibilidad insuficiente de observaciones SST de alta calidad durante la ventana analizada.

---

## 22. Auditoría Satelital Comparativa Multievento (E1–E6)

Se evaluaron los seis principales eventos de discrepancia extrema detectados en la serie 2015–2025 mediante observaciones independientes VIIRS S-NPP y MODIS Aqua. La disponibilidad de observaciones infrarrojas de alta calidad fue heterogénea entre eventos, por lo que la interpretación se realizó considerando tanto la cobertura disponible como las métricas de comparación colocalizada.

| Evento | Periodo | $\text{RMSE}_{\max}$ (°C) | Evidencia VIIRS S-NPP | Evidencia MODIS Aqua | Dictamen |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **E1** | 2015-10-17 → 2015-10-21 | **2.276** | Cobertura QL=5 extremadamente limitada (0.11%); los 10 píxeles disponibles fueron localmente más próximos a OISST/BIL | Sin observaciones de alta calidad utilizables durante la ventana principal | **INCONCLUSO** |
| **E2** | 2021-11-17 → 2021-11-19 | **1.463** | Cobertura QL=5 insuficiente para una evaluación regional robusta | Cobertura de alta calidad insuficiente | **INCONCLUSO** |
| **E3** | 2024-10-19 → 2024-10-20 | **1.292** | Sin cobertura de alta calidad suficiente para una comparación regional | Sin cobertura de alta calidad suficiente para una comparación regional | **INCONCLUSO** |
| **E4** | 2015-08-03 → 2015-08-06 | **1.102** | La comparación colocalizada favoreció OISST/BIL sobre una fracción considerable de celdas MUR muestreadas | La muestra disponible, considerablemente menor, favoreció MUR | **HETEROGÉNEO; evidencia VIIRS favorable a BIL** |
| **E5** | 2016-06-04 → 2016-06-08 | **1.059** | Comparación favorable a OISST/BIL en la fecha pico | Comparación favorable a MUR en la fecha pico | **MIXTO / INCONCLUSO** |
| **E6** | 2019-06-14 → 2019-06-17 | **1.030** | Diferencias pequeñas en la fecha pico; considerando el evento completo, la comparación favorece OISST/BIL | Muestra reducida y comportamiento mixto en la fecha pico; el evento completo presenta mayor proximidad a BIL | **HETEROGÉNEO; tendencia hacia BIL en el evento completo** |

> **Nota metodológica:** Los porcentajes de cobertura empleados en esta auditoría representan la fracción de celdas oceánicas de la cuadrícula MUR que contienen al menos una observación satelital independiente válida y no deben interpretarse como una equivalencia exacta de área geográfica observada.

> **Conclusión de la Validación Satelital:** La evidencia infrarroja obtenida es **HETEROGÉNEA Y, EN VARIOS EVENTOS, INCONCLUSA**. Los eventos E1–E3 carecen de cobertura independiente suficiente para confirmar o refutar regionalmente las discrepancias MUR–OISST. En E4 existe evidencia VIIRS sobre una fracción considerable del dominio observado que favorece OISST/BIL, mientras que E5 presenta resultados contradictorios entre sensores. E6 muestra comportamiento mixto en la fecha pico, aunque el análisis del evento completo tiende a favorecer OISST/BIL. En consecuencia, no existe evidencia suficiente para clasificar de manera general los eventos extremos como errores de MUR ni para modificar el dataset armonizado de la Fase C.2.
---

## 23. Incertidumbre MUR (`analysis_error`)

Inicialmente, la serie local de `analysis_error` presentaba un hueco temporal entre `2016-01-01` y `2019-07-22` debido a la descarga histórica previa de sólo SST. Dicho hueco fue recuperado de manera completa y directa desde NASA PO.DAAC / NOAA CoastWatch ERDDAP sin utilizar intermediarios Harmony:

- **Producto:** `MUR-JPL-L4-GLOB-v4.1` (DOI: 10.5067/GHGMR-4FJ04).
- **Periodo Final Disponible:** **2015-01-01 a 2025-12-31** (4018 / 4018 días).
- **Faltantes / Duplicados:** **0 faltantes, 0 duplicados**.
- **Archivo Consolidado:** [`DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc) ($4018 \times 86 \times 96$, 133 MB).

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
> - **Eventos E1–E3:** Presentan discrepancias extremas MUR–OISST coincidentes con el valor superior de saturación de `analysis_error` ($0.4100^\circ\text{C}$) en la totalidad del dominio. No obstante, la evidencia independiente disponible no permite confirmar por sí sola que estos eventos sean artefactos instrumentales de MUR.
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
- `DATASET_TESIS/archive/figure_versions/`: Contiene versiones gráficas intermedias de figuras (e.g. `figura7_delta_analysis_error_2015_2025_legend_old.png`).
- `DATASET_TESIS/archive/validacion_infrarroja_pre_correccion_final/`: Manifiesto y respaldos de validaciones satelitales preliminares.

## 30. Estado Científico y Conclusiones de la Auditoría

```text
============================================================
ESTADO DE LA AUDITORÍA DE analysis_error: CERRADA
============================================================
```

1. **`analysis_error` aporta información diagnóstica complementaria** sobre determinados episodios de discrepancia entre MUR y OISST. A escala diaria existe una asociación positiva entre la incertidumbre media reportada por MUR y la magnitud de la discrepancia MUR–OISST; sin embargo, dicha asociación es moderada y no determinista.

2. **`analysis_error` no constituye una medida directa del error verdadero de MUR** y, por tanto, no puede utilizarse de forma aislada para clasificar observaciones como correctas o incorrectas, identificar automáticamente artefactos ni justificar la eliminación de fechas del dataset.

3. **Los eventos E1–E3 presentan simultáneamente discrepancias MUR–OISST extremas y `analysis_error = 0.4100 °C` en el 100% del dominio**, mientras que E4 demuestra que una discrepancia extrema también puede producirse con niveles ordinarios de `analysis_error`. Por ello, no existe una firma universal de incertidumbre asociada a todos los eventos extremos.

4. **El episodio del 23 de mayo de 2016 constituye una anomalía específica del campo `analysis_error`**, presente directamente en el producto MUR v4.1 original. Los 744 valores iguales a 0.000 °C no corresponden a `_FillValue`, errores de decodificación, máscara o concatenación y no están acompañados por una discontinuidad equivalente en `analysed_sst`. Su origen algorítmico específico no puede determinarse con la evidencia actualmente disponible.

5. **No existe evidencia suficiente para eliminar o corregir manualmente observaciones de MUR.** Los eventos investigados se conservan en su forma original para mantener la integridad y trazabilidad del producto utilizado.

6. **El producto consolidado `faseC2_2015_2025.nc` permanece íntegro y sin modificaciones ad hoc.** Ninguno de los resultados de las auditorías justifica modificar la Fase C.2.

7. **La auditoría de `analysis_error` se considera cerrada para efectos de control de calidad del dataset.** Las particularidades identificadas quedan documentadas y podrán utilizarse posteriormente para análisis de sensibilidad durante la modelación.

---

## 31. Próxima Fase: Tratamiento de Incertidumbre y Fase D (ML)

```text
============================================================
PRÓXIMA FASE: DISEÑO Y CONSTRUCCIÓN DEL DATASET PARA ML
============================================================
```

### Directrices Metodológicas para el Dataset ML

1. **Variable objetivo:** Se mantiene la formulación residual:

   $$
   R(t,x,y)=\text{SST}_{\text{MUR}}(t,x,y)
   -\text{SST}_{\text{BIL}}(t,x,y)
   $$

2. **Predictores disponibles operacionalmente:** La construcción de la matriz $\mathbf{X}$ priorizará variables que puedan estar disponibles independientemente de la SST objetivo de MUR, incluyendo inicialmente:

   - $\text{SST}_{\text{BIL}}$;
   - `depth`;
   - `distance_coast_km`;
   - `ocean_fraction`;
   - componentes temporales cíclicas (`DOY_sin`, `DOY_cos`);
   - otras covariables que posteriormente sean justificadas y verificadas.

3. **Tratamiento de `analysis_error`:** En el modelo principal, `analysis_error` **no se incorporará inicialmente como predictor directo**, debido a que es una variable derivada del propio producto MUR utilizado como referencia y podría introducir dependencia respecto al target o limitar la aplicabilidad posterior del modelo.

   Su utilización se evaluará inicialmente como:

   - variable de control para estratificar el desempeño;
   - indicador auxiliar para análisis de sensibilidad;
   - posible esquema de ponderación de muestras (`sample weights`), únicamente si su beneficio se demuestra experimentalmente sin comprometer la independencia metodológica.

4. **Prohibición de filtros rígidos:** No se eliminarán fechas o muestras exclusivamente por presentar valores elevados, bajos o excepcionales de `analysis_error`. Cualquier tratamiento especial deberá justificarse mediante experimentos controlados y análisis de sensibilidad.

5. **Partición temporal estricta:**

   - **TRAIN:** 2015–2021 (2557 días).
   - **VALIDATION:** 2022–2023 (730 días).
   - **TEST:** 2024–2025 (731 días).

6. **Prevención de fuga de información:** Toda transformación que dependa de la distribución de los datos —normalización, estandarización, selección de variables, ajuste de hiperparámetros o esquemas de ponderación— deberá estimarse exclusivamente con el conjunto de entrenamiento y aplicarse posteriormente, sin reajuste, a validación y prueba.

7. **Estado actual:** **La modelación Machine Learning todavía NO ha iniciado.** El siguiente paso corresponde al diseño formal y ensamblado reproducible del dataset de la Fase D.

## 32. Reproducibilidad y Guía de Ejecución

### Entorno de Ejecución
- **Sistema Operativo:** macOS Darwin (`arm64`) / Linux x86_64
- **Python:** 3.11.16 en `/Users/mariajosenande/Documents/Lole/.venv/bin/python`

### Guía de Comandos Reproducibles

1. **Ejecución de Fase C.2 (Armonización completa 4018 días):**
   ```bash
   python DATASET_TESIS/fase_c2_armonizacion_2015_2025.py
   ```
2. **Recuperación histórica de `analysis_error` (PO.DAAC OPeNDAP):**
   ```bash
   python DATASET_TESIS/descargar_analysis_error_mur_opendap.py
   ```
3. **Auditoría final consolidada de `analysis_error` (N = 4018 días):**
   ```bash
   python DATASET_TESIS/auditar_analysis_error_mur_FINAL.py
   ```
4. **Control temporal de saltos diarios ($\Delta\text{AE}$):**
   ```bash
   python DATASET_TESIS/auditar_cambios_abruptos_analysis_error.py
   ```
5. **Diagnóstico puntual de `analysis_error = 0` (2016-05-23):**
   ```bash
   python DATASET_TESIS/diagnosticar_zero_analysis_error_20160523.py
   ```

---

## 33. Catálogo de Artefactos Generados

### Productos NetCDF Principales
- [`DATASET_TESIS/outputs/faseC2_2015_2025.nc`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/outputs/faseC2_2015_2025.nc): Cubo consolidado 4018 días (567 MB).
- [`DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc): Serie continua de `analysis_error` (133 MB).

### Catálogo de Figuras Oficiales de Auditoría (`DATASET_TESIS/auditoria_analysis_error/figures/`)
- **[Figura 1: `figura1_serie_temporal_rmse_analysis_error.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura1_serie_temporal_rmse_analysis_error.png):** Serie temporal completa 2015–2025 de RMSE vs `analysis_error`.
- **[Figura 2: `figura2_scatter_rmse_analysis_error.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura2_scatter_rmse_analysis_error.png):** Dispersión entre RMSE y `mean_AE` con ajuste de regresión ($r = +0.3338$).
- **[Figura 3: `figura3_scatter_absbias_analysis_error.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura3_scatter_absbias_analysis_error.png):** Dispersión entre $|Bias|$ y `mean_AE` ($r = +0.2780$).
- **[Figura 4: `figura4_boxplot_analysis_error_grupos.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura4_boxplot_analysis_error_grupos.png):** Distribución de `analysis_error` por grupos de severidad de RMSE.
- **[Figura 5: `figura5_distribucion_fraction_at_041.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura5_distribucion_fraction_at_041.png):** Frecuencia y dispersión de la fracción en saturación ($0.4100^\circ\text{C}$).
- **[Figura 6: `figura6_comparacion_eventos_E1_E6.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura6_comparacion_eventos_E1_E6.png):** Comparación cuantitativa de los eventos extremos E1 a E6.
- **[Figura 7: `figura7_delta_analysis_error_2015_2025.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura7_delta_analysis_error_2015_2025.png):** Serie temporal de cambios abruptos $\Delta\text{AE}$ con Top 10 transiciones marcadas y leyendas en cuadrante inferior derecho.
- **[Figura 8: `figura8_analysis_error_20160522_24.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura8_analysis_error_20160522_24.png):** Evolución espacial en 3 paneles del episodio del 23 de mayo de 2016.
- **[Figura 9: `figura9_mascara_analysis_error_zero_20160523.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura9_mascara_analysis_error_zero_20160523.png):** Mapa binario de celdas con $\text{AE} = 0.00^\circ\text{C}$ superpuesto a batimetría GEBCO.
- **[Figura 10: `figura10_analysis_error_zero_event_temporal.png`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/figures/figura10_analysis_error_zero_event_temporal.png):** Diagnóstico temporal de 4 paneles del episodio de mayo de 2016.

### Reportes Oficiales (`DATASET_TESIS/auditoria_analysis_error/reports/`)
- [`auditoria_analysis_error_FINAL_4018dias.md`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md): Informe técnico de la auditoría final de 4018 días.
- [`auditoria_cambios_abruptos_analysis_error.md`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/reports/auditoria_cambios_abruptos_analysis_error.md): Informe de control temporal de saltos diarios.
- [`diagnostico_analysis_error_zero_20160523.md`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/auditoria_analysis_error/reports/diagnostico_analysis_error_zero_20160523.md): Diagnóstico exhaustivo del valor cero del 23 de mayo de 2016.

---

## 34. Limitaciones Metodológicas

1. **Naturaleza analizada de MUR:** MUR SST v4.1 es un producto L4 interpolado mediante análisis multiescala. Una resolución de malla de ~1 km (0.01°) no garantiza resolución dinámica efectiva equivalente en periodos con cobertura nubosa prolongada.
2. **Limitación de sensores infrarrojos:** Los radiómetros infrarrojos (VIIRS, MODIS) están severamente restringidos por nubosidad, lo que impidió una validación infrarroja concluyente en varios eventos extremos.
3. **Disparidad de magnitudes físicas:** VIIRS y MODIS L2P miden temperatura de piel/subpiel ($\text{SST}_{\text{skin}} / \text{SST}_{\text{subskin}}$) instantánea, sujeta al ciclo diurno, mientras que MUR representa $\text{SST}_{\text{foundation}}$ diaria libre de calentamiento diurno.
4. **Incertidumbre vs Error:** `analysis_error` cuantifica la varianza posterior del estimador multiescala de MUR, no el error residual verdadero frente a la temperatura in situ.

---

## 35. Deuda de Documentación Interna

1. [`DATASET_TESIS/config.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/config.py): Especificar unidades físicas y sistemas de referencia geodésica a nivel de constante.
2. [`DATASET_TESIS/modules/bathymetry.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/bathymetry.py): Documentar explícitamente la cobertura exhaustiva en bordes costeros.
3. [`DATASET_TESIS/modules/residual.py`](file:///Users/mariajosenande/Documents/Lole/DATASET_TESIS/modules/residual.py): Registrar nota de deprecación interna señalando a `interpolate_strategy_a_coastal_support` como función canónica.

---

> **Última actualización:** 2026-08-31  
> **Estado:** FASE C.2 = COMPLETA | VALIDACIÓN SATELITAL = COMPLETA | RECUPERACIÓN `analysis_error` = COMPLETA | AUDITORÍA `analysis_error` = CERRADA | ML = NO INICIADO

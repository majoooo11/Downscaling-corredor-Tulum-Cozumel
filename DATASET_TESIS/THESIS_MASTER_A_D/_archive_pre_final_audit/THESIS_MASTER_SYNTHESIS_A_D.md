# Master Thesis Synthesis — Stages A to D
## Reconstrucción Metodológica, Trazabilidad Científica y Archivo Definitivo para Tesis

**Proyecto:** Downscaling espacial de Sea Surface Temperature (SST) mediante aprendizaje residual en el Caribe mexicano  
**Área de Estudio:** Corredor Tulum–Cozumel, Quintana Roo, México  
**Periodo Decenal:** 2015-01-01 a 2025-12-31 (4,018 días continuos)  
**Nivel del Documento:** Síntesis Maestra Canónica Definitiva (Capa de Documentación de Tesis)  
**Estado:** COMPLETO Y CONGELADO PARA REDACCIÓN DE TESIS

---

## 1. Scientific Objective

El objetivo científico central de la tesis es desarrollar, validar y auditar rigurosamente un marco de downscaling espacial estadístico para proyectar campos diarios de temperatura superficial del mar (SST) desde una escala gruesa de 0.25° (~27 km, provista por NOAA OISST v2.1) hasta una resolución ultra-alta de 0.01° (~1 km, compatible con MUR SST v4.1) sobre el corredor marino Tulum–Cozumel, empleando un enfoque de aprendizaje residual supervisado con modelos basados en árboles de decisión e incorporando covariables fisiográficas estáticas (batimetría GEBCO) y temporales cíclicas (DOY).

---

## 2. Study Area

El corredor marino Tulum–Cozumel se localiza en la costa caribeña de la Península de Yucatán:
- **Latitud:** $19.90^\circ\text{N}$ a $20.75^\circ\text{N}$
- **Longitud:** $-87.60^\circ\text{W}$ a $-86.65^\circ\text{W}$
- **Complejidad oceanográfica:** El área alberga una estrecha plataforma somera (0–20 m), lagunas arrecifales protegidas del Sistema Arrecifal Mesoamericano, la isla de Cozumel, el canal de Cozumel y cañones oceánicos profundos (>1000 m) sujetos al flujo forzado de la corriente de Yucatán.

---

## 3. Overall Pipeline

El pipeline integral se articula en cuatro etapas metodológicas consecutivas y no circulares:
1. **Etapa A:** Adquisición, verificación temporal y auditoría de integridad decenal (MUR, OISST, GEBCO, L2P).
2. **Etapa B & B.1:** Definición del marco espacial estándar (86 × 96 celdas a 0.01°) y construcción de la máscara oceánica corregida (5,279 celdas marinas).
3. **Etapa C (C.1, C.1b, C.1c, C.2):** Armonización multirresolución, diagnóstico y solución del soporte costero (Estrategia A), generación del cubo decenal `faseC2_2015_2025.nc` y auditorías complementarias (Satélite L2P y MUR `analysis_error`).
4. **Etapa D (D.1, D31, D32, D33, D34, D35, D36):** Construcción del dataset tabular Parquet, diagnóstico de predictibilidad, selección de la formulación parsimoniosa `E3b-C0` en Holdout 2021, validación externa temporal en 2022–2023, auditoría diagnóstica no adaptativa, evaluación terminal ciega en Final Test 2024–2025 y consolidación final.

---

## 4. Stage A — Data Acquisition

Se verificaron exhaustivamente las cuatro fuentes de datos para el periodo 2015–2025:
- **MUR SST v4.1:** 5 archivos NetCDF históricos (2015–2019) y 1,710 gránulos diarios L4 (2019–2025). 4,018 fechas continuas disponibles.
- **NOAA OISST v2.1:** Descarga vía ERDDAP en cuadrícula de 0.25° con halo de 0.5° y recuperación local en 2025-01-14.
- **GEBCO 2026 Grid:** Archivo `gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc` a 15 arc-segundos.
- **VIIRS S-NPP / MODIS Aqua L2P:** Gránulos orbitales independientes para auditoría radiométrica.

---

## 5. Stage A — Quality Control

- **Completitud:** 4,018 días esperados = 4,018 días encontrados (0 faltantes, 0 duplicados, 0 corrupciones).
- **Consistencia física:** Rangos térmicos validados en [24.0 °C, 32.5 °C] en mar abierto y lagunas.
- **Conclusión de Etapa A:** Datos de entrada científicamente completos, temporalmente continuos y aprobados.

---

## 6. Stage B — Master Spatial Grid

- **Malla maestra:** Basada en la cuadrícula de MUR SST v4.1 a 0.01° de resolución.
- **Dimensiones:** 86 celdas latitudinales × 96 celdas longitudinales = **8,256 celdas totales**.
- **Proyección cartográfica:** UTM Zona 16 Norte (**EPSG:32616**).

---

## 7. Stage B.1 — Ocean Mask and Static Covariates

- **Máscara corregida:** $M_{\text{final}} = M_{\text{MUR}} \land (\text{ocean\_fraction} \ge 0.5)$.
- **Censo espacial base:** **5,279 celdas oceánicas** y **2,977 celdas terrestres** (383 celdas continentales reclasificadas a tierra respecto a la máscara nativa de MUR; interior de Cozumel asignado estrictamente a tierra).
- **Covariables fisiográficas:** Profundidad positiva ($depth$), distancia mínima euclidiana a la costa ($distance\_coast\_km$) y fracción oceánica subpíxel ($ocean\_fraction$).

---

## 8. Stage C.1 — Initial Harmonization

- En el ensayo piloto del 2015-01-01, la interpolación bilineal directa de OISST arrojó **1,292 NaNs costeros** (pérdida del **24.47%** de celdas oceánicas).
- Causa: Presencia de 20 nodos terrestres en la porción occidental del halo OISST (Península de Yucatán).

---

## 9. Stage C.1b — Coastal Coverage Problem

- **Comparación cuantitativa:**
  - Estrategia A (extensión costera auxiliar nearest-ocean en 0.25°): 5,279 / 5,279 celdas válidas.
  - Estrategia B (triangulación 2D Delaunay sobre océano): 5,279 / 5,279 celdas válidas.
  - Discrepancia A vs B: $\text{MAE} = 0.0038^\circ\text{C}, \, P_{95} = 0.0214^\circ\text{C}, \, \text{Máxima} = 0.0626^\circ\text{C}$.
- Se ratificó la equivalencia numérica entre ambos métodos.

---

## 10. Stage C.1c — Adopted Solution

- Se adoptó la **Estrategia A** por garantizar regularidad cartesiana, eficiencia algorítmica para 4,018 días y trazabilidad explícita mediante la máscara `oisst_coastal_support_mask`.
- Se estableció el Baseline E0 piloto para 2015-01-01 ($	ext{RMSE} = 0.2367^\circ	ext{C}$). Se aprobó la ejecución decenal completa.

---

## 11. Stage C.2 — Full 2015–2025 Harmonization

- **Ejecución decenal:** 4,018 días procesados sin interrupciones.
- **Volumen observacional:** $21,211,022$ observaciones espaciotemporales acumuladas.
- **Métricas decenales globales del Baseline E0:**
  $$\text{RMSE} = 0.3426^\circ\text{C}, \quad \text{MAE} = 0.2631^\circ\text{C}, \quad \text{Bias} = +0.0133^\circ\text{C}, \quad R^2 = 0.9016$$
- **Artefactos congelados:** `outputs/faseC2_2015_2025.nc` (540.82 MB) y 11 archivos anuales en `outputs/fase_c2/`.

---

## 12. Satellite Validation

- **Auditoría independiente:** VIIRS S-NPP v2.80 y MODIS Aqua v2019.0 en eventos de discrepancia extrema E1–E6.
- **Hallazgo:** Bloqueo nuboso casi total (>80–100% de píxeles con $QL < 5$). En el pico de octubre 2015, MODIS tuvo 0.0% de datos limpios y VIIRS 0.1% (10 píxeles aislados).
- **Dictamen:** Evidencia satelital independiente clasificada como **heterogénea e inconclusa**. Se confirma mantener `faseC2_2015_2025.nc` intacto.

---

## 13. MUR Uncertainty Audit

- Auditoría de la variable `analysis_error` sobre las 4,018 fechas ($21,211,022$ puntos).
- **Correlación:** Spearman $\rho = +0.2853$ ($p = 4.25 \times 10^{-76}$).
- **Techo algorítmico:** Confirmación de saturación asintótica en $0.4100^\circ\text{C}$ durante vacíos observacionales infrarrojos directos.
- **Decisión:** Excluir `analysis_error` de las variables predictoras del modelo ML y preservarla como variable de control diagnóstico.

---

## 14. Transition from Harmonized Cube to ML Dataset

- **Transición geométrica 5,279 -> 5,275 celdas:** En la subfase D32, la evaluación de ablaciones con features espaciales 2D (`grad_mag`) requirió diferencias finitas no nulas en $X$ e $Y$. Cuatro celdas costeras/insulares aisladas (índices 0, 161, 4437, 4472) arrojaron `grad_mag = NaN`. La máscara común `COMMON_VALID_MASK` excluyó estas cuatro celdas, congelando el dominio evaluativo de Machine Learning en exactamente **5,275 celdas marinas** (`frozen_cell_ids.csv`, SHA-256: `6f046931...`).

---

## 15. Stage D — ML Dataset Construction

- Formato tabular analítico Apache Parquet comprimido con Snappy en `ml_dataset/`.
- 13 columnas estructuradas: `date`, `year`, `doy`, `sst_bil`, `depth`, `distance_coast_km`, `ocean_fraction`, `doy_sin`, `doy_cos`, `sst_mur`, `residual`, `analysis_error`, `split`.
- Partición temporal estricta anti-fuga (Development 2015–2020, Holdout 2021, Validation 2022–2023, Test 2024–2025).

---

## 16. D31 — Predictability Diagnostics

- Demostración empírica de que el residual fino $R = \text{sst\_mur} - \text{sst\_bil}$ no es ruido blanco estocástico desestructurado.
- Autocorrelación temporal reproducible ($r \approx 0.68$ a lag 1 día) y correlación espacial sistemática con la topografía submarina justificaron avanzar al modelado no lineal tabular.

---

## 17. D32 — Model Selection

- Evaluación de seis formulaciones prespecificadas (`E3b-C0`, `E3b-T1`, `E3b-T3`, `E3b-S`, `E3b-TS`, `E3b-ALL`) sobre Holdout 2021 ($1,925,375$ obs).
- **Modelo seleccionado:** **`E3b-C0`** (XGBoost con 4 features: `sst_bil`, `doy_sin`, `doy_cos`, `depth`).
- **Desempeño Holdout 2021:** B0 RMSE = $0.359493^\circ\text{C}$ -> C0 RMSE = $0.349274^\circ\text{C}$ (**+2.84243%** de mejora).
- Las variantes con lags y vecindad espacial obtuvieron menor mejora (+2.21% a +2.63%). Selección por **parsimonia y desempeño**. Estado: **METHODOLOGICALLY CLOSED**.

---

## 18. D33 — External Validation

- Reajuste de la especificación `E3b-C0` en datos previos 2015–2021 ($13,488,175$ obs) y evaluación fuera de muestra en el bienio **2022–2023** ($3,850,750$ obs).
- **Resultados:** B0 RMSE = $0.335666^\circ\text{C}$ -> C0 RMSE = $0.323838^\circ\text{C}$ (**+3.5237%** de mejora global). 16/24 meses mejorados (66.7%); 4,755/5,275 celdas mejoradas (90.14%).
- **Dictamen formal:** **D33-B — Partial/Mixed Generalization** (mejora global positiva pero estabilidad mensual sub-umbral predeclarado de 18 meses). Se retiene avance a Final Test.

---

## 19. D34 — Diagnostic Audit

- Auditoría diagnóstica no adaptativa sobre las predicciones de validación.
- Hallazgos: Fuerte compresión de amplitud (ratio de std ~0.25); degradación leve en regímenes de bajo residual ($|R| < 0.2066^\circ\text{C}$); alta exactitud en discrepancias moderadas a severas; correlación positiva de ganancias con proximidad a costa y baja profundidad.
- Recomendación histórica: **PREPARE FINAL TEST**. Estado: **INTERPRETATIONALLY CLOSED**.

---

## 20. D35 — Final Test

- Reajuste final de la especificación `E3b-C0` en el periodo 2015–2023 ($17,338,925$ obs) e inferencia única sobre el conjunto ciego **2024–2025** ($3,856,025$ obs, 731 días).
- **Resultado primario confirmado:**
  - B0 RMSE: $0.357317^\circ\text{C}$ -> C0 RMSE: **$0.331502^\circ\text{C}$** (**+7.2247%** de mejora global)
  - MAE: $0.272713^\circ\text{C}$ -> $0.256325^\circ\text{C}$ (+6.01%)
  - Bias: $-0.062425^\circ\text{C}$ -> $-0.014197^\circ\text{C}$ (reducción de 77.3%)
  - $R^2$ SST: $0.892661$ -> **$0.907610$**
  - Desglose anual: 2024 = **+9.3986%**; 2025 = **+4.4704%**
  - Robustez temporal: **18 de 24 meses mejorados** (75.0%); **484 de 731 días mejorados** (66.21%)
  - Inferencia bootstrap 14d $CI_{95}$: **$[-0.042485, -0.009792]^\circ\text{C}$**
  - Cobertura espacial: **5,273 de 5,275 celdas mejoradas** (**99.96%**)
- **Dictamen formal:** **D35-A — FINAL GENERALIZATION CONFIRMED**.
- **Condición de cierre:** **FINAL TEST CONSUMED**.

---

## 21. D36 — Final ML Synthesis

- Consolidación transversal definitiva de resultados D31–D35.
- Cero reentrenamientos, cero modificaciones de hiperparámetros, cero aperturas lógicas raw a test.
- Declaración terminal: **ML MODEL DEVELOPMENT SCIENTIFICALLY CLOSED**.

---

## 22. Main Methodological Decisions

1. Cuadrícula de MUR como malla maestra del proyecto (DEC-01).
2. Corrección de máscara con GEBCO $\text{ocean\_fraction} \ge 0.5$ (DEC-02).
3. Adopción de Estrategia A para soporte costero bilineal (DEC-03).
4. Exclusión de `analysis_error` del vector de predictores (DEC-05).
5. Partición cronológica estricta out-of-sample (DEC-06).
6. Selección de `E3b-C0` por parsimonia y desempeño (DEC-07).
7. Dictamen D33-B y pausa diagnóstica (DEC-08).
8. Dictamen confirmatorio D35-A y consumo de test (DEC-10).
9. Cierre científico definitivo del bloque de Machine Learning (DEC-11).

---

## 23. Main Scientific Results

1. La interpolación bilineal directa de OISST v2.1 deja un error basal promedio decenal de $\text{RMSE} = 0.3426^\circ\text{C}$.
2. El aprendizaje residual tabular con 4 predictores físicos y temporales (`E3b-C0`) reduce el error cuadrático medio de downscaling en un **+7.22% global** en evaluación ciega fuera de muestra.
3. La mejora es espacialmente cuasi-universal (**99.96% de celdas beneficiadas**) y se concentra fuertemente en aguas someras (0–20 m: $-0.0383^\circ\text{C}$ de reducción media; $\rho = +0.7376$).
4. El modelo opera como un estimador conservador de contracción hacia la media condicional (compresión de amplitud de ~0.25), alcanzando alta exactitud de signo (>70–87%) en discrepancias térmicas moderadas a severas.

---

## 24. Limitations

1. Falta de red de boyas in situ de alta frecuencia en el canal de Cozumel.
2. Incertidumbre en eventos nubosos prolongados donde la referencia MUR depende de persistencia.
3. Incapacidad del modelo estático para predecir variabilidad instantánea forzada por vientos o corrientes dinámicas (explicabilidad del residual en test = 11.22%).
4. Sobrecorrección leve (-20.99%) en el régimen de baja discrepancia ($|R| < 0.2066^\circ\text{C}$).

---

## 25. Reproducibility

- Código modular completamente versionado bajo Git.
- Entorno de ejecución estandarizado (Python 3.11.16 en macOS arm64).
- Semillas estocásticas fijadas (`random_state = 42`).
- Hashes criptográficos de celdas comunes (`frozen_cell_ids.csv`: `6f046931...`).
- Final Test bloqueado e inmutable.

---

## 26. Canonical Artifacts

- **NetCDF Decenal:** `outputs/faseC2_2015_2025.nc` (540.82 MB).
- **Máscara congelada:** `ml_results/E3b_D32/tables/frozen_cell_ids.csv` (5,275 celdas).
- **Tablas de Test:** `ml_results/E3b_D35_final_test/tables/final_test_summary.csv`.
- **Suite de Consolidación:** `ml_results/E3b_FINAL_SYNTHESIS/` (tables, figures, reports).
- **Capa Maestra de Tesis:** `DATASET_TESIS/THESIS_MASTER_A_D/` (tables, figures, reports).

---

## 27. Documentation Gaps

Todos los gaps detectados fueron catalogados y resueltos en `DOCUMENTATION_GAPS_A_D.md`:
- Censo 5,279 -> 5,275 explicado por diferencias finitas en `grad_mag`.
- GEBCO 2026 Grid ratificado como versión física real frente a citas históricas de 2024.
- Orientación de la regresión de calibración aclarada como $\hat{R} = a + b R$ ($b = 0.0891$).
- Seis formulaciones candidatas ratificadas en D32.
- Concepto de especificación congelada aclarado frente a reutilización de pesos.

---


- GAP-09: Umbrales DEV canónicos ratificados: DEV-P50 = 0.2066 °C, DEV-P75 = 0.3604 °C, DEV-P90 = 0.5377 °C, DEV-P95 = 0.6652 °C, DEV-P99 = 0.9659 °C.
- GAP-10: Desglose anual de meses mejorados verificado en monthly_metrics.csv: exactamente 9/12 en 2024 y 9/12 en 2025 (18/24 total, 75.0%).
- GAP-11: Hash SHA-256 canónico pre-test de `frozen_spatial_metadata.csv` ratificado como `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`.
- GAP-12: Consistencia matemática de C.2 aclarada: RMSE agrupado (pooled) decenal es 0.3426 °C; el promedio de RMSEs diarios es 0.3019 °C; los promedios anuales diarios varían de 0.2724 a 0.3257 °C; los RMSEs agrupados anuales varían de 0.3034 a 0.3811 °C.

## 28. Thesis-Ready Conclusions

1. **Viabilidad Metodológica:** Es técnicamente viable y científicamente riguroso downescalar OISST a 0.01° en el Caribe mexicano mediante aprendizaje residual tabular acoplado a batimetría y periodicidad astronómica.
2. **Parsimonia:** Modelos de gradiente boosting compactos (XGBoost con 19 árboles y profundidad 4) superan a formulaciones densas con lags autoregresivos o vecindades espaciales 2D.
3. **Generalización Terminal:** La generalización temporal out-of-sample fue comprobada estadísticamente en Final Test 2024–2025 con un $CI_{95}$ de $[-0.0425, -0.0098]^\circ\text{C}$ que excluye estrictamente el cero.
4. **Dominio de Aplicabilidad:** El modelo es altamente eficaz para la corrección de gradientes costeros persistentes en aguas someras, debiendo aplicarse con cautela en condiciones oceánicas homogéneas de baja discrepancia.

---

## 29. Final Project Status

El estado científico terminal de las Etapas A–D queda formalmente cerrado y archivado:

| Stage | Scientific function | Final status |
| :--- | :--- | :--- |
| **A** | Data acquisition and audit | **COMPLETED** |
| **B** | Spatial framework | **COMPLETED** |
| **B.1** | Final land/ocean mask | **COMPLETED** |
| **C.1** | Initial harmonization | **COMPLETED** |
| **C.1b** | Coastal support diagnosis | **COMPLETED** |
| **C.1c** | Coastal solution validation | **COMPLETED** |
| **C.2** | Full harmonization 2015–2025 | **COMPLETED** |
| **Satellite audit** | Independent L2P radiometry verification | **COMPLETED / EVIDENCE HETEROGENEOUS** |
| **MUR uncertainty audit** | Analysis error decenal characterization | **CLOSED** |
| **D dataset** | Tabular ML analytical dataset construction | **COMPLETED** |
| **D31** | Residual predictability diagnostics | **CLOSED** |
| **D32** | Parsimonious model selection & ablation | **METHODOLOGICALLY CLOSED** |
| **D33** | External temporal validation 2022–2023 | **D33-B — UNCHANGED** |
| **D34** | Diagnostic audit & microaudit | **INTERPRETATIONALLY CLOSED** |
| **D35** | Blind out-of-sample evaluation 2024–2025 | **D35-A — FINAL GENERALIZATION CONFIRMED** |
| **D36** | Final ML block synthesis | **FINAL SYNTHESIS COMPLETED** |
| **FINAL TEST** | Terminal out-of-sample evaluation | **CONSUMED** |
| **ML MODEL DEVELOPMENT** | Machine Learning development & tuning | **CLOSED** |

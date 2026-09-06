# Master Methodology Synthesis (Stages A to D)
## Reconstrucción Metodológica Integral para Redacción de Tesis

**Documento:** `THESIS_METHODS_MASTER_A_D.md`  
**Estado:** AUDITED & FROZEN  
**Regla Epistemológica:** Estricta separación entre MÉTODO y RESULTADO. Este documento describe exhaustivamente el marco teórico-metodológico, las decisiones de diseño, las formulaciones matemáticas y las salvaguardas de reproducibilidad sin incrustar tablas de resultados ni discusiones interpretativas.

---

## 1. Study Area

El área de estudio corresponde al sector marino **Tulum–Cozumel**, situado en la costa oriental de la Península de Yucatán (Quintana Roo, México), en el Caribe occidental. Los límites geográficos del dominio se establecen formalmente en:

$$\text{Latitud}: 19.90^\circ\text{N} \text{ a } 20.75^\circ\text{N}$$
$$\text{Longitud}: -87.60^\circ\text{W} \text{ a } -86.65^\circ\text{W}$$

El dominio abarca el canal de Cozumel, la plataforma arrecifal somera del Sistema Arrecifal Mesoamericano (SAM) y cuencas oceánicas profundas (>1000 m). El canal de Yucatán propiamente dicho se localiza al norte de los 21.5°N y no forma parte del dominio delimitado.

---

## 2. Data Sources

El pipeline integra tres fuentes primarias de información satelital y fisiográfica:
1. **MUR SST v4.1 (NASA/JPL PO.DAAC):** Producto L4 global de resolución ultra-alta (0.01°, ~1 km) que asimila observaciones infrarrojas (MODIS, VIIRS, AVHRR) y de microondas (AMSR2, WindSat) mediante una técnica de interpolación multiescala (*Chin et al., 2017*).
2. **NOAA OISST v2.1 (NOAA NCEI):** Producto L4 global diario sobre grilla regular de 0.25° (~27 km), basado en interpolación óptima (*Huang et al., 2021*). Constituye el predictor térmico de baja resolución a ser downescalado.
3. **GEBCO 2026 Grid (BODC / Nippon Foundation-GEBCO Seabed 2030):** Modelo continuo de elevación terreno/océano a 15 arc-segundos (~450 m de resolución), derivado de la fusión de batimetría acústica multihaz y altimetría satelital SRTM15+ v2.8 (publicado en abril 2026, DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa).
4. **Radiometría Infrarroja L2P (VIIRS S-NPP y MODIS Aqua):** Gránulos orbitales independientes L2P no interpolados, utilizados exclusivamente para auditorías de calidad radiométrica en eventos extremos.

---

## 3. Data Acquisition and Quality Control

El periodo experimental abarca exactamente 11 años completos: del **2015-01-01 al 2025-12-31**, totalizando **4,018 días astronómicos continuos**.
- **Integridad temporal:** Se realizó una auditoría de ingesta donde cada fecha fue verificada contra el índice temporal del producto. No se detectaron días faltantes ni duplicados (cobertura temporal = 100.0%).
- **Conversión de unidades:** Las observaciones de MUR SST se transformaron de Kelvin a Celsius mediante $T_{\text{C}} = T_{\text{K}} - 273.15$.
- **Recuperación local:** Para OISST, un fallo puntual de conexión ERDDAP en la fecha 2025-01-14 fue subsanado mediante ingestión del archivo NetCDF local correspondiente.

---

## 4. Master Spatial Grid

Para posibilitar la armonización multirresolución, se seleccionó la cuadrícula espacial de MUR SST como malla maestra de referencia. Las características de la grilla son:
- Dimensiones: **86 celdas en latitud × 96 celdas en longitud** ($N_{\text{total}} = 8,256$ celdas).
- Resolución angular regular: $0.0100^\circ$ (~1.11 km en meridiano; ~1.04 km en paralelo zonal).
- Coordenadas proyectadas de soporte: Proyección Universal Transversa de Mercator (UTM) Zona 16 Norte, datum WGS84 (**EPSG:32616**).

---

## 5. Land/Ocean Masking

La máscara nativa de tierra/océano de MUR clasifica como agua varias celdas con influencia costera mixta o lagunas interiores de la isla de Cozumel. Para evitar contaminación por firmas térmicas terrestres, se construyó una máscara corregida:
1. Se calculó la fracción oceánica subpíxel ($\text{ocean\_fraction} \in [0, 1]$) agregando las celdas batimétricas de GEBCO contenidas en cada píxel de 0.01°.
2. Se formuló la regla booleana canónica:
   $$M_{\text{final}} = M_{\text{MUR}} \land (\text{ocean\_fraction} \ge 0.5)$$
3. Se verificó que el interior continental de Cozumel quedara asignado estrictamente a tierra.
4. Resultado: **5,279 celdas oceánicas** y **2,977 celdas terrestres** (383 celdas modificadas respecto a MUR nativo).

---

## 6. Bathymetry and Static Covariates

Sobre la cuadrícula maestra se calcularon tres covariables fisiográficas estáticas:
1. **Profundidad batimétrica ($depth$):** Definida como profundidad positiva bajo el nivel del mar en metros ($depth = -elevation$ para $elevation < 0$; NaN sobre tierra).
2. **Distancia euclidiana a la costa ($distance\_coast\_km$):** Calculada en coordenadas métricas proyectadas (EPSG:32616) como la distancia euclidiana mínima desde el centro de cada celda oceánica hasta el polígono costero continental o insular más cercano.
3. **Fracción oceánica ($ocean\_fraction$):** Proporción de superficie de agua marina en el subpíxel.

---

## 7. OISST Interpolation

Para proyectar el campo térmico grueso OISST (0.25°) a la malla fina (0.01°), se extrae diariamente una ventana espacial con halo de protección de $0.5^\circ$ ($7 \times 7$ nodos OISST) para evitar discontinuidades de contorno. Sobre esta cuadrícula se aplica un operador de interpolación bilineal regular en 2D:

$$\text{SST}_{\text{BIL}}(x, y) = \sum_{i=1}^2 \sum_{j=1}^2 w_{ij} \, \text{OISST}(x_i, y_j)$$

---

## 8. Coastal Support Strategy

Debido a que 20 nodos occidentales del halo OISST corresponden a la Península de Yucatán (tierra continental), la interpolación bilineal directa produjo **1,292 celdas oceánicas con valor NaN** (pérdida del 24.47% del dominio marino).

Para resolver esta frontera matemática sin alterar la física oceánica:
1. **Estrategia A (Adoptada):** Se extendieron los nodos terrestres OISST asignándoles el valor del nodo oceánico válido más cercano (*nearest-ocean coastal extension*). Estos nodos extendidos actúan **únicamente como soporte matemático envolvente** para que la interpolación bilineal en las 5,279 celdas marinas disponga de 4 esquinas cuadriláteras válidas. Posteriormente, el campo resultante se recorta estrictamente con $M_{\text{final}}$, garantizando que ninguna celda terrestre conserve valores de SST.
2. **Estrategia B (Evaluada):** Triangulación 2D de Delaunay basada exclusivamente en nodos oceánicos de OISST.
3. **Criterio de selección:** La Estrategia A fue seleccionada por preservar la regularidad cartesiana de la grilla, mantener costo computacional bajo y asegurar trazabilidad explícita mediante la máscara booleana `oisst_coastal_support_mask`.

---

## 9. Full SST Harmonization

La armonización espaciotemporal completa se ejecutó sobre los 4,018 días del periodo 2015–2025. Cada día genera un campo tridimensional que contiene:
- $\text{sst\_mur}$: SST de alta resolución (~1 km).
- $\text{sst\_bil}$: SST bilineal extendida (~1 km).
- $R$: Campo de residual fino, definido por la identidad aditiva fundamental:
  $$R(x, y, t) = \text{sst\_mur}(x, y, t) - \text{sst\_bil}(x, y, t)$$
- Cobertura espacial: Exactamente 5,279 celdas oceánicas válidas por día ($21,211,022$ observaciones espaciotemporales acumuladas).

---

## 10. Reference Uncertainty Audit

El algoritmo de asimilación de MUR v4.1 reporta diariamente la desviación estándar estimada del error de análisis (`analysis_error`). Se ejecutó una auditoría exhaustiva sobre los 4,018 días para evaluar su comportamiento:
- Se evaluó la correlación entre `analysis_error` y la magnitud de discrepancia MUR–BIL.
- Se examinaron eventos de saturación donde `analysis_error` alcanza su límite asintótico superior en $0.4100^\circ\text{C}$ (*Chin et al., 2017*).
- **Decisión metodológica:** La variable `analysis_error` se incorpora en las tablas maestras exclusivamente como variable de control y filtro diagnóstico post-hoc, **excluyéndola taxativamente del vector de predictores del modelo ML** para evitar circularidad en la estimación de la referencia.

---

## 11. Residual-Learning Formulation

El problema de downscaling espacial se formula bajo el paradigma de **aprendizaje residual** (*residual learning*):
1. En lugar de predecir directamente el campo continuo absoluto de temperatura $\text{SST}_{\text{high}}$, el modelo de Machine Learning predice el campo de discrepancia fina:
   $$\hat{R} = f(X)$$
2. La reconstrucción de alta resolución final se obtiene mediante la suma del campo grueso bilineal y la corrección residual estimada:
   $$\text{SST}_{\text{downscaled}} = \text{SST}_{\text{BIL}} + \hat{R}$$
3. Algebraicamente, si $\hat{R} = 0$, entonces $\text{SST}_{\text{downscaled}} = \text{SST}_{\text{BIL}}$, garantizando que una predicción nula del residual preserva de forma exacta el baseline de interpolación bilineal.

---

## 12. Machine-Learning Dataset

Los campos espaciotemporales del cubo armonizado se estructuran en formato tabular analítico:
- **Unidad observacional:** Registro diario por celda ($date \times cell\_id$).
- **Variables de entrada ($X$):** $\text{sst\_bil}$, $\text{doy\_sin}$, $\text{doy\_cos}$, $depth$.
- **Target ($y$):** Residual fino $R = \text{sst\_mur} - \text{sst\_bil}$ (°C).
- **Almacenamiento:** Formato Apache Parquet particionado anualmente con compresión Snappy.

---

## 13. Temporal Partitioning

Para prevenir la filtración espuria de información por persistencia sinóptica y autocorrelación temporal, se prohibió el uso de muestreo aleatorio (*random train/test split*). El protocolo define cuatro ventanas temporales estrictamente cronológicas y mutuamente excluyentes:

$$\text{Development}: 2015-01-01 \text{ a } 2020-12-31 \quad (6 \text{ años}, 2,192 \text{ días})$$
$$\text{Diagnostic Holdout}: 2021-01-01 \text{ a } 2021-12-31 \quad (1 \text{ año}, 365 \text{ días})$$
$$\text{External Validation}: 2022-01-01 \text{ a } 2023-12-31 \quad (2 \text{ años}, 730 \text{ días})$$
$$\text{Final Test}: 2024-01-01 \text{ a } 2025-12-31 \quad (2 \text{ años}, 731 \text{ días, previamente retenido y actualmente CONSUMIDO})$$

---

## 14. Predictability Diagnostics

En la subfase D31 se evaluó si el residual fino $R$ exhibe estructura predictiva reproducible frente a un baseline nulo o de no-habilidad (*no-skill baseline*). Se implementaron baselines de persistencia temporal, climatología residual local, modelos lineales y pruebas de información mutua sobre el conjunto Development (2015–2020).

---

## 15. Model Selection

En la subfase D32 se prespecificaron **SEIS configuraciones candidatas** derivadas de la familia `E3b`:
1. `E3b-C0` (Core, 4 features): $[\text{sst\_bil}, \text{doy\_sin}, \text{doy\_cos}, \text{depth}]$
2. `E3b-T1` (Temporal 1 lag, 6 features): Core + $[\text{sst\_bil\_lag1}, \Delta\text{sst\_1d}]$
3. `E3b-T3` (Temporal 3 lags, 10 features): Core + lags 1, 2, 3 y deltas asociados.
4. `E3b-S` (Espacial 2D, 7 features): Core + $[\text{grad\_mag}, \text{local\_std\_3x3}, \text{local\_contrast}]$
5. `E3b-TS` (Espaciotemporal, 13 features): Core + Lags T3 + Features espaciales.
6. `E3b-ALL` (Full, 16 features): Todas las anteriores + $[\text{distance\_coast\_km}, \text{ocean\_fraction}, \text{local\_range}]$.

- **Máscara evaluativa común (`COMMON_VALID_MASK`):** Para garantizar comparación idéntica y justa, las 6 variantes se evaluaron sobre las celdas con gradientes y lags completos. Esto fijó el dominio en **5,275 celdas congeladas** (`frozen_cell_ids.csv`, SHA-256: `6f046931...`).
- **Arquitectura:** `XGBRegressor` con algoritmo de división de histogramas (`tree_method = "hist"`).
- **Hiperparámetros congelados en D32:**
  $$\text{n\_estimators} = 19, \quad \text{max\_depth} = 4, \quad \text{learning\_rate} = 0.10$$
  $$\text{subsample} = 0.8, \quad \text{colsample\_bytree} = 0.8, \quad \text{min\_child\_weight} = 5$$
  $$\text{random\_state} = 42, \quad \text{objective} = \text{"reg:squarederror"}$$

---

## 16. External Validation

En la subfase D33, la especificación congelada de `E3b-C0` fue evaluada fuera de muestra en el bienio **2022–2023**:
- **Protocolo de ajuste:** La especificación (4 features e hiperparámetros congelados) fue reajustada utilizando todos los datos previos a la validación: **2015–2021** (2,557 días, $13,488,175$ observaciones).
- **Evaluación:** Aplicación en el bienio 2022–2023 (730 días, $3,850,750$ observaciones).
- **Criterio formal predeclarado:** Dictamen formal inmutable categorizado en D33-A (Confirmada), D33-B (Parcial/Inestable) o D33-C (Fallo).

---

## 17. Diagnostic Robustness Analysis

Tras la emisión del dictamen D33-B, la subfase D34 ejecutó una auditoría diagnóstica no adaptativa sobre las predicciones de validación:
- **Estratificación por regímenes de residual:** Partición del conjunto de prueba según los percentiles canónicos del residual absoluto congelados en Development:
  $$\text{DEV-P50} = 0.2066^\circ\text{C}, \, \text{DEV-P75} = 0.3604^\circ\text{C}, \, \text{DEV-P90} = 0.5377^\circ\text{C}, \, \text{DEV-P95} = 0.6652^\circ\text{C}, \, \text{DEV-P99} = 0.9659^\circ\text{C}$$
- **Compresión de amplitud:** Relación $\text{std}(\hat{R}) / \text{std}(R)$ y regresión de calibración.
- **Exactitud de signo (*Sign Accuracy*):** Proporción de observaciones donde $\text{sign}(\hat{R}) = \text{sign}(R)$, comparada contra el baseline de clase mayoritaria.
- **Correlación de rangos espacial:** Asociación no paramétrica (Spearman) entre $\Delta\text{RMSE}$ por celda y las covariables fisiográficas ($depth$, $distance\_coast\_km$).

---

## 18. Final Refit

Antes de abrir el conjunto Final Test 2024–2025, el protocolo congeló el procedimiento de reentrenamiento final:
- **Periodo de entrenamiento final:** Todos los datos históricos pre-test: **2015-01-01 a 2023-12-31** (9 años completos, 3,287 días astronómicos).
- **Volumen de datos:** Exactamente **17,338,925 observaciones tabulares**.
- **Modelo resultante:** `E3b-C0_FINALREFIT_2015_2023`, manteniendo estrictamente invariables las 4 features y los hiperparámetros $\theta^*$.

---

## 19. Final Test

La evaluación terminal fuera de muestra se ejecutó en la subfase D35:
- **Periodo de prueba:** **2024-01-01 a 2025-12-31** (bienio completo previamente retenido, 731 días astronómicos).
- **Volumen evaluado:** Exactamente **3,856,025 observaciones**.
- **Protocolo ciego:** Ejecución única sin afinamiento, sin selección adaptativa de umbrales y sin modificaciones de código post-acceso.
- **Condición de estado:** Tras la inferencia, el conjunto 2024–2025 queda registrado formalmente como **TEST CONSUMED**, perdiendo su cualidad de conjunto ciego.

---

## 20. Evaluation Metrics

El desempeño del modelo downescalado ($C_0$) frente al baseline bilineal ($B_0$) se cuantifica mediante métricas canónicas:

1. **Error Cuadrático Medio (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
2. **Error Absoluto Medio (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Sesgo Medio (*Mean Bias*):**
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
4. **Mejora Relativa de RMSE (Skill):**
   $$\text{Impr\_RMSE\_pct} = \left(\frac{\text{RMSE}_{B0} - \text{RMSE}_{C0}}{\text{RMSE}_{B0}}\right) \times 100\%$$
   $$\Delta\text{RMSE} = \text{RMSE}_{C0} - \text{RMSE}_{B0} \quad (\Delta\text{RMSE} < 0 \implies \text{Mejora})$$
5. **Inferencia Robusta por Bloques (*Moving Block Bootstrap*):** Remuestreo no paramétrico con 1,000 iteraciones utilizando bloques temporales continuos de **14 días** para estimar el intervalo de confianza al 95% ($CI_{95}$) preservando la dependencia temporal de corto alcance (*short-range temporal dependence under the evaluated block lengths*).

---

## 21. Reproducibility and Leakage Prevention

Para garantizar reproducibilidad absoluta y blindaje metodológico:
1. **Semillas fijas:** `random_state = 42` fijado en todas las operaciones estocásticas de división y ajuste de árboles.
2. **Inmutabilidad de artefactos:** Los pesos del modelo, el censo de celdas (`frozen_cell_ids.csv`: `6f046931...`), los metadatos espaciales (`frozen_spatial_metadata.csv`: `8cc02f86...`) y los archivos de métricas intermedias están protegidos por hashes criptográficos SHA-256.
3. **Cero fugas temporales:** Ningún estimador estadístico utilizado en normalización o evaluación en Holdout, Validation o Test utilizó información posterior al periodo de ajuste correspondiente.

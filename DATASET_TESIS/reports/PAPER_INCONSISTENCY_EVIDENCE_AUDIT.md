# Paper Inconsistency Evidence Audit

**Target Publication:** ICITS'27 Conference Paper & Master's Thesis  
**Date of Audit:** 2026-09-12  
**Audit Protocol:** Deep Archaeological Code, Metadata, Table, and Artifact Audit  
**Target File:** `DATASET_TESIS/reports/PAPER_INCONSISTENCY_EVIDENCE_AUDIT.md`  
**Execution Authority:** Canonical Artifact Synthesis (Phases D.3.1–D.3.6 and Thesis Master Framework)

---

## 1. Executive Summary

| Issue | Canonical Evidence | Correction Required | Confidence |
|:---|:---|:---|:---:|
| **Bootstrap Schemes** | Evaluated block lengths were strictly **1-day cluster**, **7-day**, and **14-day circular moving-block bootstrap** ($B=1{,}000$ paired resamples using daily sum of squares). Never 5 or 10 days. | State "1-day complete-field cluster, 7-day, and 14-day moving-block bootstrap" ($B=1{,}000$). Primary confirmatory scheme is 14 days. | **VERY HIGH** |
| **Residual Terminology** | Stratification variable is strictly $|R| = \|\mathrm{SST}_{\mathrm{MUR}} - \mathrm{SST}_{\mathrm{BIL}}\|$. Thresholds originate from the frozen 2015–2020 Development quantiles (DEV-P50 to DEV-P99). No climatological anomaly was computed. | Replace "thermal anomaly", "anomaly regime", or "extreme anomaly" with "residual-magnitude regimes", "large-residual regime", or "large MUR–BIL departures". | **VERY HIGH** |
| **Bathymetry Causality** | `depth` is in the 4-feature model (`gain = 0.2420`). However, **no ablation without depth** was performed. Spatial correlation $\rho = +0.7376$ ($p < 10^{-15}$) is strictly descriptive. | Reframe bathymetric associations as purely descriptive; avoid claims that bathymetry "guided" or "drove" the model causally. | **VERY HIGH** |
| **GEBCO Version** | The actual physical file processed is `gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc` (`title: The GEBCO_2026 Grid`, created 2026-04-17, DOI: `10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa`). | Correct "GEBCO 2024" (a documented early typographical gap, GAP-02) to **GEBCO 2026 Grid**. | **VERY HIGH** |
| **Predictor Set** | Exactly four features: `sst_bil`, `doy_sin`, `doy_cos`, `depth`. No lat/lon, no distance to coast, no ocean fraction, no lags. | Confirm and maintain the exact phrasing: "parsimonious four-predictor feature set". | **VERY HIGH** |
| **Regularization Interpretation** | Model used `XGBRegressor` with default `reg_lambda = 1.0` (unspecified in params). No experimental ablation of regularization was performed. | Rephrase from causal assertion ("joint consequence of...") to prudent hypothesis: "This variance compression may reflect...". | **VERY HIGH** |
| **Table 1 Values** | Confirmed: 2021 Holdout (+2.84%), 2022–2023 Validation (+3.52%), 2024–2025 Test (+7.22%), 2024 (+9.40%), 2025 (+4.47%). | Reconstruct Table 1 with canonical figures to 4–6 decimal places; note distinct fitted estimators. | **VERY HIGH** |
| **Table 2 Values** | Months improved: 18/24 (75.0%), Days: 484/731 (66.21%), Cells: 5,273/5,275 (99.96%). Regimes: P0–P50 (-20.99%), P50–P75 (+3.51%), P75–P90 (+7.48%), P90–P95 (+9.21%), P95–P99 (+12.07%), P99+ (+11.49%). | Reconstruct Table 2; reconcile rounding of fractions (48.16%, 24.35%, 15.68%, 5.29%, 4.71%, 1.80% summing to 100.0%). | **VERY HIGH** |
| **Bootstrap CI** | 14-day moving-block 95% CI is strictly **$[-0.042485, -0.009792]^\circ\mathrm{C}$**. | Confirm that $[-0.0425, -0.0098]^\circ\mathrm{C}$ is the 14-day moving-block CI for $\Delta\mathrm{RMSE} = \mathrm{RMSE}_{C0} - \mathrm{RMSE}_{B0}$. | **VERY HIGH** |
| **BibTeX Duplicates** | No local `.bib` exists in workspace. BibTeX engines duplicate DOI if both `doi = {...}` and `url = {https://doi.org/...}` are included. | Provide unified, non-duplicated BibTeX entries for MUR and Liu et al. | **VERY HIGH** |
| **Broken Citation Candidate** | In Discussion, context on downscaling under extremes/nonstationarity points to candidates in the local `PAPERS` library. | Report Cyriac et al. (2025), Reder et al. (2025), or Kalmus et al. (2022) as candidate references. | **HIGH** |

---

## 2. Canonical ML Configuration

The canonical pipeline is governed by the following mathematical and algorithmic specifications:

1. **Baseline Formulation ($B_0$):**
   $$B_0(s,t) = \mathrm{SST}_{\mathrm{BIL}}(s,t)$$
   where $\mathrm{SST}_{\mathrm{BIL}}$ represents the bilinear interpolation of NOAA OISST (0.25° grid) onto the 5,275 ocean cells of the master 0.01° grid.

2. **Residual Definition ($R$):**
   $$R(s,t) = \mathrm{SST}_{\mathrm{MUR}}(s,t) - \mathrm{SST}_{\mathrm{BIL}}(s,t)$$
   Sign convention: **MUR reference minus OISST bilinear baseline**. Positive values indicate areas where the fine-resolution reference is warmer than the coarse interpolation.

3. **Statistical Learner Output ($\hat{R}$):**
   $$\hat{R}(s,t) = f(\mathbf{x}(s,t))$$
   approximated using extreme gradient boosting (`XGBRegressor`, `tree_method='hist'`).

4. **Reconstructed Downscaled Field ($\widehat{\mathrm{SST}}$ or Model $C_0$):**
   $$\widehat{\mathrm{SST}}(s,t) = \mathrm{SST}_{\mathrm{BIL}}(s,t) + \hat{R}(s,t) = B_0(s,t) + \hat{R}(s,t)$$
   When $\hat{R}(s,t) = 0$, the framework exactly recovers the bilinear baseline $B_0$.

---

## 3. Bootstrap Audit

### Exact Implementation Details
The statistical uncertainty of aggregate $\Delta\mathrm{RMSE}$ was audited from `DATASET_TESIS/fase_d35_final_test_c0.py` (lines 953–1020), `DATASET_TESIS/fase_d34_postvalidation_diagnostics.py` (lines 751–835), and `ml_results/E3b_D35_final_test/tables/bootstrap_sensitivity.csv`.

```
BOOTSTRAP_CANONICO:
- Resamples:                B = 1,000 replicates
- Esquema 1:                1-day complete-field cluster bootstrap (preserving spatial dependence across all 5,275 cells)
- Esquema 2:                7-day circular moving-block bootstrap
- Esquema 3:                14-day circular moving-block bootstrap (Primary confirmatory test criterion)
- Tipo:                     Paired resamples (identical resampled day indices applied simultaneously to daily SSE_B0 and SSE_C0)
- Acumulación:              Reconstructed from sums of squared errors (SSE) and observation counts:
                            RMSE_boot = sqrt(sum(daily_SSE[indices]) / sum(daily_N[indices]))
                            DeltaRMSE_boot = RMSE_C0_boot - RMSE_B0_boot
- Circular/non-circular:    Circular in Phase D.3.5 (code: idx_block = [(i + j) % n_days_tot for j in range(L)], allowing year-end wraparound).
                            Non-circular was evaluated in diagnostic Phase D.3.4 (code: n_blocks = 730 - L + 1).
- CI final (14-day):        [-0.042485, -0.009792] °C
- Fuente(s) exacta(s):      fase_d35_final_test_c0.py (L953–1020), tables/bootstrap_sensitivity.csv
- Versiones antiguas:       No version ever evaluated 5-day or 10-day blocks. Phase D.3.3 only evaluated 1-day clusters.
- Dictamen:
  [x] El paper debe decir 1-day, 7-day, and 14-day moving-block bootstrap
  [ ] Otra cosa
```

---

## 4. Residual-Regime Audit

### Construction and Verification
The residual stratification was audited from `final_test_freeze_manifest.json`, `paper_table3_residual_regimes.csv`, and `residual_regime_metrics.csv`.

```
RESIDUAL_REGIMES:
- Variable de estratificación:  |R| = |SST_MUR - SST_BIL|
- ¿Es |R|?:                     SÍ, estrictamente el valor absoluto de la discrepancia residual MUR–BIL.
- Definición matemática:        R(s,t) = SST_MUR(s,t) - SST_BIL(s,t); cuantiles calculados sobre |R|.
- Origen de cuantiles:          Conjunto de DEVELOPMENT (2015–2020), congelados pre-test (GAP-09).
- Cuantiles congelados:         DEV-P50 = 0.2066 °C
                                DEV-P75 = 0.3604 °C
                                DEV-P90 = 0.5377 °C
                                DEV-P95 = 0.6652 °C
                                DEV-P99 = 0.9659 °C
- Bins exactos en TEST:
  * DEV-P0-P50:                 |R| < 0.2066 °C           (1,857,229 obs, 48.16%)
  * DEV-P50-P75:                0.2066 <= |R| < 0.3604 °C  (938,991 obs, 24.35%)
  * DEV-P75-P90:                0.3604 <= |R| < 0.5377 °C  (604,739 obs, 15.68%)
  * DEV-P90-P95:                0.5377 <= |R| < 0.6652 °C  (204,020 obs,  5.29%)
  * DEV-P95-P99:                0.6652 <= |R| < 0.9659 °C  (181,676 obs,  4.71%)
  * DEV-P99+:                   |R| >= 0.9659 °C          (69,370 obs,   1.80%)
- Métricas clave:
  * DEV-P0-P50:                 RMSE Impr = -20.99%, Sign Acc = 56.4%, Overcorr = 31.08%, Amp Ratio = 0.73
  * DEV-P50-P75:                RMSE Impr = +3.51%,  Sign Acc = 65.8%, Overcorr = 1.29%,  Amp Ratio = 0.30
  * DEV-P75-P90:                RMSE Impr = +7.48%,  Sign Acc = 70.9%, Overcorr = 0.00%,  Amp Ratio = 0.20
  * DEV-P90-P95:                RMSE Impr = +9.21%,  Sign Acc = 75.7%, Overcorr = 0.00%,  Amp Ratio = 0.18
  * DEV-P95-P99:                RMSE Impr = +12.07%, Sign Acc = 82.5%, Overcorr = 0.00%,  Amp Ratio = 0.20
  * DEV-P99+:                   RMSE Impr = +11.49%, Sign Acc = 87.2%, Overcorr = 0.00%,  Amp Ratio = 0.45
- ¿Es anomalía climatológica?:  NO. En ninguna parte de la formulación de regímenes se sustrajo una climatología histórica
                                estacional o diaria. Corresponde exclusivamente al error de representación de OISST respecto a MUR.
- Terminología recomendada:
  * residual-magnitude regimes
  * large-residual regime
  * large MUR–BIL departures / discrepancies
  * low-discrepancy regime (para |R| < 0.2066 °C)
- Fuente(s):                    final_test_freeze_manifest.json, paper_table3_residual_regimes.csv
- Dictamen final:               Sustituir cualquier mención de "thermal anomaly" por terminología de residuales.
```

---

## 5. Spatial/Bathymetry Claims Audit

```
DEPTH_CAUSALITY:
- ¿depth está en el modelo final?:  SÍ (como cuarto predictor estático).
- Feature set final:                ['sst_bil', 'doy_sin', 'doy_cos', 'depth']
- ¿Existe ablación con/sin depth?:  NO. En la Fase D.3.2 todas las 6 variantes (C0, T1, T3, S, TS, ALL) incluyeron depth.
                                    Nunca se entrenó un modelo sin batimetría sobre el conjunto de desarrollo.
- ¿Existe feature importance?:      SÍ. En E3b_D32/tables/feature_importance.csv:
                                    depth gain relativo = 0.2420 (rango 2 tras doy_sin con 0.3158),
                                    permutation delta MSE = 0.000846 (1.75% relativo).
- ¿Existe SHAP?:                    NO. No se calcularon valores SHAP en ninguna fase del proyecto.
- ¿Puede afirmarse causalidad?:     NO. No existen experimentos de intervención ni ablaciones aisladas.
- ¿Qué evidencia sí existe?:        Asociación espacial descriptiva:
                                    1. Correlación de Spearman entre profundidad y DeltaRMSE por celda: rho = +0.7376 (p < 1e-15).
                                    2. Correlación con distancia a la costa: rho = +0.5336 (p < 1e-15).
                                    3. Mayor reducción de RMSE en estratos someros:
                                       0–20 m:  -0.0383 °C (100% celdas mejoradas)
                                       20–50 m: -0.0327 °C (100% celdas mejoradas)
                                       >500 m:  -0.0145 °C (99.91% celdas mejoradas)
- Redacción defendible:
  “The largest improvements occurred along shallow nearshore areas and across parts of the Cozumel Channel. These spatial patterns are descriptive and do not by themselves establish a causal contribution of bathymetric depth.”
- Evaluación de consistencia:       TOTALMENTE CONSISTENTE. Es exactamente la postura adoptada en paper_results_ML.md (L49)
                                    y paper_discussion_ML.md (L33, L76).
```

---

## 6. GEBCO Version Audit

```
GEBCO_VERSION_AUDIT:
- Nombre exacto del archivo:    GEBCO/gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc
- Atributos NetCDF verificados: * title: The GEBCO_2026 Grid - a continuous terrain model for oceans and land at 15 arc-second intervals
                                * source: The GEBCO_2026 Grid is the 2026 release of the global bathymetric product...
                                * date_created: 2026-04-17
                                * id / DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa
                                * base-map: Version 2.8 of SRTM15+ (Tozer et al., 2019)
- ¿Aparece GEBCO 2023?:         NO.
- ¿Aparece GEBCO 2024?:         SÍ, en borradores intermedios tempranos y en paper_methods_ML.md (L37).
                                Documentado exhaustivamente como GAP-02 en DOCUMENTATION_GAPS_A_D.md: fue una errata tipográfica.
- ¿Aparece GEBCO 2026?:         SÍ, es la versión física procesada de forma continua e inmutable desde la Fase A.
- Versión real procesada:       GEBCO 2026 Grid (publicado en abril 2026).
- Recomendación para el paper:  Corregir todas las menciones a "GEBCO 2026 Grid (DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa)".
```

---

## 7. Model Predictor and Hyperparameter Audit

```
FEATURE_SET_FINAL:
- Modelo final:                 E3b-C0_FINALREFIT_2015_2023 (serializado en final_test_freeze_manifest.json y E3b-C0_FINALREFIT_2015_2023.json)
- Tipo de modelo:               XGBRegressor (algoritmo hist de partición de histogramas)
- Target:                       R = SST_MUR - SST_BIL
- Predictores exactos:
  1. sst_bil                    (SST interpolada bilinealmente desde OISST 0.25°, en °C)
  2. doy_sin                    (Representación armónica seno: sin(2*pi*DOY/365.25))
  3. doy_cos                    (Representación armónica coseno: cos(2*pi*DOY/365.25))
  4. depth                      (Profundidad batimétrica marina estática en metros positivos, derivada de GEBCO 2026 Grid)
- Total predictores:            4
- ¿Incluye lat/lon?:            NO
- ¿Incluye distance_to_coast?:  NO
- ¿Incluye ocean_fraction?:     NO
- ¿Incluye depth?:              SÍ
- ¿Incluye temporal sin/cos?:   SÍ (doy_sin, doy_cos)
- ¿Incluye SST_BIL?:            SÍ
- ¿Incluye analysis_error?:     NO
- ¿Expresión correcta?:         SÍ. “parsimonious four-predictor feature set” es 100% fiel al modelo.

REGULARIZATION_EVIDENCE:
- Hiperparámetros congelados:   n_estimators: 19
                                max_depth: 4
                                learning_rate: 0.10
                                subsample: 0.80
                                colsample_bytree: 0.80
                                min_child_weight: 5
                                tree_method: 'hist'
                                objective: 'reg:squarederror'
                                random_state: 42
- ¿Hubo L2 explícita?:          XGBoost utiliza reg_lambda=1.0 por defecto; no fue modificada ni sintonizada explícitamente.
- ¿Ablación de regularización?: NO.
- ¿Prueba de compresión?:       NO se probó formalmente que la compresión de varianza provenga de la regularización L2 frente al feature set o el ruido.
- Dictamen:                     [x] Sólo interpretación plausible
- Redacción prudente:           “This variance compression may reflect the parsimonious predictor set, model regularization,
                                and unresolved high-frequency variability in the MUR--BIL residual.”
- Evaluación de consistencia:   TOTALMENTE CONSISTENTE con la evidencia del proyecto.
```

---

## 8. Table 1 Reconstruction

Reconstrucción de desempeño histórico del modelo residual a través de las diferentes fases temporales. Los estimadores evaluados en 2021, 2022–2023 y 2024–2025 fueron reajustados con ventanas de entrenamiento progresivamente mayores (2015–2020, 2015–2021, y 2015–2023, respectivamente); por lo tanto, representan una comparación descriptiva y no la trayectoria de un único estimador fijo.

| Evaluation Period | Training Window | Days / Obs | Baseline $B_0$ RMSE (°C) | Model $C_0$ RMSE (°C) | $\Delta\mathrm{RMSE}$ (°C) | RMSE Impr (%) | $B_0$ MAE (°C) | $C_0$ MAE (°C) | MAE Impr (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2021 Diagnostic Holdout** | 2015–2020 | 365 d / 1.93M | 0.359493 | 0.349274 | -0.010219 | **+2.8424%** | 0.277337 | 0.266264 | +3.99% |
| **2022–2023 Validation** | 2015–2021 | 730 d / 3.85M | 0.335666 | 0.323838 | -0.011828 | **+3.5237%** | 0.263594 | 0.254457 | +3.47% |
| **2024–2025 Final Test** | 2015–2023 | 731 d / 3.86M | 0.357317 | 0.331502 | -0.025815 | **+7.2247%** | 0.272713 | 0.256325 | +6.01% |
| *— 2024 (leap year)* | 2015–2023 | 366 d / 1.93M | 0.379710 | 0.344023 | -0.035687 | **+9.3986%** | 0.285148 | 0.264325 | +7.30% |
| *— 2025 (regular year)* | 2015–2023 | 365 d / 1.93M | 0.333355 | 0.318453 | -0.014902 | **+4.4704%** | 0.260243 | 0.248303 | +4.59% |

*Fuentes canónicas:* `paper_table1_final_performance.csv`, `historical_skill_summary.csv`, `ml_phase_summary.csv`.  
*Discrepancias encontradas:* Ninguna. Las métricas coinciden de forma idéntica en todos los reportes finales.

---

## 9. Table 2 / Residual Regime Reconstruction

Reconstrucción canónica de la evaluación por regímenes de residual $|R|$ en el conjunto ciego Final Test 2024–2025 ($N = 3{,}856{,}025$ observaciones).

| Residual Regime | Threshold Range | Observations ($N$) | Fraction of Test (%) | $B_0$ RMSE (°C) | $C_0$ RMSE (°C) | $\Delta\mathrm{RMSE}$ (°C) | Relative Improvement (%) | Sign Accuracy (%) | Overcorrection (%) | Amplitude Ratio $\frac{\mathrm{std}(\hat{R})}{\mathrm{std}(R)}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DEV-P0-P50** | $< 0.2066^\circ\mathrm{C}$ | 1,857,229 | 48.16% | 0.1154 | 0.1396 | +0.0242 | **-20.99%** | 56.4% | 31.08% | 0.73 |
| **DEV-P50-P75** | $0.2066–0.3604^\circ\mathrm{C}$ | 938,991 | 24.35% | 0.2808 | 0.2710 | -0.0098 | **+3.51%** | 65.8% | 1.29% | 0.30 |
| **DEV-P75-P90** | $0.3604–0.5377^\circ\mathrm{C}$ | 604,739 | 15.68% | 0.4400 | 0.4071 | -0.0329 | **+7.48%** | 70.9% | 0.00% | 0.20 |
| **DEV-P90-P95** | $0.5377–0.6652^\circ\mathrm{C}$ | 204,020 | 5.29% | 0.5961 | 0.5412 | -0.0549 | **+9.21%** | 75.7% | 0.00% | 0.18 |
| **DEV-P95-P99** | $0.6652–0.9659^\circ\mathrm{C}$ | 181,676 | 4.71% | 0.7877 | 0.6926 | -0.0951 | **+12.07%** | 82.5% | 0.00% | 0.20 |
| **DEV-P99+** | $\ge 0.9659^\circ\mathrm{C}$ | 69,370 | 1.80% | 1.1469 | 1.0151 | -0.1317 | **+11.49%** | 87.2% | 0.00% | 0.45 |
| **Total Test** | Full Domain | 3,856,025 | 100.00% | 0.3573 | 0.3315 | -0.0258 | **+7.22%** | 63.8% | 15.28% | 0.25 |

### Aclaración de Proporciones y Suma
- Suma exacta: $48.1643\% + 24.3513\% + 15.6830\% + 5.2909\% + 4.7115\% + 1.7990\% = 100.0000\%$.
- La mención en borradores preliminares de 4.67% y 1.85% obedeció a un redondeo inicial no corregido; la proporción canónica evaluada sobre los 3,856,025 registros de TEST es estrictamente **4.71%** para P95–P99 y **1.80%** para P99+.
- **Comportamiento monotónico:** El beneficio absoluto ($|\Delta\mathrm{RMSE}|$) aumenta de forma estrictamente monótona con la discrepancia $|R|$ (+0.0242 degradación en P0–P50; -0.0098 en P50–P75; -0.0329 en P75–P90; -0.0549 en P90–P95; -0.0951 en P95–P99; -0.1317 °C en P99+). La mejora relativa crece desde -20.99% hasta +12.07% (P95–P99).

---

## 10. Bootstrap Confidence Intervals

| Resampling Scheme | Block Length $L$ | Median $\Delta\mathrm{RMSE}$ (°C) | Empirical 95% CI Lower (°C) | Empirical 95% CI Upper (°C) | $P(\Delta\mathrm{RMSE} < 0)$ | Two-Sided Tail Fraction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Complete-Field Cluster** | 1 day | -0.025705 | -0.031932 | -0.019440 | 1.0000 | $1.998 \times 10^{-3}$ |
| **7-Day Moving Block** | 7 days | -0.025563 | -0.038870 | -0.012558 | 1.0000 | $1.998 \times 10^{-3}$ |
| **14-Day Moving Block (Primary)** | 14 days | -0.025395 | **-0.042485** | **-0.009792** | 1.0000 | $1.998 \times 10^{-3}$ |

- **Unidad:** Grados Celsius (°C).
- **Definición:** $\Delta\mathrm{RMSE} = \mathrm{RMSE}_{C0} - \mathrm{RMSE}_{B0}$. Valores negativos denotan reducción del error de reconstrucción respecto a la interpolación bilineal.
- **Intervalo Canónico:** El intervalo $[-0.042485, -0.009792]^\circ\mathrm{C}$ corresponde sin ambigüedad al **bootstrap de bloques móviles de 14 días** evaluado en la Fase D.3.5 sobre el periodo ciego 2024–2025.

---

## 11. Terminology Search

Auditoría de ocurrencias del término "anomalía" a lo largo de los reportes del proyecto:

| Archivo | Frase / Contexto | Línea | Clasificación | Sustitución Recomendada |
|:---|:---|:---:|:---:|:---|
| `faseD32_E3b_tabular.md` | "RMSE por regímenes de anomalía térmica (0-P50 a >=P99)" | L399 | **B (Incorrecta)** | "Within-regime RMSE across residual-magnitude bins (DEV-P0–P50 to DEV-P99+)" |
| `faseD32_E3b_tabular.md` | "se evalúa la respuesta según la escala de la anomalía" | L240 | **B (Incorrecta)** | "evaluated across residual-magnitude regimes" |
| `faseD33_external_validation_C0.md` | "En días con anomalías sub-mesoescala muy débiles..." | L182 | **B (Incorrecta)** | "During days with small MUR–BIL discrepancies..." |
| `faseD34_postvalidation_diagnostics.md` | "anomalía de alta frecuencia" | L66 | **B (Incorrecta)** | "unresolved high-frequency residual structure / discrepancy" |
| `faseD34_postvalidation_diagnostics.md` | "anomalías residuales observadas R de signo persistente opuesto al ciclo climatológico" | L82 | **B (Incorrecta)** | "months characterized by reduced agreement in residual sign" |
| `THESIS_LIMITATIONS_A_D.md` | "No es adecuado para predecir anomalías térmicas extremas instantáneas aisladas" | L36 | **B (Incorrecta)** | "The model underrepresents residual amplitude, limiting its ability to reproduce large instantaneous MUR–BIL discrepancies" |
| `diagnosticar_zero_analysis_error_20160523.py` | "Diagnóstico de la anomalía puntual analysis_error = 0" | L383 | **A (Correcta)** | Mantener (describe un comportamiento anómalo en el flujo de datos del producto satelital) |
| `diagnosticar_extremos_faseC2.py` | "criterios de detección de anomalías (IQR, MAD, Percentiles)" | L188 | **A (Correcta)** | Mantener (detección formal de outliers estadísticos en series de error) |

---

## 12. Bibliography Audit

### Archivos `.bib` en el Espacio de Trabajo
- **Estado verificado:** No existe ningún archivo `.bib` dentro de `/Users/mariajosenande/Documents/Lole` (verificado mediante búsqueda completa de Git y del sistema de archivos).
- **Razón del DOI duplicado en manuscritos compilados:**  
  Al redactar entradas BibTeX para productos satelitales o revistas de acceso abierto, es común declarar tanto el campo `doi = {...}` como `url = {https://doi.org/...}`. En paquetes como `natbib` o estilos como `IEEEtran` y `elsarticle`, esto provoca que el compilador imprima el DOI dos veces consecutivas en la bibliografía.

### Entradas BibTeX Canónicas y Desduplicadas

#### 1. MUR SST Level-4
```bibtex
@misc{JPL_MUR_2015,
  author       = {{JPL MUR MEaSUREs Project}},
  title        = {{GHRSST Level 4 MUR Global Foundation Sea Surface Temperature Analysis (v4.1)}},
  year         = {2015},
  publisher    = {Physical Oceanography Distributed Active Archive Center (PO.DAAC)},
  address      = {Pasadena, CA, USA},
  doi          = {10.5067/GHGMR-4FJ04},
  note         = {Dataset accessed 2026-08-15}
}
```

#### 2. Liu et al. (Remote Sensing, 2026)
Verificado contra los metadatos CrossRef del DOI `10.3390/rs18091346` y el archivo físico `Xiaoyu2026.pdf` en `PAPERS`:
```bibtex
@article{Liu2026,
  author    = {Liu, Xiaoyu and Wang, Xuan and Tong, Yicong and Li, Wei and Han, Guijun},
  title     = {Reconstructing High-Resolution Coastal Water Quality Data Based on a Deep Learning Multivariate Downscaling Approach},
  journal   = {Remote Sensing},
  volume    = {18},
  number    = {9},
  pages     = {1346},
  year      = {2026},
  doi       = {10.3390/rs18091346}
}
```

#### 3. GEBCO 2026 Grid
Verificado contra los metadatos globales del NetCDF `gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc`:
```bibtex
@misc{GEBCO2026,
  author       = {{GEBCO Compilation Group}},
  title        = {{The GEBCO\_2026 Grid---A continuous terrain model for oceans and land at 15 arc-second intervals}},
  year         = {2026},
  publisher    = {British Oceanographic Data Centre (BODC)},
  doi          = {10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa}
}
```

### Candidatas para la Referencia Rota `[?]` en Discussion
Dentro del repositorio local de bibliografía en `/Users/mariajosenande/Documents/PAPERS/`, se identificaron las siguientes referencias directamente pertinentes a downscaling estadístico bajo regímenes extremos, no-estacionariedad o extrapolación climática:

1. **Cyriac et al. (2025):**
   - **Referencia:** Cyriac, A., Sun, C., Taylor, M., et al. (2025). *A Machine Learning Approach to Rapidly Downscale Sea Surface Temperature Extremes and Heat Stress on the Great Barrier Reef*. **Geophysical Research Letters**, 52, e2024GL114521. DOI: `10.1029/2024GL114521`.
   - **Por qué podría corresponder:** Evalúa explícitamente el uso de Machine Learning para downscaling de extremos térmicos superficiales marinos y estrés por calor, abordando la capacidad de los modelos estadísticos para resolver eventos extremos.
2. **Reder et al. (2025):**
   - **Referencia:** Reder, A., Fedele, A., Manco, A., et al. (2025). *Estimating pros and cons of statistical downscaling based on EQM bias adjustment as a complementary method to dynamical downscaling*. **Scientific Reports**, 15, 621. DOI: `10.1038/s41598-024-84527-5`.
   - **Por qué podría corresponder:** Examina detalladamente las limitaciones y ventajas del downscaling estadístico frente al dinámico bajo eventos extremos y no-estacionariedad climática, abordando los límites de extrapolación.
3. **Kalmus et al. (2022):**
   - **Referencia:** Kalmus, P., Ekanayaka, R., Kang, E. L., et al. (2022). *Past the Precipice? Projected Coral Habitability Under Global Heating*. **Earth's Future**, 10, e2021EF002608. DOI: `10.1029/2021EF002608`.
   - **Por qué podría corresponder:** Discute explícitamente el impacto del calentamiento global acelerado, extremos térmicos e incertidumbre en proyecciones climáticas sobre ecosistemas de arrecifes de coral.

---

## 13. Canonical vs Obsolete Results

| Concept | Obsolete / Draft Status | Canonical Final Status | Primary Audit Source |
|:---|:---|:---|:---|
| **GEBCO Model** | GEBCO 2024 (mencionado por errata tipográfica en borradores iniciales). | **GEBCO 2026 Grid** (publicado en abril 2026, 15 arc-sec, NetCDF con hash verificado). | `DOCUMENTATION_GAPS_A_D.md` (GAP-02), `config.py` |
| **Residual Quantiles** | P50≈0.177, P75≈0.354, P90≈0.536, P95≈0.697, P99≈1.054 °C (borrador exploratorio). | **DEV-P50 = 0.2066, DEV-P75 = 0.3604, DEV-P90 = 0.5377, DEV-P95 = 0.6652, DEV-P99 = 0.9659 °C**. | `final_test_freeze_manifest.json`, `GAP-09` |
| **Residual Terminology** | "Thermal anomaly", "anomaly regime", "weak submesoscale anomalies". | **"Residual-magnitude regimes"**, **"large MUR–BIL departures"**, **"small discrepancies"**. | `terminology_audit_ML_figures.md`, `d34_microaudit_changes.csv` |
| **Bootstrap Blocks** | Bloques hipotéticos de 5 o 10 días, o suposición de muestreo no bloqueado. | **1-day cluster**, **7-day**, y **14-day circular moving-block** ($B=1{,}000$). | `fase_d35_final_test_c0.py`, `bootstrap_sensitivity.csv` |
| **Tree Count in C0** | n_estimators = 18 (confusión con `best_iteration = 18` indexado en 0). | **n_estimators = 19** (19 árboles reales persistidos en `E3b-C0_FINALREFIT_2015_2023.json`). | `fase_d32_e3b_tabular.py` (L1566), `manifest.json` |
| **Spatial Feature Claims** | Afirmaciones preliminares de que C0 incluía gradientes espaciales locales o contraste. | C0 contiene **exclusivamente 4 predictores** (`sst_bil`, `doy_sin`, `doy_cos`, `depth`). | `fase_d32_e3b_tabular.py` (L1556), `paper_methods_ML.md` |

---

## 14. Exact Corrections Supported by Evidence

1. **Bootstrap:**
   Escribir: *"Statistical uncertainty was assessed using 1-day complete-field cluster, 7-day, and 14-day moving-block bootstrap with $B=1{,}000$ paired replicates reconstructed from daily sums of squared errors. The 14-day scheme was the primary confirmatory criterion ($95\%\text{ CI} = [-0.0425, -0.0098]^\circ\mathrm{C}$)."*
2. **Residual Regimes:**
   Escribir: *"Observations were stratified into six residual-magnitude regimes based on the absolute MUR–BIL discrepancy ($|R|$), using quantiles predeclared on the Development set ($0.2066, 0.3604, 0.5377, 0.6652, 0.9659^\circ\mathrm{C}$)."*
3. **Bathymetry:**
   Escribir: *"The largest improvements occurred along shallow nearshore areas and across parts of the Cozumel Channel. These spatial patterns are descriptive and do not by themselves establish a causal contribution of bathymetric depth."*
4. **GEBCO Version:**
   Escribir: *"Static water depth was extracted from the GEBCO 2026 Grid at 15 arc-second resolution (DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa)."*
5. **Feature Set:**
   Escribir: *"The model utilized a parsimonious four-predictor feature set comprising bilinearly interpolated SST, harmonic day-of-year terms (sine and cosine), and bathymetric depth."*
6. **Variance Compression:**
   Escribir: *"This variance compression may reflect the parsimonious predictor set, model regularization, and unresolved high-frequency variability in the MUR--BIL residual."*

---

## 15. Items That Cannot Be Verified

1. **Texto del Manuscrito Externo:** No se puede verificar la redacción exacta en el archivo fuente del manuscrito (LaTeX / Word / Overleaf) debido a que no reside dentro del espacio de trabajo del asistente, tal como fue indicado explícitamente en el protocolo de auditoría.
2. **Archivo `.bib` Original del Usuario:** La verificación de si en el archivo `.bib` del autor los campos DOI de MUR y Liu et al. están físicamente duplicados no puede comprobarse directamente sobre su archivo local por estar ausente del repositorio. No obstante, se entregaron las entradas BibTeX corregidas y desduplicadas en la Sección 12.
3. **Identidad Unívoca de la Cita Rota `[?]`:** No es posible determinar con certeza algorítmica matemática cuál de las tres referencias candidatas (`Cyriac2025`, `Reder2025` o `Kalmus2022`) tenía en mente el autor en la sección de discusión, por lo que se listaron las candidatas más consistentes con la literatura disponible en el proyecto.

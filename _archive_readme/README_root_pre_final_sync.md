# Residual Machine Learning for Coastal SST Downscaling in the Mexican Caribbean
### Master's Thesis Project — Tulum–Cozumel, Quintana Roo, Mexico

[![Python](https://img.shields.io/badge/Python-3.11.16-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-E3b--C0%20(XGBoost)-brightgreen.svg)]()
[![D35 Decision](https://img.shields.io/badge/Decision-D35--A%20Confirmed-success.svg)]()
[![Documentation Layer](https://img.shields.io/badge/Documentation%20Layer-AUDITED%20%26%20FROZEN-teal.svg)]()
[![Status](https://img.shields.io/badge/Experimental%20Status-SCIENTIFICALLY%20CLOSED-lightgrey.svg)]()

---

## 1. Project Overview

This repository contains the complete, reproducible computational framework developed for the Master's Thesis on Sea Surface Temperature (SST) statistical downscaling in the **Tulum–Cozumel marine corridor** (Mexican Caribbean). The scientific objective is to downscale daily coarse-resolution SST from **NOAA OISST v2.1** (~0.25°, ~27 km) to a fine-resolution grid (~0.01°, ~1 km) compatible with **MUR SST v4.1** by coupling physical spatial predictors (GEBCO bathymetry), astronomical periodicity (day of year), and supervised decision-tree algorithms.

The operational architecture adopts a **residual-learning formulation**:
1. **Bilinear Baseline ($B_0$):** Coarse OISST is projected onto the fine grid via bilinear interpolation with coastal extension support (Strategy A): $\text{SST}_{\text{BIL}}$.
2. **Residual Target ($R$):** The model predicts the fine-scale thermal discrepancy relative to the high-resolution reference:
   $$R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$$
3. **Reconstructed Field ($C_0$):** High-resolution SST is reconstructed by adding the predicted residual correction:
   $$\hat{\text{SST}} = \text{SST}_{\text{BIL}} + \hat{R}$$

Under this formulation, if $\hat{R} = 0$, then $\hat{\text{SST}} \equiv \text{SST}_{\text{BIL}}$, ensuring that a zero-residual prediction exactly preserves the bilinear baseline. The confirmed operational model is **`E3b-C0`** (compact XGBoost regressor).

> [!NOTE]
> **Operational Role of MUR SST:** In this study, MUR SST v4.1 serves as the **high-resolution operational reference**, not as absolute ground truth. In-situ buoy networks are scarce in this reef corridor, and satellite infrared measurements are frequently occluded by tropical cloud cover.

---

## 2. Study Area

The domain covers the complex coastal and reef ecosystem of the **Tulum–Cozumel sector** in Quintana Roo, Mexico (western Caribbean):
- **Latitude:** $19.90^\circ\text{N}$ to $20.75^\circ\text{N}$
- **Longitude:** $-87.60^\circ\text{W}$ to $-86.65^\circ\text{W}$
- **Grid Dimensions:** 86 latitude rows $\times$ 96 longitude columns = 8,256 total grid cells at 0.01° (~1 km) resolution.
- **Ocean / Land Mask:** 5,279 harmonized marine cells (63.94% of the domain) and 2,977 land cells (36.06%), derived from GEBCO ocean fraction $\ge 0.5$.
- **Geographic Scope:** The domain includes the Mesoamerican Barrier Reef reef lagoons, the Cozumel Island shelf, the Cozumel Channel, and deep oceanic basins (>1000 m). *The Yucatán Channel proper is located north of 21.5°N and is outside the study domain.*

![Pipeline and Study Domain](DATASET_TESIS/THESIS_MASTER_A_D/figures/fig_master_pipeline_A_D.png)

---

## 3. Data Sources

The full study period spans **2015-01-01 to 2025-12-31** (**4,018 continuous astronomical days**, 100.0% temporal completeness, 0 missing dates, 0 duplicates):

| Dataset | Provider / Sensor | Spatial Resolution | Temporal Cadence | Role in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **MUR SST v4.1** | NASA JPL (Multi-sensor L4) | 0.01° (~1 km) | Daily | High-resolution SST reference & target residual |
| **NOAA OISST v2.1** | NOAA NCEI (AVHRR-only L4) | 0.25° (~27 km) | Daily | Coarse-scale SST predictor & bilinear baseline |
| **GEBCO 2026** | IHO / IOC GEBCO Consortium | 15 arc-seconds (~450 m) | Static | Bathymetric depth, coast distance & land/ocean mask |
| **VIIRS S-NPP L2P** | NASA / NOAA (Orbital Radiometer) | 750 m nominal | Swaths | Independent radiometric validation audit |
| **MODIS Aqua L2P** | NASA OB.DAAC (Thermal IR) | 1 km nominal | Swaths | Independent radiometric validation audit |

---

## 4. Methodological Pipeline Summary

The project was executed across four chronological, methodologically sealed stages:

```
[ Stage A: Data Acquisition & Audit ] 
                 │
                 ▼
[ Stage B: Spatial Framework & Master Grid (86x96, 5279 ocean cells) ]
                 │
                 ▼
[ Stage C: Temporal Harmonization (2015–2025, Strategy A, 21.2M obs) ]
                 │
                 ▼
[ Stage D: Machine Learning Modeling (D31 -> D32 -> D33 -> D34 -> D35 -> D36) ]
```

### Stage A — Data Acquisition and Audit
- Daily ingestion of 4,018 NetCDF scenes for MUR SST and OISST v2.1 via ERDDAP / JPL OPeNDAP.
- Extraction of regional GEBCO 2026 Grid sub-dataset.
- Multi-sensor orbital satellite audit (VIIRS L2P v2.80 and MODIS Aqua L2P v2019.0).
- **Status:** **COMPLETED**

### Stage B — Spatial Framework and Mask Harmonization
- Construction of master regular grid (86 $\times$ 96 cells at 0.01°).
- Mask correction based on physical GEBCO shoreline ($\text{ocean\_fraction} \ge 0.5$).
- Calculation of static physiographic covariates: bathymetric depth ($depth$) and euclidean distance to coast ($distance\_coast\_km$).
- Official marine census: **5,279 harmonized ocean cells**.
- **Status:** **COMPLETED**

### Stage C — Spatiotemporal Harmonization (2015–2025)
- **Coastal boundary problem:** Direct bilinear interpolation from coarse 0.25° OISST produced 1,292 coastal NaNs (24.47% domain loss) due to continental grid nodes on the Yucatán Peninsula.
- **Coastal Strategy A:** Auxiliary nearest-ocean extrapolation onto 20 continental halo nodes, enabling 100% complete bilinear interpolation across all 5,279 ocean cells without coastal data loss.
- Construction of the decadal data cube `faseC2_2015_2025.nc` (540.82 MB, **21,211,022 spacetime observations**).
- Independent radiometric audit across extreme discrepancy events (E1–E6).
- Decadal characterization of MUR uncertainty (`analysis_error`).
- **Status:** **COMPLETED**

### Stage D — Machine Learning Modeling & Confirmatory Evaluation
- **D.1:** Analytical tabular dataset construction with strict anti-leakage chronological partitions.
- **D31:** Predictability diagnostics confirming reproducible signal in the fine residual beyond persistence and local climatology baselines.
- **D32:** Model selection and feature ablation on Development Holdout (2021). Selection of parsimonious `E3b-C0` over 5 expanded spatio-temporal variants.
- **D33:** External out-of-sample temporal validation on 2022–2023. Positive generalization confirmed under formal ruling **D33-B — UNCHANGED**.
- **D34:** Post-validation spatial, regime, and calibration micro-audits.
- **D35:** Terminal confirmatory evaluation on previously withheld Final Test (2024–2025). Unanimous confirmation under formal ruling **D35-A — FINAL GENERALIZATION CONFIRMED**.
- **D36:** Paper-ready figure and table consolidation.
- **Status:** **SCIENTIFICALLY CLOSED**

---

## 5. Grid Census: 5,279 vs 5,275 Marine Cells

A key methodological distinction documented in [DOCUMENTATION_GAPS_A_D.md](DATASET_TESIS/THESIS_MASTER_A_D/reports/DOCUMENTATION_GAPS_A_D.md) (GAP-01):
- **Stage B & C Harmonized Cube:** Contains **5,279 ocean cells** (physical ocean census).
- **Stage D Machine-Learning Domain:** Evaluates **5,275 frozen cells** (`frozen_cell_ids.csv`).

**Reason for Difference:** During Stage D32, spatial ablations (`E3b-S`, `E3b-TS`, `E3b-ALL`) required computing 2D horizontal thermal gradients (`grad_mag`). Four isolated boundary cells (indices 0, 161, 4437, 4472) lacked sufficient adjacent ocean neighbors for valid finite differences. To guarantee an identical, strictly fair evaluation mask across all candidate variants, `COMMON_VALID_MASK` excluded those 4 cells, freezing the ML domain at exactly 5,275 cells.

---

## 6. Temporal Evaluation Design

To prevent temporal leakage caused by strong serial autocorrelation in oceanic temperatures, data were split chronologically without random shuffling:

| Period | Dates | Days | ML Rows | Role in Framework | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Development** | 2015-01-01 to 2020-12-31 | 2,192 | 11,562,800 | Model training & hyperparameter tuning | Archived |
| **Holdout** | 2021-01-01 to 2021-12-31 | 365 | 1,925,375 | Diagnostic model comparison & ablation (D32) | Archived |
| **Validation** | 2022-01-01 to 2023-12-31 | 730 | 3,850,750 | Out-of-sample external validation (D33) | Archived |
| **Final Test** | 2024-01-01 to 2025-12-31 | 731 | 3,856,025 | Confirmatory terminal generalization (D35) | **CONSUMED** |

> [!CAUTION]
> **FINAL TEST 2024–2025 IS PERMANENTLY CONSUMED.**
> The 2024–2025 period was held completely blind until the single inference pass in Stage D.3.5. Model development is permanently closed. D35 scripts may only be rerun as a technical post-consumption reproducibility run, never as a new confirmatory test.

---

## 7. Confirmed Model Specification: E3b-C0

`E3b-C0` was selected over five more complex temporal-lag and spatial-neighborhood configurations (`E3b-T1`, `E3b-T3`, `E3b-S`, `E3b-TS`, `E3b-ALL`) based on out-of-sample skill and strict parsimony principles:

- **Algorithm:** Gradient Boosted Trees via `xgboost.XGBRegressor` (`tree_method='hist'`)
- **Input Features ($k = 4$):**
  1. `sst_bil`: Bilinearly interpolated coarse OISST (°C)
  2. `doy_sin`: Sine harmonic of astronomical day of year ($\sin(2\pi \cdot \text{DOY} / 365.25)$)
  3. `doy_cos`: Cosine harmonic of astronomical day of year ($\cos(2\pi \cdot \text{DOY} / 365.25)$)
  4. `depth`: GEBCO 2026 bathymetric depth (meters, negative below sea level)
- **Frozen Hyperparameters ($\theta^*$):**
  `n_estimators=19`, `max_depth=4`, `learning_rate=0.10`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `objective='reg:squarederror'`, `random_state=42`.

---

## 8. Key Results

All numerical values are verified against canonical tables and frozen manifests:

### A. Stage Progression Summary

| Evaluation Stage | Period | Baseline ($B_0$) RMSE | Model ($C_0$) RMSE | Relative Skill ($\Delta\text{RMSE}\%$) | Temporal Robustness | Ruling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diagnostic Holdout (D32)** | 2021 | 0.359493 °C | 0.349274 °C | **+2.8424%** | — | Parsimony Winner |
| **External Validation (D33)** | 2022–2023 | 0.335666 °C | 0.323838 °C | **+3.5237%** | 16 / 24 months (66.7%) | **D33-B — UNCHANGED** |
| **Final Test (D35)** | 2024–2025 | 0.357317 °C | 0.331502 °C | **+7.2247%** | 18 / 24 months (75.0%) | **D35-A — CONFIRMED** |

### B. Final Test 2024–2025 Detailed Metrics (Stage D35)
- **Sample Size:** 3,856,025 out-of-sample predictions (731 days $\times$ 5,275 cells).
- **Error Reduction:** $\Delta\text{RMSE} = -0.025815^\circ\text{C}$ (+7.2247% relative improvement).
- **Mean Absolute Error (MAE):** Reduced from $0.272713^\circ\text{C}$ to $0.256325^\circ\text{C}$ (+6.01%).
- **Mean Bias:** Reduced from $-0.062425^\circ\text{C}$ to $-0.014197^\circ\text{C}$ (77.3% bias reduction).
- **Reconstructed SST $R^2$:** Increased from $0.892661$ ($B_0$) to **$0.907610$** ($C_0$).
- **Annual Breakdown:**
  - **Year 2024:** $B_0 = 0.3797^\circ\text{C} \to C_0 = 0.3440^\circ\text{C}$ (**+9.3986%** skill).
  - **Year 2025:** $B_0 = 0.3334^\circ\text{C} \to C_0 = 0.3185^\circ\text{C}$ (**+4.4704%** skill).
- **Monthly Improvement Allocation (GAP-10 Canonical Truth):**
  - Exactly **18 of 24 months improved** ($\Delta\text{RMSE} < 0$, **75.0%** consistency).
  - **2024:** **9 / 12 months improved** (non-improved: Jan, Apr, Oct).
  - **2025:** **9 / 12 months improved** (non-improved: Sep, Oct, Dec).
- **Daily Robustness:** **484 of 731 days improved** (66.21%).
- **Spatial Coverage:** **5,273 of 5,275 ocean cells improved** (**99.96%** spatial coverage).
- **Block Bootstrap Invariance:** 14-day moving-block bootstrap (1,000 resamples):
  $$CI_{95}(\Delta\text{RMSE}) = [-0.042485, -0.009792]^\circ\text{C}$$
  The interval lies strictly below zero, demonstrating statistical robustness under short-range temporal dependence.

### C. Residual-Regime Dependence
Evaluated using pre-test frozen percentiles of absolute residual discrepancy $|R|$:

| Regime | Cutoff ($|R|$) | % Test Data | Baseline RMSE | Model RMSE | Skill ($\Delta\text{RMSE}\%$) | Sign Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DEV-P0–P50** | $< 0.2066^\circ\text{C}$ | 48.16% | 0.1154 °C | 0.1396 °C | **-20.99%** | 56.4% |
| **DEV-P50–P75** | $0.2066\text{–}0.3604^\circ\text{C}$ | 24.35% | 0.2808 °C | 0.2710 °C | **+3.51%** | 65.8% |
| **DEV-P75–P90** | $0.3604\text{–}0.5377^\circ\text{C}$ | 15.68% | 0.4400 °C | 0.4071 °C | **+7.48%** | 70.9% |
| **DEV-P90–P95** | $0.5377\text{–}0.6652^\circ\text{C}$ | 5.29% | 0.5961 °C | 0.5412 °C | **+9.21%** | 75.7% |
| **DEV-P95–P99** | $0.6652\text{–}0.9659^\circ\text{C}$ | 4.71% | 0.7877 °C | 0.6926 °C | **+12.07%** | 82.5% |
| **DEV-P99+** | $\ge 0.9659^\circ\text{C}$ | 1.80% | 1.1469 °C | 1.0151 °C | **+11.49%** | 87.2% |

- In the **small MUR–BIL discrepancy regime** ($|R| < 0.2066^\circ\text{C}$), the baseline is already accurate and there is less margin for beneficial correction, leading to mild degradation (-20.99%).
- In **larger MUR–BIL discrepancy regimes** ($|R| \ge 0.2066^\circ\text{C}$), sign agreement exceeds 65–87% and the model achieves strong improvements (+3.51% to +12.07%).

---

## 9. Important Interpretation Boundaries

To maintain strict scientific integrity during thesis drafting and review:

| What the Study DOES Demonstrate | What the Study DOES NOT Claim |
| :--- | :--- |
| Reproducible reduction in SST reconstruction error relative to bilinear baseline (+7.22% global RMSE skill). | Reconstruction of submesoscale ocean dynamics or turbulence. |
| Confirmed temporal generalization across an independent, previously withheld 2-year period (2024–2025). | Causal attribution of skill to specific oceanographic mechanisms (e.g. upwelling, fronts, currents). |
| Spatially widespread improvement (99.96% of ocean cells improved; maximum benefit in shallow coastal waters <20 m). | Direct identification or enhanced detection of Marine Heatwaves (MHW) without specialized extreme metrics. |
| Regime-dependent correction behavior (amplitude compression consistent with regularized MSE loss: $\text{std}(\hat{R})/\text{std}(R) \approx 0.25$). | MUR SST acting as absolute ground truth (MUR is an L4 satellite product with its own interpolation uncertainties). |

---

## 10. Note on C.2 Baseline E0 Metrics (GAP-12 Clarification)

When reviewing historical Phase C.2 reports:
- **Pooled Spatiotemporal RMSE decenal:** **$0.342596^\circ\text{C} \approx 0.3426^\circ\text{C}$**  
  $$\text{RMSE}_{\text{pooled}} = \sqrt{\frac{1}{N} \sum_{t=1}^{4018} \sum_{c=1}^{5279} (T_{\text{BIL}} - T_{\text{MUR}})^2}$$
  Annual pooled RMSE ranges from **$0.3034^\circ\text{C}$ (2018)** to **$0.3811^\circ\text{C}$ (2015)**, with 2024 at **$0.3797^\circ\text{C}$**.
- **Mean Daily Spatial RMSE decenal:** **$0.301914^\circ\text{C} \approx 0.3019^\circ\text{C}$**  
  $$\overline{\text{RMSE}}_{\text{daily}} = \frac{1}{4018} \sum_{d=1}^{4018} \text{RMSE}_d$$
  Annual mean daily RMSE ranges from **$0.2724^\circ\text{C}$ (2018)** to **$0.3257^\circ\text{C}$ (2024)**.

By Jensen's inequality, $\text{mean}(\text{RMSE}_d) \le \sqrt{\text{mean}(\text{RMSE}_d^2)}$ ($0.3019^\circ\text{C} \le 0.3426^\circ\text{C}$). These are distinct mathematical aggregations and must not be conflated (see [FINAL_DOCUMENTATION_AUDIT_A_D.md](DATASET_TESIS/THESIS_MASTER_A_D/reports/FINAL_DOCUMENTATION_AUDIT_A_D.md)).

---

## 11. Canonical Documentation Layer: `THESIS_MASTER_A_D`

The directory [`DATASET_TESIS/THESIS_MASTER_A_D/`](DATASET_TESIS/THESIS_MASTER_A_D/) serves as the single source of truth for all thesis writing, figures, and tables. All documents have been audited against frozen experimental manifests:

| Master Document | Path | Scope and Purpose | Canonical Status |
| :--- | :--- | :--- | :---: |
| **Master Synthesis** | [`THESIS_MASTER_SYNTHESIS_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_MASTER_SYNTHESIS_A_D.md) | Complete 29-section narrative synthesis of Stages A–D | **AUDITED & FROZEN** |
| **Methods Master** | [`THESIS_METHODS_MASTER_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_METHODS_MASTER_A_D.md) | Formulations, equations, features, split dates, protocols | **AUDITED & FROZEN** |
| **Results Master** | [`THESIS_RESULTS_MASTER_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_RESULTS_MASTER_A_D.md) | Audited quantitative tables, metric comparisons, bootstrap | **AUDITED & FROZEN** |
| **Discussion Notes** | [`THESIS_DISCUSSION_NOTES_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_DISCUSSION_NOTES_A_D.md) | Category I, II, III claims classification & jury Q&A guide | **AUDITED & FROZEN** |
| **Limitations Master** | [`THESIS_LIMITATIONS_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_LIMITATIONS_A_D.md) | Rigorous inventory of dataset, masking, and model bounds | **AUDITED & FROZEN** |
| **Gaps Inventory** | [`DOCUMENTATION_GAPS_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/DOCUMENTATION_GAPS_A_D.md) | Full audit of 12 documentation conflicts (GAPs 01–12) | **RESOLVED & FROZEN** |
| **Reproducibility Map** | [`REPRODUCIBILITY_MAP_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/REPRODUCIBILITY_MAP_A_D.md) | SHA-256 hashes, environments, seeds, and execution map | **AUDITED & FROZEN** |
| **Thesis Writing Map** | [`THESIS_WRITING_MAP.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/THESIS_WRITING_MAP.md) | Chapter-by-chapter section mapping for thesis drafting | **AUDITED & FROZEN** |
| **Final Audit Report** | [`FINAL_DOCUMENTATION_AUDIT_A_D.md`](DATASET_TESIS/THESIS_MASTER_A_D/reports/FINAL_DOCUMENTATION_AUDIT_A_D.md) | Official 17-section audit closing report | **AUDITED & FROZEN** |

---

## 12. Repository Structure

```text
Downscaling-corredor-Tulum-Cozumel/
├── README.md                                <- Main repository entry point
├── requirements.txt                         <- Exact Python package dependencies
├── config.py                                <- Central geographic and pipeline constants
├── _archive_readme/                         <- Historical README backups
│   └── README_pre_final_update.md
│
├── DATASET_TESIS/
│   ├── outputs/
│   │   ├── faseC2_2015_2025.nc              <- Harmonized decadal NetCDF cube (540.8 MB)
│   │   └── fase_c2/                         <- Yearly harmonized NetCDF files
│   │
│   ├── ml_dataset/                          <- Analytical ML Parquet splits
│   │   ├── train_2015_2020.parquet
│   │   ├── holdout_2021.parquet
│   │   ├── val_2022_2023.parquet
│   │   └── test_2024_2025/                  <- Consumed Final Test (read-only)
│   │
│   ├── ml_results/
│   │   ├── E3b_D32/                         <- Model selection & ablation results
│   │   ├── E3b_D33_external_validation/     <- Validation 2022–2023 results
│   │   ├── E3b_D34_postvalidation/          <- Diagnostic and micro-audit tables
│   │   ├── E3b_D35_final_test/              <- Final test manifest, logs, and metrics
│   │   │   ├── final_test_freeze_manifest.json
│   │   │   ├── frozen_spatial_metadata.csv
│   │   │   └── tables/
│   │   └── E3b_FINAL_SYNTHESIS/             <- Consolidated paper-ready tables & figures
│   │
│   ├── THESIS_MASTER_A_D/                   <- CANONICAL FROZEN DOCUMENTATION LAYER
│   │   ├── reports/                         <- Audited master markdown synthesis reports
│   │   ├── tables/                          <- Verified CSV audit logs and metrics
│   │   │   ├── final_numerical_audit.csv
│   │   │   ├── final_claims_audit.csv
│   │   │   └── final_documentation_corrections.csv
│   │   ├── figures/                         <- Canonical master figures
│   │   └── _archive_pre_final_audit/        <- Immutable pre-audit document backups
│   │
│   └── scripts/                             <- Canonical execution pipeline scripts
│       ├── armonizar_datos_tesis.py         <- Stages A and B
│       ├── fase_c2_armonizacion_2015_2025.py<- Stage C.2 full harmonization
│       ├── fase_d1_construir_dataset_ml.py  <- Stage D.1 dataset construction
│       ├── fase_d32_e3b_tabular.py          <- Stage D32 ablation
│       ├── fase_d33_external_validation_c0.py <- Stage D33 validation
│       └── fase_d35_final_test_c0.py        <- Stage D35 final test runner
```

---

## 13. Computational Environment & Reproducibility

- **Operating System:** macOS 26.6.2 (Darwin Kernel arm64 Apple Silicon)
- **Python Version:** 3.11.16
- **Primary Dependencies:** `xgboost==2.1.4` (or compatible 3.x runtime), `scikit-learn==1.6.1`, `pandas==3.0.5`, `numpy==2.4.6`, `scipy==1.15.2`, `xarray==2026.7.0`, `netcdf4==1.7.2`, `pyarrow==25.0.1`, `matplotlib==3.10.0`.
- **Primary Seed:** `random_state = 42` (fixed across all randomized routines).
- **Core Cryptographic Hashes:**
  - `frozen_cell_ids.csv` (5,275 cells): `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb`
  - `frozen_spatial_metadata.csv`: `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`

### Reproducing Technical Audits
To inspect the master numerical audit tables:
```bash
# View audited numerical metrics (Stages A–D)
head -n 20 DATASET_TESIS/THESIS_MASTER_A_D/tables/final_numerical_audit.csv

# View claims audit and replacement policy
cat DATASET_TESIS/THESIS_MASTER_A_D/tables/final_claims_audit.csv

# View formal documentation gaps resolution (GAP-01 to GAP-12)
cat DATASET_TESIS/THESIS_MASTER_A_D/reports/DOCUMENTATION_GAPS_A_D.md
```

---

## 14. Academic Citation & Thesis Defense

When citing or consulting results from this repository for the master's thesis dissertation or conference publications (e.g. ICITS '27), refer directly to the audited reports in `DATASET_TESIS/THESIS_MASTER_A_D/reports/`.

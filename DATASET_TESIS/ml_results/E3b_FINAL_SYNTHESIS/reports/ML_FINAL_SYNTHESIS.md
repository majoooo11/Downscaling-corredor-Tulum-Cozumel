# Master Machine Learning Synthesis: Phases D.3.1 to D.3.5

## Final Synthesis of Residual SST Downscaling for the Quintana Roo Marine Corridor

---

## 1. Scientific Objective

The primary objective of the Machine Learning experimental track was to determine whether a statistical model trained on coarse-resolution SST and static geophysical information can systematically improve high-resolution SST reconstruction relative to standard bilinear interpolation, and whether such improvement generalizes to temporally withheld evaluation periods under a strictly chronological experimental design.

---

## 2. Residual-Learning Formulation

Rather than predicting absolute SST directly, the framework adopts an additive residual formulation:

$$
R(s,t)
=
\mathrm{SST}_{\mathrm{MUR}}(s,t)
-
\mathrm{SST}_{\mathrm{BIL}}(s,t)
$$

The reconstructed high-resolution SST field is then defined as:

$$
\widehat{\mathrm{SST}}(s,t)
=
\mathrm{SST}_{\mathrm{BIL}}(s,t)
+
\hat{R}(s,t)
$$

where $R$ represents the discrepancy between the MUR high-resolution reference and the bilinearly interpolated OISST baseline.

This formulation preserves the bilinear baseline by construction: when $\hat{R}=0$,

$$
\widehat{\mathrm{SST}}
=
\mathrm{SST}_{\mathrm{BIL}},
$$

which corresponds exactly to baseline $B_0$.

The residual should therefore be interpreted as the **MUR–BIL discrepancy**, which may contain spatially structured and temporally recurring components not captured by bilinear interpolation. It is not assumed a priori to represent a specific physical process.

---

## 3. Datasets and Temporal Partition

The final common evaluation domain comprises **5,275 fixed ocean cells** at $0.01^\circ$ spatial resolution (approximately 1 km), covering the Tulum–Cozumel sector of the Mexican Caribbean.

- **High-resolution reference:** MUR Level-4 SST at $0.01^\circ$.
- **Coarse SST input:** OISST Level-4 daily SST at $0.25^\circ$.
- **Static geophysical predictor:** bathymetric depth derived from GEBCO.

The experimental design used strictly chronological and non-overlapping evaluation windows:

- **2015–2020:** model development, feature exploration, tuning, and ablation analysis.
- **2021:** diagnostic/model-selection holdout.
- **2022–2023:** out-of-development temporal validation.
- **2024–2025:** previously withheld final test.

The 2021, 2022–2023, and 2024–2025 evaluation periods contained:

- **2021:** 365 days, 1,925,375 observations.
- **2022–2023:** 730 days, 3,850,750 observations.
- **2024–2025:** 731 days, 3,856,025 observations.

During D.3.2, comparisons among formulations were conducted on a common-valid subset to ensure fair evaluation of models requiring temporal lags; therefore, development-stage row counts may differ from the nominal $2,192 \times 5,275$ grid count.

---

## 4. D31 Predictability Diagnostics

Phase D.3.1 investigated whether the MUR–BIL residual contained sufficient predictable structure to justify a supervised residual-learning approach.

The main findings were:

1. Simple climatological baselines explained only a very small fraction of residual variability.

2. The four-feature tabular formulation combining bilinear SST, annual phase, and bathymetric depth,

   ```text
   sst_bil
   doy_sin
   doy_cos
   depth
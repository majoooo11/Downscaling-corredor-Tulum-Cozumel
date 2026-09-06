# Master Machine Learning Synthesis: Phases D.3.1 to D.3.5
## Final Synthesis of Residual SST Downscaling for the Quintana Roo Marine Corridor

---

## 1. Scientific Objective
The primary objective of the Machine Learning experimental track was to determine whether a statistical model trained on coarse satellite SST and static geophysical features can systematically improve high-resolution SST downscaling relative to standard spatial interpolation, and whether that improvement robustly generalizes out-of-sample to previously withheld temporal periods.

---

## 2. Residual-Learning Formulation
Rather than predicting absolute SST directly, the framework adopts an additive residual formulation:
$$R(s, t) = \mathrm{SST}_{\mathrm{MUR}}(s, t) - \mathrm{SST}_{\mathrm{BIL}}(s, t)$$
$$\widehat{\mathrm{SST}}(s, t) = \mathrm{SST}_{\mathrm{BIL}}(s, t) + \hat{R}(s, t)$$
This formulation guarantees identity preservation: when $\hat{R} = 0$, the reconstruction defaults identically to standard bilinear interpolation ($B_0$).

---

## 3. Datasets and Temporal Partition
The domain encompasses **5,275 fixed ocean cells** ($0.01^\circ$ resolution, ~1 km) covering the coastal waters of Tulum, Playa del Carmen, Cozumel, and Puerto Morelos.
- **Reference Target:** MUR Level-4 0.01° global foundation SST.
- **Coarse Input:** OISST Level-4 0.25° daily foundation SST.
- **Temporal Windows:**
  - `2015–2020` (2,192 days, 11,562,800 rows): Model development and ablation exploration.
  - `2021` (365 days, 1,925,375 rows): Diagnostic holdout and model selection.
  - `2022–2023` (730 days, 3,850,750 rows): Out-of-development external validation.
  - `2024–2025` (731 days, 3,856,025 rows): Withheld final test.

---

## 4. D31 Predictability Diagnostics
Phase D.3.1 established that:
1. Rigid climatological baselines ($B_3$) explain less than 0.25% of residual variance.
2. The combination of annual phase (`doy_sin`, `doy_cos`) and bathymetric depth (`depth`) captures significant deterministic structure (+4.94% gain over $B_0$).
3. Additional static descriptors (distance to coast, ocean fraction) provided no incremental gain once depth was included.

---

## 5. D32 Model Selection
Phase D.3.2 evaluated eight prespecified tabular configurations on the 2021 holdout:
- Core 4-feature model `E3b-C0` (`sst_bil`, `doy_sin`, `doy_cos`, `depth`) achieved **RMSE = 0.349274 °C** (+2.8424% improvement vs $B_0 = 0.359493^\circ	ext{C}$).
- Complex variants incorporating multi-day temporal lags (`E3b-T1`, `E3b-T3`) and spatial gradient/contrast kernels (`E3b-S`, `E3b-ALL`) yielded inferior improvements (+2.21% to +2.63%).
- Consequently, `E3b-C0` was selected as the optimal, parsimonious architecture, and development was declared **METHODOLOGICALLY CLOSED**.

---

## 6. D33 External Validation (2022–2023)
Phase D.3.3 tested frozen `E3b-C0` weights on the two-year external period 2022–2023:
- Global RMSE improved from 0.335666 °C to 0.323838 °C (**+3.5237% gain**).
- However, monthly consistency was 16 / 24 months (66.7%), failing the strict predeclared D33-A threshold ($\ge 18/24$).
- Dictamen: **`D33-B — PARTIAL / MIXED GENERALIZATION`**.

---

## 7. D34 Post-Validation Diagnostic Audit
Phase D.3.4 conducted an exhaustive diagnostic audit of D.3.3:
- Verified zero code bugs or data contamination.
- Demonstrated that moving-block bootstrap (7d and 14d) confirmed statistical significance.
- Identified the low-discrepancy trade-off ($|R| < 0.20^\circ	ext{C}$) as the primary driver of negative months.
- Unanimously issued recommendation: **`PREPARE FINAL TEST`**.

---

## 8. D35 Final Refit (2015–2023)
Incorporating all pre-test data, `E3b-C0` was refitted on 9 continuous years (3,287 dates, 17,338,925 observations) under the frozen specification (19 trees, `max_depth=4`, `learning_rate=0.10`). Weights were serialized to `E3b-C0_FINALREFIT_2015_2023.json` (`SHA256: 395638980b...`).

---

## 9. D35 Final Test (2024–2025)
The refit model was evaluated in a single irreversible pass on 2024–2025:
- **`TEST CONSUMED = YES`**.
- Exact censo: 731 days, 3,856,025 observations, 0 duplicates, 0 NaNs.

---

## 10. Final-Test Performance
- **Baseline $B_0$ RMSE:** 0.357317 °C
- **Model $C_0$ RMSE:** **0.331502 °C**
- **Relative Improvement:** **+7.2247%** ($\Delta = -0.025815^\circ	ext{C}$)
- **MAE:** 0.272713 °C $	o$ 0.256325 °C (+6.0092%)
- **Bias:** -0.062425 °C $	o$ -0.014197 °C
- **$R^2$ SST:** 0.892661 $	o$ 0.907610

---

## 11. Temporal Robustness
- **Year 2024:** +9.3986% improvement ($B_0: 0.3797^\circ	ext{C} 	o C_0: 0.3440^\circ	ext{C}$).
- **Year 2025:** +4.4704% improvement ($B_0: 0.3334^\circ	ext{C} 	o C_0: 0.3185^\circ	ext{C}$).
- **Monthly:** 18 / 24 months improved (75.0%).
- **Daily:** 484 / 731 days improved (66.21%).
- **14-day Moving Block Bootstrap:** 95% CI = **[-0.042485, -0.009792] °C** ($P(\Delta < 0) = 1.000$).

---

## 12. Spatial Robustness
- **Cells improved:** **5,273 / 5,275 cells (99.96%)**.
- **Median spatial $\Delta\mathrm{RMSE}$:** -0.0223 °C.
- Bathymetric correlation: $ho = +0.7376$ ($p < 10^{-15}$). Benefits concentrate strongly in coastal lagoons and shallow shelves (0–20 m: -0.0383 °C mean reduction).

---

## 13. Residual Explanatory Skill
- $R^2_{\mathrm{residual}} = 0.112175$.
- Pearson $r = 0.3512$; Spearman $ho = 0.3671$.
- Amplitude compression ratio $\mathrm{std}(\hat{R})/\mathrm{std}(R) = 0.2536$.
- Calibration slope $b = 0.0891$.

---

## 14. Sign Predictability
- Sign accuracy: **63.79%** (vs majority baseline 52.45%, a +11.34 pp gain).
- Balanced sign accuracy: **63.27%**.

---

## 15. Residual-Regime Dependence
- `DEV-P0-P50` ($|R| < 0.21^\circ	ext{C}$): -20.99% (low-discrepancy shrinkage penalty).
- `DEV-P50-P75` ($0.21–0.36^\circ	ext{C}$): +3.51%.
- `DEV-P75-P90` ($0.36–0.54^\circ	ext{C}$): +7.48%.
- `DEV-P90-P95` ($0.54–0.67^\circ	ext{C}$): +9.21%.
- `DEV-P95-P99` ($0.67–0.97^\circ	ext{C}$): +12.07%.
- `DEV-P99+` ($\ge 0.97^\circ	ext{C}$): +11.49%.

---

## 16. Scientific Interpretation
The residual tabular approach functions as a structured spatial-climatological bias corrector. It reliably improves downscaled fields by suppressing coarse coastal errors where depth gradients and seasonal forcing create persistent offsets, while conservatively shrinking predictions toward the conditional mean through amplitude compression.

---

## 17. Limitations
1. Ineffective for near-zero residuals ($|R| < 0.21^\circ	ext{C}$).
2. Unexplained variance remains high ($88.8\%$ residual variance remains unmodeled).
3. Residual amplitude is compressed by ~75%.
4. No explicit atmospheric or hydrodynamic terms are modeled.
5. MUR is treated as an operational benchmark, not absolute truth.

---

## 18. Implications for ICITS’27
Provides a rigorous, publication-ready story: an autonomous, parsimonious model that demonstrates verifiable out-of-sample generalization (+7.22% RMSE) without overclaiming physical causality or hydrodynamic resolution.

---

## 19. Implications for the Master's Thesis
Closes the machine learning core of the thesis with an unassailable methodological trajectory: Prespecified development (D32) $	o$ Frozen validation (D33) $	o$ Diagnostic audit (D34) $	o$ Blind confirmatory test (D35) $	o$ Master synthesis (D36).

---

## 20. Final Methodological Status
```
============================================================
FINAL ML SYNTHESIS STATUS:
D31: CLOSED
D32: METHODOLOGICALLY CLOSED
D33: D33-B — UNCHANGED
D34: INTERPRETATIONALLY CLOSED
D35: D35-A — FINAL GENERALIZATION CONFIRMED
FINAL TEST: CONSUMED
FINAL MODEL DEVELOPMENT: CLOSED
ML BLOCK STATUS: SCIENTIFICALLY CLOSED
============================================================
```

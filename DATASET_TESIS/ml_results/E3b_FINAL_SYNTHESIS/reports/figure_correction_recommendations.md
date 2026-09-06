# Figure Correction Recommendations: ICITS'27 Pre-Publication Package

**Artifact ID:** `figure_correction_recommendations.md`  
**Target Publication:** ICITS'27 Conference Paper / Master's Thesis  
**Date:** 2026-09-05  
**Status:** DRAFT AUDIT REPORT — WAITING FOR HUMAN APPROVAL (FIRST PASS)  
**Safety Constraints:** Models retrained = 0 | Predictions modified = 0 | Final-test metrics modified = 0 | Figures regenerated = 0

---

## 1. Audit Summary & Overall Verdict

A comprehensive numerical provenance, terminology, and visual coherence audit was performed across Figures 1–5 of the Machine Learning downscaling synthesis (`E3b-C0`), referencing canonical tables (`paper_table1_final_performance.csv`, `paper_table2_robustness.csv`, `paper_table3_residual_regimes.csv`, `historical_skill_summary.csv`, `spatial_depth_diagnostics.csv`).

| Figure | Artifact File | Audit Status | Key Finding / Issue Identified | Recommended Action |
|:---:|:---|:---:|:---|:---|
| **Fig 1** | `fig1_final_performance.png` | **PASS** | Exact numerical match for RMSE/MAE across Combined, 2024, 2025. Predeclared +1.00% threshold correctly rendered. | Maintain data. Keep current layout. |
| **Fig 2** | `fig2_temporal_robustness.png` | **PASS** | 18/24 months, 484/731 days (66.21%), daily median $\Delta\text{RMSE} = -0.0108^\circ\text{C}$, 14-day moving block bootstrap $[-0.0425, -0.0098]^\circ\text{C}$ strictly below 0. | Reiterate 14-day block as primary confirmatory test. Maintain explicit caveat of non-uniform daily skill. |
| **Fig 3** | `fig3_spatial_skill.png` | **PASS (Numeric)**<br>`REVIEW (Title)` | 5,273/5,275 cells improved (99.96%), median $\Delta\text{RMSE} = -0.0228^\circ\text{C}$. All 5 depth strata match table counts. | **Change Title:** Replace `"Bathymetric Dependence"` with descriptive `"Skill Across Bathymetric Strata"` to eliminate implicit causality. |
| **Fig 4** | `fig4_residual_regimes.png` | **PASS (Numeric)**<br>`REVIEW (Title/Caption)` | Regimes P0–P50 through P99+ match table3. Sign accuracy matches baseline (52.45%). Panel C provenance mathematically verified. | **Change Title:** Replace `"Trade-Off Between Noise and Structure"` with `"Residual-Magnitude Dependence and Prediction Behavior"`. Clarify within-regime vs global ratio in caption. |
| **Fig 5** | `fig5_historical_skill.png` | **PASS (Provenance)** | 2021 (+2.84%), 2022–2023 (+3.52%), 2024–2025 (+7.22%) match historical summary. Distinct fitted estimators correctly noted. | **Retain in Supplement/Thesis only.** Do not promote to ICITS core paper. Maintain estimator caveat. |

---

## 2. In-Depth Provenance Verification: Figure 4C

### 2.1 The Apparent Paradox
A potential question arises when comparing the documented global TEST standard deviation ratio against the values plotted in Figure 4C:
- **Global TEST Ratio:**
  $$\text{std}(\hat{R}) = 0.0892^\circ\text{C},\quad \text{std}(R) = 0.3518^\circ\text{C} \implies \frac{\text{std}(\hat{R})}{\text{std}(R)} = 0.2536$$
- **Figure 4C Bar Heights:**
  - `DEV-P0-P50`: $\approx 0.73$
  - `DEV-P50-P75`: $\approx 0.30$
  - `DEV-P75-P90`: $\approx 0.20$
  - `DEV-P90-P95`: $\approx 0.18$
  - `DEV-P95-P99`: $\approx 0.20$
  - `DEV-P99+`: $\approx 0.45$

### 2.2 Mathematical Verification & Source Code Proof
Audit of `DATASET_TESIS/fase_d35_final_test_c0.py` (lines 861–863) and `paper_table3_residual_regimes.csv` confirms:
1. **Numerator:** $\text{std}(\hat{R} \mid \text{regime})$ (sample standard deviation of model predictions evaluated strictly within the subset of observations falling into that residual bin).
2. **Denominator:** $\text{std}(R \mid \text{regime})$ (sample standard deviation of true MUR–BIL residuals evaluated strictly within that same subset).
3. **Population:** The 3,856,025 valid Final Test observations (2024–2025) partitioned by $|R_{\text{MUR-BIL}}|$ into bins predeclared from Development percentiles ($P_{50}=0.2066, P_{75}=0.3604, P_{90}=0.5377, P_{95}=0.6652, P_{99}=0.9659^\circ\text{C}$).

**Why Within-Regime Ratios Differ from Global Ratio:**
Within-regime variance conditioning naturally differs from the global variance:
- In the narrow lower slice (`DEV-P0-P50`, where $|R| \le 0.2066^\circ\text{C}$), the reference residual $R$ is constrained within a tight band, so its conditional standard deviation $\text{std}(R \mid \text{P0-P50})$ is small ($\sim 0.055^\circ\text{C}$). The model predictions $\hat{R}$ have a conditional standard deviation of $\sim 0.040^\circ\text{C}$, yielding a ratio of $0.040 / 0.055 \approx 0.7342$.
- In the intermediate and large discrepancy regimes (`DEV-P75-P90`, `DEV-P90-P95`), $R$ has a very broad conditional spread, while the regularized tree ensemble shrinks predictions toward the conditional mean, resulting in low ratios ($\sim 0.18\text{--}0.20$).
- In the extreme upper tail (`DEV-P99+`, $|R| > 0.9659^\circ\text{C}$), larger predictions are produced, raising the ratio back to $0.4527$.

**Audit Verdict:**
$$\text{FIGURE 4C NUMERIC AUDIT} = \mathbf{PASS}$$
The values are mathematically exact, reproducible, and internally consistent.

---

## 3. Detailed Figure Recommendations

### Figure 1: Out-of-Sample Performance on Withheld FINAL TEST (2024–2025)
- **Status:** PASS. No numerical changes required.
- **Visuals:** Bar labels accurately show 3 decimal places for RMSE/MAE and 2 decimal places for percentage improvements (+7.22%, +9.40%, +4.47%).
- **Criterium:** The horizontal red dashed line at $+1.00\%$ accurately represents the predeclared success threshold established prior to Final Test unblinding.

### Figure 2: Temporal Robustness Across Monthly, Daily, and Resampled Scales
- **Status:** PASS. No numerical changes required.
- **Narrative Clarification:** Panel C must clearly be identified as the confirmatory statistical defense against serial correlation. The 14-day moving block bootstrap 95% CI ($[-0.0425, -0.0098]^\circ\text{C}$) excludes zero, confirming aggregate robustness despite daily non-uniformity (66.21% days improved, 18/24 months improved).
- **Caption Note:** Ensure the caption clearly defines $\Delta\text{RMSE} = \text{RMSE}_{C_0} - \text{RMSE}_{B_0}$ (negative indicates improvement).

### Figure 3: Spatial Consistency and Skill Across Bathymetric Strata
- **Status:** PASS (Numeric) / ACTION REQUIRED (Title).
- **Proposed Title Modification:**
  - *Current:* `"FIGURE 3: Spatial Consistency and Bathymetric Dependence Across 5,275 Cells"`
  - *Recommended:* `"FIGURE 3: Spatial Consistency and Skill Across Bathymetric Strata Across 5,275 Cells"` (or `"Spatial Consistency and Skill Across Bathymetric Strata"`).
- **Rationale:** Replacing "Dependence" with "Skill Across Bathymetric Strata" avoids any reader misinterpretation that shallow bathymetry causally triggers model improvement.
- **Caption Text:** Reiterate that Spearman correlations ($\rho_{\text{depth}} = +0.7376$, $\rho_{\text{dist}} = +0.5336$) are descriptive characterizations of spatial distribution, reflecting higher baseline errors near complex coastal topography rather than direct physical causation.

### Figure 4: Residual-Magnitude Dependence and Prediction Behavior
- **Status:** PASS (Numeric) / ACTION REQUIRED (Title & Caption).
- **Proposed Title Modification:**
  - *Current:* `"FIGURE 4: Residual Regime Dependence and Trade-Off Between Noise and Structure"`
  - *Recommended:* `"FIGURE 4: Residual-Magnitude Dependence and Prediction Behavior"`
- **Rationale:** The analysis does not prove that residuals in $|R| < 0.2066^\circ\text{C}$ constitute pure physical or observational noise. They represent discrepancies that fall within nominal sensor uncertainty and grid interpolation tolerances.
- **Panel A Labeling:** Maintain `"DEV-P0-P50 ($|R| < 0.21^\circ\text{C}$)"` or `"Low-Discrepancy Regime"`. Never use `"Noise Regime"`.
- **Panel B Guidance:** In the caption, clarify that Sign Accuracy ($\text{accuracy}(\text{sign}(\hat{R}) == \text{sign}(R))$) evaluates directional concordance against the majority-sign baseline (52.45%), not calibrated posterior probabilities.
- **Panel C Caption Formulation (MANDATORY):**
  > *"Panel C shows within-regime amplitude ratios, $\text{std}(\hat{R} \mid \text{regime}) / \text{std}(R \mid \text{regime})$, illustrating conditional shrinkage behavior across residual bins; the corresponding global TEST ratio across all pooled observations was 0.2536."*

### Figure 5: Descriptive Historical Skill Across Experimental Phases
- **Status:** PASS (Provenance) / ALLOCATION CONFIRMED.
- **Publication Target:** **Supplement / Master's Thesis only.**
- **Rationale:** Figure 5 documents three different models fitted on increasing training sets (2015–2020: +2.84%; 2015–2021: +3.52%; 2015–2023: +7.22%). It does not represent a single model's trajectory over time and cannot be used to prove that larger training samples caused the gain. It is valuable historical context for the thesis, but would invite methodological confusion if included in the 4-page ICITS conference paper.
- **Mandatory Caveat:** The existing banner note must remain: *"Different fitted estimators and progressively larger training windows were used. This comparison is strictly descriptive and does not represent a same-estimator generalization trajectory."*

---

## 4. Proposed Figure Package Allocation for ICITS'27

```
ICITS'27 CONFERENCE PAPER (CORE 4-FIGURE SUITE):
├── Figure 1: Out-of-Sample Performance on Withheld FINAL TEST (2024–2025)
│   └── Answers: Does the final tuned model beat the baseline on withheld test data?
├── Figure 2: Temporal Robustness Across Monthly, Daily, and Resampled Scales
│   └── Answers: Is the aggregate benefit robust to serial correlation and seasonality?
├── Figure 3: Spatial Consistency and Skill Across Bathymetric Strata
│   └── Answers: Is the skill widespread across all cells, or confined to specific zones?
└── Figure 4: Residual-Magnitude Dependence and Prediction Behavior
    └── Answers: What is the mechanistic explanation for when the model helps or hurts?

THESIS / EXTENDED SUPPLEMENT:
└── Figure 5: Descriptive Historical Skill Across Experimental Phases
    └── Function: Historical context documenting progression across diagnostic phases.
```

---

## 5. Execution Plan (Post-Approval Only)

Upon human approval of this audit:
1. Update `DATASET_TESIS/fase_d36_final_synthesis.py` to apply the recommended title changes for Figure 3 and Figure 4.
2. Regenerate the high-resolution PNG figures (`fig1`–`fig4` in main paper folder, `fig5` in supplement folder).
3. Synchronize captions in `paper_results_ML.md` and `paper_discussion_ML.md` with the verified wording.
4. Commit updated figures with complete provenance traceability.

# Figure Editorial Corrections & Publication Changes Log

**Artifact ID:** `figure_publication_changes.md`  
**Target Publication:** ICITS'27 Manuscript & Master's Thesis  
**Date:** 2026-09-05  
**Audit Protocol:** Evidence-First Pre-Publication Audit & Final Regeneration  
**Integrity Checks:** Numeric values changed = 0 | Models retrained = 0 | Predictions modified = 0

---

## 1. Summary of Editorial Adjustments

In accordance with user instructions and the approved pre-publication audit, editorial, typographic, and epistemic adjustments were executed across the figure generation script (`fase_d36_final_synthesis.py`), the selection guide (`FIGURE_SELECTION_GUIDE.md`), and synthesis documentation. All underlying datasets, test metrics, and predictions remained strictly frozen.

---

## 2. Granular Change Log

| File | Old Text | New Text | Reason |
|:---|:---|:---|:---|
| `fase_d36_final_synthesis.py` (L787) | `fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)` | `fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)` | Harmonize horizontal aspect ratio across Figures 1, 2, 3, and 4 for visual consistency in publication. |
| `fase_d36_final_synthesis.py` (L852) | `plt.savefig(fig1_path, dpi=300, bbox_inches="tight")` | Adds `fig1_final_performance_PUBLICATION.png` and vector `fig1_final_performance_PUBLICATION.pdf` exports | Provide uncompressed high-resolution raster and direct vector PDF assets for camera-ready submission. |
| `fase_d36_final_synthesis.py` (L910) | `plt.savefig(fig2_path, dpi=300, bbox_inches="tight")` | Adds `fig2_temporal_robustness_PUBLICATION.png` and vector `fig2_temporal_robustness_PUBLICATION.pdf` exports | Provide uncompressed high-resolution raster and direct vector PDF assets for camera-ready submission. |
| `fase_d36_final_synthesis.py` (L963) | `"FIGURE 3: Spatial Consistency and Bathymetric Dependence Across 5,275 Cells"` | `"FIGURE 3: Spatial Consistency and Skill Across Bathymetric Strata (5,275 Cells)"` | Eliminate causal connotation implied by the word "Dependence"; strictly describe skill across depth strata. |
| `fase_d36_final_synthesis.py` (L965) | `plt.savefig(fig3_path, dpi=300, bbox_inches="tight")` | Adds `fig3_spatial_skill_PUBLICATION.png` and vector `fig3_spatial_skill_PUBLICATION.pdf` exports | Provide uncompressed high-resolution raster and direct vector PDF assets for camera-ready submission. |
| `fase_d36_final_synthesis.py` (L1015) | `axes[2].set_title("Panel C: Amplitude Compression Ratio")` | `axes[2].set_title("Panel C: Within-Regime Amplitude Ratio")` | Clarify unequivocally that bar ratios are calculated within each residual regime, not across pooled data. |
| `fase_d36_final_synthesis.py` (L1021) | *(None - unannotated)* | `axes[2].text(0.5, 0.05, "Global TEST ratio across all obs = 0.2536", ...)` | Document the distinct global TEST standard deviation ratio directly on the figure to prevent confusion. |
| `fase_d36_final_synthesis.py` (L1023) | `"FIGURE 4: Residual Regime Dependence and Trade-Off Between Noise and Structure"` | `"FIGURE 4: Residual-Magnitude Dependence and Prediction Behavior"` | Eliminate unsupported claim that low-discrepancy observations ($|R| < 0.21^\circ\text{C}$) are proven "noise". |
| `fase_d36_final_synthesis.py` (L1025) | `plt.savefig(fig4_path, dpi=300, bbox_inches="tight")` | Adds `fig4_residual_regimes_PUBLICATION.png` and vector `fig4_residual_regimes_PUBLICATION.pdf` exports | Provide uncompressed high-resolution raster and direct vector PDF assets for camera-ready submission. |
| `fase_d36_final_synthesis.py` (L1061) | `plt.savefig(fig5_path, dpi=300, bbox_inches="tight")` | Adds `fig5_historical_skill_SUPPLEMENT.png` and vector `fig5_historical_skill_SUPPLEMENT.pdf` exports | Formally demarcate Figure 5 as a Supplement / Thesis artifact rather than main conference paper core. |
| `fase_d36_final_synthesis.py` (L1084) | `\| **Figure 3** \| Spatial Skill & Bathymetry \|` | `\| **Figure 3** \| Spatial Consistency & Skill Across Bathymetric Strata (5,275 Cells) \|` | Align selection guide table with approved Figure 3 banner. |
| `fase_d36_final_synthesis.py` (L1085) | `\| **Figure 4** \| Residual Regime Behavior \|` | `\| **Figure 4** \| Residual-Magnitude Dependence and Prediction Behavior \|` | Align selection guide table with approved Figure 4 banner. |
| `fase_d36_final_synthesis.py` (L1097) | `Mechanistic regime breakdown showing why the model helps large errors but degrades tiny noise.` | `Mechanistic regime breakdown showing why the model provides gains during large baseline discrepancies while regularizing in the low-discrepancy regime.` | Remove "noise" overclaim in narrative guide; replace with precise statistical shrinkage description. |
| `fase_d36_final_synthesis.py` (L1404-1492) | Direct overwriting of `paper_methods_ML.md`, `paper_results_ML.md`, `paper_discussion_ML.md` | Guard with existence check: `if not (...).exists(): write(...) else: print("Conservado reporte curado: ...")` | Protect comprehensive, audited paper-ready text files from being overwritten by older short drafts. |
| `fase_d36_final_synthesis.py` (L1561) | `Identified the low-residual noise trade-off` | `Identified the low-discrepancy trade-off` | Remove informal "noise" terminology from synthesis master summary. |
| `fase_d36_final_synthesis.py` (L1619) | `- DEV-P0-P50 ($|R| < 0.21^\circ\text{C}$): -20.99% (noise degradation)` | `- DEV-P0-P50 ($|R| < 0.21^\circ\text{C}$): -20.99% (low-discrepancy shrinkage penalty)` | Replace "noise degradation" with precise characterization of conditional regularization penalty. |
| `fase_d36_final_synthesis.py` (L1629) | `while conservatively damping noise through amplitude compression` | `while conservatively shrinking predictions toward the conditional mean through amplitude compression` | Ground explanation in mathematical regression toward conditional mean under regularized squared error. |
| `fase_d36_final_synthesis.py` (L1635) | `88.8% residual variance is stochastic` | `88.8% residual variance remains unmodeled` | Accurately reflect epistemic boundaries: unmodeled variance cannot be conclusively proven purely stochastic. |
| `FIGURE_SELECTION_GUIDE.md` | Entire file | Regenerated with approved titles, exact publication filenames (PNG/PDF), and neutral descriptions | Complete synchronization between repository documentation and generated visual assets. |

---

## 3. Numeric Invariance Confirmation

Post-regeneration verification against `figure_metric_audit.csv` and underlying canonical tables confirmed:
- Figure 1 Combined RMSE Improvement: **+7.22%** (exact: 7.2247%) $\to$ **PASS**
- Figure 2 Temporal Consistency: **18/24 months (75.0%)**, **484/731 days (66.21%)** $\to$ **PASS**
- Figure 3 Spatial Consistency: **5,273/5,275 cells (99.96%)** $\to$ **PASS**
- Figure 4 DEV-P0-P50 Improvement: **-20.99%** $\to$ **PASS**
- Figure 4 DEV-P95-P99 Improvement: **+12.07%** $\to$ **PASS**
- Figure 4C DEV-P0-P50 Within-Regime Ratio: **0.73** (exact: 0.7342) $\to$ **PASS**
- Figure 4C DEV-P99+ Within-Regime Ratio: **0.45** (exact: 0.4527) $\to$ **PASS**
- Global TEST Amplitude Ratio: **0.2536** $\to$ **PASS**
- Figure 5 Final TEST Improvement: **+7.22%** (exact: 7.22466%) $\to$ **PASS**

$$\mathbf{NUMERIC\ VALUES\ CHANGED} = \mathbf{0}$$
$$\mathbf{MODELS\ RETRAINED} = \mathbf{0}$$
$$\mathbf{PREDICTIONS\ MODIFIED} = \mathbf{0}$$

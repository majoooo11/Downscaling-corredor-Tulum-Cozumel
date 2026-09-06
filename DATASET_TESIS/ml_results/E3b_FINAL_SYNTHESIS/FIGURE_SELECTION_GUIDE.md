# Figure Selection Guide — Machine Learning Block (E3b-C0)

This guide establishes the assignment and placement of figures across the thesis, the ICITS'27 conference manuscript, and supplementary materials, following the strict 4–5 figure limit for standard conference papers.

---

## 1. Summary Classification

| Figure ID | Short Name | Panels | Recommended Venue | Primary Purpose |
|---|---|:---:|---|---|
| **Figure 1** | Final Test Global Performance | 3 (RMSE, MAE, % Impr) | **PAPER MAIN** / THESIS | Documents primary confirmatory results on withheld 2024–2025 test. |
| **Figure 2** | Temporal Robustness | 3 (Monthly, Daily, Bootstrap) | **PAPER MAIN** / THESIS | Visualizes day-to-day variability and 14-day moving block robustness. |
| **Figure 3** | Spatial Consistency & Skill Across Bathymetric Strata (5,275 Cells) | 3 (Map, Density, Depth Bins) | **PAPER MAIN** / THESIS | Confirms 99.96% spatial generalization and depth-dependent skill across strata. |
| **Figure 4** | Residual-Magnitude Dependence and Prediction Behavior | 3 (Impr, Sign Acc, Shrinkage) | **PAPER MAIN** / THESIS | Explains the mechanistic trade-off across $|R|$ discrepancy regimes. |
| **Figure 5** | Descriptive Historical Skill | 1 (Bar Comparison) | **PAPER SUPPLEMENT** / THESIS | Contextualizes progression from D32 to D35 with explicit caveats. |

---

## 2. Venue-Specific Packages

### Package A: ICITS'27 Conference Paper (Compact 4-Figure Core)
- **Main Text Figures:**
  1. `Figure 1`: Core confirmatory outcome (2024, 2025, Combined RMSE and MAE).
     - Files: `fig1_final_performance_PUBLICATION.png` / `fig1_final_performance_PUBLICATION.pdf`
  2. `Figure 2`: Temporal generalization (Monthly bar chart, daily rolling series, moving-block bootstrap).
     - Files: `fig2_temporal_robustness_PUBLICATION.png` / `fig2_temporal_robustness_PUBLICATION.pdf`
  3. `Figure 3`: Spatial consistency and skill across bathymetric strata (5,275 cells).
     - Files: `fig3_spatial_skill_PUBLICATION.png` / `fig3_spatial_skill_PUBLICATION.pdf`
  4. `Figure 4`: Residual-magnitude dependence and prediction behavior (gain in large baseline discrepancies, shrinkage in low-discrepancy regime).
     - Files: `fig4_residual_regimes_PUBLICATION.png` / `fig4_residual_regimes_PUBLICATION.pdf`
- **Supplementary / Appendix:**
  - `Figure 5`: Descriptive historical comparison across experimental phases (distinct fitted estimators).
    - Files: `fig5_historical_skill_SUPPLEMENT.png` / `fig5_historical_skill_SUPPLEMENT.pdf`
  - Phase D31 diagnostic feature importance & ablation charts.
  - Phase D34 detailed negative-month diagnostic time series.

### Package B: Master's Thesis (Full Comprehensive Package)
- **Chapter on Machine Learning Results:**
  - Embed Figures 1, 2, 3, 4, and 5 directly within the main narrative.
- **Thesis Appendix:**
  - D.3.1 Predictability & Climatological Baseline figures.
  - D.3.2 8-model ablation comparisons.
  - D.3.3 External validation monthly breakdown.
  - D.3.4 Post-validation diagnostic audit and low-residual decomposition.

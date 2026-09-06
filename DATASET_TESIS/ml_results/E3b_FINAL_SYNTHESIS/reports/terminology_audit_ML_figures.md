# Terminology and Epistemic Audit: Machine Learning Figures and Narrative

**Artifact ID:** `terminology_audit_ML_figures.md`  
**Target Publication:** ICITS'27 Conference / Master's Thesis  
**Date:** 2026-09-05  
**Scope:** Figures 1–5, Figure Titles, Panel Labels, Captions, Synthesis Reports (`ML_FINAL_SYNTHESIS.md`, `paper_methods_ML.md`, `paper_results_ML.md`, `paper_discussion_ML.md`, `FIGURE_SELECTION_GUIDE.md`), and Generation Scripts (`fase_d36_final_synthesis.py`).

---

## 1. Executive Summary & Epistemic Framework

In accordance with strict methodological rigor for the pre-publication submission to ICITS'27, a systematic audit of sensitive terms was conducted. The goal is to enforce conservative, descriptive, and strictly defensible interpretations of model behavior. Specifically:

1. **No Unproven "Noise" Attributions:** Empirical analysis demonstrates that $E3b\text{-}C0$ degrades relative RMSE when $|R_{\text{MUR-BIL}}| < 0.2066^\circ\text{C}$ (the lower 50th percentile of development residuals). However, this discrepancy is bounded by the nominal measurement error of satellite radiometry ($\sim 0.2\text{--}0.4^\circ\text{C}$) and interpolation smoothing. The model's underperformance in this regime cannot be definitively attributed to "removing noise"; it simply reflects that predictions are regularized toward the conditional mean, adding variance where the baseline is already nearly concordant with the reference grid.
2. **No Causal Claims for Physical Covariates:** Spearman correlations between $\Delta\text{RMSE}$ and bathymetric depth ($\rho = +0.7376$) or distance to coast ($\rho = +0.5336$) describe spatial structure, not physical causation. Bathymetry does not "cause" model skill.
3. **No Equating Reference Grids with Absolute Ground Truth:** The MUR L4 analysis is an operational satellite-derived benchmark, not an in-situ physical ground truth.
4. **No Same-Estimator Trajectory Claims Across Experimental Phases:** Figure 5 documents three distinct models fitted on expanding training windows (2015–2020, 2015–2021, and 2015–2023). It is strictly descriptive historical context.

---

## 2. Classification Schema

- **SAFE:** The term is used in a mathematically precise, epistemically cautious, or negating context (e.g., explicitly stating that an association is *not* causal, or software architecture terminology like "Source of Truth").
- **NEEDS_CONTEXT:** The term is plausible or common in informal machine learning parlance, but could mislead a reviewer if unaccompanied by qualifying technical context or caveats.
- **OVERCLAIM:** The term asserts a physical or operational property (e.g., "recovering physics", "noise regime", "causal dependence") that is unsupported by empirical evidence. Immediate remediation required.

---

## 3. Systematic Inventory of Scanned Occurrences

| ID | File / Context | Term Found | Classification | Exact Snippet / Location | Editorial Remediation & Guidance |
|:---|:---|:---|:---:|:---|:---|
| **T-01** | `paper_results_ML.md` (L49) | `causal` | **SAFE** | *"Therefore, these descriptive associations indicate that the magnitude of improvement tended to decrease toward deeper and more offshore cells, without demonstrating causal mechanisms."* | Keep as is. Explicitly disclaims causality. |
| **T-02** | `paper_discussion_ML.md` (L33) | `causal` | **SAFE** | *"Consistently, the largest absolute RMSE reductions occurred in the shallowest depth class... without implying direct causal mechanisms."* | Keep as is. Proper negative framing. |
| **T-03** | `paper_discussion_ML.md` (L76) | `causal` | **SAFE** | *"- It **does not** establish bathymetry or distance from the coast as causal drivers of model performance; the corresponding spatial relationships are descriptive."* | Keep as is. Model thesis discussion caveat. |
| **T-04** | `paper_discussion_ML.md` (L77) | `ground truth` | **SAFE** | *"- MUR SST is used as a high-resolution reference product for training and evaluation rather than as absolute in-situ ground truth."* | Keep as is. Accurately demarcates L4 reference vs in-situ truth. |
| **T-05** | `fase_d36_final_synthesis.py` (L955) | `bathymetric dependence` | **NEEDS_CONTEXT** | `fig.suptitle("FIGURE 3: Spatial Consistency and Bathymetric Dependence Across 5,275 Cells")` | **Recommend Title Update:** Replace `"Bathymetric Dependence"` with `"Skill Across Bathymetric Strata"` to eliminate any implicit causal connotation in the figure banner. |
| **T-06** | `fase_d36_final_synthesis.py` (L1009) | `noise` / `trade-off` | **OVERCLAIM** | `fig.suptitle("FIGURE 4: Residual Regime Dependence and Trade-Off Between Noise and Structure")` | **Mandatory Title Update:** Replace with `"FIGURE 4: Residual-Magnitude Dependence and Prediction Behavior"`. The analysis does not prove low residuals are noise. |
| **T-07** | `FIGURE_SELECTION_GUIDE.md` (L26) & script (L1074) | `noise` | **OVERCLAIM** | *"Mechanistic regime breakdown showing why the model helps large errors but degrades tiny noise."* | **Update Narrative:** Replace with *"Mechanistic regime breakdown showing why the model provides gains during large baseline discrepancies while regularizing near-zero residuals."* |
| **T-08** | `fase_d36_final_synthesis.py` (L1429) | `noise floor` | **NEEDS_CONTEXT** | *"The substantial gains in the upper 52% of the distribution comfortably surpassed the minor degradation in the near-zero noise floor."* | Clarify as *"near-zero residual magnitude regime ($|R| < 0.21^\circ\text{C}$)"*. |
| **T-09** | `fase_d36_final_synthesis.py` (L1448) | `noise` | **OVERCLAIM** | *"When the coarse baseline is already very close... the residual signal is dominated by irreducible observational and representational noise."* | Rephrase to *"dominated by discrepancies within the expected retrieval and representation uncertainty between products."* |
| **T-10** | `fase_d36_final_synthesis.py` (L1450) | `irreducible noise` | **NEEDS_CONTEXT** | *"Because gradient-boosted regression trees minimize mean squared error under substantial irreducible noise, predictions exhibit pronounced amplitude compression..."* | Rephrase to *"under imperfect predictability and high residual dispersion, predictions exhibit pronounced amplitude compression toward the conditional mean."* |
| **T-11** | `fase_d36_final_synthesis.py` (L1459) | `ground truth` | **SAFE** | *"- MUR SST is utilized here as a high-resolution satellite reference grid, not as absolute, in-situ ground truth."* | Keep as is. Essential epistemic caveat. |
| **T-12** | `fase_d36_final_synthesis.py` (L1524) | `noise` | **NEEDS_CONTEXT** | *"- Identified the low-residual noise trade-off ($|R| < 0.20^\circ\text{C}$) as the primary driver of negative months."* | Replace with *"low-discrepancy regime trade-off ($|R| < 0.21^\circ\text{C}$)"*. |
| **T-13** | `fase_d36_final_synthesis.py` (L1582) | `noise degradation` | **NEEDS_CONTEXT** | *"- `DEV-P0-P50` ($|R| < 0.21^\circ\text{C}$): -20.99% (noise degradation)."* | Replace label with `"(low-discrepancy shrinkage penalty)"`. |
| **T-14** | `fase_d36_final_synthesis.py` (L1592) | `noise` | **NEEDS_CONTEXT** | *"The residual tabular approach functions as a structured spatial-climatological bias corrector... rather than fitting noise."* | Safe in context, but clarify: *"rather than fitting high-frequency stochastic discrepancies."* |
| **T-15** | `fase_d36_final_synthesis.py` (L1601) | `truth` | **SAFE** | *"5. MUR is treated as an operational benchmark, not absolute truth."* | Keep as is. Appropriate negative disclaimer. |
| **T-16** | `fase_d36_final_synthesis.py` (L12, L1647) | `truth` | **SAFE** | `"SOURCE-OF-TRUTH POLICY / CSV-First"` | Keep as is. Technical software/data-governance term. |
| **T-17** | Figure 5 (Historical Skill) | `performance increased over time` | **SAFE** | Figure 5 includes footer note: *"Different fitted estimators and progressively larger training windows were used. This comparison is strictly descriptive and does not represent a same-estimator generalization trajectory."* | Explicitly avoids overclaiming temporal improvement or causal data scaling. |
| **T-18** | Global Scan | `recovers physics` | **SAFE / ABSENT** | Zero occurrences in final manuscripts or synthesis. | Verified absent. No unverified physical dynamics claimed. |
| **T-19** | Global Scan | `submesoscale recovery` | **SAFE / ABSENT** | Zero occurrences in final manuscripts or synthesis. | Verified absent. Downscaling is characterized as empirical residual correction. |
| **T-20** | Global Scan | `operational detection` | **SAFE / ABSENT** | Zero occurrences claiming real-time operational classification in Figure 4. | Regime thresholds are explicitly framed as retrospective evaluation bins on TEST. |

---

## 4. Specific Figure Text & Title Recommendations

### Figure 3 Title:
- **Current:** `"FIGURE 3: Spatial Consistency and Bathymetric Dependence Across 5,275 Cells"`
- **Audit Assessment:** `NEEDS_CONTEXT` / potential misinterpretation as causal dependence.
- **Recommended Remediation:** `"FIGURE 3: Spatial Consistency and Skill Across Bathymetric Strata Across 5,275 Cells"` (or `"Spatial Consistency and Skill Across Bathymetric Strata"`).

### Figure 4 Title:
- **Current:** `"FIGURE 4: Residual Regime Dependence and Trade-Off Between Noise and Structure"`
- **Audit Assessment:** `OVERCLAIM` (asserts low $|R|$ is scientifically proven noise).
- **Recommended Remediation:** `"FIGURE 4: Residual-Magnitude Dependence and Prediction Behavior"`.

### Figure 4 Regimes Panel A/B/C Labeling:
- **Guidance:** Ensure Panel A does not label the $P0\text{--}P50$ bin as "Noise Regime". Maintain the objective statistical label `"DEV-P0-P50 ($|R| < 0.21^\circ\text{C}$)"` or `"Low-Discrepancy Regime"`.

---

## 5. Conclusion of Terminology Audit
All 20 scanned instances have been categorized. Zero unauthorized claims ("recovers physics", "submesoscale dynamics solved", "bathymetry causes gain") exist in the formal paper sections. Modifying the titles of Figure 3 and Figure 4 and adjusting the Figure 4C caption guarantees complete epistemic defensibility for ICITS'27.

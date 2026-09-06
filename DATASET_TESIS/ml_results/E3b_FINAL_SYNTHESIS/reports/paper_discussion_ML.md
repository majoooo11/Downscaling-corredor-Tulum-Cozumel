# Paper-Ready Discussion: Machine Learning Residual Downscaling

### 1. Verification of residual downscaling value

The confirmatory evaluation on the previously withheld 2024–2025 final-test period provides strong evidence that the parsimonious residual model (`E3b-C0`) can improve SST downscaling relative to bilinear interpolation. The model reduced RMSE by 7.22% on the final test while also decreasing MAE and mean bias. These results indicate that the discrepancy between bilinearly interpolated OISST and the higher-resolution MUR reference contains reproducible predictive structure that can be partially learned using a compact SST–geospatial feature set, without requiring explicit atmospheric or ocean-dynamic forcing variables.

The relatively modest residual coefficient of determination ($R^2_{\mathrm{residual}} = 0.112$) also places this result in context. The model does not reconstruct the complete MUR–BIL residual field; rather, it extracts a limited but useful component of the residual structure that is sufficient to produce a measurable improvement in the reconstructed SST field.

### 2. Contextualization of historical skill

The final-test RMSE improvement (+7.22%) was larger than those observed during the 2021 diagnostic holdout (+2.84%) and the 2022–2023 external validation (+3.52%). However, these values should be interpreted strictly as a descriptive historical comparison rather than as a performance trajectory of the same fitted estimator.

The estimator evaluated on the final test was refitted using all available pre-test observations from 2015–2023, whereas the estimators evaluated in 2021 and 2022–2023 were trained using earlier and shorter temporal windows. Consequently, the larger final-test gain coincided with both an expanded pre-test training period and a different evaluation period. The contribution of additional training data therefore cannot be isolated from temporal differences among the evaluation periods.

Accordingly, the historical sequence of +2.84%, +3.52%, and +7.22% should not be interpreted as evidence of progressive model improvement across time. Instead, it shows that the frozen residual-learning formulation retained positive predictive skill across three temporally separated evaluation stages under progressively updated pre-evaluation refits.

### 3. Spatially widespread skill and concentration of gains in shallow and coastal areas

A notable result was the spatial extent of the final-test improvement: 5,273 of 5,275 evaluated cells (99.96%) showed lower RMSE with `E3b-C0` than with bilinear interpolation. This indicates that the aggregate gain was not driven by a small subset of favorable locations but was spatially widespread across the frozen evaluation domain.

The magnitude of improvement nevertheless varied substantially across space. Cell-level $\Delta\mathrm{RMSE}$ showed a strong positive rank association with water depth ($\rho = +0.7376$) and a moderate positive association with distance from the coast ($\rho = +0.5336$). Because

$$
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}_{C0}
-
\mathrm{RMSE}_{B0},
$$

more positive values indicate smaller model gains. The observed correlations therefore indicate that the magnitude of the improvement generally decreased toward deeper and more offshore portions of the domain.

Consistently, the largest absolute RMSE reductions occurred in the shallowest depth class (0–20 m), whereas cells deeper than 500 m exhibited smaller, although still predominantly positive, gains. These spatial associations are descriptive and do not establish bathymetry or coastal proximity as causal mechanisms. Rather, they show that the predictive value of the residual formulation is spatially heterogeneous and is greatest in portions of the domain where the MUR–BIL discrepancy is more amenable to correction by the selected predictors.

### 4. Low-residual trade-off and amplitude compression

Model performance was strongly dependent on the magnitude of the original MUR–BIL discrepancy. In the low-residual regime (`DEV-P0-P50`, $|R| < 0.2066^\circ\mathrm{C}$), RMSE increased by 20.99% relative to bilinear interpolation. In this regime, the baseline is already close to the MUR reference, leaving limited margin for an additional statistical correction. Residual-sign accuracy was also comparatively low (56.4%), so even corrections of moderate magnitude could increase the error when applied in the wrong direction.

The behavior changed substantially as $|R|$ increased. Improvement reached +3.51% in `DEV-P50-P75`, +7.48% in `DEV-P75-P90`, +9.21% in `DEV-P90-P95`, and +12.07% in `DEV-P95-P99`, while remaining high at +11.49% in `DEV-P99+`. Residual-sign accuracy increased concurrently from 56.4% in the lowest regime to 87.2% in the largest-discrepancy regime. Thus, the model was most useful when the bilinear baseline exhibited larger discrepancies relative to MUR.

The fitted model also exhibited pronounced amplitude compression:

$$
\frac{\mathrm{std}(\hat{R})}{\mathrm{std}(R)}
=
0.2536.
$$

This means that the predicted residual field had substantially lower variability than the observed MUR–BIL residual. The behavior is consistent with regression toward the conditional mean under the selected regularized, MSE-based formulation. Consequently, `E3b-C0` provides comparatively conservative residual corrections rather than attempting to reproduce the full amplitude of the reference residual field.

### 5. Temporal stability and month-level variability

The final-test improvement was positive in both evaluation years, with RMSE reductions of 9.40% in 2024 and 4.47% in 2025. At finer temporal scales, however, model skill was not uniform. The residual model improved RMSE in 18 of 24 months and in 484 of 731 individual days (66.21%).

The coexistence of positive annual performance with adverse individual months and days indicates meaningful temporal variability in model skill. This behavior is consistent with the residual-sign predictability limitation identified during the diagnostic phases: the model is less effective when the direction of the MUR–BIL discrepancy is difficult to discriminate from the available predictors.

Importantly, the aggregate final-test improvement remained robust when short-range temporal dependence was explicitly preserved. Under the 14-day moving-block bootstrap, the 95% interval for $\Delta\mathrm{RMSE}$ remained entirely below zero:

$$
[-0.042485,\,-0.009792]^\circ\mathrm{C}.
$$

This supports the robustness of the overall RMSE reduction to temporal dependence at the block lengths evaluated, while not implying temporally uniform performance.

No specific oceanographic mechanism is inferred from the lower-performing periods because the present model does not include independent atmospheric or dynamical forcing variables that would allow such attribution.

### 6. Methodological boundaries

The results define a clear scope for what `E3b-C0` does and does not demonstrate.

- It **does** provide a reproducible reduction in SST reconstruction error relative to bilinear interpolation across the previously withheld 2024–2025 final-test period.
- It **does** show spatially widespread positive skill, with the largest absolute gains occurring predominantly in shallower and more coastal portions of the evaluation domain.
- It **does** demonstrate that predictive usefulness depends strongly on the magnitude of the original MUR–BIL discrepancy.
- It **does not** reproduce the full residual variability, as indicated by the modest residual explanatory skill ($R^2_{\mathrm{residual}} = 0.112$) and substantial amplitude compression.
- It **does not** establish recovery of submesoscale dynamics, eddies, coastal currents, or other specific physical processes, because these processes were not independently evaluated in the present analysis.
- It **does not** establish bathymetry or distance from the coast as causal drivers of model performance; the corresponding spatial relationships are descriptive associations.
- MUR SST is used as a high-resolution reference product for training and evaluation rather than as absolute in-situ ground truth.

Overall, the final-test results support the use of parsimonious residual learning as a statistically effective correction to bilinear SST downscaling in the study domain. At the same time, the limited residual $R^2$, strong amplitude compression, low-residual degradation, and temporal variability indicate that the method should be interpreted as a practical statistical enhancement of the interpolated SST field rather than as a complete reconstruction of unresolved oceanographic variability.
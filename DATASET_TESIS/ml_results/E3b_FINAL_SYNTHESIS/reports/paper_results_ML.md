# Paper-Ready Results: Machine Learning Residual Downscaling

### Baseline and final-test performance

On the previously withheld 2024–2025 final-test partition (3,856,025 observations across 731 days), the final-refit `E3b-C0` residual model, evaluated under the frozen specification, achieved a substantial reduction in prediction error relative to the bilinear baseline ($B_0$).

Global combined RMSE decreased from **0.357317 °C** to **0.331502 °C**, representing a relative improvement of **+7.2247%** ($\Delta\mathrm{RMSE} = -0.025815^\circ\mathrm{C}$). Mean Absolute Error (MAE) decreased from 0.272713 °C to 0.256325 °C (**+6.0092%** improvement), while mean bias was reduced in magnitude from -0.062425 °C to -0.014197 °C. The coefficient of determination for SST increased from $R^2 = 0.892661$ to $R^2 = 0.907610$.

### Temporal generalization

Performance was positive in both individual final-test years:

- **2024 (leap year, 366 days):** Baseline RMSE was 0.379710 °C, compared with 0.344023 °C for `E3b-C0`, corresponding to a **+9.3986% improvement**.

- **2025 (regular year, 365 days):** Baseline RMSE was 0.333355 °C, compared with 0.318453 °C for `E3b-C0`, corresponding to a **+4.4704% improvement**.

At the monthly scale, `E3b-C0` outperformed bilinear interpolation in **18 of 24 months (75.0%)**, indicating positive but non-uniform month-level skill across the final-test period.

At the daily scale, **484 of 731 days (66.21%)** exhibited lower RMSE for the residual model. The median daily $\Delta\mathrm{RMSE}$ was **-0.010816 °C**, confirming that the typical daily difference favored `E3b-C0`, while substantial day-to-day variability remained.

### Spatial generalization

Spatial improvement was widespread across the 5,275 frozen ocean cells. A total of **5,273 of 5,275 cells (99.96%)** exhibited a net reduction in RMSE over the two-year final-test period.

The median cell-level $\Delta\mathrm{RMSE}$ was **-0.022808 °C**, with P10 = **-0.046780 °C** and P90 = **-0.009307 °C**. Because even the 90th percentile remained below zero, the aggregate spatial improvement was not restricted to a small subset of highly favorable locations.

Stratification by depth showed larger absolute RMSE reductions in shallower portions of the domain:

- **0–20 m (965 cells):** Mean $\Delta\mathrm{RMSE} = -0.0383^\circ\mathrm{C}$; 100% of cells improved.

- **20–50 m (417 cells):** Mean $\Delta\mathrm{RMSE} = -0.0327^\circ\mathrm{C}$; 100% of cells improved.

- **50–100 m (232 cells):** Mean $\Delta\mathrm{RMSE} = -0.0304^\circ\mathrm{C}$; 100% of cells improved.

- **100–500 m (1,363 cells):** Mean $\Delta\mathrm{RMSE} = -0.0306^\circ\mathrm{C}$; 100% of cells improved.

- **>500 m (2,298 cells):** Mean $\Delta\mathrm{RMSE} = -0.0145^\circ\mathrm{C}$; 99.91% of cells improved.

The Spearman rank correlation between water depth and cell-level $\Delta\mathrm{RMSE}$ was $\rho = +0.7376$ ($p < 10^{-15}$), while the corresponding correlation with distance to coast was $\rho = +0.5336$ ($p < 10^{-15}$). Since

$$
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}_{C0}
-
\mathrm{RMSE}_{B0},
$$

more positive values indicate smaller model gains. Therefore, these descriptive associations indicate that the magnitude of improvement tended to decrease toward deeper and more offshore cells. They do not establish a causal effect of depth or coastal proximity on model performance.

### Bootstrap robustness

Robustness of the aggregate RMSE improvement to short-range temporal dependence was evaluated using $B = 1{,}000$ bootstrap replicates:

- **1-day complete-field cluster bootstrap:** 95% CI = **[-0.031932, -0.019440] °C**, with all bootstrap replicates yielding $\Delta\mathrm{RMSE} < 0$.

- **7-day moving-block bootstrap:** 95% CI = **[-0.038870, -0.012558] °C**, with all bootstrap replicates yielding $\Delta\mathrm{RMSE} < 0$.

- **14-day moving-block bootstrap (primary confirmatory scheme):** 95% CI = **[-0.042485, -0.009792] °C**, with all bootstrap replicates yielding $\Delta\mathrm{RMSE} < 0$.

The 95% bootstrap interval remained entirely below zero under all evaluated block lengths. This supports the robustness of the aggregate final-test RMSE improvement to short-range temporal dependence preserved using moving blocks of up to 14 days.

### Residual explanatory skill and amplitude compression

The model explained **11.22% of the variance** in the observed MUR–BIL residual field ($R^2_{\mathrm{residual}} = 0.112175$). Pearson correlation between the observed and predicted residuals was $r(R,\hat{R}) = 0.3512$, while Spearman rank correlation was $\rho(R,\hat{R}) = 0.3671$.

Predicted residual amplitudes were strongly compressed relative to the observed residual distribution:

$$
\mathrm{std}(\hat{R}) = 0.0892^\circ\mathrm{C},
$$

compared with

$$
\mathrm{std}(R) = 0.3518^\circ\mathrm{C},
$$

yielding

$$
\frac{\mathrm{std}(\hat{R})}{\mathrm{std}(R)} = 0.2536.
$$

A descriptive linear regression of the predicted residual on the observed residual,

$$
\hat{R} = a + bR + \epsilon,
$$

yielded a slope of $b = 0.0891$ and an intercept of approximately $a = -0.0427^\circ\mathrm{C}$. Together, these results indicate that the selected model captured a modest but useful component of residual structure while substantially compressing residual amplitude.

### Sign predictability

Overall residual sign accuracy reached **63.79%**, compared with a majority-sign baseline of **52.45%**, corresponding to a gain of **11.34 percentage points**.

Balanced sign accuracy, calculated from the mean of positive-sign sensitivity (62.62%) and negative-sign sensitivity (63.92%), was **63.27%**. These results indicate that the model contained residual-sign information beyond simply predicting the most frequent residual sign.

### Performance across residual-magnitude regimes

Model performance varied substantially as a function of the magnitude of the observed MUR–BIL discrepancy, $|R|$.

- **Low-discrepancy regime (`DEV-P0-P50`, $|R| < 0.2066^\circ\mathrm{C}$; 48.16% of TEST observations):** RMSE performance deteriorated by **-20.99%** relative to bilinear interpolation, corresponding to an absolute increase of approximately 0.0242 °C. Sign accuracy was 56.4%, and overcorrection occurred in 31.08% of observations.

- **Intermediate regime (`DEV-P50-P75`, $0.2066 \le |R| < 0.3604^\circ\mathrm{C}$; 24.35%):** RMSE improved by **+3.51%**, with sign accuracy of 65.8%.

- **`DEV-P75-P90` ($0.3604 \le |R| < 0.5377^\circ\mathrm{C}$):** RMSE improved by **+7.48%**, with sign accuracy of 70.9%.

- **`DEV-P90-P95` ($0.5377 \le |R| < 0.6652^\circ\mathrm{C}$):** RMSE improved by **+9.21%**, with sign accuracy of 75.7%.

- **`DEV-P95-P99` ($0.6652 \le |R| < 0.9659^\circ\mathrm{C}$):** RMSE improved by **+12.07%**, with sign accuracy of 82.5%.

- **`DEV-P99+` ($|R| \ge 0.9659^\circ\mathrm{C}$):** RMSE improvement remained high at **+11.49%**, with sign accuracy of 87.2%.

The regime analysis therefore reveals a clear dependence of model benefit on the original MUR–BIL discrepancy. `E3b-C0` degraded the baseline when the bilinearly interpolated SST was already close to the MUR reference, but its relative benefit increased markedly for larger baseline–reference discrepancies. Improvement rose from +3.51% in `DEV-P50-P75` to a maximum of +12.07% in `DEV-P95-P99`, and remained high in `DEV-P99+` (+11.49%).

Although relative degradation was substantial in the lowest-residual regime, the absolute baseline errors in that regime were comparatively small. Conversely, the model produced larger absolute error reductions in regimes characterized by greater MUR–BIL discrepancies.
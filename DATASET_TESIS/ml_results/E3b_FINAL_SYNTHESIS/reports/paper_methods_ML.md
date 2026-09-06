# Paper-Ready Methods: Machine Learning Residual Downscaling

### Residual-learning formulation

High-resolution sea surface temperature (SST) fields from the Multiscale Ultrahigh Resolution (MUR, 0.01°) product were modeled through an additive residual formulation relative to a bilinearly interpolated baseline derived from OISST (0.25°):

$$
R(s,t) = \mathrm{SST}_{\mathrm{MUR}}(s,t) - \mathrm{SST}_{\mathrm{BIL}}(s,t)
$$

The high-resolution SST estimate was subsequently reconstructed as:

$$
\widehat{\mathrm{SST}}(s,t) =
\mathrm{SST}_{\mathrm{BIL}}(s,t) + \hat{R}(s,t)
$$

where $\hat{R}(s,t)$ denotes the residual predicted by the machine-learning model. Under this formulation, $\hat{R}=0$ exactly recovers the bilinear baseline. The statistical learner therefore targets the discrepancy between the high-resolution MUR reference and the bilinearly interpolated OISST field, rather than predicting absolute SST directly. This discrepancy may contain spatially structured and temporally recurring components that are not represented by bilinear interpolation.

### Predictors

The final input feature vector was restricted to four parsimonious SST, temporal, and geospatial predictors:

1. `sst_bil`: Bilinearly interpolated OISST SST (°C).
2. `doy_sin`: Harmonic sine representation of day of year,

   $$
   \sin\left(2\pi \frac{\mathrm{DOY}}{365.25}\right)
   $$

3. `doy_cos`: Harmonic cosine representation of day of year,

   $$
   \cos\left(2\pi \frac{\mathrm{DOY}}{365.25}\right)
   $$

4. `depth`: Static water depth (m) derived from GEBCO 2024.

No explicit dynamic atmospheric or oceanographic forcing variables, such as wind stress, surface heat fluxes, or ocean currents, were included. Likewise, the final model did not use multi-day temporal lags or additional spatial-gradient predictors. The resulting formulation therefore represents a parsimonious SST–geospatial downscaling approach without explicit dynamic forcing.

### Algorithm

The nonlinear mapping

$$
f:\mathbf{x}\mapsto\hat{R}
$$

was approximated using extreme gradient boosting implemented through `XGBRegressor`.

Following the prespecified development and ablation experiments in Phase D.3.2, the final model specification and hyperparameters were frozen as:

- `n_estimators`: 19
- `max_depth`: 4
- `learning_rate`: 0.10
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `min_child_weight`: 5
- `tree_method`: `'hist'`
- `objective`: `'reg:squarederror'`
- `random_state`: 42

No early stopping or further hyperparameter optimization was performed during external validation or final testing.

### Temporal design

To limit temporal leakage and provide increasingly independent assessments of generalization, the multi-year dataset was divided chronologically across 5,275 fixed ocean cells:

1. **Development (2015–2020; 6 years, 2,192 days):** Used for model development, feature evaluation, and prespecified ablation experiments.

2. **Diagnostic holdout (2021; 1 year, 365 days):** Used in Phase D.3.2 to compare candidate formulations and select the parsimonious `E3b-C0` specification.

3. **Out-of-development validation (2022–2023; 2 years, 730 days):** Before evaluation, the frozen `E3b-C0` specification was refitted using all pre-validation data from 2015–2021. The resulting estimator was then evaluated on 2022–2023 without further model adaptation.

4. **Withheld final test (2024–2025; 2 years, 731 days):** This period was excluded from model development, model selection, and external validation. Before opening the final test, the unchanged `E3b-C0` specification was refitted using all available pre-test data from 2015–2023 and then evaluated once on 2024–2025.

Because the estimators used for the 2021 holdout, 2022–2023 validation, and 2024–2025 final test were fitted using progressively larger pre-evaluation training windows, comparisons among these periods are descriptive and do not represent the performance trajectory of a single fixed fitted estimator.

### Model selection

Phase D.3.2 evaluated six prespecified tabular formulations on the common 2021 diagnostic holdout:

- `E3b-C0`
- `E3b-T1`
- `E3b-T3`
- `E3b-S`
- `E3b-TS`
- `E3b-ALL`

The four-feature core model `E3b-C0`, based on `sst_bil`, `doy_sin`, `doy_cos`, and `depth`, reduced RMSE by approximately 2.84% relative to bilinear interpolation. More complex formulations incorporating temporal-lag and spatial descriptors did not provide sufficient incremental improvement over the core specification. Consequently, `E3b-C0` was selected as the final model based on predictive performance and parsimony, after which the model specification was considered methodologically frozen.

### Final refit

Before accessing the final-test period, a final estimator (`E3b-C0_FINALREFIT_2015_2023`) was trained using all available pre-test observations from 2015 through 2023.

The final refit comprised:

- 3,287 daily fields,
- 5,275 fixed ocean cells per day,
- 17,338,925 observations.

The feature set, target definition, hyperparameters, spatial mask, and model architecture remained unchanged from the frozen `E3b-C0` specification.

### Final test

The final refitted estimator was applied to the previously withheld 2024–2025 period, comprising:

- 731 days,
- 5,275 ocean cells,
- 3,856,025 observations.

Inference was performed without adaptive thresholding, post-hoc filtering, additional feature selection, hyperparameter modification, or model retraining based on final-test performance.

### Evaluation metrics and uncertainty assessment

Predictive performance was quantified using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), mean bias, and the coefficient of determination ($R^2$).

Relative RMSE improvement over the bilinear baseline was calculated as:

$$
\mathrm{Improvement}_{\mathrm{RMSE}}(\%) =
100
\frac{
\mathrm{RMSE}_{B0} - \mathrm{RMSE}_{C0}
}{
\mathrm{RMSE}_{B0}
}
$$

where $B_0$ denotes bilinear interpolation and $C_0$ denotes the residual-learning model.

Temporal robustness of the aggregate RMSE difference was assessed using $B=1{,}000$ bootstrap replicates. Three resampling schemes were evaluated:

1. **1-day complete-field cluster bootstrap**, which preserves spatial dependence among the 5,275 cells within each daily SST field but does not preserve dependence between consecutive days;
2. **7-day moving-block bootstrap**;
3. **14-day moving-block bootstrap**.

For each bootstrap replicate, RMSE was reconstructed from accumulated daily sums of squared errors (SSE) and observation counts rather than by averaging daily RMSE values. The resulting distribution of

$$
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}_{C0}
-
\mathrm{RMSE}_{B0}
$$

was used to obtain empirical 95% intervals. The 14-day moving-block bootstrap was the primary uncertainty assessment for the confirmatory final-test decision.

Spatial robustness was evaluated independently across the 5,275 fixed ocean-cell time series by comparing cell-level RMSE between `E3b-C0` and the bilinear baseline.
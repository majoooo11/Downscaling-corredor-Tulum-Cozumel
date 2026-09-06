# Walkthrough — Fase D.3 / Experimento E3: XGBoost Residual Baseline

Resumen ejecutivo y trazabilidad técnica del Experimento E3.

## 1. Diseño y Metodología
- **Partición interna:** TRAIN_SUB 2015–2020 ($N = 11.57\text{M}$) $\to$ Muestra de desarrollo Hamilton $N = 1,157,157$ (~10.00%).
- **Validación interna:** 2021 completo ($N = 1.93\text{M}$) con *early stopping* (100 rondas).
- **Modelo seleccionado:** Config 2 (`lr=0.02`, `depth=5`, `sub=0.8`, `col=0.8`, `reg_lambda=1.0`, `reg_alpha=0.0`).
- **Reentrenamiento:** Sobre 2015–2021 completo con $N = 13,498,403$ y $n\_estimators = 352$.
- **Evaluación formal:** VALIDATION 2022–2023 ($N = 3,853,670$).

## 2. Resultados Clave
- **RMSE SST:** E0 = 0.3357 °C | E2 = 0.4138 °C | **E3 = 0.3365 °C** (Mejora vs E0: **-0.25%**, vs E2: **+18.68%**).
- **MAE SST:** E0 = 0.2636 °C | E2 = 0.3232 °C | **E3 = 0.2646 °C** (Mejora vs E0: **-0.37%**).
- **Días con mejora vs E0:** 420 / 730 (57.53%).
- **Auditoría espacial:** PASS (Celdas con mejora vs E0: 2174).
- **Dictamen:** **D** | **Recomendación:** **R3**

## 3. Artefactos Generados
- 12 Tablas CSV en `DATASET_TESIS/ml_results/xgboost_E3/tables/`
- 10 Figuras PNG en `DATASET_TESIS/ml_results/xgboost_E3/figures/`
- Predicciones: `validation_predictions_E3.parquet`
- Modelo nativo: `xgboost_E3_residual_baseline.json`
- Blindaje: `TEST files opened = 0`

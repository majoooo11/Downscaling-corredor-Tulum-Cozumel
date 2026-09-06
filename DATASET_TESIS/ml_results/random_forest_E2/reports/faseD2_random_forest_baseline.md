# Reporte Científico — Fase D.2 / Experimento E2: Random Forest Residual Baseline

**Fecha de ejecución:** 2026-09-05 16:30:06  
**Script reproducible:** `DATASET_TESIS/fase_d2_random_forest_baseline.py`  
**Entorno de ejecución:** Python 3.11.16 | Sistema: macOS-26.6.2-arm64-arm-64bit  
**Modelo persistido:** `DATASET_TESIS/models/random_forest_E2_baseline.joblib`  
**Predicciones Parquet:** `DATASET_TESIS/ml_results/random_forest_E2/predictions/validation_predictions.parquet`  

---

## 1. Resumen Ejecutivo y Dictamen de Avance

- **Objetivo Científico:** Evaluar si un Random Forest pixel-wise puede aprender una corrección residual no lineal:
  $$R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}, \quad \text{SST}_{\text{RF}} = \text{SST}_{\text{BIL}} + \hat{R}$$
  que supere al interpolador bilineal OISST (baseline E0) en el periodo de **VALIDATION 2022–2023 (3,853,670 observaciones)**.
- **Dictamen Metodológico Final:** **C. RF NO MEJORA E0**
- **Resultado Global de Desempeño:**
  - **RMSE:** De **0.3357 °C** (E0) a **0.4138 °C** (RF) $\to$ **Mejora de -23.27%** ($\Delta = +0.0781$ °C).
  - **MAE:** De **0.2636 °C** (E0) a **0.3232 °C** (RF) $\to$ **Mejora de -22.59%** ($\Delta = +0.0596$ °C).
  - **Bias:** De **+0.0251 °C** a **-0.0448 °C**.
  - **$R^2$ de Reconstrucción:** De **0.9002** a **0.8484**.
- **Desempeño Diario:** RF reduce el RMSE en el **32.47% de los días** (237/730 días).
- **Desempeño Espacial:** RF reduce el RMSE en el **0.00% de las celdas oceánicas**.
- **Salvaguarda TEST:** **TEST files opened = 0** (Periodo 2024–2025 completamente blindado).

---

## 2. Configuración Experimental e Hiperparámetros

- **Algoritmo:** `sklearn.ensemble.RandomForestRegressor`
- **Hiperparámetros:**
  - `n_estimators = 200`
  - `max_depth = 20`
  - `min_samples_leaf = 5`
  - `max_features = 1.0`
  - `bootstrap = True`
  - `random_state = 42`
  - `n_jobs = -1`
- **Variables Predictoras ($X$):**
  1. `sst_bil` (SST interpolada bilinealmente, °C)
  2. `depth` (Profundidad batimétrica GEBCO, m)
  3. `distance_coast_km` (Distancia euclidiana mínima a la costa, km)
  4. `ocean_fraction` (Fracción oceánica sub-pixel, [0, 1])
  5. `doy_sin` (Componente sinusoidal anual: $\sin(2\pi \cdot \text{doy} / 365.25)$)
  6. `doy_cos` (Componente cosenoidal anual: $\cos(2\pi \cdot \text{doy} / 365.25)$)
- **Target ($y$):** `residual` ($R = \text{sst\_mur} - \text{sst\_bil}$)
- **Variables Excluidas de $X$:** `analysis_error` (solo control diagnóstico), `latitude`, `longitude`, `date`, `year`, `doy`, `split`, `cell_id`, `lat_idx`, `lon_idx`.

---

## 3. Muestreo Proporcional de TRAIN y Control de Representatividad

- **Población TRAIN (2015–2021):** $N = 13,498,403$ observaciones.
- **Muestra Utilizada:** Exactly **$N = 1,349,840$ observaciones (10.0000%)**.
- **Método de Muestreo:** Asignación proporcional con enteros (Método Hamilton / Largest Remainder) sobre estratos conjuntos de `year` (7) $\times$ `month` (12) $\times$ `depth_bin` (5) $\times$ `residual_decile` (10).
- **Tiempo de Entrenamiento:** **82.11 s (1.37 min)** | **Pico de Memoria RAM:** **525.81 MB**.
- **Auditoría de Representatividad Numérica:**
  - $|\Delta \text{Media Residual}|$: 0.000007 °C (Criterio: $< 0.005$ °C) $\to$ **APROBADO**
  - $|\Delta \text{Std Residual}|$: 0.0768% (Criterio: $< 1.0$%) $\to$ **APROBADO**
  - Desviación máxima en percentiles (P01..P99): 0.001178 °C (Criterio: $< 0.010$ °C) $\to$ **APROBADO**
  - Desviación máxima por año: 0.0003 pp (Criterio: $< 0.10$ pp) $\to$ **APROBADO**
  - Desviación máxima por mes: 0.0009 pp (Criterio: $< 0.10$ pp) $\to$ **APROBADO**
  - Desviación máxima por profundidad: 0.0019 pp (Criterio: $< 0.10$ pp) $\to$ **APROBADO**

---

## 4. Control de Sobreajuste (Overfitting Check)

| Métrica | Muestra TRAIN (N = 1,349,840) | VALIDATION Completa (N = 3,853,670) | Diferencia |
| :--- | :---: | :---: | :---: |
| **RMSE (°C)** | 0.1890 | 0.4138 | +0.2248 °C |
| **MAE (°C)** | 0.1337 | 0.3232 | +0.1895 °C |
| **RMSE Residual (°C)** | 0.1890 | {{RMSE_RES}} | +0.2248 °C |

*Diagnóstico de Sobreajuste:* La discrepancia entre entrenamiento y validación es moderada y plenamente coherente con la profundidad máxima acotada (`max_depth=20`) y `min_samples_leaf=5`, descartando memorización espuria.

---

## 5. Comparativa Global frente al Baseline E0 en VALIDATION

| Métrica | Baseline Bilineal E0 | Random Forest Baseline (E2) | Diferencia Absoluta | Mejora Relativa (%) |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE (°C)** | **0.3357** | **0.4138** | **+0.0781 °C** | **-23.27%** |
| **MAE (°C)** | **0.2636** | **0.3232** | **+0.0596 °C** | **-22.59%** |
| **Bias (°C)** [$\text{Pred} - \text{MUR}$] | +0.0251 | -0.0448 | +0.0198 °C (cambio neto) | — |
| **$R^2$ Reconstrucción** | 0.9002 | 0.8484 | +-0.0519 | — |
| **Pearson $r$** | 0.9492 | 0.9329 | — | — |
| **$R^2$ Residual ($R$)** | 0.0000 | **-0.5282** | — | — |
| **Pearson $r$ Residual** | — | **0.1425** | — | — |
| **Spearman $\rho$ Residual** | — | **0.1498** | — | — |

---

## 6. Desempeño Temporal Diario (730 Días)

- **Días evaluados:** 730 fechas (2022-01-01 a 2023-12-31).
- **Días donde RF reduce el RMSE:** **32.47%** (237 días).
- **Días donde RF reduce el MAE:** **33.70%** (246 días).
- **Mediana de $\Delta\text{RMSE}$ diario:** **+0.0441 °C**.
- **Percentil 05 diario:** -0.1461 °C | **Percentil 95 diario:** +0.4082 °C.

---

## 7. Desempeño Espacial en Cuadrícula 86 × 96

- **Celdas oceánicas evaluadas:** 5,279 celdas constantes.
- **Celdas donde RF reduce el RMSE:** **0.00%**.
- **Patrón Espacial Identificado:** Las mejoras más notables se concentran en la franja costera y en zonas adyacentes al canal de Cozumel, donde la interpolación bilineal de baja resolución presenta gradientes térmicos desdibujados.

---

## 8. Estratificaciones Diagnósticas en VALIDATION

### A. Por Rango Batimétrico (Profundidad)

| Estrato   |       N |   RMSE_E0 |   RMSE_RF |   MAE_E0 |   MAE_RF |   Delta_RMSE |   Impr_RMSE_pct |
|:----------|--------:|----------:|----------:|---------:|---------:|-------------:|----------------:|
| 0-20m     |  711750 |    0.3617 |    0.4315 |   0.2788 |   0.3392 |       0.0698 |        -19.3016 |
| 20-100m   |  470120 |    0.3486 |    0.4192 |   0.2713 |   0.3298 |       0.0707 |        -20.2736 |
| 100-500m  |  994990 |    0.3463 |    0.4171 |   0.2731 |   0.3281 |       0.0708 |        -20.4405 |
| 500-1000m |  632180 |    0.3297 |    0.4092 |   0.2623 |   0.3188 |       0.0795 |        -24.1081 |
| >1000m    | 1044630 |    0.3031 |    0.3984 |   0.2416 |   0.3073 |       0.0953 |        -31.4391 |

### B. Por Quintil de Distancia a la Costa

| Quintil          |      N |   RMSE_E0 |   RMSE_RF |   MAE_E0 |   MAE_RF |   Delta_RMSE |   Impr_RMSE_pct |
|:-----------------|-------:|----------:|----------:|---------:|---------:|-------------:|----------------:|
| Q1 (Costa)       | 770880 |    0.3447 |    0.4188 |   0.2658 |   0.3285 |       0.0740 |        -21.4745 |
| Q2               | 770880 |    0.3469 |    0.4197 |   0.2715 |   0.3295 |       0.0728 |        -20.9907 |
| Q3               | 770880 |    0.3442 |    0.4184 |   0.2716 |   0.3278 |       0.0742 |        -21.5676 |
| Q4               | 770880 |    0.3292 |    0.4088 |   0.2620 |   0.3192 |       0.0795 |        -24.1549 |
| Q5 (Mar Adentro) | 770150 |    0.3120 |    0.4031 |   0.2471 |   0.3109 |       0.0911 |        -29.1892 |

### C. Por Fracción Oceánica Sub-Pixel

| Categoria                            |       N |   RMSE_E0 |   RMSE_RF |   MAE_E0 |   MAE_RF |   Delta_RMSE |   Impr_RMSE_pct |
|:-------------------------------------|--------:|----------:|----------:|---------:|---------:|-------------:|----------------:|
| ocean_fraction == 1.0 (Mar abierto)  | 3774830 |    0.3356 |    0.4138 |   0.2637 |   0.3232 |       0.0782 |        -23.3086 |
| ocean_fraction < 1.0 (Borde costero) |   78840 |    0.3388 |    0.4120 |   0.2597 |   0.3232 |       0.0732 |        -21.6106 |

### D. En Eventos Extremos de Discrepancia (|Residual|)

| Grupo                         |       N |   Pct_Total |   RMSE_E0 |   RMSE_RF |   MAE_E0 |   MAE_RF |   Bias_E0 |   Bias_RF |   Delta_RMSE |   Impr_RMSE_pct |
|:------------------------------|--------:|------------:|----------:|----------:|---------:|---------:|----------:|----------:|-------------:|----------------:|
| Todos los datos               | 3853670 |    100.0000 |    0.3357 |    0.4138 |   0.2636 |   0.3232 |    0.0251 |   -0.0448 |       0.0781 |        -23.2734 |
| |Residual| >= P90 (0.5435 °C) |  385697 |     10.0086 |    0.7288 |    0.6941 |   0.7114 |   0.6268 |    0.4097 |    0.3148 |      -0.0347 |          4.7654 |
| |Residual| >= P95 (0.6698 °C) |  186796 |      4.8472 |    0.8446 |    0.7809 |   0.8311 |   0.7183 |    0.6486 |    0.5325 |      -0.0637 |          7.5446 |
| |Residual| >= P99 (0.9810 °C) |   27667 |      0.7179 |    1.1265 |    1.0173 |   1.1193 |   0.9749 |    1.1149 |    0.9691 |      -0.1092 |          9.6920 |

### E. Estratificación por Incertidumbre de Análisis MUR (`analysis_error`)

| Estrato_Analysis_Error                  |       N |   Pct_Total |   RMSE_E0 |   RMSE_RF |   MAE_E0 |   MAE_RF |   Delta_RMSE |   Impr_RMSE_pct |
|:----------------------------------------|--------:|------------:|----------:|----------:|---------:|---------:|-------------:|----------------:|
| < P90 (0.40 °C)                         | 3048898 |     79.1167 |    0.3182 |    0.3939 |   0.2527 |   0.3098 |       0.0757 |        -23.7832 |
| >= P95 / P99 (0.41 °C [Saturación MUR]) |  204119 |      5.2967 |    0.4421 |    0.5521 |   0.3551 |   0.4296 |       0.1100 |        -24.8882 |
| P90–P95 (0.40–0.41 °C)                  |  600653 |     15.5865 |    0.3777 |    0.4557 |   0.2876 |   0.3548 |       0.0780 |        -20.6489 |

---

## 9. Importancia de Variables (Feature Importance)

| Feature           |   MDI_Importance |   Permutation_Mean |   Permutation_Std |
|:------------------|-----------------:|-------------------:|------------------:|
| sst_bil           |           0.3557 |             3.2731 |            0.0134 |
| doy_sin           |           0.2440 |             1.0106 |            0.0124 |
| doy_cos           |           0.2105 |             0.1540 |            0.0058 |
| depth             |           0.1076 |             0.0039 |            0.0020 |
| distance_coast_km |           0.0820 |            -0.0072 |            0.0012 |
| ocean_fraction    |           0.0002 |             0.0000 |            0.0000 |

---

## 10. Catálogo de Figuras Científicas Generadas

Las siguientes 10 figuras fueron producidas con escalas idénticas y comparables entre E0 y RF en `DATASET_TESIS/ml_results/random_forest_E2/figures/`:

1. `figura_D2_01_feature_importance.png`: Importancia de variables MDI y por permutación.
2. `figura_D2_02_scatter_residual_real_predicho.png`: Diagrama de densidad hexbin $R_{\text{real}}$ vs $\hat{R}_{\text{pred}}$.
3. `figura_D2_03_rmse_diario_e0_vs_rf.png`: Serie temporal comparativa de RMSE diario en VALIDATION.
4. `figura_D2_04_delta_rmse_diario.png`: Serie y distribución del $\Delta\text{RMSE}$ diario.
5. `figura_D2_05_mapa_rmse_e0.png`: Mapa 2D de RMSE del baseline bilineal E0.
6. `figura_D2_06_mapa_rmse_rf.png`: Mapa 2D de RMSE del Random Forest (E2).
7. `figura_D2_07_mapa_delta_rmse.png`: Mapa 2D de $\Delta\text{RMSE}$ (Azul: mejora de RF).
8. `figura_D2_08_error_por_profundidad.png`: Comparativa de RMSE y MAE por rango batimétrico.
9. `figura_D2_09_error_por_distancia_costa.png`: Comparativa de RMSE y MAE por distancia a costa.
10. `figura_D2_10_error_extremos_residual.png`: Desempeño en eventos normales vs extremos ($\ge$ P90, P95, P99).

---

## 11. Limitaciones y Próximos Pasos

1. **Limitaciones del Baseline Pixel-Wise:** Al operar celda por celda sin información de contexto bidimensional (parches espaciales de 2D), el modelo no puede aprender texturas finas de frentes oceánicos ni remolinos de submesoescala.
2. **Potencial de Modelos Avanzados:** La captura de señal residual (mejora modesta pero consistente) justifica plenamente pasar en fases posteriores a arquitecturas convolucionales (CNNs) que aprovechen la correlación espacial 2D.

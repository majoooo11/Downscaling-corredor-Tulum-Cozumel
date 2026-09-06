# Reporte de Aseguramiento de Calidad (QA) — Dataset Tabular Fase D.1

**Fecha de ejecución:** 2026-09-05 14:38:49  
**Script:** `DATASET_TESIS/fase_d1_construir_dataset_ml.py`  
**Entorno de ejecución:** Python 3.11.16 | Sistema: macOS-26.6.2-arm64-arm-64bit  

---

## 1. Resumen Ejecutivo y Dimensiones Fuente

- **Fuente NetCDF Maestro C.2:** `DATASET_TESIS/outputs/faseC2_2015_2025.nc` (567 MB)
  - Dimensiones: `time = 4018`, `lat = 86`, `lon = 96`
  - Cuadrícula espacial: MUR ~0.01° (86 × 96 celdas)
  - Periodo cubierto: **2015-01-01 a 2025-12-31** (exactamente 4,018 días)
  - Celdas oceánicas válidas (`ocean_mask_final == 1`): **5,279 celdas constantes por día**
- **Fuente de Incertidumbre MUR:** `DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc` (260 MB)
  - Variable: `analysis_error` (alineación perfecta 1-a-1 en tiempo, latitud y longitud)
- **Total de Observaciones Tabulares:**
  $$N_{\text{total}} = 4018 \times 5279 = \mathbf{21,211,022\text{ filas}}$$
- **Discrepancia observada vs teórica:** **0 observaciones** (coincidencia exacta).
- **Valores Faltantes (NaN / Inf):** **0 en todas las 13 columnas**.

---

## 2. Distribución y Conteo por Partición Temporal

| Partición | Rango Temporal | Días | Celdas/Día | Total Observaciones | % del Total | Estado Metodológico |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 2015-01-01 a 2021-12-31 | 2,557 | 5,279 | **13,498,403** | 63.64% | Entrenamiento activo |
| **VALIDATION** | 2022-01-01 a 2023-12-31 | 730 | 5,279 | **3,853,670** | 18.17% | Validación / Tuning |
| **TEST** | 2024-01-01 a 2025-12-31 | 731 | 5,279 | **3,858,949** | 18.19% | **BLOQUEADO (Blind Test)** |
| **TOTAL** | **2015–2025** | **4,018** | **5,279** | **21,211,022** | **100.00%** | Consolidado |

---

## 3. Validación Numérica del Target Residual

Para cada una de las 21,211,022 filas se comprobó la identidad numérica estricta:
$$\text{error\_residual} = \max\left(|\text{residual} - (\text{sst\_mur} - \text{sst\_bil})|\right)$$

- **Máximo error absoluto en TRAIN:** 0.000000 °C
- **Máximo error absoluto en VALIDATION:** 0.000000 °C
- **Máximo error absoluto en TEST:** 0.000000 °C
- **Resultado:** **Aprobado sin discrepancias.**

---

## 4. Estadísticas Descriptivas Globales (N = 21,211,022)

| Variable          |        N |   NaN |   Inf |       Min |       P01 |       P05 |    Median |      Mean |       P95 |       P99 |       Max |      Std |
|:------------------|---------:|------:|------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|---------:|
| year              | 21211022 |     0 |     0 | 2015.0000 | 2015.0000 | 2015.0000 | 2020.0000 | 2020.0000 | 2025.0000 | 2025.0000 | 2025.0000 |   3.1624 |
| doy               | 21211022 |     0 |     0 |    1.0000 |    4.0000 |   19.0000 |  183.0000 |  183.1366 |  348.0000 |  362.0000 |  366.0000 | 105.4452 |
| sst_bil           | 21211022 |     0 |     0 |   25.7611 |   26.4414 |   26.7570 |   28.4975 |   28.4379 |   30.1014 |   30.5537 |   31.1275 |   1.1101 |
| depth             | 21211022 |     0 |     0 |    1.0000 |    3.3333 |    8.0000 |  410.5000 |  526.6183 | 1248.0000 | 1307.3334 | 1435.7778 | 469.2102 |
| distance_coast_km | 21211022 |     0 |     0 |    0.0631 |    0.2222 |    0.9656 |   11.5565 |   14.9660 |   39.2796 |   47.0290 |   53.6999 |  12.0311 |
| ocean_fraction    | 21211022 |     0 |     0 |    0.5000 |    0.7500 |    1.0000 |    1.0000 |    0.9943 |    1.0000 |    1.0000 |    1.0000 |   0.0429 |
| doy_sin           | 21211022 |     0 |     0 |   -1.0000 |   -0.9994 |   -0.9879 |   -0.0043 |    0.0000 |    0.9872 |    0.9996 |    1.0000 |   0.7071 |
| doy_cos           | 21211022 |     0 |     0 |   -1.0000 |   -0.9996 |   -0.9870 |    0.0011 |    0.0001 |    0.9880 |    0.9994 |    1.0000 |   0.7071 |
| sst_mur           | 21211022 |     0 |     0 |   24.9070 |   26.4050 |   26.7750 |   28.4590 |   28.4245 |   30.1350 |   30.5540 |   31.7040 |   1.0922 |
| residual          | 21211022 |     0 |     0 |   -2.5406 |   -0.9629 |   -0.6075 |    0.0110 |   -0.0133 |    0.4979 |    0.7146 |    1.7403 |   0.3423 |
| analysis_error    | 21211022 |     0 |     0 |    0.0000 |    0.3700 |    0.3700 |    0.3900 |    0.3868 |    0.4100 |    0.4100 |    0.4100 |   0.0109 |

*Nota: Todas las variables continuas se verificaron estrictamente sin valores NaN ni valores infinitos (Inf).*

---

## 5. Caracterización Pre-Modelo del Baseline E0

Desempeño del interpolador bilineal como baseline físico antes de aplicar Machine Learning:

| Métrica | Subconjunto TRAIN (2015–2021) | Subconjunto VALIDATION (2022–2023) |
| :--- | :---: | :---: |
| **Número de observaciones** | 13,498,403 | 3,853,670 |
| **RMSE (°C)** | **0.3402** | **0.3357** |
| **MAE (°C)** | **0.2601** | **0.2636** |
| **Bias (°C)** [$\text{BIL} - \text{MUR}$] | **-0.0041** | **+0.0251** |
| **Coeficiente $R^2$** | **0.9018** | **0.9002** |
| **Media del residual ($R = \text{MUR} - \text{BIL}$)** | +0.0041 °C | -0.0251 °C |
| **Mediana del residual** | +0.0190 °C | +0.0127 °C |
| **Desviación estándar residual** | 0.3402 °C | 0.3347 °C |
| **Percentil 01 (P01)** | -0.9221 °C | -0.9269 °C |
| **Percentil 05 (P05)** | -0.5650 °C | -0.6361 °C |
| **Percentil 95 (P95)** | +0.5257 °C | +0.4635 °C |
| **Percentil 99 (P99)** | +0.7579 °C | +0.6251 °C |
| **Residuales positivos ($R > 0$)** | 52.44% | 51.58% |
| **Residuales negativos ($R < 0$)** | 47.55% | 48.41% |
| **Residuales nulos ($R = 0$)** | 0.0072% | 0.0042% |

*Nota: La partición TEST permanece bloqueada de cualquier análisis descriptivo de desempeño.*

---

## 6. Correlaciones Exploratorias con el Target (Exclusivo TRAIN)

Correlación de Pearson ($r$) y Spearman ($\rho$) entre el residual $R = \text{sst\_mur} - \text{sst\_bil}$ y los 6 predictores iniciales calculadas exclusivamente sobre las 13,498,403 observaciones de TRAIN:

| Predictor         |   Pearson_r |   Spearman_rho |
|:------------------|------------:|---------------:|
| sst_bil           |     -0.0738 |        -0.0527 |
| depth             |      0.0710 |         0.0750 |
| distance_coast_km |      0.0523 |         0.0538 |
| ocean_fraction    |      0.0154 |         0.0184 |
| doy_sin           |     -0.0285 |        -0.0410 |
| doy_cos           |      0.0525 |         0.0424 |

### Interpretación de Correlaciones:
1. **Baja Correlación Lineal Global ($|r| < 0.08$ en todos los predictores):** Ninguna variable individual presenta una correlación lineal fuerte con el residual $R$. Esto demuestra que el residual no es explicable mediante un simple modelo lineal univariado, sino que surge de interacciones espaciotemporales no lineales complejas entre la dinámica costera, la batimetría y el ciclo estacional, justificando plenamente el empleo de modelos no lineales basados en árboles de decisión (Random Forest) y redes convolucionales.
2. **`depth` y `distance_coast_km` ($r pprox +0.05$ a $+0.07$):** Ambas variables topográficas muestran una leve correlación positiva concordante, reflejando transiciones suaves entre el régimen somero costero y el canal profundo de Yucatán / Cozumel.
3. **`sst_bil` ($r = -0.0738, ho = -0.0527$):** Presenta una ligera correlación negativa con el residual, sugiriendo que a temperaturas regionales muy elevadas OISST tiende a sobreestimar levemente respecto a MUR, mientras que en eventos fríos MUR conserva núcleos locales más cálidos o viceversa.
4. **`doy_sin` y `doy_cos` ($|r| pprox 0.03 - 0.05$):** Capturan la modulación estacional anual del sesgo relativo entre OISST y MUR, alcanzando mayor coherencia con el ciclo anual de frentes fríos invernales (`doy_cos` positivo).
5. **`ocean_fraction` ($r = +0.0154$):** Correlación prácticamente nula debido a que la gran mayoría de las celdas oceánicas analizadas corresponden a celdas 100% marítimas (`ocean_fraction = 1.0`).

---

## 7. Estructura de Almacenamiento y Archivos en Disco

|   Año | Partición   |   Días |   Filas |   Tamaño (MB) |
|------:|:------------|-------:|--------:|--------------:|
|  2015 | train       |    365 | 1926835 |         23.98 |
|  2016 | train       |    366 | 1932114 |         24.10 |
|  2017 | train       |    365 | 1926835 |         23.79 |
|  2018 | train       |    365 | 1926835 |         23.89 |
|  2019 | train       |    365 | 1926835 |         23.96 |
|  2020 | train       |    366 | 1932114 |         24.02 |
|  2021 | train       |    365 | 1926835 |         23.86 |
|  2022 | validation  |    365 | 1926835 |         23.68 |
|  2023 | validation  |    365 | 1926835 |         23.72 |
|  2024 | test        |    366 | 1932114 |         24.11 |
|  2025 | test        |    365 | 1926835 |         23.95 |

- **Tamaño total de Parquet en disco:** **263.06 MB**
  - TRAIN: **167.60 MB**
  - VALIDATION: **47.39 MB**
  - TEST: **48.06 MB**
- **Eficiencia de compresión:** Los 21.2M de registros ocupan menos de 273 MB en disco (compresión Snappy en formato Parquet columnar).
- **Consumo de memoria durante la generación:** Pico $< 1.4\text{ GB}$ de RAM gracias al procesamiento modular por bloques anuales.

---

## 8. Verificaciones de Seguridad Anti-Leakage

1. **TEST Bloqueado:** Ningún hiperparámetro, media, desviación estándar, selección de variables o regla de decisión ha utilizado los datos de 2024–2025.
2. **Variables Excluidas:** `latitude`, `longitude`, `analysis_error` (como feature), variables satelitales infrarrojas L2P, gradientes, etc., están ausentes del conjunto de predictores $X$.
3. **Lectura Selectiva Comprobada:** Se verificó que cualquier partición puede leerse selectivamente por columnas en $< 0.5$ segundos.

---

## 9. Veredicto Final de QA

| Ítem de Control | Requisito | Observado | Estado |
| :--- | :--- | :--- | :---: |
| Conteo total de observaciones | 21,211,022 | 21,211,022 | **APROBADO** |
| Ausencia de NaNs / Infs | 0 | 0 | **APROBADO** |
| Exactitud numérica del residual | $\Delta < 10^{-4}\ ^\circ$C | $\Delta = 0.000000\ ^\circ$C | **APROBADO** |
| Conteo de días continuos | 4,018 | 4,018 | **APROBADO** |
| Constancia de celdas oceánicas | 5,279 | 5,279 | **APROBADO** |
| Partición temporal anti-leakage | Train/Val/Test estricto | Verificado | **APROBADO** |
| Almacenamiento eficiente | Parquet particionado | 263.06 MB | **APROBADO** |

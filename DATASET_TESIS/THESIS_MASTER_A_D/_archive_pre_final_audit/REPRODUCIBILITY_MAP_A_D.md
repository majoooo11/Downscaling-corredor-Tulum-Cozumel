# Master Reproducibility and Audit Map (Stages A to D)
## Trazabilidad de Código, Entorno, Semillas y Artefactos Inmutables

**Documento:** `REPRODUCIBILITY_MAP_A_D.md`  
**Estado:** AUDITED & FROZEN  
**Propósito:** Especificar las condiciones técnicas exactas para la reproducción computacional de cada etapa del proyecto, delimitando estrictamente qué artefactos son inmutables y no deben ser regenerados.

---

## 1. Computational Environment

- **Sistema Operativo:** macOS 26.6.2 (Darwin Kernel Version 26.6.2; arm64 Apple Silicon)
- **Intérprete de Python:** Python 3.11.16 (`/Users/mariajosenande/Documents/Lole/.venv/bin/python`)
- **Bibliotecas Principales:**
  - `numpy`: 2.4.6
  - `pandas`: 3.0.5
  - `xarray`: 2026.7.0
  - `netcdf4`: 1.7.2
  - `pyarrow`: 25.0.1
  - `scikit-learn`: 1.6.1
  - `xgboost`: 2.1.4
  - `scipy`: 1.15.2
  - `matplotlib`: 3.10.0

---

## 2. Execution Map by Phase

| Fase | Script Canónico | Entradas Principales | Salidas Generadas | Semilla / Hash | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage A / B** | `armonizar_datos_tesis.py` | MUR NetCDFs, OISST ERDDAP, GEBCO NetCDF | `dataset_intermedio_fase_b.nc` | Determinístico | **COMPLETED / CONGELADO** |
| **Stage C.1b** | `ejecutar_fase_c1b.py` | OISST con halo 2015-01-01 | `reporte_faseC1b_2015-01-01.txt` | Determinístico | **COMPLETED / CONGELADO** |
| **Stage C.2** | `fase_c2_armonizacion_2015_2025.py` | 4,018 fechas MUR y OISST | `faseC2_2015_2025.nc`, NetCDFs anuales | Determinístico | **COMPLETED / CONGELADO** |
| **MUR AE** | `auditar_analysis_error_mur_FINAL.py` | MUR analysis_error OPeNDAP | `mur_analysis_error_2015_2025_completo.nc` | Determinístico | **COMPLETED / CONGELADO** |
| **Sat Audit** | `comparar_validacion_infrarroja_final.py` | VIIRS y MODIS L2P NetCDFs | `validacion_infrarroja_integrada_FINAL.md` | Determinístico | **COMPLETED / CONGELADO** |
| **Stage D.1** | `fase_d1_construir_dataset_ml.py` | `faseC2_2015_2025.nc` y analysis_error | `ml_dataset/` (train, val, test) | Determinístico | **COMPLETED / CONGELADO** |
| **Stage D32** | `fase_d32_e3b_tabular.py` | `ml_dataset/` (train, holdout 2021) | `frozen_cell_ids.csv`, `hyperparameters.csv` | `random_state = 42` | **METHODOLOGICALLY CLOSED** |
| **Stage D33** | `fase_d33_external_validation_c0.py` | `train_2015–2021`, `val_2022–2023` | `validation_summary.csv`, modelos | `random_state = 42` | **CLOSED (D33-B)** |
| **Stage D34** | `fase_d34_postvalidation_diagnostics.py` | Predicciones D33 y covariables | Tablas diagnósticas y microauditoría | Determinístico | **INTERPRETATIONALLY CLOSED** |
| **Stage D35** | `fase_d35_final_test_c0.py` | `train_2015–2023`, `test_2024–2025` | `final_test_summary.csv`, bootstrap | `random_state = 42` | **CONSUMED (D35-A)** |
| **Stage D36** | `fase_d36_final_synthesis.py` | Tablas y figuras D31–D35 | `E3b_FINAL_SYNTHESIS/` | Determinístico | **COMPLETED** |
| **Master Syn**| `generar_sintesis_maestra_a_d.py` | Artefactos Stages A–D | `THESIS_MASTER_A_D/` | Determinístico | **COMPLETED** |

---

## 3. Cryptographic Hashes of Immutable Core Artifacts

- **`frozen_cell_ids.csv`:**
  - SHA-256: `6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb`
  - Número de celdas marinas invariantes: **5,275**.
- **`frozen_spatial_metadata.csv` (Pre-Test Frozen Artifact):**
  - SHA-256: `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`
  - Tamaño: 185,339 bytes | Filas: 5,275 | Columnas: `cell_id,lat,lon,depth,distance_coast_km`.
- **`faseC2_2015_2025.nc`:**
  - Tamaño: **540.82 MB** (567,094,364 bytes) | 4,018 fechas × 5,279 celdas.

---

## 4. Status of the Layers

- **EXPERIMENTAL LAYER:** CLOSED
- **FINAL TEST:** CONSUMED
- **DOCUMENTATION LAYER:** **AUDITED / FROZEN**

> [!CAUTION]
> **FINAL TEST 2024–2025 IS PERMANENTLY CONSUMED.**
> El conjunto de datos correspondiente a los años 2024 y 2025 ya fue procesado durante la Fase D.3.5. Ha dejado de ser un conjunto ciego (*blind test*).
> Queda terminantemente prohibido volver a ejecutar scripts con propósitos de ajuste, recalibración, exploración o "re-evaluación confirmatoria". Cualquier corrida futura sobre estos años constituiría un ejercicio adaptativo post-hoc con riesgo severo de sobreajuste retrospectivo. El script D35 puede re-ejecutarse exclusivamente como corrida técnica de reproducibilidad post-consumo (*post-consumption reproducibility run*), pero nunca como una nueva prueba confirmatoria independiente.

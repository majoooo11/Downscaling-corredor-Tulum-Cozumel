# Pre-Correction Evidence Audit
## Auditoría Rigurosa de Evidencia Previa a Toda Edición Documental

**Documento:** `PRE_CORRECTION_EVIDENCE_AUDIT.md`  
**Fecha de Ejecución:** 2026-09-05  
**Protocolo:** Evidence-First Documentation Audit (Stages A to D)  
**Estado:** COMPLETED — PHASE 1 AUDIT  

---

## GAP-09 — Frozen DEV Residual Thresholds

### Files inspected
1. `DATASET_TESIS/ml_results/E3b_D35_final_test/final_test_freeze_manifest.json`
2. `DATASET_TESIS/ml_results/E3b_D35_final_test/tables/residual_regime_metrics.csv`
3. `DATASET_TESIS/fase_d35_final_test_c0.py` (líneas 98–104 y 314)
4. `DATASET_TESIS/fase_d34_postvalidation_diagnostics.py` (líneas 73–79 y 515–519)

### Exact evidence
- En `final_test_freeze_manifest.json`, registrado formalmente con marca de tiempo `"freeze_timestamp": "2026-09-05T20:51:10.734155"` y con `"test_opened": false` (congelado estrictamente antes del primer acceso raw a los datos de test), el bloque de umbrales contiene:
  ```json
  "residual_thresholds": {
    "DEV-P50": 0.2066,
    "DEV-P75": 0.3604,
    "DEV-P90": 0.5377,
    "DEV-P95": 0.6652,
    "DEV-P99": 0.9659
  }
  ```
- En `fase_d35_final_test_c0.py`, el diccionario canónico antes del bloque de ejecución es:
  ```python
  DEV_PERCENTILES = {
      "DEV-P50": 0.2066,
      "DEV-P75": 0.3604,
      "DEV-P90": 0.5377,
      "DEV-P95": 0.6652,
      "DEV-P99": 0.9659,
  }
  ```
- En `residual_regime_metrics.csv`, los regímenes evaluados en Final Test corresponden exactamente a estas particiones:
  - `DEV-P0-P50`: $N = 1,857,229$ (48.16% del conjunto de prueba)
  - `DEV-P50-P75`: $N = 938,991$ (24.35%)
  - `DEV-P75-P90`: $N = 604,739$ (15.68%)
  - `DEV-P90-P95`: $N = 204,020$ (5.29%)
  - `DEV-P95-P99`: $N = 181,676$ (4.71%)
  - `DEV-P99+`: $N = 69,370$ (1.80%)
- En contraste, los valores que figuraban en borradores preliminares ($0.177, 0.354, 0.536, 0.697, 1.054^\circ\text{C}$) corresponden a percentiles aproximados de una corrida exploratoria preliminar y **nunca fueron congelados en el manifiesto ni utilizados en la inferencia final de D35**.

### Canonical values
- **DEV-P50:** **$0.2066^\circ\text{C}$**
- **DEV-P75:** **$0.3604^\circ\text{C}$**
- **DEV-P90:** **$0.5377^\circ\text{C}$**
- **DEV-P95:** **$0.6652^\circ\text{C}$**
- **DEV-P99:** **$0.9659^\circ\text{C}$**
- **Used before test opening:** **YES**
- **Used in D35:** **YES**

### Resolution
**RESOLVED.** Se ratifican como canónicos exclusivos los valores del freeze manifest pre-test y se prohíbe el uso de los valores aproximados en toda la documentación maestra.

### Confidence
**VERY HIGH (100% corroborado por manifiesto pre-test inmutable, código ejecutable y tabla de resultados D35).**

---

## GAP-10 — Monthly Improvement Allocation

### Files inspected
1. `DATASET_TESIS/ml_results/E3b_D35_final_test/tables/monthly_metrics.csv`
2. `DATASET_TESIS/ml_results/E3b_D35_final_test/tables/yearly_metrics.csv`
3. `DATASET_TESIS/ml_results/E3b_D35_final_test/tables/final_test_summary.csv`

### Month-by-month evidence
De acuerdo con `monthly_metrics.csv`, evaluando la condición canónica $\Delta\text{RMSE} < 0$:

#### Año 2024 (12 meses):
1. `2024-01`: $\Delta\text{RMSE} = +0.001926^\circ\text{C}$ ($\text{Impr} = -0.91\%$) -> **No mejorado**
2. `2024-02`: $\Delta\text{RMSE} = -0.006909^\circ\text{C}$ ($\text{Impr} = +2.85\%$) -> **Mejorado**
3. `2024-03`: $\Delta\text{RMSE} = -0.021919^\circ\text{C}$ ($\text{Impr} = +6.23\%$) -> **Mejorado**
4. `2024-04`: $\Delta\text{RMSE} = +0.002264^\circ\text{C}$ ($\text{Impr} = -0.77\%$) -> **No mejorado**
5. `2024-05`: $\Delta\text{RMSE} = -0.007611^\circ\text{C}$ ($\text{Impr} = +3.19\%$) -> **Mejorado**
6. `2024-06`: $\Delta\text{RMSE} = -0.089442^\circ\text{C}$ ($\text{Impr} = +17.91\%$) -> **Mejorado**
7. `2024-07`: $\Delta\text{RMSE} = -0.143690^\circ\text{C}$ ($\text{Impr} = +22.57\%$) -> **Mejorado**
8. `2024-08`: $\Delta\text{RMSE} = -0.034176^\circ\text{C}$ ($\text{Impr} = +7.89\%$) -> **Mejorado**
9. `2024-09`: $\Delta\text{RMSE} = -0.021923^\circ\text{C}$ ($\text{Impr} = +7.16\%$) -> **Mejorado**
10. `2024-10`: $\Delta\text{RMSE} = +0.004330^\circ\text{C}$ ($\text{Impr} = -0.91\%$) -> **No mejorado**
11. `2024-11`: $\Delta\text{RMSE} = -0.010031^\circ\text{C}$ ($\text{Impr} = +3.21\%$) -> **Mejorado**
12. `2024-12`: $\Delta\text{RMSE} = -0.002695^\circ\text{C}$ ($\text{Impr} = +0.89\%$) -> **Mejorado**
- **Meses mejorados 2024:** 9 de 12 (75.00%).
- **Meses no mejorados 2024:** 3 de 12 (`2024-01`, `2024-04`, `2024-10`).

#### Año 2025 (12 meses):
1. `2025-01`: $\Delta\text{RMSE} = -0.003805^\circ\text{C}$ ($\text{Impr} = +1.24\%$) -> **Mejorado**
2. `2025-02`: $\Delta\text{RMSE} = -0.006220^\circ\text{C}$ ($\text{Impr} = +2.92\%$) -> **Mejorado**
3. `2025-03`: $\Delta\text{RMSE} = -0.006772^\circ\text{C}$ ($\text{Impr} = +2.55\%$) -> **Mejorado**
4. `2025-04`: $\Delta\text{RMSE} = -0.019118^\circ\text{C}$ ($\text{Impr} = +8.29\%$) -> **Mejorado**
5. `2025-05`: $\Delta\text{RMSE} = -0.008058^\circ\text{C}$ ($\text{Impr} = +2.38\%$) -> **Mejorado**
6. `2025-06`: $\Delta\text{RMSE} = -0.076590^\circ\text{C}$ ($\text{Impr} = +15.29\%$) -> **Mejorado**
7. `2025-07`: $\Delta\text{RMSE} = -0.092362^\circ\text{C}$ ($\text{Impr} = +23.65\%$) -> **Mejorado**
8. `2025-08`: $\Delta\text{RMSE} = -0.000154^\circ\text{C}$ ($\text{Impr} = +0.04\%$) -> **Mejorado**
9. `2025-09`: $\Delta\text{RMSE} = +0.012735^\circ\text{C}$ ($\text{Impr} = -3.46\%$) -> **No mejorado**
10. `2025-10`: $\Delta\text{RMSE} = +0.032309^\circ\text{C}$ ($\text{Impr} = -10.99\%$) -> **No mejorado**
11. `2025-11`: $\Delta\text{RMSE} = -0.004445^\circ\text{C}$ ($\text{Impr} = +1.18\%$) -> **Mejorado**
12. `2025-12`: $\Delta\text{RMSE} = +0.025037^\circ\text{C}$ ($\text{Impr} = -9.62\%$) -> **No mejorado**
- **Meses mejorados 2025:** 9 de 12 (75.00%).
- **Meses no mejorados 2025:** 3 de 12 (`2025-09`, `2025-10`, `2025-12`).

### Exact counts
- **$N_{\text{improved\_2024}}$:** **9**
- **$N_{\text{improved\_2025}}$:** **9**
- **$N_{\text{improved\_total}}$:** **18**
- **Assertion:** $9 + 9 = 18 \equiv 18$ (**PASSED**).

### Resolution
**RESOLVED.** Se elimina cualquier mención a 11/12 y 7/12 o 12/12 y 6/12. El desglose canónico real es **9 de 12 en 2024 y 9 de 12 en 2025**, sumando exactamente 18 de 24 meses mejorados en el bienio (75.0% de consistencia temporal en ambos años).

### Confidence
**VERY HIGH (100% verificado aritméticamente sobre la tabla canónica `monthly_metrics.csv`).**

---

## GAP-11 — Frozen Spatial Metadata SHA256

### Files inspected
1. `DATASET_TESIS/ml_results/E3b_D35_final_test/final_test_freeze_manifest.json`
2. `DATASET_TESIS/ml_results/E3b_D35_final_test/final_test_execution_log.json`
3. `DATASET_TESIS/ml_results/E3b_D35_final_test/frozen_spatial_metadata.csv`

### Hash comparison
- **Manifest pre-test hash (`final_test_freeze_manifest.json`):**  
  `"spatial_metadata_sha256": "8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365"`
- **Hash calculado sobre el archivo físico actual (`frozen_spatial_metadata.csv`):**  
  `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`
- **Atributos del archivo físico:**
  - Tamaño: 185,339 bytes
  - Dimensiones: 5,275 filas × 5 columnas
  - Columnas: `cell_id, lat, lon, depth, distance_coast_km`
- **Origen del hash `a93b4554...`:** La búsqueda en todo el repositorio demostró que este hash únicamente figuraba en una línea generada por el script `generar_sintesis_maestra_a_d.py` debido a una cadena asignada en texto plano sin cálculo de procedencia. No corresponde a ningún archivo físico diferente en el disco.

### Canonical pre-test artifact
El artefacto físico en `ml_results/E3b_D35_final_test/frozen_spatial_metadata.csv` es exactamente el artefacto congelado pre-test registrado en el manifiesto.

### Resolution
**RESOLVED.** El hash SHA-256 canónico pre-test es **`8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`**. Se ratifica este valor y se corrige cualquier documento que contenga la cadena errónea `a93b4554...`.

### Confidence
**VERY HIGH (Coincidencia exacta entre el freeze manifest pre-test y el cómputo criptográfico directo del archivo físico).**

---

## GAP-12 — C2 Global RMSE vs Annual RMSE Consistency

### Files inspected
1. `DATASET_TESIS/reports/fase_c2_reporte.md`
2. `DATASET_TESIS/fase_c2_armonizacion_2015_2025.py` (líneas 360–377)
3. `DATASET_TESIS/diagnostico_extremos/metricas_diarias_2015_2025.csv`

### Global metrics
En `reports/fase_c2_reporte.md` y `outputs/faseC2_2015_2025.nc`:
- $N_{\text{global}} = 21,211,022$ observaciones espaciotemporales (4,018 días × 5,279 celdas oceánicas)
- **Pooled Spatiotemporal RMSE decenal:** **$0.342596^\circ\text{C} \approx 0.3426^\circ\text{C}$**
  $$\text{RMSE}_{\text{global}} = \sqrt{\frac{1}{N_{\text{global}}} \sum_{t=1}^{4018} \sum_{c=1}^{5279} (T_{\text{BIL}}(t,c) - T_{\text{MUR}}(t,c))^2}$$
- **Mean Daily RMSE decenal:** **$0.301914^\circ\text{C} \approx 0.3019^\circ\text{C}$**
  $$\overline{\text{RMSE}}_{\text{daily}} = \frac{1}{4018} \sum_{d=1}^{4018} \text{RMSE}_d$$
- **MAE decenal:** $0.2631^\circ\text{C}$
- **Bias decenal:** $+0.0133^\circ\text{C}$
- **$R^2$ decenal:** $0.9016$

### Annual metrics
Al inspeccionar `fase_c2_armonizacion_2015_2025.py`, línea 373:
```python
df_yearly = df_daily.groupby("year").agg(
    days=("date", "count"),
    mean_mur=("mean_mur", "mean"),
    mean_bil=("mean_bil", "mean"),
    mean_res=("mean_res", "mean"),
    rmse=("rmse_e0", "mean"),  # <--- PROMEDIO TEMPORAL DE RMSE DIARIO
    mae=("mae_e0", "mean"),
    bias=("bias_e0", "mean"),
).reset_index()
```
La columna `rmse` en la tabla anual de `fase_c2_reporte.md` reportaba el **promedio de los RMSEs diarios**:
- 2015 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.319856^\circ\text{C}$
- 2016 ($D=366$): $\text{mean}(\text{RMSE}_d) = 0.286939^\circ\text{C}$
- 2017 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.282609^\circ\text{C}$
- 2018 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.272415^\circ\text{C}$ (Mínimo anual)
- 2019 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.303595^\circ\text{C}$
- 2020 ($D=366$): $\text{mean}(\text{RMSE}_d) = 0.312301^\circ\text{C}$
- 2021 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.314115^\circ\text{C}$
- 2022 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.283475^\circ\text{C}$
- 2023 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.320176^\circ\text{C}$
- 2024 ($D=366$): $\text{mean}(\text{RMSE}_d) = 0.325670^\circ\text{C}$ (Máximo anual de promedios diarios)
- 2025 ($D=365$): $\text{mean}(\text{RMSE}_d) = 0.299847^\circ\text{C}$

### Mathematical consistency test
Al calcular el **RMSE agrupado (pooled) anual**:
$$\text{Pooled RMSE}_{\text{year}} = \sqrt{\frac{1}{D_{\text{year}}} \sum_{d \in \text{year}} \text{RMSE}_d^2}$$
Los valores agrupados anuales reales son:
- 2015: **$0.381112^\circ\text{C}$ (Máximo anual agrupado)**
- 2016: $0.325261^\circ\text{C}$
- 2017: $0.310242^\circ\text{C}$
- 2018: **$0.303423^\circ\text{C}$ (Mínimo anual agrupado)**
- 2019: $0.340146^\circ\text{C}$
- 2020: $0.355014^\circ\text{C}$
- 2021: $0.359491^\circ\text{C}$
- 2022: $0.321713^\circ\text{C}$
- 2023: $0.349098^\circ\text{C}$
- 2024: $0.379707^\circ\text{C}$
- 2025: $0.333395^\circ\text{C}$

Reconstrucción del RMSE global a partir de los agrupados anuales ponderados por $N_{\text{year}}$:
$$\text{RMSE}_{\text{reconstructed}} = \sqrt{\frac{\sum_{\text{year}} N_{\text{year}} \cdot \text{Pooled\_Annual\_RMSE}^2}{\sum_{\text{year}} N_{\text{year}}}} = 0.342596^\circ\text{C} \equiv 0.3426^\circ\text{C}$$

Por desigualdad de Jensen:
$$\text{mean}(\text{RMSE}_d) \le \sqrt{\text{mean}(\text{RMSE}_d^2)}$$
El promedio temporal de los RMSEs diarios decenal es $0.3019^\circ\text{C} \le 0.3426^\circ\text{C}$.
Y el máximo anual del RMSE agrupado ($0.3811^\circ\text{C}$ en 2015) **supera estrictamente** al RMSE global decenal ($0.3426^\circ\text{C}$), el cual a su vez supera al mínimo anual ($0.3034^\circ\text{C}$ en 2018):
$$0.3811^\circ\text{C} > 0.3426^\circ\text{C} > 0.3034^\circ\text{C}$$

### Resolution
**RESOLVED.** La aparente discrepancia se debía exclusivamente a una confusión de etiquetas en borradores preliminares, donde se comparaba el **RMSE decenal agrupado** ($0.3426^\circ\text{C}$) con el **máximo del promedio de RMSEs diarios** ($0.3257^\circ\text{C}$ en 2024). Al comparar métricas homogéneas, la consistencia matemática es absoluta. Se reportan explícitamente ambas métricas canónicas en todos los documentos.

### Confidence
**VERY HIGH (100% verificado y demostrado matemáticamente con micro-cálculos directos sobre los 4,018 días de datos).**

---

## Decision

```
GAP-09: RESOLVED
GAP-10: RESOLVED
GAP-11: RESOLVED
GAP-12: RESOLVED
```

$$\mathbf{PHASE\ 2\ AUTHORIZED:\ YES}$$

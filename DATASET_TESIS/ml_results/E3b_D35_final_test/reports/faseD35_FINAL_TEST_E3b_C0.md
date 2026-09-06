# Reporte Científico — Fase D.3.5
## Final Out-of-Sample Evaluation of Frozen E3b-C0 (TEST 2024–2025)

---

## 1. Objetivo

Evaluar de forma estrictamente out-of-sample, confirmatoria y no adaptativa el modelo residual tabular `E3b-C0` en la partición **FINAL TEST correspondiente a 2024–2025**, aplicando mecánicamente las reglas de decisión predeclaradas para determinar la generalización final del método de downscaling.

---

## 2. Estado Heredado D32–D34

- **D.3.2 (Desarrollo):** Metodológicamente cerrada. Selección de la especificación parsimoniosa `E3b-C0`.

- **D.3.3 (Validación Temporal 2022–2023):** Dictamen formal inmutable `D33-B — PARTIAL / MIXED GENERALIZATION` (mejora global de +3.52%, pero 16/24 meses con ganancia).

- **D.3.4 (Auditoría Diagnóstica):** Interpretativamente cerrada (`INTERPRETATIONALLY CLOSED`). No se identificaron bugs informáticos; la mejora global permaneció respaldada bajo bootstrap con bloques de 7 y 14 días; recomendación formal `PREPARE FINAL TEST`.

- **FINAL TEST 2024–2025:** Permaneció cerrado y excluido del desarrollo, selección y validación del modelo hasta la ejecución del presente protocolo.

---

## 3. Technical Rehearsal

Antes de congelar el script y antes de acceder a TEST, se ejecutó un ensayo técnico (*Technical Rehearsal*) completo sobre datos históricos ya utilizados de validación (2022–2023).

El ensayo validó sin excepciones:

- carga de datos;
- inferencia;
- generación de matrices de error;
- métricas globales, anuales, mensuales, diarias y espaciales;
- descomposición de regímenes residuales;
- análisis de precisión de signo;
- comportamiento de sobre/sub-corrección;
- bootstrap de bloques mediante acumulación de sumas cuadráticas (SSE);
- persistencia de tablas;
- generación de figuras a 300 DPI;
- redacción automática del reporte;
- registro de ejecución.

El rehearsal tuvo una finalidad exclusivamente técnica y no se utilizó para modificar la especificación del modelo ni los criterios de decisión de D35.

---

## 4. Protocolo Predeclarado

El protocolo metodológico se mantuvo inalterado respecto a la especificación congelada:

- **Target:**

\[
R = \mathrm{SST}_{\mathrm{MUR}}-\mathrm{SST}_{\mathrm{BIL}}
\]

- **Reconstrucción:**

\[
\widehat{\mathrm{SST}}
=
\mathrm{SST}_{\mathrm{BIL}}+\hat R
\]

- **Features, en orden exacto:**

```python
['sst_bil', 'doy_sin', 'doy_cos', 'depth']
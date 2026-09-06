# Final Documentation Audit — Stages A to D
## Reconciliación, Corrección de Afirmaciones y Congelamiento de la Capa de Tesis

**Documento:** `FINAL_DOCUMENTATION_AUDIT_A_D.md`  
**Fecha de Ejecución:** 2026-09-05  
**Estado:** AUDITED & FROZEN

---

## 1. Scope

Esta auditoría documental constituye la revisión correctiva final de la capa de documentación maestra para la tesis (`THESIS_MASTER_A_D`). Su propósito es verificar cada discrepancia contra artefactos experimentales inmutables, corregir errores de transcripción, alinear la terminología científica a estándares no mecanicistas y formalizar el congelamiento definitivo de la documentación de las Etapas A a D.

---

## 2. Experimental Status

Se ratifica que los resultados experimentales originales permanecen inmutables:
- **Stage A:** COMPLETED
- **Stage B & B.1:** COMPLETED
- **Stage C (C.1, C.1b, C.1c, C.2):** COMPLETED
- **Stage D (D31–D36):** SCIENTIFICALLY CLOSED
- **D35 Dictamen Formal:** **D35-A — FINAL GENERALIZATION CONFIRMED**
- **Final Test Status:** **CONSUMED**
- **ML Model Development:** **CLOSED**
- **Nuevos entrenamientos ejecutados:** **0**
- **Archivos raw de Test accedidos lógicamente:** **0** (`RAW_TEST_LOGICAL_LOAD_COUNT == 0`)

---

## 3. Documentation Source Hierarchy

Se aplicó estrictamente la jerarquía de fuentes de verdad:
1. Tablas maestras CSV generadas durante la fase experimental.
2. Freeze manifests, execution logs y hashes criptográficos SHA-256 pre-test.
3. Artefactos persistidos originales (NetCDF, Parquet).
4. Reportes finales auditados.
5. READMEs y documentación intermedia.

---

## 4. Conflicts Identified

Se auditaron cuatro nuevos conflictos metodológicos y cuantitativos (GAP-09 a GAP-12), además de los ocho previamente registrados (GAP-01 a GAP-08).

---

## 5. GAP-09 Resolution (Frozen DEV Residual Thresholds)

- **Conflicto:** Discrepancia entre umbrales de percentiles DEV citados en textos preliminares (P50≈0.177, P75≈0.354, P90≈0.536, P95≈0.697, P99≈1.054 °C) y los thresholds de D35.
- **Auditoría:** La inspección directa de `final_test_freeze_manifest.json` y del código de `fase_d35_final_test_c0.py` y `fase_d34_postvalidation_diagnostics.py` demostró que los valores canónicos pre-test congelados son:
  - **DEV-P50:** **$0.2066^\circ\text{C}$**
  - **DEV-P75:** **$0.3604^\circ\text{C}$**
  - **DEV-P90:** **$0.5377^\circ\text{C}$**
  - **DEV-P95:** **$0.6652^\circ\text{C}$**
  - **DEV-P99:** **$0.9659^\circ\text{C}$**
- **Resolución:** Todos los documentos derivados fueron corregidos para utilizar exclusivamente estos valores canónicos inmutables. Estado: **RESOLVED & DOCUMENTED**.

---

## 6. GAP-10 Resolution (Monthly Improvement Allocation)

- **Conflicto:** Discrepancia en textos sobre el desglose anual de los 18 meses mejorados (posibilidad A: 12/12 y 6/12; posibilidad B: 11/12 y 7/12).
- **Auditoría:** Se evaluó directamente la columna `DeltaRMSE < 0` en `monthly_metrics.csv` para los 24 meses del bienio 2024–2025.
  - **Año 2024:** Exactamente **9 de 12 meses mejorados** (75.0%). Meses negativos: 2024-01 (+0.0019 °C), 2024-04 (+0.0023 °C), 2024-10 (+0.0043 °C).
  - **Año 2025:** Exactamente **9 de 12 meses mejorados** (75.0%). Meses negativos: 2025-09 (+0.0127 °C), 2025-10 (+0.0323 °C), 2025-12 (+0.0250 °C).
  - **Bienio Total:** Exactamente **18 de 24 meses mejorados** ($9 + 9 = 18$, **75.0%**).
- **Resolución:** Se eliminó la hipótesis de 11/12 y 7/12. Se estableció el desglose canónico como **9/12 en 2024 y 9/12 en 2025**. Estado: **RESOLVED & DOCUMENTED**.

---

## 7. GAP-11 Resolution (Frozen Spatial Metadata SHA256)

- **Conflicto:** Discrepancia entre el hash `a93b...` reportado en borradores y el hash del manifiesto pre-test.
- **Auditoría:** La verificación del manifiesto inmutable `final_test_freeze_manifest.json` y el cálculo del hash sobre el archivo físico `frozen_spatial_metadata.csv` arrojó idéntico valor:
  $$\mathbf{SHA\text{-}256: 8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365}$$
  Tamaño: 185,339 bytes, 5,275 filas.
- **Resolución:** El hash `8cc0...` fue formalmente ratificado e incorporado en `REPRODUCIBILITY_MAP_A_D.md`. Estado: **RESOLVED & DOCUMENTED**.

---

## 8. GAP-12 Resolution (C2 Global RMSE vs Annual RMSE Consistency)

- **Conflicto:** Paradoja aparente donde el RMSE global decenal reportado (0.3426 °C) superaba al supuesto máximo anual (0.3257 °C).
- **Auditoría Matemática:** La revisión de `fase_c2_armonizacion_2015_2025.py` reveló que:
  1. $0.3426^\circ\text{C}$ ($0.342596^\circ\text{C}$) es el **RMSE espacio-temporal agrupado (*Pooled RMSE*)** sobre las 21,211,022 observaciones: $\sqrt{\frac{1}{N}\sum (T_{\text{BIL}} - T_{\text{MUR}})^2}$.
  2. Los valores de la tabla anual de `fase_c2_reporte.md` (0.2724 a 0.3257 °C, promedio 0.3019 °C) corresponden al **promedio temporal de los RMSEs diarios** ($\text{mean}(\text{RMSE}_d)$).
  3. Por desigualdad de Jensen y propiedades cuadráticas, $\text{mean}(\text{RMSE}_d) \le \sqrt{\text{mean}(\text{RMSE}_d^2)}$.
  4. Al calcular el RMSE agrupado anual ($\sqrt{\frac{1}{D}\sum \text{RMSE}_d^2}$), los valores anuales varían entre **0.3034 °C (2018) y 0.3811 °C (2015)**, situándose 2024 en **0.3797 °C**.
  5. Como $\max(\text{Pooled Annual RMSE}) = 0.3811^\circ\text{C} > 0.3426^\circ\text{C} > \min(\text{Pooled Annual RMSE}) = 0.3034^\circ\text{C}$, la consistencia matemática es absoluta.
- **Resolución:** Se clarificó la distinción entre RMSE agrupado y promedio diario en todos los documentos. Estado: **RESOLVED & DOCUMENTED**.

---

## 9. Numerical Consistency Audit

Se auditaron 34 métricas clave en `tables/final_numerical_audit.csv`. Todas las métricas mostraron coincidencia exacta (`MATCH`) o diferencias atribuibles exclusivamente a redondeo (`ROUNDING_ONLY`). No quedó ningún conflicto numérico crítico sin resolver.

---

## 10. Claims Audit

Se auditaron 15 afirmaciones en `tables/final_claims_audit.csv`. Se eliminaron todas las afirmaciones causales o especulativas sin evidencia independiente (e.g. upwelling, MHW, noise floor, persistencia estacional espuria). Se sustituyeron por lenguaje rigurosamente descriptivo.

---

## 11. Methods Corrections

- Se corrigió la delimitación geográfica en `THESIS_METHODS_MASTER_A_D.md`, eliminando la mención errónea al Canal de Yucatán.
- Se aclaró la formulación residual como garantía algebraica ($\hat{R}=0 \implies 	ext{SST}_{	ext{downscaled}}=	ext{SST}_{	ext{BIL}}$).
- Se precisó el bootstrap como preservación de dependencia temporal de corto alcance.

---

## 12. Results Corrections

- Se actualizaron los umbrales de regímenes DEV canónicos (0.2066, 0.3604, 0.5377, 0.6652, 0.9659 °C).
- Se corrigió el desglose mensual a 9/12 en 2024 y 9/12 en 2025.
- Se integró la doble métrica de C.2 (pooled RMSE = 0.3426 °C; mean daily RMSE = 0.3019 °C).

---

## 13. Discussion Corrections

- Se reformuló la discusión de parsimonia de D32 para basarse exclusivamente en la falta de skill incremental.
- Se reformuló la respuesta al jurado sobre la varianza de OISST (competitividad del baseline bilineal).
- Se justificó la exclusión de CNNs mediante diagnósticos de predictibilidad y principios de parsimonia.

---

## 14. Limitations Corrections

- Se evitó atribuir la varianza residual no explicada exclusivamente a forzantes dinámicos faltantes.
- Se acotó la discusión de compresión de amplitud sin hacer afirmaciones sobre eventos térmicos extremos no evaluados.

---

## 15. Reproducibility Corrections

- Se incorporó el hash canónico de metadatos espaciales (`8cc0...`).
- Se declaró que el conjunto Test está permanentemente consumido y no puede usarse para nuevas pruebas confirmatorias.

---

## 16. Writing-Map Corrections

- Se añadieron referencias canónicas a GAP-09 a GAP-12 y la columna `CANONICAL STATUS` (`VERIFIED` / `RESOLVED`).

---

## 17. Remaining Unresolved Issues

**NINGUNO.** Todos los gaps detectados fueron formalmente resueltos con evidencia de procedencia canónica.

---

## 18. Final Documentation Status

$$\mathbf{DOCUMENTATION\_LAYER: FROZEN}$$

La capa documental `THESIS_MASTER_A_D` queda formalmente auditada, corregida y congelada como referencia canónica definitiva para la redacción de la tesis de maestría y publicaciones científicas.

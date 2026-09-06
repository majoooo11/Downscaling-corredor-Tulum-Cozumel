# Final Documentation Audit — Stages A–D
## Reconciliación, Corrección de Afirmaciones y Congelamiento de la Capa de Tesis

**Documento:** `FINAL_DOCUMENTATION_AUDIT_A_D.md`  
**Fecha de Ejecución:** 2026-09-05  
**Protocolo:** Evidence-First Final Documentation Audit  
**Estado:** AUDITED AND FROZEN  

---

## 1. Scope

Esta auditoría documental constituye la revisión correctiva final sobre la capa de documentación maestra para la tesis (`THESIS_MASTER_A_D`). Su alcance es estrictamente documental y de verificación de consistencia científica entre las etapas A, B, C y D del proyecto. Su objetivo prioritario es verificar cada discrepancia contra artefactos experimentales canónicos inmutables (manifests de congelamiento, logs de ejecución, tablas maestras CSV), corregir afirmaciones mecanicistas no demostradas y formalizar el congelamiento definitivo de la documentación de tesis.

---

## 2. Experimental layer status

Se ratifica que todos los estados experimentales del proyecto permanecen completamente inmutables:
- **Stage A:** COMPLETED
- **Stage B & B.1:** COMPLETED
- **Stage C (C.1, C.1b, C.1c, C.2):** COMPLETED
- **Stage D.1 (Dataset ML):** COMPLETED
- **Stage D31 (Predictibilidad):** CLOSED
- **Stage D32 (Selección de modelo):** METHODOLOGICALLY CLOSED
- **Stage D33 (Validación externa):** D33-B — UNCHANGED
- **Stage D34 (Diagnósticos & Microauditoría):** INTERPRETATIONALLY CLOSED
- **Stage D35 (Final Test 2024–2025):** D35-A — FINAL GENERALIZATION CONFIRMED
- **Stage D36 (Síntesis terminal):** FINAL SYNTHESIS COMPLETED
- **FINAL TEST:** CONSUMED
- **ML MODEL DEVELOPMENT:** CLOSED

Restricciones de blindaje observadas:
- Nuevos entrenamientos ejecutados: **0**
- Modelos modificados: **0**
- Hiperparámetros modificados: **0**
- Aperturas lógicas raw a `test_2024.parquet` o `test_2025.parquet`: **0 (`RAW_TEST_LOGICAL_LOAD_COUNT == 0`)**
- Resultados experimentales alterados: **NO**

---

## 3. Evidence-first protocol

Bajo el protocolo de auditoría basado en evidencia (*evidence-first*):
1. No se modificó ningún documento maestro hasta que cada discrepancia fue localizada, abierta y contrastada contra artefactos primarios del repositorio.
2. Se completó la Fase 1 generando el reporte `PRE_CORRECTION_EVIDENCE_AUDIT.md` y la tabla `pre_correction_conflict_evidence.csv`.
3. Al resolverse unívocamente los cuatro gaps (GAP-09, GAP-10, GAP-11 y GAP-12), se autorizó formalmente la Fase 2 (`PHASE 2 AUTHORIZED: YES`).
4. Se generaron copias de seguridad en `_archive_pre_final_audit/` antes de aplicar las correcciones sistemáticas.

---

## 4. GAP-09 resolution (Frozen DEV Residual Thresholds)

- **Conflicto:** Existencia de textos preliminares con percentiles aproximados ($0.177, 0.354, 0.536, 0.697, 1.054^\circ\text{C}$).
- **Evidencia canónica:** El freeze manifest inmutable pre-test `ml_results/E3b_D35_final_test/final_test_freeze_manifest.json` (congelado el 2026-09-05T20:51:10.734155 antes de abrir el test), el código de `fase_d35_final_test_c0.py` y la tabla `residual_regime_metrics.csv` demuestran que los umbrales DEV congelados pre-test son:
  - **DEV-P50:** **$0.2066^\circ\text{C}$**
  - **DEV-P75:** **$0.3604^\circ\text{C}$**
  - **DEV-P90:** **$0.5377^\circ\text{C}$**
  - **DEV-P95:** **$0.6652^\circ\text{C}$**
  - **DEV-P99:** **$0.9659^\circ\text{C}$**
- **Resolución:** Se ratificaron estos valores como canónicos absolutos. Se corrigieron todos los documentos maestros para utilizar únicamente estos valores. Estado: **RESOLVED & DOCUMENTED**.

---

## 5. GAP-10 resolution (Monthly Improvement Allocation)

- **Conflicto:** Discrepancia sobre la distribución anual de los 18 meses con mejora entre 2024 y 2025 (posibilidad A: 12/12 y 6/12; posibilidad B: 11/12 y 7/12).
- **Evidencia canónica:** La evaluación directa de la columna `DeltaRMSE < 0` en `ml_results/E3b_D35_final_test/tables/monthly_metrics.csv` demuestra:
  - **Año 2024:** Exactamente **9 de 12 meses mejorados** (75.0%). Meses no mejorados: `2024-01` ($+0.0019^\circ\text{C}$), `2024-04` ($+0.0023^\circ\text{C}$), `2024-10` ($+0.0043^\circ\text{C}$).
  - **Año 2025:** Exactamente **9 de 12 meses mejorados** (75.0%). Meses no mejorados: `2025-09` ($+0.0127^\circ\text{C}$), `2025-10` ($+0.0323^\circ\text{C}$), `2025-12` ($+0.0250^\circ\text{C}$).
  - **Total:** Exactamente **18 de 24 meses mejorados** ($9 + 9 = 18$, **75.0%** de consistencia temporal en ambos años).
- **Resolución:** Se descartaron formalmente las hipótesis de 11/12 y 7/12. Se unificó el desglose canónico a **9/12 en 2024 y 9/12 en 2025**. Estado: **RESOLVED & DOCUMENTED**.

---

## 6. GAP-11 resolution (Frozen Spatial Metadata SHA256)

- **Conflicto:** Discrepancia entre el hash `a93b4554...` citado en borradores y el hash del manifiesto inmutable pre-test.
- **Evidencia canónica:** El cómputo criptográfico directo sobre el archivo físico `frozen_spatial_metadata.csv` (185,339 bytes, 5,275 filas) y la verificación de `final_test_freeze_manifest.json` arrojan exactamente:
  $$\mathbf{SHA\text{-}256: 8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365}$$
  Se verificó que `a93b...` provino de una asignación en texto plano de un script generador de resúmenes y no correspondió a ningún artefacto físico diferente.
- **Resolución:** Se ratificó el hash `8cc0...` en `REPRODUCIBILITY_MAP_A_D.md`. Estado: **RESOLVED & DOCUMENTED**.

---

## 7. GAP-12 resolution (C2 Global RMSE vs Annual RMSE Consistency)

- **Conflicto:** Paradoja aparente donde el RMSE global decenal reportado ($0.3426^\circ\text{C}$) superaba al supuesto máximo anual ($0.3257^\circ\text{C}$).
- **Evidencia canónica y matemática:**
  1. $0.3426^\circ\text{C}$ ($0.342596^\circ\text{C}$) es el **RMSE espacio-temporal agrupado (*Pooled Spatiotemporal RMSE*)** sobre la totalidad de los 21,211,022 puntos: $\sqrt{\frac{1}{N}\sum (T_{\text{BIL}} - T_{\text{MUR}})^2}$.
  2. Los valores anuales reportados en la tabla de `fase_c2_reporte.md` (0.2724 a 0.3257 °C, promedio 0.3019 °C) corresponden al **promedio temporal de los RMSEs diarios** ($\text{mean}(\text{RMSE}_d)$).
  3. Por la desigualdad de Jensen, $\text{mean}(\text{RMSE}_d) \le \sqrt{\text{mean}(\text{RMSE}_d^2)}$.
  4. Al calcular el **RMSE agrupado anual** ($\sqrt{\frac{1}{D}\sum \text{RMSE}_d^2}$), los valores oscilan entre **$0.3034^\circ\text{C}$ (2018)** y **$0.3811^\circ\text{C}$ (2015)**, situándose 2024 en **$0.3797^\circ\text{C}$**.
  5. Como $\max(\text{Pooled Annual RMSE}) = 0.3811^\circ\text{C} > 0.3426^\circ\text{C} > \min(\text{Pooled Annual RMSE}) = 0.3034^\circ\text{C}$, la coherencia matemática es perfecta.
- **Resolución:** Se clarificó la distinción entre RMSE agrupado y promedio diario en todos los documentos maestros. Estado: **RESOLVED & DOCUMENTED**.

---

## 8. Numerical consistency

Se auditaron 35 métricas clave en `tables/final_numerical_audit.csv` cubriendo las Etapas A a D35.
- Estado de métricas:
  - `MATCH`: 22 métricas.
  - `ROUNDING_ONLY`: 7 métricas (discrepancias menores a $10^{-4}$ por decimales de redondeo).
  - `CORRECTED`: 6 métricas (GAP-09 thresholds, GAP-10 monthly allocation, GAP-11 hash).
  - `UNRESOLVED`: **0**.
- Criterio de consistencia numérica: **PASS**.

---

## 9. Claims audit

Se auditaron 15 aseveraciones en `tables/final_claims_audit.csv` evaluando la presencia de terminología mecanicista, causal o física no fundamentada empíricamente:
- Se eliminaron afirmaciones de "ruido radiométrico de sensor", "piso de ruido" y "gradientes físicos verdaderos", sustituyéndolas por "régimen de baja discrepancia MUR–BIL" y "discrepancias mayores MUR–BIL".
- Se eliminó la afirmación causal de que los lags o vecindades espaciales introducen "persistencia espuria" o "ruido de frontera".
- Se corrigió la atribución causal de la batimetría y distancia a la costa, definiéndolas como asociaciones descriptivas de rango no paramétrico.
- Se eliminó la afirmación de que el modelo opera exclusivamente sobre el 10.7% restante de la varianza de OISST.
- Criterio de auditoría de afirmaciones: **PASS**.

---

## 10. Methods corrections

En `THESIS_METHODS_MASTER_A_D.md`:
1. **Área de estudio:** Se eliminó la mención al Canal de Yucatán, delimitando el dominio al sector Tulum–Cozumel, canal de Cozumel y Caribe mexicano.
2. **Formulación residual:** Se sustituyó la afirmación de convergencia teórica por la identidad algebraica estricta: $\hat{R} = 0 \implies \text{SST}_{\text{downscaled}} = \text{SST}_{\text{BIL}}$.
3. **Predictibilidad D31:** Se sustituyó "estructura determinística" por "estructura predictiva reproducible frente a un baseline nulo o de no-habilidad".
4. **Bootstrap D35:** Se sustituyó "dependencia de mesoescala" por "dependencia temporal de corto alcance (*short-range temporal dependence*) bajo la longitud de bloque evaluada".
5. **Final Test:** Se precisó que el periodo 2024–2025 se mantuvo ciego hasta D35 y se declaró en estado actual como `TEST CONSUMED`.

---

## 11. Results corrections

En `THESIS_RESULTS_MASTER_A_D.md`:
1. **Armonización C.2:** Se integró formalmente la doble métrica de RMSE decenal (pooled = 0.3426 °C; promedio diario = 0.3019 °C; pooled anual entre 0.3034 y 0.3811 °C).
2. **Predictibilidad D31:** Lenguaje ajustado a "reproducible predictive structure".
3. **Robustez temporal D35:** Desglose anual fijado en 9/12 en 2024 (75.0%) y 9/12 en 2025 (75.0%).
4. **Regímenes residuales D35:** Umbrales DEV actualizados a 0.2066, 0.3604, 0.5377, 0.6652, 0.9659 °C.
5. **Inferencia bootstrap D35:** Intervalo $[-0.0425, -0.0098]^\circ\text{C}$ descrito como evidencia de robustez ante dependencia serial de corto alcance.
6. **Asociación espacial:** Interpretación descriptiva del signo positivo de Spearman ($\Delta\text{RMSE}$ menos negativo mar adentro).
7. **Dictamen terminal:** Ratificación de `D35-A — FINAL GENERALIZATION CONFIRMED`.

---

## 12. Discussion corrections

En `THESIS_DISCUSSION_NOTES_A_D.md`:
1. **Parsimonia D32:** Redacción basada en la ausencia de skill incremental de las extensiones complejas frente a E3b-C0, sin postular mecanismos no aislados.
2. **Regímenes de discrepancia:** Análisis centrado en margen de corrección y exactitud de signo.
3. **Preguntas al jurado:**
   - Reducción del 7.22% explicada por la fortaleza del baseline bilineal ($R^2 \approx 0.893$).
   - Ausencia de CNNs fundamentada en diagnósticos de predictibilidad tabular D31, desempeño en ablaciones y parsimonia metodológica.

---

## 13. Limitations corrections

En `THESIS_LIMITATIONS_A_D.md`:
1. **Varianza residual no explicada:** Atribuida con cautela a diferencias entre productos, incertidumbre de recuperación y procesos dinámicos no observados.
2. **Compresión de amplitud:** Explicada como regresión a la media condicional bajo pérdida cuadrática regularizada ($\text{std}(\hat{R})/\text{std}(R) \approx 0.25$).
3. **Extremos térmicos:** Descrita como subrepresentación de amplitud residual sin aseveraciones categóricas sobre eventos térmicos extremos no evaluados con métricas específicas.
4. **Variabilidad mensual:** Caracterizada como no uniformidad temporal del skill a escala de mes.

---

## 14. Reproducibility corrections

En `REPRODUCIBILITY_MAP_A_D.md`:
1. **Hash espacial:** Actualizado al valor canónico pre-test `8cc02f86b14d5bc0bd255bbe896b3d237df5fbf219ce3d1a5eec4283d0a1e365`.
2. **Reproducibilidad post-consumo:** Declarado formalmente que cualquier re-ejecución técnica de D35 constituye una corrida de reproducibilidad post-consumo y no un nuevo blind test.
3. **Estado de capas:**
   - `EXPERIMENTAL LAYER: CLOSED`
   - `FINAL TEST: CONSUMED`
   - `DOCUMENTATION LAYER: AUDITED AND FROZEN`

---

## 15. Writing map corrections

En `THESIS_WRITING_MAP.md`:
1. Se añadió la columna `CANONICAL STATUS` (`VERIFIED`, `RESOLVED`, `HISTORICAL`, `DO NOT USE`).
2. Se vincularon explícitamente las fuentes canónicas para thresholds, asignación mensual, hash espacial y RMSE de C2.
3. Se marcaron borradores antiguos con valores preliminares como `DO NOT USE`.

---

## 16. Remaining unresolved issues

**NINGUNO.**
- GAP-09: RESOLVED
- GAP-10: RESOLVED
- GAP-11: RESOLVED
- GAP-12: RESOLVED
- Conflictos numéricos críticos: 0
- Afirmaciones canónicas sin sustento: 0

---

## 17. Freeze decision

$$\mathbf{THESIS\ MASTER\ A\text{–}D:\ FROZEN}$$

Se declara formalmente que la capa documental `THESIS_MASTER_A_D` está completamente reconciliada, auditada, basada en evidencia experimental canónica y **CONGELADA** como referencia oficial para la redacción de la tesis de maestría y artículos científicos derivados.

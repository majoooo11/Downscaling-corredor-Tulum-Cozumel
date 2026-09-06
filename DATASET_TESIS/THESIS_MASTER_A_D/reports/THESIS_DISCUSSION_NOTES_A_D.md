# Master Thesis Discussion Notes (Stages A to D)
## Estructura de Argumentación Científica para Capítulos de Discusión

**Documento:** `THESIS_DISCUSSION_NOTES_A_D.md`  
**Estado:** AUDITED & FROZEN  
**Propósito:** Proveer la base conceptual, analítica y crítica para la redacción posterior del capítulo de discusión de la tesis y artículos derivados, clasificando estrictamente las aseveraciones según su nivel de sustento probatorio.

---

## 1. Clasificación Epistemológica de Afirmaciones

Para garantizar rigor académico y evitar sobreinterpretaciones, los temas de discusión se dividen en tres categorías formales:
- **CATEGORY I: SUPPORTED DIRECTLY BY PROJECT DATA (Sustento Empírico Directo):** Afirmaciones respaldadas de forma unívoca por las tablas maestras, logs y scripts congelados del repositorio.
- **CATEGORY II: REQUIRES LITERATURE SUPPORT (Requiere Soporte Bibliográfico Externo):** Decisiones de diseño estándar o interpretaciones metodológicas que deben ser justificadas citando literatura oceanográfica o de aprendizaje automático.
- **CATEGORY III: SPECULATIVE / DO NOT CLAIM (Especulativo — Prohibido Afirmar como Hecho Demostrado):** Hipótesis físicas o mecanicistas plausibles pero que no cuentan con corroboración empírica independiente dentro de los datos del proyecto.

---

## 2. Category I: Findings Supported Directly by Project Data

1. **Superioridad del Modelo Parsimonioso (`E3b-C0`):**
   - *Evidencia Canónica:* En Diagnostic Holdout 2021, la formulación parsimoniosa `E3b-C0` (4 features: `sst_bil`, `doy_sin`, `doy_cos`, `depth`) logró la mayor reducción relativa del error (+2.8424% de RMSE). Las extensiones temporales y espaciales evaluadas (`E3b-T1`, `E3b-T3`, `E3b-S`, `E3b-TS`, `E3b-ALL`) obtuvieron mejoras inferiores (+2.21% a +2.63%).
   - *Interpretación Empírica:* The evaluated temporal-lag and spatial-neighborhood extensions did not provide incremental predictive skill relative to `E3b-C0`. The specific mechanism responsible for their lower performance was not isolated within the experimental protocol.

2. **Compresión de Amplitud (*Amplitude Compression*):**
   - *Evidencia Canónica:* En Final Test 2024–2025, la relación de desviaciones estándar fue $\text{std}(\hat{R}) / \text{std}(R) = 0.2536$, y la regresión de calibración $\hat{R} = a + b R$ arrojó pendiente $b = 0.0891$ e intercepto $a = -0.0427^\circ\text{C}$.
   - *Interpretación Empírica:* The fitted model exhibited pronounced amplitude compression, a pattern consistent with regression toward the conditional mean under the selected regularized MSE-based formulation.

3. **Dependencia Crítica del Régimen de Discrepancia (*Regime Dependence*):**
   - *Evidencia Canónica:* En el régimen de baja discrepancia (DEV-P0–P50, $|R| < 0.2066^\circ\text{C}$), el RMSE de `C0` aumentó en un -20.99% respecto a `B0`. En cambio, en regímenes de discrepancia mayor (P75–P99+), `C0` redujo el RMSE entre +7.48% y +12.07%, alcanzando exactitudes de signo de 70.9% a 87.2%.
   - *Interpretación Empírica:* When the MUR–BIL discrepancy was small, there was less margin for a beneficial correction and lower residual-sign agreement was associated with relative degradation. Sign agreement and RMSE improvement increased across larger MUR–BIL discrepancy regimes.

4. **Amplitud y Asociación Espacial del Desempeño:**
   - *Evidencia Canónica:* El 99.96% de las celdas marinas evaluadas (5,273 de 5,275) mejoraron su RMSE. Las correlaciones de rangos de Spearman entre $\Delta\text{RMSE}$ por celda y las covariables fueron positivas: $\rho = +0.7376$ con profundidad y $\rho = +0.5336$ con distancia a la costa.
   - *Interpretación Empírica:* The magnitude of improvement decreased toward deeper and more offshore cells (since $\Delta\text{RMSE}$ became less negative). These rank associations are descriptive and do not establish physical causality.

---

## 3. Category II: Decisions Requiring Literature Support

1. **Uso de MUR SST como Verdad de Referencia Operacional:**
   - *Citar:* Chin et al. (2017), Armstrong et al. (2012). Justificar que los productos L4 fusionados multiescala son la referencia estándar operativa en regiones tropicales sin cobertura de boyas in situ densa, reconociendo que no constituyen ground truth absoluta.
2. **Formulación de Aprendizaje Residual vs Downscaling Directo:**
   - *Citar:* He et al. (2016) para fundamentos de residual learning; Cyriac et al. (2025) para downscaling de SST. Justificar la identidad aditiva y la conservación del baseline bilineal cuando $\hat{R} = 0$.
3. **Partición Cronológica Estricta para Prevenir Fugas de Información:**
   - *Citar:* Roberts et al. (2017) sobre estrategias de validación en datos espaciotemporales con fuerte autocorrelación serial.
4. **Validación Infrarroja Bloqueada por Cobertura Nubosa:**
   - *Citar:* Kilpatrick et al. (2015), Minnett et al. (2019) sobre las limitaciones físicas inherentes a la radiometría térmica satelital en el Caribe noroccidental.

---

## 4. Category III: Speculative Claims — Prohibido Afirmar como Hechos Demostrados

1. **PROHIBIDO:** Atribuir la discrepancia MUR–BIL o la corrección de `E3b-C0` a "surgencias costeras (*upwelling*)", "frentes térmicos de mesoescala", "ondas de calor marinas (MHW)", o "mezcla por vientos del Norte", salvo que se incorporen datos oceanográficos in situ independientes.
   - *Fraseo correcto:* "Asociaciones estadísticas consistentes con gradientes térmicos locales cercanos a la costa".
2. **PROHIBIDO:** Afirmar que el régimen bajo ($|R| < 0.2066^\circ\text{C}$) representa estrictamente el "piso de ruido del sensor (*noise floor*)".
   - *Fraseo correcto:* "Régimen de discrepancia reducida MUR–BIL (*small discrepancy regime*)".
3. **PROHIBIDO:** Afirmar que el modelo realizó "reconstrucción de dinámicas submesoescala".
   - *Fraseo correcto:* "Reducción reproducible del error de reconstrucción de SST frente a interpolación bilineal a escala de ~1 km".

---

## 5. Preguntas Clave para el Jurado de Tesis y Respuestas Metodológicas

1. **¿Por qué la mejora global en RMSE es de ~7.2% y no de magnitudes superiores?**
   - *Respuesta Canónica:* The bilinear baseline already showed a high level of agreement with MUR ($R^2 \approx 0.893$), so the residual model was evaluated against a strong baseline rather than against an uninformative predictor. The final 7.22% RMSE reduction therefore represents incremental improvement over an already competitive baseline across 3.86 million out-of-sample observations.
2. **¿Por qué empeora el modelo en el régimen de baja discrepancia ($|R| < 0.2066^\circ\text{C}$)?**
   - *Respuesta Canónica:* When the baseline-reference discrepancy is already small (mean of $0.099^\circ\text{C}$), the baseline is already accurate ($B_0 \text{ RMSE} = 0.115^\circ\text{C}$). Under regularized MSE loss, small sign errors in residual prediction penalize the squared metric, causing minor overcorrections where little learnable signal exists.
3. **¿Por qué no se utilizaron redes neuronales convolucionales (CNN)?**
   - *Respuesta Canónica:* CNNs were not pursued because the prespecified predictability diagnostics and tabular ablation results supported a parsimonious tabular formulation, and the incremental value of engineered spatial information did not exceed the predefined threshold required to justify substantially greater model complexity.

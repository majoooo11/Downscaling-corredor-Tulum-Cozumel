# Master Inventory of Scientific Limitations (Stages A to D)
## Registro Sistemático de Limitaciones Metodológicas para Tesis

**Documento:** `THESIS_LIMITATIONS_A_D.md`  
**Estado:** AUDITED & FROZEN  
**Propósito:** Agrupar con transparencia y rigor científico todas las limitaciones inherentes a los datos, métodos, suposiciones y modelos evaluados a lo largo de las Etapas A, B, C y D.

---

## 1. Stage A: Data-Product Limitations

1. **Falta de Ground Truth Absoluto In Situ:** El proyecto carece de una red densa de boyas oceanográficas fijas o derivadoras (*drifters*) con cobertura diaria continua en el corredor Tulum–Cozumel. MUR SST actúa como la referencia de mayor resolución disponible (~1 km), pero es a su vez un producto L4 interpolado multiescala sujeto a sus propios algoritmos de asimilación.
2. **Inconsistencias Radiométricas en Eventos Nubosos:** En situaciones de nubosidad persistente asociada a sistemas tropicales, las observaciones infrarrojas directas quedan enmascaradas, provocando que MUR dependa de microondas (~25 km) o de persistencia temporal, incrementando su incertidumbre interna (`analysis_error`).

---

## 2. Stage B: Spatial Framework and Masking Limitations

1. **Resolución Discreta de la Fracción Oceánica:** La máscara oceánica corregida empleó un umbral binario $\text{ocean\_fraction} \ge 0.5$. Las celdas con fracción mixta entre 50% y 99% contienen tierra residual sub-píxel que puede introducir sesgos leves en la radiometría costera.
2. **Suavizado Batimétrico en Cañones Estrechos:** Aunque GEBCO 2026 Grid provee celdas a 15 arc-segundos, la topografía marina submarina en el canal de Cozumel contiene pendientes verticales abruptas que quedan suavizadas al promediarse a la resolución de 0.01° de la cuadrícula maestra.

---

## 3. Stage C: Interpolation and Harmonization Limitations

1. **Naturaleza Matemática del Soporte Costero (Estrategia A):** Los 20 nodos extendidos de OISST sobre la Península de Yucatán no representan temperatura superficial terrestre real; funcionan exclusivamente como soporte matemático para permitir la interpolación bilineal en la costa. Aunque el producto final se recorta con la máscara oceánica, la solución en celdas a <10 km de la costa está condicionada por la técnica de extensión utilizada.
2. **Pérdida de Información por Bloqueo Nuboso en Validación Satelital:** La auditoría con radiómetros L2P (VIIRS y MODIS) durante eventos de discrepancia extrema resultó inconclusa en más del 90% de los pasos orbitales debido al bloqueo por nubes ($QL < 5$), impidiendo una validación radiométrica cruzada cuantitativa en los picos térmicos más agudos.
3. **Comportamiento Asintótico de MUR Analysis Error:** La variable de incertidumbre de MUR satura en un techo fijo de $0.4100^\circ\text{C}$ durante vacíos observacionales prolongados, lo cual impide discriminar gradaciones de incertidumbre por encima de dicho umbral.

---

## 4. Stage D: Machine Learning Model Limitations

1. **Explicabilidad Parcial de la Varianza Residual ($R^2_{\text{residual}} = 11.22\%$):** Most residual variance remained unexplained by the selected predictor set. Potential contributors include unrepresented dynamic processes, product differences, retrieval uncertainty, and variability not captured by the available predictors.
2. **Compresión de Amplitud de Predicción:** The fitted model exhibited pronounced amplitude compression ($\text{std}(\hat{R})/\text{std}(R) \approx 0.25$), consistent with regression toward the conditional mean under the selected regularized MSE formulation. The model underrepresented residual amplitude, which may limit its ability to reproduce large instantaneous MUR–BIL discrepancies.
3. **Degradación en Regímenes de Baja Discrepancia:** En el 48.2% de los datos donde OISST y MUR difieren en menos de $0.2066^\circ\text{C}$ (DEV-P0–P50), el modelo introduce sobrecorrecciones leves que incrementan el RMSE en un 20.99%, careciendo de un mecanismo intrínseco de abstención o umbralización adaptable.
4. **Variabilidad Mensual del Skill:** Month-level skill remained variable, indicating that aggregate final-test improvement was not temporally uniform (9 de 12 meses mejorados en 2024; 9 de 12 meses mejorados en 2025).
5. **Ausencia de Covariables Atmosféricas y de Corrientes:** El modelo no incluye velocidad ni dirección del viento (e.g. ERA5), radiación solar incidente, ni velocidad de corrientes marinas (CMEMS), limitando su capacidad para modelar fenómenos advectivos o eventos de mezcla inducidos por viento.

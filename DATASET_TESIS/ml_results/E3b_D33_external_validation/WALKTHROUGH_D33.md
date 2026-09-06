# Walkthrough — Fase D.3.3: External Validation (E3b-C0)

## Resumen Ejecutivo
Se completó de forma rigurosa y reproducible la Fase D.3.3 de Validación Externa del modelo tabular parsimonioso `E3b-C0` sobre el periodo independiente 2022–2023.

### Estado del Blindaje de Datos
- **Archivos de VALIDATION abiertos:** **2** (`validation_2022.parquet`, `validation_2023.parquet`)
- **Archivos de TEST abiertos:** **0** (Blindaje estricto 2024–2025 preservado)

### Resultados Clave (2022–2023 Combinado)
- **B0 RMSE:** 0.335666 °C
- **E3b-C0 RMSE:** 0.323838 °C
- **Mejora RMSE vs B0:** **+3.5237%**
- **B0 MAE:** 0.263594 °C | **C0 MAE:** 0.254457 °C
- **Mejora 2022:** +2.2375% | **Mejora 2023:** +4.6298%
- **Meses mejorados:** 16/24 (66.7%)
- **Días mejorados:** 478/730 (65.5%)
- **Celdas mejoradas:** 4755/5275 (90.1%)
- **Bootstrap IC 95% $\Delta\text{RMSE}$:** [-0.016446 °C, -0.007224 °C]
- **Probabilidad $\Delta\text{RMSE} < 0$:** 1.0000
- **Dictamen D33:** **D33-B** (Positive but partial or unstable external generalization.)
- **Recomendación:** **HOLD FINAL TEST**

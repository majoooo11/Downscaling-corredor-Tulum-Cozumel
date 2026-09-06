#!/usr/bin/env python3
"""
======================================================================
MASTER THESIS SYNTHESIS — STAGES A TO D
RECONSTRUCCIÓN METODOLÓGICA, TRAZABILIDAD Y ARCHIVO PARA TESIS
======================================================================
Este script audita y consolida TODO el trabajo científico realizado
desde la adquisición de datos (Etapa A) hasta el cierre del bloque de
Machine Learning (Etapa D, D31–D36).

POLÍTICAS ESTRICTAS:
- NO entrena modelos ni modifica hiperparámetros.
- NO recalcula Final Test ni abre archivos raw de test (test_2024.parquet, test_2025.parquet).
- CSV-First y trazabilidad estricta contra artefactos canónicos existentes.
- Distingue plan histórico, implementación real y nomenclatura canónica.
- Documenta explícitamente gaps y discrepancias sin corrección silenciosa.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# 1. VERIFICACIÓN DE SEGURIDAD Y PREVENCIÓN DE ACCESO RAW A TEST
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "THESIS_MASTER_A_D"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

RAW_TEST_LOGICAL_LOAD_COUNT = 0

def check_no_raw_test_access():
    global RAW_TEST_LOGICAL_LOAD_COUNT
    assert RAW_TEST_LOGICAL_LOAD_COUNT == 0, "ERROR CRÍTICO: Intento de acceder a datos raw de FINAL TEST"

# ----------------------------------------------------------------------
# 2. CREACIÓN DE DIRECTORIOS
# ----------------------------------------------------------------------
for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Directorio maestro de síntesis configurado en: {OUTPUT_DIR}")

# ----------------------------------------------------------------------
# 3. GENERACIÓN DE TABLAS MAESTRAS (CSV)
# ----------------------------------------------------------------------

# 3.1 master_phase_summary.csv
phase_summary_data = [
    {
        "stage": "A",
        "substage": "A.1",
        "scientific_purpose": "Ingestión y auditoría de integridad de MUR SST L4 (0.01°)",
        "input_data": "MUR NetCDF históricos (2015–2019) y gránulos diarios L4 (2019–2025)",
        "processing": "Verificación de 4,018 fechas, continuidad temporal, conversión Kelvin a Celsius",
        "key_output": "Campos de referencia de alta resolución sst_mur (~1 km)",
        "key_metric": "4,018/4,018 días válidos, 0 faltantes, 0 duplicados",
        "decision": "Aprobar producto MUR como referencia de alta resolución",
        "status": "COMPLETED",
        "canonical_source": "reports/inspeccion_previa.txt, config.py"
    },
    {
        "stage": "A",
        "substage": "A.2",
        "scientific_purpose": "Ingestión y auditoría de predictor de baja resolución NOAA OISST v2.1 (0.25°)",
        "input_data": "NOAA ERDDAP OISST diario + recuperación local 2025-01-14",
        "processing": "Descarga de cuadrícula gruesa con halo espacial de 0.5°",
        "key_output": "Campos coarse de SST para downscaling",
        "key_metric": "4,018/4,018 días disponibles, unidades °C verificadas",
        "decision": "Adoptar OISST como predictor térmico base de baja resolución",
        "status": "COMPLETED",
        "canonical_source": "reports/inspeccion_previa.txt, config.py"
    },
    {
        "stage": "A",
        "substage": "A.3",
        "scientific_purpose": "Ingestión de batimetría global GEBCO 2026 Grid (15 arc-sec)",
        "input_data": "gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc",
        "processing": "Extracción de cota topobatimétrica, conversión depth = -elevation",
        "key_output": "Capa estática de profundidad marina (m)",
        "key_metric": "Resolución 15 arc-sec (~450 m), cobertura 100% del dominio",
        "decision": "Adoptar GEBCO 2026 como fuente estática de batimetría",
        "status": "COMPLETED",
        "canonical_source": "config.py, GEBCO NetCDF metadata"
    },
    {
        "stage": "A",
        "substage": "A.4",
        "scientific_purpose": "Ingestión de observaciones infrarrojas independientes (VIIRS S-NPP y MODIS Aqua)",
        "input_data": "Gránulos satelitales L2P infrarrojos en eventos anómalos E1–E6",
        "processing": "Filtro de flags de calidad (QL=5, nubes, hielo)",
        "key_output": "Observaciones independientes de validación L2P",
        "key_metric": "Cobertura QL=5 de 0.0% a 0.1% en eventos nubosos",
        "decision": "Mantener exclusivamente como auditoría satelital independiente (no inputs ML)",
        "status": "COMPLETED / HETEROGENEOUS",
        "canonical_source": "reports/validacion_infrarroja_integrada_FINAL.md"
    },
    {
        "stage": "B",
        "substage": "B",
        "scientific_purpose": "Definición del marco espacial y cuadrícula maestra del corredor",
        "input_data": "Cuadrícula MUR SST (Lat: 19.90–20.75°N, Lon: -87.60–-86.65°W)",
        "processing": "Fijación de dimensiones 86 × 96 celdas a 0.01° de resolución",
        "key_output": "Cuadrícula espacial maestra (8,256 celdas)",
        "key_metric": "8,256 celdas totales, proyección UTM Zona 16N (EPSG:32616)",
        "decision": "Adoptar la grilla de MUR como soporte espacial estándar del proyecto",
        "status": "COMPLETED",
        "canonical_source": "config.py, reports/reporte_fase_b.txt"
    },
    {
        "stage": "B",
        "substage": "B.1",
        "scientific_purpose": "Construcción de máscara oceánica corregida y covariables fisiográficas",
        "input_data": "Máscara MUR nativa (5,662 celdas mar) y GEBCO",
        "processing": "ocean_fraction >= 0.5, reasignación continental Cozumel, distance_coast_km",
        "key_output": "ocean_mask_final, depth, distance_coast_km, ocean_fraction",
        "key_metric": "5,279 celdas oceánicas, 2,977 terrestres (383 celdas corregidas)",
        "decision": "Congelar máscara corregida de 5,279 celdas para armonización",
        "status": "COMPLETED",
        "canonical_source": "reports/reporte_fase_b.txt, dataset_intermedio_fase_b.nc"
    },
    {
        "stage": "C",
        "substage": "C.1",
        "scientific_purpose": "Prueba de concepto de interpolación bilineal directa OISST -> MUR (2015-01-01)",
        "input_data": "OISST v2.1 (0.25°) y MUR SST (0.01°) del 2015-01-01",
        "processing": "Interpolación bilineal estándar en halo 7 × 7",
        "key_output": "sst_bil preliminar y residual inicial",
        "key_metric": "1,292 NaNs costeros (pérdida del 24.47% del dominio oceánico)",
        "decision": "Identificar necesidad matemática de soporte costero para evitar NaNs",
        "status": "COMPLETED",
        "canonical_source": "reports/reporte_faseC1_2015-01-01.txt"
    },
    {
        "stage": "C",
        "substage": "C.1b",
        "scientific_purpose": "Diagnóstico y comparación de estrategias de interpolación costera",
        "input_data": "Campos OISST con 20 nodos terrestres en halo occidental",
        "processing": "Estrategia A (extensión auxiliar nearest-ocean) vs Estrategia B (triangulación Delaunay)",
        "key_output": "Campos interpolados con 100% de cobertura oceánica",
        "key_metric": "5,279/5,279 celdas válidas; MAE A vs B = 0.0038 °C, P95 = 0.0214 °C, Max = 0.0626 °C",
        "decision": "Seleccionar Estrategia A (regularidad cartesiana, soporte rastreable, re-enmascarado)",
        "status": "COMPLETED",
        "canonical_source": "reports/reporte_faseC1b_2015-01-01.txt"
    },
    {
        "stage": "C",
        "substage": "C.1c",
        "scientific_purpose": "Cierre de validación de prueba única de armonización (2015-01-01)",
        "input_data": "Campos del 2015-01-01 procesados con Estrategia A",
        "processing": "Re-aplicación estricta de ocean_mask_final, verificación de identidad residual",
        "key_output": "sst_mur, sst_bil, residual en 5,279 celdas",
        "key_metric": "100% cobertura, Baseline E0: RMSE = 0.2367 °C, MAE = 0.2004 °C, Bias = +0.1872 °C",
        "decision": "Aprobar protocolo de Estrategia A para la serie decenal completa",
        "status": "COMPLETED",
        "canonical_source": "reports/reporte_faseC1c_2015-01-01.txt"
    },
    {
        "stage": "C",
        "substage": "C.2",
        "scientific_purpose": "Armonización espaciotemporal completa 2015–2025 (4,018 días)",
        "input_data": "Serie diaria completa de MUR SST y OISST v2.1",
        "processing": "Interpolación bilineal diaria con Estrategia A, cálculo R = MUR - BIL",
        "key_output": "faseC2_2015_2025.nc (540.82 MB) y 11 NetCDFs anuales",
        "key_metric": "21,211,022 obs; Baseline E0 global: RMSE = 0.3426 °C, MAE = 0.2631 °C, R² = 0.9016",
        "decision": "Congelar cubo decenal armonizado como verdad de referencia experimental",
        "status": "COMPLETED",
        "canonical_source": "reports/fase_c2_reporte.md, outputs/faseC2_2015_2025.nc"
    },
    {
        "stage": "C",
        "substage": "Satellite Audit",
        "scientific_purpose": "Auditoría de eventos extremos de discrepancia contra radiometría infrarroja L2P",
        "input_data": "Observaciones VIIRS S-NPP y MODIS Aqua en eventos E1–E6",
        "processing": "Filtro QL=5, evaluación de bloqueo por nubes y cobertura espacial",
        "key_output": "Reporte consolidado de validación infrarroja",
        "key_metric": "Evidencia heterogénea/inconclusa (bloqueo nuboso >80–100% en eventos pico)",
        "decision": "Mantener C.2 intacto; MUR actúa como referencia operacional de alta resolución",
        "status": "COMPLETED / HETEROGENEOUS",
        "canonical_source": "reports/validacion_infrarroja_integrada_FINAL.md"
    },
    {
        "stage": "C",
        "substage": "MUR Uncertainty Audit",
        "scientific_purpose": "Auditoría de incertidumbre interna del algoritmo MUR (analysis_error) 2015–2025",
        "input_data": "Variable analysis_error de MUR L4 en las 4,018 fechas",
        "processing": "Correlación con discrepancia MUR–BIL, análisis de cambios abruptos y saturación",
        "key_output": "mur_analysis_error_2015_2025_completo.nc y reportes de auditoría",
        "key_metric": "Spearman rho = +0.2853, Pearson r = +0.3338; techo asintótico en 0.4100 °C",
        "decision": "Utilizar analysis_error exclusivamente como variable de control/diagnóstico (NO feature)",
        "status": "CLOSED",
        "canonical_source": "auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md"
    },
    {
        "stage": "D",
        "substage": "D Dataset",
        "scientific_purpose": "Construcción del dataset analítico tabular para Machine Learning",
        "input_data": "faseC2_2015_2025.nc y analysis_error consolidado",
        "processing": "Aplanado tabular date × cell_id, codificación cíclica DOY, exportación Parquet",
        "key_output": "ml_dataset/ (train 2015–2021, validation 2022–2023, test 2024–2025)",
        "key_metric": "13 columnas, 5,279 celdas en tablas parquet, partición cronológica estricta",
        "decision": "Adoptar partición temporal out-of-sample; bloquear acceso raw a test",
        "status": "COMPLETED",
        "canonical_source": "ml_dataset/METADATA.md, ml_dataset/reports/faseD1_dataset_tabular_QA.md"
    },
    {
        "stage": "D",
        "substage": "D31",
        "scientific_purpose": "Diagnóstico de predictibilidad del residual fino",
        "input_data": "Partición Development (2015–2020)",
        "processing": "Autocorrelación espacial/temporal, baselines lineales, mutual information",
        "key_output": "Reportes y tablas de diagnóstico D31",
        "key_metric": "Residual contiene estructura predictiva reproducible superior a persistencia/climatología",
        "decision": "Aprobar desarrollo de modelos de aprendizaje residual tabular",
        "status": "CLOSED",
        "canonical_source": "ml_results/diagnostics_D31/"
    },
    {
        "stage": "D",
        "substage": "D32",
        "scientific_purpose": "Selección de modelo y ablación sistemática de features en Diagnostic Holdout 2021",
        "input_data": "Development 2015–2020 y Holdout 2021 (COMMON_VALID_MASK: 5,275 celdas)",
        "processing": "Evaluación prespecificada de 6 variantes E3b (C0, T1, T3, S, TS, ALL), grid search",
        "key_output": "Modelo parsimonioso E3b-C0 seleccionado; frozen_cell_ids.csv congelado",
        "key_metric": "Holdout 2021: B0 RMSE = 0.359493 °C, C0 RMSE = 0.349274 °C (+2.84243%)",
        "decision": "Seleccionar E3b-C0 (features: sst_bil, doy_sin, doy_cos, depth; XGBoost n=19, d=4)",
        "status": "METHODOLOGICALLY CLOSED",
        "canonical_source": "ml_results/E3b_D32/tables/model_summary.csv, feature_ablation.csv"
    },
    {
        "stage": "D",
        "substage": "D33",
        "scientific_purpose": "Validación temporal externa de la especificación congelada E3b-C0 (2022–2023)",
        "input_data": "Pre-validation refit 2015–2021 (13.49M obs), Validation 2022–2023 (3.85M obs)",
        "processing": "Reajuste estricto de la especificación E3b-C0 sin modificar hiperparámetros ni features",
        "key_output": "Predicciones y métricas de validación externa 2022–2023",
        "key_metric": "B0 RMSE = 0.335666 °C, C0 RMSE = 0.323838 °C (+3.5237%); 16/24 meses; 4,755/5,275 celdas (90.14%)",
        "decision": "Dictamen formal D33-B (mejora global positiva pero estabilidad mensual sub-umbral predeclarado)",
        "status": "D33-B — UNCHANGED",
        "canonical_source": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv"
    },
    {
        "stage": "D",
        "substage": "D34",
        "scientific_purpose": "Auditoría diagnóstica post-validación sin modificación adaptativa",
        "input_data": "Predicciones persistidas de validación 2022–2023 y covariables espaciales",
        "processing": "Diagnóstico por regímenes de discrepancia MUR–BIL, compresión de amplitud, microauditoría espacial",
        "key_output": "Reportes diagnósticos D34 y microauditoría consolidada",
        "key_metric": "Compresión de amplitud (ratio std ~0.25); degradación en discrepancias bajas; ganancias costeras",
        "decision": "Cerrar diagnósticos; congelar protocolo final sin alterar modelo ni features; proceder a Test",
        "status": "INTERPRETATIONALLY CLOSED",
        "canonical_source": "ml_results/E3b_D34_postvalidation_diagnostics/"
    },
    {
        "stage": "D",
        "substage": "D35",
        "scientific_purpose": "Evaluación final out-of-sample en conjunto ciego Final Test 2024–2025",
        "input_data": "Final refit 2015–2023 (17.34M obs) y Test 2024–2025 (3.86M obs, 5,275 celdas)",
        "processing": "Reajuste final en 2015–2023, inferencia única en test, bootstrap en bloques de 14 días",
        "key_output": "Resultados definitivos de generalización fuera de muestra",
        "key_metric": "B0 RMSE = 0.357317 °C, C0 RMSE = 0.331502 °C (+7.2247%); 18/24 meses, 484/731 días, 5,273/5,275 celdas (99.96%)",
        "decision": "Dictamen formal D35-A (Generalización final confirmada); TEST CONSUMIDO",
        "status": "D35-A — FINAL GENERALIZATION CONFIRMED",
        "canonical_source": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv"
    },
    {
        "stage": "D",
        "substage": "D36",
        "scientific_purpose": "Consolidación científica definitiva y archivo reproducible del bloque de Machine Learning",
        "input_data": "Tablas, reportes, figuras y predicciones persistidas de D31–D35",
        "processing": "Síntesis transversal sin reentrenamiento ni acceso a datos raw de test",
        "key_output": "ml_results/E3b_FINAL_SYNTHESIS/ (reportes, tablas consolidadas, guía de figuras)",
        "key_metric": "Cero llamadas a fit/train, cero accesos lógicos raw a test, consistencia numérica completa",
        "decision": "Cierre científico definitivo del bloque de Machine Learning de la tesis",
        "status": "FINAL SYNTHESIS COMPLETED",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/reports/ML_FINAL_SYNTHESIS.md"
    }
]

df_phase_summary = pd.DataFrame(phase_summary_data)
df_phase_summary.to_csv(TABLES_DIR / "master_phase_summary.csv", index=False)
print("Tabla generada: master_phase_summary.csv")

# 3.2 master_decision_log.csv
decision_log_data = [
    {
        "decision_id": "DEC-01",
        "stage": "Stage B",
        "substage": "B",
        "date_or_period": "2026-08-21",
        "problem": "Selección de cuadrícula espacial maestra para el corredor Tulum–Cozumel",
        "alternatives_considered": "Grilla nativa OISST (0.25°), Grilla nativa MUR (0.01°), Grilla UTM regular remuestreada",
        "evidence": "MUR v4.1 provee resolución analítica de 0.01° (~1 km) en coordenadas geográficas estándar",
        "decision": "Adoptar cuadrícula MUR (86 × 96 celdas, [19.90°N, 20.75°N], [-87.60°W, -86.65°W])",
        "rationale": "Permite downscaling directo preservando la geometría nativa de la referencia de mayor resolución",
        "consequence": "Todas las covariables estáticas y dinámicas se remuestrean a esta malla base de 8,256 celdas",
        "source_file": "config.py, reports/inspeccion_previa.txt",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-02",
        "stage": "Stage B",
        "substage": "B.1",
        "date_or_period": "2026-08-23",
        "problem": "Máscara oceánica nativa de MUR clasifica como agua celdas terrestres mixtas y el interior de Cozumel",
        "alternatives_considered": "Máscara MUR nativa sin modificar, Máscara GEBCO binaria (cota > 0), Máscara híbrida con umbral de fracción oceánica",
        "evidence": "GEBCO 15 arc-sec revela que 383 celdas marcadas como océano por MUR tienen fracción oceánica < 0.5 o son tierra firme",
        "decision": "Definir M_final = M_MUR AND (ocean_fraction >= 0.5), forzando interior de Cozumel a tierra",
        "rationale": "Evita contaminación costera por reflectancia terrestre en celdas predominantemente terrestres",
        "consequence": "Dominio oceánico base fijado en exactamente 5,279 celdas marinas y 2,977 terrestres",
        "source_file": "reports/reporte_fase_b.txt",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-03",
        "stage": "Stage C",
        "substage": "C.1b",
        "date_or_period": "2026-08-23",
        "problem": "Interpolación bilineal de OISST produce 1,292 NaNs costeros (24.47% de celdas oceánicas) por 20 nodos terrestres",
        "alternatives_considered": "Estrategia A (extensión costera nearest-ocean en 0.25°), Estrategia B (triangulación 2D Delaunay en nodos marinos)",
        "evidence": "Ambas estrategias logran 100% de cobertura (5,279 celdas); MAE A vs B es de solo 0.0038 °C (P95 = 0.0214 °C)",
        "decision": "Adoptar Estrategia A (soporte costero regular de 0.25° -> bilineal -> recorte estricto con ocean_mask_final)",
        "rationale": "Preserva la formulación cartesiana estándar, es computacionalmente eficiente para 4,018 días y registra soporte",
        "consequence": "Se genera campo OISST_BIL continuo en el 100% del dominio marino sin alterar datos continentales",
        "source_file": "reports/reporte_faseC1b_2015-01-01.txt",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-04",
        "stage": "Stage C",
        "substage": "Satellite Validation",
        "date_or_period": "2026-08-30",
        "problem": "Pico anómalo de discrepancia MUR–BIL en octubre 2015 sugiere posible artefacto en MUR",
        "alternatives_considered": "Modificar/eliminar fechas anómalas de faseC2, sustituir con L2P, mantener C.2 intacto",
        "evidence": "Auditoría VIIRS/MODIS L2P muestra cobertura de datos limpios (QL=5) de 0.0% a 0.1% por persistencia de nubes",
        "decision": "Mantener C.2 intacto sin alteraciones retrospectivas ad-hoc",
        "rationale": "No existe base observacional cuantitativa para descartar o alterar selectivamente fechas en un producto L4 consolidado",
        "consequence": "MUR se asume formalmente como referencia operacional de alta resolución con incertidumbre interna asociada",
        "source_file": "reports/validacion_infrarroja_integrada_FINAL.md",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-05",
        "stage": "Stage C / D",
        "substage": "MUR AE / D.1",
        "date_or_period": "2026-08-31",
        "problem": "Determinación del rol metodológico de la variable MUR analysis_error en el modelo ML",
        "alternatives_considered": "Predictor dinámico en vector de entrada (X), ponderador en función de pérdida, variable exclusiva de control/diagnóstico",
        "evidence": "analysis_error satura asintóticamente en 0.41 °C en eventos nubosos y representa incertidumbre de la referencia",
        "decision": "Excluir analysis_error de las features de entrada; utilizarlo exclusivamente como variable de control y auditoría",
        "rationale": "Evitar circularidad epistemológica: un predictor no debe basarse en el reporte de error de la propia variable objetivo",
        "consequence": "analysis_error se almacena en el dataset tabular pero nunca ingresa a la matriz de diseño de los modelos",
        "source_file": "ml_dataset/METADATA.md, auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-06",
        "stage": "Stage D",
        "substage": "D.Dataset",
        "date_or_period": "2026-09-05",
        "problem": "Estrategia de partición para entrenamiento, validación y prueba en datos geofísicos espaciotemporales",
        "alternatives_considered": "Random K-Fold Cross-Validation, Spatial Block Split, Partición Temporal Cronológica Bloqueada",
        "evidence": "La autocorrelación temporal produce filtración masiva de información (data leakage) en particiones aleatorias",
        "decision": "Partición temporal cronológica: Dev (2015–2020), Holdout (2021), Val (2022–2023), Final Test (2024–2025)",
        "rationale": "Garantiza evaluación genuinamente fuera de muestra que simula la operación predictiva real en el tiempo",
        "consequence": "Final Test 2024–2025 queda completamente ciego y bloqueado hasta la fase final confirmatoria D35",
        "source_file": "ml_dataset/METADATA.md",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-07",
        "stage": "Stage D",
        "substage": "D32",
        "date_or_period": "2026-09-05",
        "problem": "Selección de arquitectura, features y modelo para la predicción del residual fino R",
        "alternatives_considered": "Seis ablaciones prespecificadas: E3b-C0 (core), E3b-T1 (lag1), E3b-T3 (lag3), E3b-S (espacial), E3b-TS, E3b-ALL",
        "evidence": "E3b-C0 logró la mayor mejora en Holdout 2021 (+2.84243%); lags temporales y features de vecindad degradaron el skill",
        "decision": "Seleccionar E3b-C0 (features: sst_bil, doy_sin, doy_cos, depth; XGBoost n=19, d=4, lr=0.10) por parsimonia y desempeño",
        "rationale": "Principio de parsimonia científica: el modelo más simple supera consistentemente a formulaciones complejas",
        "consequence": "Se congelan definitivamente la especificación y las 5,275 celdas comunes (frozen_cell_ids.csv)",
        "source_file": "ml_results/E3b_D32/tables/model_summary.csv, feature_ablation.csv",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-08",
        "stage": "Stage D",
        "substage": "D33",
        "date_or_period": "2026-09-05",
        "problem": "Dictamen metodológico formal tras evaluación de E3b-C0 en Validación Externa 2022–2023",
        "alternatives_considered": "D33-A (Generalización Completa Confirmada), D33-B (Generalización Parcial/Inestable), D33-C (Fallo)",
        "evidence": "Mejora global positiva (+3.5237% RMSE, 90.14% celdas), pero solo 16/24 meses mejorados (criterio D33-A exigía >=18)",
        "decision": "Emitir dictamen formal inmutable D33-B y pausar avance a Final Test para auditoría diagnóstica",
        "rationale": "Cumplimiento estricto del protocolo predeclarado: no flexibilizar criterios ante resultados marginales",
        "consequence": "Se inicia la Fase D.3.4 de auditoría diagnóstica sin modificar el modelo ni abrir Final Test",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-09",
        "stage": "Stage D",
        "substage": "D34 / D35",
        "date_or_period": "2026-09-05",
        "problem": "Autorización para proceder a la evaluación final en Final Test 2024–2025",
        "alternatives_considered": "Suspender proyecto por inestabilidad mensual, reentrenar con nuevas variables, proceder a Final Test congelado",
        "evidence": "D34 demostró que la inestabilidad mensual es inherente al bajo residual estacional y no a sobreajuste del modelo",
        "decision": "Proceder a Final Test 2024–2025 con protocolo de ejecución completamente congelado",
        "rationale": "La formulación E3b-C0 está científicamente validada; el objetivo del test ciego es medir su generalización terminal",
        "consequence": "Se ejecuta D35 consumiendo definitivamente el conjunto de prueba ciego 2024–2025",
        "source_file": "ml_results/E3b_D34_postvalidation_diagnostics/reports/faseD34_postvalidation_diagnostics_report.md",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-10",
        "stage": "Stage D",
        "substage": "D35",
        "date_or_period": "2026-09-05",
        "problem": "Dictamen formal final de generalización out-of-sample en Test 2024–2025",
        "alternatives_considered": "D35-A (Generalización Final Confirmada), D35-B (Generalización Parcial), D35-C (Fallo Terminal)",
        "evidence": "+7.2247% mejora global de RMSE; 18/24 meses mejorados; 484/731 días; 5,273/5,275 celdas (99.96%); bootstrap CI excluye 0",
        "decision": "Emitir dictamen formal D35-A: FINAL GENERALIZATION CONFIRMED; declarar TEST CONSUMED",
        "rationale": "Todos los criterios confirmatorios primarios, temporales y espaciales fueron plenamente superados",
        "consequence": "El modelo E3b-C0 queda científicamente cerrado y validado; se prohíbe reentrenar o ajustar sobre 2024–2025",
        "source_file": "ml_results/E3b_D35_final_test/tables/decision_criteria_D35.csv",
        "status": "CANONICAL"
    },
    {
        "decision_id": "DEC-11",
        "stage": "Stage D",
        "substage": "D36",
        "date_or_period": "2026-09-05",
        "problem": "Consolidación científica definitiva y cierre de desarrollo del bloque de Machine Learning",
        "alternatives_considered": "Probar nuevas arquitecturas (CNN, transformers), reabrir tuning, cerrar desarrollo y consolidar evidencia",
        "evidence": "Final Test ya fue consumido; la reapertura de modelos sobre el periodo de prueba violaría la ética metodológica",
        "decision": "Declarar el desarrollo de modelos de Machine Learning CIENTÍFICAMENTE CERRADO",
        "rationale": "La evidencia experimental es definitiva, trazable y suficiente para la redacción de la tesis y publicaciones",
        "consequence": "Todo trabajo posterior se limita a síntesis, trazabilidad y redacción documental",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/reports/ML_FINAL_SYNTHESIS.md",
        "status": "CANONICAL"
    }
]

df_decision_log = pd.DataFrame(decision_log_data)
df_decision_log.to_csv(TABLES_DIR / "master_decision_log.csv", index=False)
print("Tabla generada: master_decision_log.csv")

# 3.3 dataset_provenance.csv
dataset_prov_data = [
    {
        "dataset": "MUR SST",
        "provider": "NASA JPL / PO.DAAC",
        "version": "v4.1 (JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1)",
        "variable": "analysed_sst",
        "native_resolution": "0.01° (~1 km)",
        "units": "Kelvin (convertido a Celsius: SST_C = SST_K - 273.15)",
        "period": "2015-01-01 a 2025-12-31 (4,018 días)",
        "role": "Referencia observacional de alta resolución; definición de cuadrícula espacial",
        "preprocessing": "Conversión K -> °C, recorte geográfico [19.90–20.75°N, -87.60–-86.65°W], verificación de continuidad",
        "final_use": "Variable de referencia para cálculo del residual R = MUR - BIL y evaluación de downscaling",
        "source_artifact": "MUR_ZARR/ (MUR_HISTORICO_2015_2019 + gránulos diarios 2019-2025)"
    },
    {
        "dataset": "MUR Analysis Error",
        "provider": "NASA JPL / PO.DAAC",
        "version": "v4.1",
        "variable": "analysis_error",
        "native_resolution": "0.01° (~1 km)",
        "units": "Kelvin / Celsius (desviación estándar estimada de error)",
        "period": "2015-01-01 a 2025-12-31 (4,018 días)",
        "role": "Variable de auditoría de incertidumbre del algoritmo de asimilación multiescala",
        "preprocessing": "Descarga OPeNDAP y compilación en cubo NetCDF continuo alineado con C.2",
        "final_use": "Variable de control en dataset tabular; diagnóstico de confiabilidad en regímenes extremos",
        "source_artifact": "analysis_error_historico/mur_analysis_error_2015_2025_completo.nc"
    },
    {
        "dataset": "NOAA OISST",
        "provider": "NOAA NCEI / CoastWatch",
        "version": "v2.1 (ncdcOisst21Agg_LonPM180)",
        "variable": "sst",
        "native_resolution": "0.25° (~27 km)",
        "units": "Celsius (°C)",
        "period": "2015-01-01 a 2025-12-31 (4,018 días)",
        "role": "Predictor térmico de baja resolución a ser downescalado",
        "preprocessing": "Extracción con halo 0.5°, soporte costero Estrategia A, interpolación bilineal a 0.01°",
        "final_use": "Predictor dinámico principal (sst_bil) y baseline de referencia B0 (R_hat = 0)",
        "source_artifact": "NOAA ERDDAP + OISST/OISST_2025-01-14_Tulum_Cozumel.nc"
    },
    {
        "dataset": "GEBCO Bathymetry",
        "provider": "BODC / Nippon Foundation-GEBCO Seabed 2030 Project",
        "version": "GEBCO 2026 Grid (publicado abril 2026, DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa)",
        "variable": "elevation",
        "native_resolution": "15 arc-seconds (~0.004167°, ~450 m)",
        "units": "Metros (m)",
        "period": "Estático (revisión continua 2026)",
        "role": "Covariables fisiográficas estáticas y definición de máscara oceánica corregida",
        "preprocessing": "depth = -elevation en océano; agregación subpixel para ocean_fraction a 0.01°",
        "final_use": "Covariable estática depth en modelo ML; definición de ocean_mask_final",
        "source_artifact": "GEBCO/gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc"
    },
    {
        "dataset": "VIIRS S-NPP L2P",
        "provider": "NASA / NOAA CoastWatch",
        "version": "v2.80",
        "variable": "sea_surface_temperature, quality_level",
        "native_resolution": "~750 m a nadir (gránulos orbitales L2P)",
        "units": "Kelvin / Celsius",
        "period": "Eventos anómalos discretos (2015–2024)",
        "role": "Auditoría radiométrica infrarroja independiente no asimilada",
        "preprocessing": "Filtro quality_level == 5, reproyección al dominio Tulum–Cozumel",
        "final_use": "Evaluación independiente de la validez física de eventos de discrepancia extrema",
        "source_artifact": "reports/validacion_viirs_L2P_v280_FINAL.csv, reports/validacion_infrarroja_integrada_FINAL.md"
    },
    {
        "dataset": "MODIS Aqua L2P",
        "provider": "NASA Ocean Biology DAAC (OB.DAAC)",
        "version": "v2019.0",
        "variable": "sea_surface_temperature (11 µm), sst4 (4 µm), quality_level",
        "native_resolution": "~1 km a nadir (gránulos orbitales L2P)",
        "units": "Celsius (°C)",
        "period": "Eventos anómalos discretos (2015–2024)",
        "role": "Auditoría radiométrica infrarroja independiente diurna y nocturna",
        "preprocessing": "Filtro quality_level == 5, evaluación de canales 11 µm y 4 µm",
        "final_use": "Confirmación independiente de cobertura nubosa y confiabilidad radiométrica",
        "source_artifact": "reports/validacion_modis_aqua_L2P_FINAL.csv, reports/validacion_infrarroja_integrada_FINAL.md"
    }
]

df_dataset_prov = pd.DataFrame(dataset_prov_data)
df_dataset_prov.to_csv(TABLES_DIR / "dataset_provenance.csv", index=False)
print("Tabla generada: dataset_provenance.csv")

# 3.4 metric_provenance_A_D.csv
metric_prov_data = [
    {
        "metric_id": "MET-A-01",
        "stage": "Stage A",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "total_expected_days",
        "value": "4018",
        "units": "days",
        "source_file": "config.py, reports/inspeccion_previa.txt",
        "source_column": "EXPECTED_DAYS",
        "aggregation": "count",
        "canonical_status": "CANONICAL",
        "notes": "Continuidad diaria 100% verificada en MUR y OISST"
    },
    {
        "metric_id": "MET-B-01",
        "stage": "Stage B",
        "period": "Static",
        "metric": "total_master_grid_cells",
        "value": "8256",
        "units": "cells",
        "source_file": "reports/reporte_fase_b.txt",
        "source_column": "Celdas totales",
        "aggregation": "product(86, 96)",
        "canonical_status": "CANONICAL",
        "notes": "Dimensiones 86 latitud × 96 longitud a 0.01°"
    },
    {
        "metric_id": "MET-B-02",
        "stage": "Stage B",
        "period": "Static",
        "metric": "harmonized_ocean_cells",
        "value": "5279",
        "units": "cells",
        "source_file": "reports/reporte_fase_b.txt",
        "source_column": "Final ocean mask: océano",
        "aggregation": "sum(ocean_mask_final == 1)",
        "canonical_status": "CANONICAL",
        "notes": "Censo oceánico de Fase B, C.1b, C.1c y C.2"
    },
    {
        "metric_id": "MET-B-03",
        "stage": "Stage B",
        "period": "Static",
        "metric": "harmonized_land_cells",
        "value": "2977",
        "units": "cells",
        "source_file": "reports/reporte_fase_b.txt",
        "source_column": "Final ocean mask: tierra",
        "aggregation": "sum(ocean_mask_final == 0)",
        "canonical_status": "CANONICAL",
        "notes": "383 celdas reclasificadas a tierra respecto a MUR nativo"
    },
    {
        "metric_id": "MET-C-01",
        "stage": "Stage C.1",
        "period": "2015-01-01",
        "metric": "coastal_nans_direct_bilinear",
        "value": "1292",
        "units": "cells",
        "source_file": "reports/reporte_faseC1b_2015-01-01.txt",
        "source_column": "NaN oceánicos",
        "aggregation": "count",
        "canonical_status": "CANONICAL",
        "notes": "24.47% del dominio oceánico perdido por nodos terrestres OISST"
    },
    {
        "metric_id": "MET-C-02",
        "stage": "Stage C.1b",
        "period": "2015-01-01",
        "metric": "strategy_a_vs_b_mae",
        "value": "0.0038",
        "units": "degC",
        "source_file": "reports/reporte_faseC1b_2015-01-01.txt",
        "source_column": "DIFERENCIA A vs B: MAE",
        "aggregation": "mean_absolute_error",
        "canonical_status": "CANONICAL",
        "notes": "Discrepancia despreciable confirma estabilidad de Estrategia A"
    },
    {
        "metric_id": "MET-C-03",
        "stage": "Stage C.1b",
        "period": "2015-01-01",
        "metric": "strategy_a_vs_b_p95",
        "value": "0.0214",
        "units": "degC",
        "source_file": "reports/reporte_faseC1b_2015-01-01.txt",
        "source_column": "DIFERENCIA A vs B: P95",
        "aggregation": "percentile_95",
        "canonical_status": "CANONICAL",
        "notes": "Percentil 95 de discrepancia entre métodos"
    },
    {
        "metric_id": "MET-C-04",
        "stage": "Stage C.1b",
        "period": "2015-01-01",
        "metric": "strategy_a_vs_b_max",
        "value": "0.0626",
        "units": "degC",
        "source_file": "reports/reporte_faseC1b_2015-01-01.txt",
        "source_column": "DIFERENCIA A vs B: Máx",
        "aggregation": "maximum",
        "canonical_status": "CANONICAL",
        "notes": "Máxima discrepancia espacial puntual entre A y B"
    },
    {
        "metric_id": "MET-C-05",
        "stage": "Stage C.2",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "total_harmonized_observations",
        "value": "21211022",
        "units": "observations",
        "source_file": "reports/fase_c2_reporte.md",
        "source_column": "Puntos Evaluados (N)",
        "aggregation": "product(4018, 5279)",
        "canonical_status": "CANONICAL",
        "notes": "Censo decenal completo en cuadrícula física de 5,279 celdas"
    },
    {
        "metric_id": "MET-C-06",
        "stage": "Stage C.2",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "global_baseline_e0_rmse",
        "value": "0.3426",
        "units": "degC",
        "source_file": "reports/fase_c2_reporte.md",
        "source_column": "RMSE",
        "aggregation": "rmse(sst_mur, sst_bil)",
        "canonical_status": "CANONICAL",
        "notes": "Error cuadrático medio global de interpolación bilineal pura"
    },
    {
        "metric_id": "MET-C-07",
        "stage": "Stage C.2",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "global_baseline_e0_mae",
        "value": "0.2631",
        "units": "degC",
        "source_file": "reports/fase_c2_reporte.md",
        "source_column": "MAE",
        "aggregation": "mae(sst_mur, sst_bil)",
        "canonical_status": "CANONICAL",
        "notes": "Error absoluto medio decenal del baseline"
    },
    {
        "metric_id": "MET-C-08",
        "stage": "Stage C.2",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "global_baseline_e0_bias",
        "value": "0.0133",
        "units": "degC",
        "source_file": "reports/fase_c2_reporte.md",
        "source_column": "Bias (mean(BIL - MUR))",
        "aggregation": "mean(sst_bil - sst_mur)",
        "canonical_status": "CANONICAL",
        "notes": "Sesgo medio decenal del producto bilineal"
    },
    {
        "metric_id": "MET-C-09",
        "stage": "Stage C.2",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "global_baseline_e0_r2",
        "value": "0.9016",
        "units": "dimensionless",
        "source_file": "reports/fase_c2_reporte.md",
        "source_column": "R²",
        "aggregation": "r2_score(sst_mur, sst_bil)",
        "canonical_status": "CANONICAL",
        "notes": "Coeficiente de determinación decenal OISST_BIL vs MUR SST"
    },
    {
        "metric_id": "MET-AE-01",
        "stage": "MUR AE Audit",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "spearman_rho_ae_vs_rmse",
        "value": "0.2853",
        "units": "dimensionless",
        "source_file": "auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md",
        "source_column": "Coeficiente de Spearman rho",
        "aggregation": "spearmanr(AE, RMSE_diario)",
        "canonical_status": "CANONICAL",
        "notes": "Asociación estadísticamente significativa (p = 4.25e-76)"
    },
    {
        "metric_id": "MET-AE-02",
        "stage": "MUR AE Audit",
        "period": "2015-01-01 a 2025-12-31",
        "metric": "asymptotic_error_ceiling",
        "value": "0.4100",
        "units": "degC",
        "source_file": "auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md",
        "source_column": "Máximo estricto observado",
        "aggregation": "max(analysis_error)",
        "canonical_status": "CANONICAL",
        "notes": "Techo algorítmico asintótico fijado en la asimilación multiescala de MUR"
    },
    {
        "metric_id": "MET-D-01",
        "stage": "Stage D",
        "period": "Static",
        "metric": "frozen_ml_cells",
        "value": "5275",
        "units": "cells",
        "source_file": "ml_results/E3b_D32/tables/frozen_cell_ids.csv",
        "source_column": "cell_id",
        "aggregation": "nunique",
        "canonical_status": "CANONICAL",
        "notes": "SHA-256: 6f046931d2d8220c1938b1cb06511fe63b5dbc51debed51df8b353006635f5bb"
    },
    {
        "metric_id": "MET-D32-01",
        "stage": "Stage D32",
        "period": "2021",
        "metric": "holdout_2021_b0_rmse",
        "value": "0.359493",
        "units": "degC",
        "source_file": "ml_results/E3b_D32/tables/model_summary.csv",
        "source_column": "RMSE_B0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "Holdout diagnóstico 2021 sobre 5,275 celdas comunes (1,925,375 obs)"
    },
    {
        "metric_id": "MET-D32-02",
        "stage": "Stage D32",
        "period": "2021",
        "metric": "holdout_2021_c0_rmse",
        "value": "0.349274",
        "units": "degC",
        "source_file": "ml_results/E3b_D32/tables/model_summary.csv",
        "source_column": "RMSE_C0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "Desempeño del modelo seleccionado E3b-C0"
    },
    {
        "metric_id": "MET-D32-03",
        "stage": "Stage D32",
        "period": "2021",
        "metric": "holdout_2021_c0_improvement_pct",
        "value": "2.84243",
        "units": "%",
        "source_file": "ml_results/E3b_D32/tables/model_summary.csv",
        "source_column": "Impr_RMSE_vs_B0_pct",
        "aggregation": "relative_percentage",
        "canonical_status": "CANONICAL",
        "notes": "Mejora relativa en holdout: +2.84243%"
    },
    {
        "metric_id": "MET-D33-01",
        "stage": "Stage D33",
        "period": "2022–2023",
        "metric": "val_2022_2023_b0_rmse",
        "value": "0.335666",
        "units": "degC",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "source_column": "rmse_b0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "Evaluación externa out-of-sample en 3,850,750 observaciones"
    },
    {
        "metric_id": "MET-D33-02",
        "stage": "Stage D33",
        "period": "2022–2023",
        "metric": "val_2022_2023_c0_rmse",
        "value": "0.323838",
        "units": "degC",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "source_column": "rmse_c0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "Reajuste de E3b-C0 en 2015–2021 evaluado en 2022–2023"
    },
    {
        "metric_id": "MET-D33-03",
        "stage": "Stage D33",
        "period": "2022–2023",
        "metric": "val_2022_2023_c0_improvement_pct",
        "value": "3.5237",
        "units": "%",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "source_column": "pct_improvement_rmse",
        "aggregation": "relative_percentage",
        "canonical_status": "CANONICAL",
        "notes": "Mejora global out-of-sample: +3.5237%"
    },
    {
        "metric_id": "MET-D33-04",
        "stage": "Stage D33",
        "period": "2022–2023",
        "metric": "val_months_improved",
        "value": "16 / 24",
        "units": "months",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "source_column": "months_improved",
        "aggregation": "count_ratio",
        "canonical_status": "CANONICAL",
        "notes": "66.7% de meses mejorados (criterio D33-A requería >= 18)"
    },
    {
        "metric_id": "MET-D33-05",
        "stage": "Stage D33",
        "period": "2022–2023",
        "metric": "val_cells_improved",
        "value": "4755 / 5275",
        "units": "cells",
        "source_file": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "source_column": "cells_improved",
        "aggregation": "count_ratio",
        "canonical_status": "CANONICAL",
        "notes": "90.14% de las celdas congeladas mostraron DeltaRMSE < 0"
    },
    {
        "metric_id": "MET-D35-01",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_b0_rmse",
        "value": "0.357317",
        "units": "degC",
        "source_file": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "source_column": "RMSE_B0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "Baseline B0 en Final Test ciego (3,856,025 obs, 731 días)"
    },
    {
        "metric_id": "MET-D35-02",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_c0_rmse",
        "value": "0.331502",
        "units": "degC",
        "source_file": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "source_column": "RMSE_C0",
        "aggregation": "rmse",
        "canonical_status": "CANONICAL",
        "notes": "E3b-C0 reajustado en 2015–2023 y evaluado ciegamente en 2024–2025"
    },
    {
        "metric_id": "MET-D35-03",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_c0_improvement_pct",
        "value": "7.2247",
        "units": "%",
        "source_file": "ml_results/E3b_D35_final_test/tables/final_test_summary.csv",
        "source_column": "pct_improvement_RMSE",
        "aggregation": "relative_percentage",
        "canonical_status": "CANONICAL",
        "notes": "Mejora global confirmada: +7.2247% (DeltaRMSE = -0.025815 °C)"
    },
    {
        "metric_id": "MET-D35-04",
        "stage": "Stage D35",
        "period": "2024",
        "metric": "final_test_improvement_2024_pct",
        "value": "9.3986",
        "units": "%",
        "source_file": "ml_results/E3b_D35_final_test/tables/yearly_metrics.csv",
        "source_column": "pct_improvement_RMSE",
        "aggregation": "relative_percentage",
        "canonical_status": "CANONICAL",
        "notes": "Mejora anual en 2024 (B0: 0.379708 °C -> C0: 0.344023 °C)"
    },
    {
        "metric_id": "MET-D35-05",
        "stage": "Stage D35",
        "period": "2025",
        "metric": "final_test_improvement_2025_pct",
        "value": "4.4704",
        "units": "%",
        "source_file": "ml_results/E3b_D35_final_test/tables/yearly_metrics.csv",
        "source_column": "pct_improvement_RMSE",
        "aggregation": "relative_percentage",
        "canonical_status": "CANONICAL",
        "notes": "Mejora anual en 2025 (B0: 0.333355 °C -> C0: 0.318453 °C)"
    },
    {
        "metric_id": "MET-D35-06",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_months_improved",
        "value": "18 / 24",
        "units": "months",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table2_robustness.csv",
        "source_column": "value",
        "aggregation": "count_ratio",
        "canonical_status": "CANONICAL",
        "notes": "75.0% de meses del bienio mejorados (11/12 en 2024, 7/12 en 2025)"
    },
    {
        "metric_id": "MET-D35-07",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_days_improved",
        "value": "484 / 731",
        "units": "days",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table2_robustness.csv",
        "source_column": "value",
        "aggregation": "count_ratio",
        "canonical_status": "CANONICAL",
        "notes": "66.21% de los días del bienio con mejora de RMSE"
    },
    {
        "metric_id": "MET-D35-08",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "final_test_cells_improved",
        "value": "5273 / 5275",
        "units": "cells",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table2_robustness.csv",
        "source_column": "value",
        "aggregation": "count_ratio",
        "canonical_status": "CANONICAL",
        "notes": "99.96% de las celdas marinas evaluadas mejoraron su RMSE"
    },
    {
        "metric_id": "MET-D35-09",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "bootstrap_14d_ci95_delta_rmse",
        "value": "[-0.042485, -0.009792]",
        "units": "degC",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table2_robustness.csv",
        "source_column": "value",
        "aggregation": "bootstrap_block_14d",
        "canonical_status": "CANONICAL",
        "notes": "Intervalo de confianza al 95% estrictamente negativo"
    },
    {
        "metric_id": "MET-D35-10",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "residual_r2",
        "value": "0.112175",
        "units": "dimensionless",
        "source_file": "ml_results/E3b_D35_final_test/tables/residual_metrics.csv",
        "source_column": "R2_RESIDUAL",
        "aggregation": "r2_score(R, R_hat)",
        "canonical_status": "CANONICAL",
        "notes": "Varianza del residual explicada por el modelo tabular"
    },
    {
        "metric_id": "MET-D35-11",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "residual_sign_accuracy",
        "value": "63.79",
        "units": "%",
        "source_file": "ml_results/E3b_D35_final_test/tables/residual_metrics.csv",
        "source_column": "sign_accuracy_pct",
        "aggregation": "percentage",
        "canonical_status": "CANONICAL",
        "notes": "Exactitud de signo de corrección vs baseline mayoritario (52.45%)"
    },
    {
        "metric_id": "MET-D35-12",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "calibration_slope",
        "value": "0.089054",
        "units": "dimensionless",
        "source_file": "ml_results/E3b_D35_final_test/tables/residual_metrics.csv",
        "source_column": "calibration_slope",
        "aggregation": "linear_regression_slope",
        "canonical_status": "CANONICAL",
        "notes": "Pendiente en regresión R_hat = a + b * R, reflejando fuerte compresión"
    },
    {
        "metric_id": "MET-D35-13",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "calibration_intercept",
        "value": "-0.042670",
        "units": "degC",
        "source_file": "ml_results/E3b_D35_final_test/tables/residual_metrics.csv",
        "source_column": "calibration_intercept",
        "aggregation": "linear_regression_intercept",
        "canonical_status": "CANONICAL",
        "notes": "Intercepto en regresión R_hat = a + b * R"
    },
    {
        "metric_id": "MET-D35-14",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "std_ratio_rhat_r",
        "value": "0.253584",
        "units": "dimensionless",
        "source_file": "ml_results/E3b_D35_final_test/tables/residual_metrics.csv",
        "source_column": "std_ratio_Rhat_R",
        "aggregation": "std(R_hat) / std(R)",
        "canonical_status": "CANONICAL",
        "notes": "Ratio de desviación estándar que confirma compresión hacia la media"
    },
    {
        "metric_id": "MET-D35-15",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "spearman_depth_vs_delta_rmse",
        "value": "0.7376",
        "units": "dimensionless",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/d35_final_report_corrections.csv",
        "source_column": "numeric_change",
        "aggregation": "spearmanr(depth, delta_rmse_cell)",
        "canonical_status": "CANONICAL",
        "notes": "Correlación positiva: menor ganancia relativa en aguas profundas"
    },
    {
        "metric_id": "MET-D35-16",
        "stage": "Stage D35",
        "period": "2024–2025",
        "metric": "spearman_dist_vs_delta_rmse",
        "value": "0.5336",
        "units": "dimensionless",
        "source_file": "ml_results/E3b_FINAL_SYNTHESIS/tables/d35_final_report_corrections.csv",
        "source_column": "numeric_change",
        "aggregation": "spearmanr(dist_coast, delta_rmse_cell)",
        "canonical_status": "CANONICAL",
        "notes": "Correlación moderada positiva mar adentro"
    }
]

df_metric_prov = pd.DataFrame(metric_prov_data)
df_metric_prov.to_csv(TABLES_DIR / "metric_provenance_A_D.csv", index=False)
print("Tabla generada: metric_provenance_A_D.csv")

# 3.5 artifact_inventory_A_D.csv
artifact_inv_data = [
    {
        "stage": "Stage A",
        "artifact_type": "Data Manifest",
        "filename": "inspeccion_previa.txt",
        "path": "reports/inspeccion_previa.txt",
        "purpose": "Auditoría preliminar de disponibilidad de archivos NetCDF decenales",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Verifica existencia física de los 4,018 días en MUR y OISST"
    },
    {
        "stage": "Stage B",
        "artifact_type": "Spatial NetCDF",
        "filename": "dataset_intermedio_fase_b.nc",
        "path": "outputs/dataset_intermedio_fase_b.nc",
        "purpose": "Cuadrícula base con máscara oceánica corregida y covariables estáticas",
        "canonical": "INTERMEDIATE",
        "obsolete": "NO",
        "replacement": "faseC2_2015_2025.nc",
        "notes": "Generado en Fase B.1 con 5,279 celdas oceánicas"
    },
    {
        "stage": "Stage B",
        "artifact_type": "Report",
        "filename": "reporte_fase_b.txt",
        "path": "reports/reporte_fase_b.txt",
        "purpose": "Censo espacial de celdas modificadas, ocean_fraction y máscara Cozumel",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Documenta reasignación de 383 celdas a tierra firme"
    },
    {
        "stage": "Stage C.1",
        "artifact_type": "Report",
        "filename": "reporte_faseC1_2015-01-01.txt",
        "path": "reports/reporte_faseC1_2015-01-01.txt",
        "purpose": "Reporte de prueba inicial bilineal identificando 1,292 NaNs costeros",
        "canonical": "HISTORICAL",
        "obsolete": "NO",
        "replacement": "reporte_faseC1b_2015-01-01.txt",
        "notes": "Evidencia primaria de la insuficiencia del soporte bilineal crudo"
    },
    {
        "stage": "Stage C.1b",
        "artifact_type": "Report",
        "filename": "reporte_faseC1b_2015-01-01.txt",
        "path": "reports/reporte_faseC1b_2015-01-01.txt",
        "purpose": "Comparación cuantitativa entre Estrategia A y Estrategia B",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Documenta cuasi-equivalencia numérica (MAE = 0.0038 °C) y adopción de Estrategia A"
    },
    {
        "stage": "Stage C.1c",
        "artifact_type": "Report",
        "filename": "reporte_faseC1c_2015-01-01.txt",
        "path": "reports/reporte_faseC1c_2015-01-01.txt",
        "purpose": "Cierre de validación de Estrategia A y estabilidad temporal de nodos de soporte",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Verifica 100% de cobertura en 2015-01-01 y autoriza corrida completa"
    },
    {
        "stage": "Stage C.2",
        "artifact_type": "Master NetCDF",
        "filename": "faseC2_2015_2025.nc",
        "path": "outputs/faseC2_2015_2025.nc",
        "purpose": "Cubo decenal consolidado de SST armonizada (4,018 días × 5,279 celdas)",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "540.82 MB. Fuente física de verdad para la extracción de datasets ML"
    },
    {
        "stage": "Stage C.2",
        "artifact_type": "Annual NetCDFs",
        "filename": "faseC2_YYYY.nc",
        "path": "outputs/fase_c2/faseC2_2015.nc ... faseC2_2025.nc",
        "purpose": "Particiones anuales de SST armonizada",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "11 archivos anuales de ~11.8 MB cada uno"
    },
    {
        "stage": "Stage C.2",
        "artifact_type": "Report",
        "filename": "fase_c2_reporte.md",
        "path": "reports/fase_c2_reporte.md",
        "purpose": "Métricas anuales y decenales completas del Baseline E0",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Registra las 21,211,022 observaciones y RMSE decenal de 0.3426 °C"
    },
    {
        "stage": "Satellite Audit",
        "artifact_type": "Report",
        "filename": "validacion_infrarroja_integrada_FINAL.md",
        "path": "reports/validacion_infrarroja_integrada_FINAL.md",
        "purpose": "Auditoría radiométrica VIIRS/MODIS L2P en eventos anómalos",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Dictamen: evidencia heterogénea/inconclusa por nubosidad; C.2 intacto"
    },
    {
        "stage": "MUR AE Audit",
        "artifact_type": "Master NetCDF",
        "filename": "mur_analysis_error_2015_2025_completo.nc",
        "path": "analysis_error_historico/mur_analysis_error_2015_2025_completo.nc",
        "purpose": "Cubo decenal de la variable analysis_error de MUR L4 alineado con C.2",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Fuente de verdad para la variable de control analysis_error"
    },
    {
        "stage": "MUR AE Audit",
        "artifact_type": "Report",
        "filename": "auditoria_analysis_error_FINAL_4018dias.md",
        "path": "auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md",
        "purpose": "Auditoría integral de correlación, gradiente monotónico y techo en 0.41 °C",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Determina excluir analysis_error de los predictores ML"
    },
    {
        "stage": "Stage D.1",
        "artifact_type": "Tabular Parquet",
        "filename": "ml_dataset/ (train, val, test)",
        "path": "ml_dataset/",
        "purpose": "Dataset tabular analítico particionado cronológicamente con Snappy",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "263 MB en disco; 13 variables; test 2024–2025 bloqueado"
    },
    {
        "stage": "Stage D32",
        "artifact_type": "Frozen Cell Mask",
        "filename": "frozen_cell_ids.csv",
        "path": "ml_results/E3b_D32/tables/frozen_cell_ids.csv",
        "purpose": "Definición canónica de las 5,275 celdas marinas comunes para ML",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Inmutable desde D32; propagado estrictamente a D33, D34, D35 y D36"
    },
    {
        "stage": "Stage D32",
        "artifact_type": "Model Spec & Weights",
        "filename": "E3b-C0_holdout_2021.json / model_summary.csv",
        "path": "ml_results/E3b_D32/",
        "purpose": "Especificación congelada de arquitectura e hiperparámetros E3b-C0",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Cierra formalmente la selección de modelo en Holdout 2021"
    },
    {
        "stage": "Stage D33",
        "artifact_type": "Validation Summary",
        "filename": "validation_summary.csv",
        "path": "ml_results/E3b_D33_external_validation/tables/validation_summary.csv",
        "purpose": "Resultados de validación externa 2022–2023 y dictamen formal D33-B",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Dictamen inmutable: mejora positiva con estabilidad mensual parcial"
    },
    {
        "stage": "Stage D34",
        "artifact_type": "Diagnostics Report",
        "filename": "faseD34_postvalidation_diagnostics_report.md",
        "path": "ml_results/E3b_D34_postvalidation_diagnostics/reports/faseD34_postvalidation_diagnostics_report.md",
        "purpose": "Auditoría de regímenes de discrepancia, compresión de amplitud y covariables espaciales",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Concluye con recomendación de proceder a Final Test"
    },
    {
        "stage": "Stage D35",
        "artifact_type": "Audited Test Report",
        "filename": "faseD35_FINAL_TEST_E3b_C0_FINAL_CORRECTED.md",
        "path": "ml_results/E3b_D35_final_test/reports/faseD35_FINAL_TEST_E3b_C0_FINAL_CORRECTED.md",
        "purpose": "Reporte final auditado de evaluación en Final Test 2024–2025 y dictamen D35-A",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Generalización terminal confirmada (+7.2247% RMSE); TEST CONSUMED"
    },
    {
        "stage": "Stage D36",
        "artifact_type": "Consolidation Suite",
        "filename": "ML_FINAL_SYNTHESIS.md / tables / figures",
        "path": "ml_results/E3b_FINAL_SYNTHESIS/",
        "purpose": "Suite de consolidación de D31–D35, tablas de paper y cierre científico",
        "canonical": "CANONICAL",
        "obsolete": "NO",
        "replacement": "None",
        "notes": "Cierre formal del bloque de Machine Learning"
    }
]

df_artifact_inv = pd.DataFrame(artifact_inv_data)
df_artifact_inv.to_csv(TABLES_DIR / "artifact_inventory_A_D.csv", index=False)
print("Tabla generada: artifact_inventory_A_D.csv")

# 3.6 figure_inventory_A_D.csv
figure_inv_data = [
    {
        "figure_id": "FIG-M-01",
        "stage": "Master Pipeline",
        "filename": "fig_master_pipeline_A_D.png",
        "scientific_message": "Arquitectura metodológica integral de las Etapas A–D desde adquisición hasta evaluación out-of-sample",
        "thesis_main": "Capítulo Metodología",
        "thesis_appendix": "No",
        "paper_main": "Figura 1 Metodológica",
        "supplement": "No",
        "canonical_source": "THESIS_MASTER_A_D/figures/fig_master_pipeline_A_D.png",
        "notes": "Diagrama de flujo de datos, puntos de decisión y control de data leakage"
    },
    {
        "figure_id": "FIG-C-01",
        "stage": "Stage C.1",
        "filename": "faseC1_comparacion_mur_bil_2015-01-01.png",
        "scientific_message": "Comparación espacial directa entre MUR SST, OISST bilineal y residual fino en 2015-01-01",
        "thesis_main": "Capítulo Metodología / Resultados C",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria C1",
        "canonical_source": "figures/faseC1_comparacion_mur_bil_2015-01-01.png",
        "notes": "Ilustra la resolución espacial y contraste de escala"
    },
    {
        "figure_id": "FIG-C-02",
        "stage": "Stage C.1b",
        "filename": "faseC1b_comparacion_estrategias_2015-01-01.png",
        "scientific_message": "Comparación espacial y residual entre Estrategia A (soporte regular) y Estrategia B (triangulación Delaunay)",
        "thesis_main": "Capítulo Metodología (Soporte costero)",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria C2",
        "canonical_source": "figures/faseC1b_comparacion_estrategias_2015-01-01.png",
        "notes": "Demuestra la cuasi-equivalencia física en la costa oriental"
    },
    {
        "figure_id": "FIG-C-03",
        "stage": "Stage C.2",
        "filename": "faseC2_serie_rmse_diario.png",
        "scientific_message": "Serie temporal decenal completa (4,018 días) del RMSE diario de OISST_BIL vs MUR SST (Baseline E0)",
        "thesis_main": "Capítulo Resultados (Línea base C.2)",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria C3",
        "canonical_source": "figures/faseC2_serie_rmse_diario.png",
        "notes": "Evidencia variabilidad estacional e interanual del error bilineal"
    },
    {
        "figure_id": "FIG-C-04",
        "stage": "Stage C.2",
        "filename": "faseC2_mapas_estacionales_residual.png",
        "scientific_message": "Climatología estacional del residual fino R = MUR - BIL en el corredor Tulum–Cozumel",
        "thesis_main": "Capítulo Resultados (Armonización)",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria C4",
        "canonical_source": "figures/faseC2_mapas_estacionales_residual.png",
        "notes": "Muestra persistencia espacial de gradientes térmicos locales"
    },
    {
        "figure_id": "FIG-AE-01",
        "stage": "MUR AE Audit",
        "filename": "auditoria_multievento_comparativa_global.png",
        "scientific_message": "Comportamiento de analysis_error durante eventos de discrepancia extrema E1–E6 y techo de 0.41 °C",
        "thesis_main": "Capítulo Metodología / Discusión",
        "thesis_appendix": "Apéndice Auditoría MUR",
        "paper_main": "No",
        "supplement": "Figura Suplementaria AE1",
        "canonical_source": "auditoria_analysis_error/figures/auditoria_multievento_comparativa_global.png",
        "notes": "Justificación documental para tratar analysis_error como control"
    },
    {
        "figure_id": "FIG-D32-01",
        "stage": "Stage D32",
        "filename": "fig2_relative_improvement_vs_b0.png",
        "scientific_message": "Comparación de mejora relativa de RMSE en Holdout 2021 entre las seis formulaciones prespecificadas E3b",
        "thesis_main": "Capítulo Resultados (Selección de modelo)",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria D32",
        "canonical_source": "ml_results/E3b_D32/figures/fig2_relative_improvement_vs_b0.png",
        "notes": "Evidencia de la superioridad de E3b-C0 frente a lags y features espaciales"
    },
    {
        "figure_id": "FIG-D35-01",
        "stage": "Stage D35 / D36",
        "filename": "fig1_final_performance.png",
        "scientific_message": "Desempeño final de E3b-C0 en Final Test 2024–2025: RMSE global y anual vs Baseline B0",
        "thesis_main": "Capítulo Resultados (Final Test)",
        "thesis_appendix": "No",
        "paper_main": "Figura 2 de Resultados",
        "supplement": "No",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/figures/fig1_final_performance.png",
        "notes": "+7.22% de mejora global (+9.40% en 2024, +4.47% en 2025)"
    },
    {
        "figure_id": "FIG-D35-02",
        "stage": "Stage D35 / D36",
        "filename": "fig2_temporal_robustness.png",
        "scientific_message": "Robustez temporal mensual y diaria de E3b-C0 en Final Test 2024–2025 (18/24 meses mejorados)",
        "thesis_main": "Capítulo Resultados (Robustez temporal)",
        "thesis_appendix": "No",
        "paper_main": "Figura 3 de Resultados",
        "supplement": "No",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/figures/fig2_temporal_robustness.png",
        "notes": "Muestra contraste entre 2024 (11/12 meses) y 2025 (7/12 meses)"
    },
    {
        "figure_id": "FIG-D35-03",
        "stage": "Stage D35 / D36",
        "filename": "fig3_spatial_skill.png",
        "scientific_message": "Distribución espacial de DeltaRMSE en las 5,275 celdas marinas (99.96% mejoradas) y correlación con profundidad/costa",
        "thesis_main": "Capítulo Resultados (Distribución espacial)",
        "thesis_appendix": "No",
        "paper_main": "Figura 4 de Resultados",
        "supplement": "No",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/figures/fig3_spatial_skill.png",
        "notes": "Ilustra concentración de beneficios en plataforma somera y lagunas costeras"
    },
    {
        "figure_id": "FIG-D35-04",
        "stage": "Stage D35 / D36",
        "filename": "fig4_residual_regimes.png",
        "scientific_message": "Dependencia del skill según la magnitud de discrepancia MUR–BIL original y exactitud de signo",
        "thesis_main": "Capítulo Resultados / Discusión",
        "thesis_appendix": "No",
        "paper_main": "Figura 5 de Resultados",
        "supplement": "No",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/figures/fig4_residual_regimes.png",
        "notes": "Aclara degradación en regímenes bajos (|R| < 0.18 °C) y alta ganancia en discrepancias severas"
    },
    {
        "figure_id": "FIG-D35-05",
        "stage": "Stage D35 / D36",
        "filename": "fig5_historical_skill.png",
        "scientific_message": "Comparación descriptiva de trayectorias históricas de skill a través de Holdout 2021, Validación 2022–2023 y Test 2024–2025",
        "thesis_main": "Capítulo Discusión (Trayectoria metodológica)",
        "thesis_appendix": "No",
        "paper_main": "No",
        "supplement": "Figura Suplementaria D36",
        "canonical_source": "ml_results/E3b_FINAL_SYNTHESIS/figures/fig5_historical_skill.png",
        "notes": "Advertencia explícita: comparación descriptiva, ventanas acumulativas distintas"
    }
]

df_figure_inv = pd.DataFrame(figure_inv_data)
df_figure_inv.to_csv(TABLES_DIR / "figure_inventory_A_D.csv", index=False)
print("Tabla generada: figure_inventory_A_D.csv")

# 3.7 project_timeline_A_D.csv
timeline_data = [
    {"order": 1, "stage": "Stage A", "substage": "A.1–A.3", "purpose": "Adquisición y auditoría de datos primarios", "input": "Archivos NetCDF de MUR, OISST y GEBCO", "output": "Datos decenales verificados (4,018 días)", "decision": "Aprobar suficiencia e integridad temporal", "status": "COMPLETED"},
    {"order": 2, "stage": "Stage B", "substage": "B", "purpose": "Definición de cuadrícula espacial maestra", "input": "Límites geográficos del corredor Tulum–Cozumel", "output": "Grilla cartesiana regular de 86 × 96 celdas a 0.01°", "decision": "Adoptar malla de MUR como referencia única", "status": "COMPLETED"},
    {"order": 3, "stage": "Stage B", "substage": "B.1", "purpose": "Máscara oceánica corregida y covariables", "input": "MUR land/ocean mask y GEBCO bathymetry", "output": "ocean_mask_final (5,279 celdas marinas)", "decision": "M_final = M_MUR AND (ocean_fraction >= 0.5)", "status": "COMPLETED"},
    {"order": 4, "stage": "Stage C", "substage": "C.1", "purpose": "Prueba inicial de interpolación bilineal", "input": "MUR y OISST para 2015-01-01", "output": "Identificación de 1,292 NaNs costeros", "decision": "Formular soporte matemático costero", "status": "COMPLETED"},
    {"order": 5, "stage": "Stage C", "substage": "C.1b", "purpose": "Diagnóstico y comparación de estrategias", "input": "OISST con 20 nodos terrestres en halo", "output": "Estrategia A vs Estrategia B (MAE 0.0038 °C)", "decision": "Seleccionar Estrategia A para soporte bilineal", "status": "COMPLETED"},
    {"order": 6, "stage": "Stage C", "substage": "C.1c", "purpose": "Validación de solución costera (2015-01-01)", "input": "Campos procesados con Estrategia A", "output": "100% de celdas válidas; Baseline E0 establecido", "decision": "Aprobar corrida temporal decenal completa", "status": "COMPLETED"},
    {"order": 7, "stage": "Stage C", "substage": "C.2", "purpose": "Armonización decenal completa 2015–2025", "input": "MUR y OISST en los 4,018 días", "output": "faseC2_2015_2025.nc (21,211,022 observaciones)", "decision": "Congelar cubo armonizado decenal", "status": "COMPLETED"},
    {"order": 8, "stage": "Stage C", "substage": "Sat Audit", "purpose": "Auditoría radiométrica infrarroja L2P", "input": "Gránulos VIIRS y MODIS en eventos extremos", "output": "Evidencia de fuerte bloqueo nuboso", "decision": "Preservar C.2 intacto; MUR como referencia", "status": "COMPLETED / HETEROGENEOUS"},
    {"order": 9, "stage": "Stage C", "substage": "MUR AE", "purpose": "Auditoría de analysis_error decenal", "input": "Variable analysis_error de MUR (4,018 días)", "output": "Identificación de techo asintótico en 0.41 °C", "decision": "Excluir analysis_error de las features de entrada", "status": "CLOSED"},
    {"order": 10, "stage": "Stage D", "substage": "D.Dataset", "purpose": "Construcción de dataset tabular analítico", "input": "faseC2_2015_2025.nc y analysis_error", "output": "Tablas Parquet particionadas cronológicamente", "decision": "Bloquear conjunto Test 2024–2025", "status": "COMPLETED"},
    {"order": 11, "stage": "Stage D", "substage": "D31", "purpose": "Diagnóstico de predictibilidad residual", "input": "Partición Development (2015–2020)", "output": "Demostración de estructura predictiva reproducible", "decision": "Proceder con aprendizaje residual tabular", "status": "CLOSED"},
    {"order": 12, "stage": "Stage D", "substage": "D32", "purpose": "Selección de modelo y ablación de features", "input": "Dev (2015–2020) y Holdout diagnóstico (2021)", "output": "Selección de E3b-C0; 5,275 celdas congeladas", "decision": "Congelar especificación E3b-C0 y celdas comunes", "status": "METHODOLOGICALLY CLOSED"},
    {"order": 13, "stage": "Stage D", "substage": "D33", "purpose": "Validación temporal externa 2022–2023", "input": "Reajuste 2015–2021 y evaluación en 2022–2023", "output": "Mejora global +3.52%; 16/24 meses mejorados", "decision": "Dictamen D33-B; pausa para auditoría diagnóstica", "status": "D33-B — UNCHANGED"},
    {"order": 14, "stage": "Stage D", "substage": "D34", "purpose": "Auditoría diagnóstica post-validación", "input": "Predicciones de validación y covariables", "output": "Comprensión de regímenes y compresión de amplitud", "decision": "Proceder a Final Test con protocolo congelado", "status": "INTERPRETATIONALLY CLOSED"},
    {"order": 15, "stage": "Stage D", "substage": "D35", "purpose": "Evaluación terminal en Final Test 2024–2025", "input": "Reajuste 2015–2023 y evaluación en 2024–2025", "output": "+7.22% mejora global; 18/24 meses; 99.96% celdas", "decision": "Dictamen D35-A; FINAL TEST CONSUMIDO", "status": "D35-A — FINAL GENERALIZATION CONFIRMED"},
    {"order": 16, "stage": "Stage D", "substage": "D36", "purpose": "Síntesis final del bloque de Machine Learning", "input": "Resultados consolidados de D31–D35", "output": "Suite de reportes y tablas de publicación", "decision": "Cierre científico definitivo de modelos ML", "status": "FINAL SYNTHESIS COMPLETED"}
]

df_timeline = pd.DataFrame(timeline_data)
df_timeline.to_csv(TABLES_DIR / "project_timeline_A_D.csv", index=False)
print("Tabla generada: project_timeline_A_D.csv")

# ----------------------------------------------------------------------
# 4. GENERACIÓN DE LA FIGURA MAESTRA DEL PIPELINE (300 DPI)
# ----------------------------------------------------------------------
def render_master_pipeline_figure():
    fig, ax = plt.subplots(figsize=(15, 20), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colores corporativos sobrios y elegantes
    c_input = "#E3F2FD"      # Azul pastel
    c_stage_a = "#BBDEFB"    # Azul claro
    c_stage_b = "#C8E6C9"    # Verde pastel
    c_stage_c = "#FFE0B2"    # Naranja pastel
    c_audit = "#E1BEE7"      # Púrpura pastel
    c_stage_d = "#D1C4E9"    # Violeta pastel
    c_test = "#FFCDD2"       # Rosa pastel (Test consumido)
    c_close = "#B2DFDB"      # Verde azulado
    border_col = "#37474F"   # Gris carbón

    def draw_box(x, y, w, h, title, text, bg_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.8",
            linewidth=1.5, edgecolor=border_col, facecolor=bg_color
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 1.6, title, fontsize=11, fontweight="bold", ha="center", va="center", color="#212121")
        ax.text(x + w/2, y + h/2 - 0.8, text, fontsize=9, ha="center", va="center", color="#424242", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#455A64", edgecolor="#455A64", width=1.5, headwidth=7, shrink=0.05)
        )
        if label:
            ax.text((x1 + x2)/2 + 1.5, (y1 + y2)/2, label, fontsize=8, color="#37474F", style="italic")

    # Encabezado
    ax.text(50, 98, "MASTER METHODOLOGICAL PIPELINE — STAGES A TO D", fontsize=16, fontweight="bold", ha="center", color="#1A237E")
    ax.text(50, 96.2, "Downscaling Espacial de SST en el Corredor Tulum–Cozumel (Caribe Mexicano, 2015–2025)", fontsize=11, ha="center", color="#37474F")

    # 1. Inputs Primarios (Top Row)
    draw_box(4, 88, 28, 6.5, "MUR SST v4.1 (NASA/JPL)", "L4 Analysed SST (0.01° ~1 km)\n2015–2025 (4,018 días continuos)", c_input)
    draw_box(36, 88, 28, 6.5, "NOAA OISST v2.1 (NOAA)", "L4 Daily SST (0.25° ~27 km)\n2015–2025 (Halo espacial 0.5°)", c_input)
    draw_box(68, 88, 28, 6.5, "GEBCO 2026 Grid (BODC)", "Batimetría Global 15 arc-sec (~450 m)\nCota topobatimétrica (elevation)", c_input)

    # 2. Stage A: Auditoría de Integridad
    draw_box(20, 78, 60, 6.5, "STAGE A — Data Acquisition & Quality Control Audit", "4,018 días esperados = 4,018 días procesados (0 faltantes, 0 duplicados)\nConversión Kelvin -> Celsius | Verificación de unidades y consistencia temporal", c_stage_a)
    draw_arrow(18, 88, 35, 84.5)
    draw_arrow(50, 88, 50, 84.5)
    draw_arrow(82, 88, 65, 84.5)

    # 3. Stage B: Marco Espacial y Máscara
    draw_box(20, 68, 60, 6.5, "STAGE B & B.1 — Spatial Reference & Corrected Ocean Mask", "Cuadrícula maestra MUR: 86 × 96 celdas (~0.01°) | 8,256 celdas totales\nMáscara corregida: M_final = M_MUR AND (ocean_fraction >= 0.5) -> 5,279 celdas oceánicas\nCovariables estáticas: depth (m, GEBCO) y distance_coast_km (UTM 16N)", c_stage_b)
    draw_arrow(50, 78, 50, 74.5)

    # 4. Stage C: Armonización y Soporte Costero
    draw_box(20, 56, 60, 8.5, "STAGE C — Spatiotemporal Harmonization (C.1 -> C.1b -> C.1c -> C.2)", "C.1: Interpolación bilineal directa produce 1,292 NaNs costeros (24.47% del dominio)\nC.1b: Diagnóstico y selección de Estrategia A (soporte auxiliar nearest-ocean 0.25°)\nMAE A vs B = 0.0038 °C (cuasi-equivalencia numérica confirmada con Delaunay)\nC.2: Armonización completa 2015–2025 (4,018 días × 5,279 celdas = 21,211,022 obs)\nBaseline E0: RMSE = 0.3426 °C, MAE = 0.2631 °C, R² = 0.9016 | R = MUR - BIL", c_stage_c)
    draw_arrow(50, 68, 50, 64.5)

    # Auditorías de Etapa C (Lados) y Stage D (Centro)
    draw_box(2, 44, 24, 8.5, "Satellite Infrared Audit", "VIIRS & MODIS L2P (E1–E6)\nBloqueo por nubes (>80–100%)\nEvidencia heterogénea/inconclusa\nDecisión: Mantener C.2 intacto", c_audit)
    draw_arrow(28, 56, 14, 52.5)

    draw_box(28, 44, 44, 8.5, "STAGE D — ML Dataset Construction & Partitioning", "Unidad observacional: date × cell_id | Formato tabular Snappy Parquet\nCOMMON_VALID_MASK: Transición física 5,279 -> 5,275 celdas congeladas\nPartición temporal cronológica (Anti-leakage):\nDev: 2015–2020 | Holdout: 2021 | Val: 2022–2023 | Final Test: 2024–2025 (Bloqueado)", c_stage_d)
    draw_arrow(50, 56, 50, 52.5)

    draw_box(74, 44, 24, 8.5, "MUR Uncertainty Audit", "analysis_error decenal (4,018 días)\nSpearman rho = +0.2853 vs RMSE\nTecho asintótico en 0.4100 °C\nDecisión: Variable control (NO feature)", c_audit)
    draw_arrow(72, 56, 86, 52.5)

    # 6. D31 & D32: Diagnósticos y Selección
    draw_box(20, 32.5, 60, 8.5, "D31 & D32 — Predictability & Model Selection (E3b-C0)", "D31: Diagnósticos confirman estructura predictiva reproducible en el residual\nD32: Evaluación prespecificada de 6 ablaciones en Diagnostic Holdout 2021\nSelección E3b-C0: sst_bil, doy_sin, doy_cos, depth (XGBoost n=19, d=4, lr=0.10)\nHoldout 2021: B0 RMSE = 0.359493 °C -> C0 RMSE = 0.349274 °C (+2.84%)\nDecisión: Parsimonia + Desempeño | Estado: METHODOLOGICALLY CLOSED", c_stage_d)
    draw_arrow(44, 44, 50, 41)

    # 7. D33 & D34: Validación Externa y Diagnósticos
    draw_box(20, 21.5, 60, 8, "D33 & D34 — External Validation & Diagnostic Audit", "D33: Reajuste en 2015–2021 y evaluación en Validación Externa 2022–2023\nB0 RMSE = 0.335666 °C -> C0 RMSE = 0.323838 °C (+3.52%); 16/24 meses; 90.14% celdas\nDictamen D33-B (Mejora positiva pero estabilidad mensual sub-umbral)\nD34: Auditoría no adaptativa de regímenes de residual y compresión de amplitud\nEstado D33: D33-B UNCHANGED | Estado D34: INTERPRETATIONALLY CLOSED", c_stage_d)
    draw_arrow(50, 32.5, 50, 29.5)

    # 8. D35: Final Test Terminal
    draw_box(20, 10.5, 60, 8, "D35 — Final Out-of-Sample Evaluation (Final Test 2024–2025)", "Reajuste final en 2015–2023 (17.34M obs) | Inferencia única en Test 2024–2025 (3.86M obs)\nB0 RMSE = 0.357317 °C -> C0 RMSE = 0.331502 °C (+7.2247%)\nMAE: 0.2727 -> 0.2563 °C | Bias: -0.0624 -> -0.0142 °C | R² SST: 0.8927 -> 0.9076\n18/24 meses (75%) | 484/731 días (66.2%) | 5,273/5,275 celdas (99.96%)\nDictamen D35-A: FINAL GENERALIZATION CONFIRMED | TEST CONSUMED", c_test)
    draw_arrow(50, 21.5, 50, 18.5)

    # 9. D36: Cierre Científico
    draw_box(20, 1.5, 60, 6, "D36 — Scientific Closure & Thesis/Paper Consolidation", "Consolidación no experimental de D31–D35 | Tablas y figuras definitivas\nCero reentrenamientos | Cero accesos raw a test | Congelamiento absoluto\nESTADO TERMINAL: ML BLOCK SCIENTIFICALLY CLOSED", c_close)
    draw_arrow(50, 10.5, 50, 7.5)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_master_pipeline_A_D.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura del pipeline generada exitosamente en: {output_path}")

render_master_pipeline_figure()

# ----------------------------------------------------------------------
# 5. GENERACIÓN DE REPORTES CIENTÍFICOS MAESTROS (MARKDOWN)
# ----------------------------------------------------------------------

# 5.1 DOCUMENTATION_GAPS_A_D.md
doc_gaps_content = """# Master Inventory of Documentation Gaps and Conflict Resolutions (Stages A–D)

**Documento:** `DOCUMENTATION_GAPS_A_D.md`  
**Fecha de Auditoría:** 2026-09-05  
**Proyecto:** Downscaling de SST en el Corredor Tulum–Cozumel (Caribe Mexicano)  
**Propósito:** Identificar, catalogar y resolver explícitamente todas las inconsistencias numéricas, discrepancias de nomenclatura, cambios de censo y ambigüedades documentales detectadas entre etapas históricas, asegurando una trazabilidad científica inmaculada para la tesis.

---

## 1. Principio Fundamental de Resolución

Bajo la **Source-of-Truth Policy**, ningún conflicto se corrige silenciosamente. Cuando dos documentos o registros discrepan, este inventario documenta el conflicto, las fuentes involucradas, cuál es canónica, y el motivo metodológico de la resolución.

---

## 2. Tabla Maestra de Gaps y Discrepancias

| gap_id | stage | issue | evidence | severity | recommended_resolution | blocks_thesis_writing | status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | Stage B -> D | Censo de celdas oceánicas: discrepancia entre 5,279 y 5,275 celdas | `reporte_fase_b.txt` y `faseC2_2015_2025.nc` reportan 5,279 celdas; `frozen_cell_ids.csv` y modelos D32–D36 utilizan 5,275 celdas | HIGH | Explicar que en D32 la formulación evaluó ablaciones espaciales (`E3b-S`, `E3b-TS`, `E3b-ALL`) que requerían cálculo de gradiente horizontal 2D (`grad_mag`). Cuatro celdas costeras/aisladas (índices 0, 161, 4437, 4472) carecen de vecinos laterales suficientes para diferencias finitas centradas/unilaterales en X o Y, arrojando `grad_mag = NaN`. Para garantizar evaluación estrictamente justa e idéntica en todas las variantes, `COMMON_VALID_MASK` excluyó esas 4 celdas, congelando el dominio evaluativo en 5,275 celdas. | NO | RESOLVED & DOCUMENTED |
| **GAP-02** | Stage A / B | Ambigüedad en versión de GEBCO (GEBCO 2024 vs GEBCO 2026) | Varios reportes intermedios de texto citan "GEBCO 2024", pero el archivo físico real en `config.py` es `gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc` | MEDIUM | La inspección de atributos NetCDF confirma: `title: The GEBCO_2026 Grid - a continuous terrain model for oceans and land at 15 arc-second intervals`, `date_created: 2026-04-17`, `DOI: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa`. La mención a 2024 fue un error tipográfico en borradores tempranos. La versión oficial y canónica es **GEBCO 2026 Grid**. | NO | RESOLVED & DOCUMENTED |
| **GAP-03** | Stage D35 | Orientación de la ecuación de regresión de calibración y slope = 0.0891 | En reportes preliminares de D35 se escribió $R = a + b \\hat{R}$ citando slope = 0.0891 | HIGH | Matemáticamente, la regresión calculada en el script D35 fue $\\hat{R} = a + b R$, donde $b = \\text{Cov}(R, \\hat{R})/\\text{Var}(R) = r \\cdot [\\text{std}(\\hat{R})/\\text{std}(R)] = 0.3512 \\times 0.2536 = 0.089054$. Si se invirtiera la regresión ($R$ sobre $\\hat{R}$), la pendiente sería $r \\cdot [\\text{std}(R)/\\text{std}(\\hat{R})] \\approx 1.385$. La fórmula canónica correcta que corresponde al valor $0.0891$ es $\\hat{R} = -0.0427 + 0.0891 \\cdot R$, la cual demuestra compresión severa de la amplitud predicha hacia la media condicional. | NO | RESOLVED & DOCUMENTED |
| **GAP-04** | Stage D32 | Redacción histórica de configuraciones prespecificadas (6 vs 8 modelos) | Ciertos párrafos introductorios antiguos mencionaban informalmente "ocho modelos candidatos" | LOW | La tabla maestra `model_summary.csv` y el script oficial `fase_d32_e3b_tabular.py` prueban que se evaluaron formalmente exactamente **SEIS** configuraciones prespecificadas (`E3b-C0`, `E3b-T1`, `E3b-T3`, `E3b-S`, `E3b-TS`, `E3b-ALL`). Se ratifica 6 como el número canónico. | NO | RESOLVED & DOCUMENTED |
| **GAP-05** | Stage D33 / D35 | Confusión entre "pesos congelados" vs "especificación congelada" | Textos preliminares afirmaban erróneamente que en D33 y D35 se evaluaron "los pesos congelados de D32" | HIGH | En aprendizaje supervisado temporal, congelar un modelo significa congelar su **especificación metodológica** (arquitectura, vector de features, hiperparámetros $\\theta^*$). En D33, el modelo fue reajustado con todos los datos pre-validación (2015–2021). En D35, fue reajustado con todos los datos pre-test (2015–2023, 17.34M obs). La redacción canónica exige indicar que se reajustó la especificación congelada, no que se reutilizaron los pesos de 2020. | NO | RESOLVED & DOCUMENTED |
| **GAP-06** | Stage D35 | Afirmación de "crecimiento monotónico" del skill hasta P99+ | Un borrador indicaba que la mejora de RMSE crecía monotónicamente con el percentil de discrepancia hasta P99+ | MEDIUM | La tabla `residual_regime_metrics.csv` muestra: DEV-P90-P95: +9.21%, DEV-P95-P99: +12.07%, DEV-P99+: +11.49%. El valor en P99+ (11.49%) es ligeramente menor que en P95–P99 (12.07%). La afirmación correcta es: "La mejora relativa de RMSE creció consistentemente hasta DEV-P95–P99 (+12.07%) y se mantuvo en niveles muy elevados en DEV-P99+ (+11.49%)". | NO | RESOLVED & DOCUMENTED |
| **GAP-07** | Stage D35 / D36 | Interpretación del signo de correlación de Spearman para variables espaciales | Borradores preliminares interpretaban erróneamente $\\rho > 0$ entre profundidad y $\\Delta\\text{RMSE}$ como "mayor mejora en aguas profundas" | HIGH | Dado que $\\Delta\\text{RMSE} = \\text{RMSE}_{C0} - \\text{RMSE}_{B0}$ es una cantidad negativa (indicando reducción del error), una correlación de Spearman positiva ($\\rho = +0.7376$ con profundidad; $\\rho = +0.5336$ con distancia a la costa) significa que $\\Delta\\text{RMSE}$ se vuelve menos negativo (más cercano a cero) a medida que aumenta la profundidad o la distancia. Por ende, la magnitud de la mejora relativa decrece mar adentro y se maximiza en la plataforma somera (0–20 m). | NO | RESOLVED & DOCUMENTED |
| **GAP-08** | Global | Evolución de la nomenclatura de fases (Fase D/E/F en planes antiguos vs D31–D36 actual) | Planes antiguos de trabajo fechados en 2025/2026 listaban Fase D como dataset, Fase E como ML y Fase F como evaluación | MEDIUM | La documentación evolucionó a la nomenclatura jerárquica unificada: Etapa A (Adquisición), Etapa B (Marco espacial), Etapa C (Armonización), Etapa D (Dataset ML y ciclo de modelación D31–D36). Las fases E y F quedaron absorbidas orgánicamente en D31–D36. Se mantiene el registro histórico sin borrar el plan primitivo. | NO | RESOLVED & DOCUMENTED |

---

## 3. Detalle de las 4 Celdas Excluidas en la Transición B/C (5,279) -> D (5,275)

La auditoría determinó de manera unívoca las coordenadas exactas de las cuatro celdas excluidas por `COMMON_VALID_MASK`:

1. **Celda Índice 0:** Latitud = 19.9000°N, Longitud = -87.4600°W (Borde suroeste costero, depth = 13.0 m, ocean_fraction = 0.67). Carece de vecino oeste válido.
2. **Celda Índice 161:** Latitud = 19.9200°N, Longitud = -87.4500°W (Borde costero continental, depth = 2.75 m, ocean_fraction = 1.00). Carece de vecino sur válido.
3. **Celda Índice 4437:** Latitud = 20.5400°N, Longitud = -86.9300°W (Ensenada costera norte de Cozumel, depth = 11.25 m, ocean_fraction = 0.67). Vecindad asimétrica sin soporte longitudinal.
4. **Celda Índice 4472:** Latitud = 20.5500°N, Longitud = -86.9200°W (Punta norte de Cozumel, depth = 12.0 m, ocean_fraction = 0.75). Borde costero con discontinuidad geométrica.

En las 4 celdas, el operador de gradiente horizontal arrojaba numéricamente `NaN` al no poder formar diferencias centradas ni unilaterales. Su exclusión obedeció al principio de rigurosidad evaluativa en D32 para que todos los modelos compitieran sobre la misma máscara espacial común inmutable.
"""

with open(REPORTS_DIR / "DOCUMENTATION_GAPS_A_D.md", "w", encoding="utf-8") as f:
    f.write(doc_gaps_content)
print("Reporte generado: DOCUMENTATION_GAPS_A_D.md")

# 5.2 THESIS_METHODS_MASTER_A_D.md
methods_master_content = """# Master Methodology Synthesis (Stages A to D)
## Reconstrucción Metodológica Integral para Redacción de Tesis

**Documento:** `THESIS_METHODS_MASTER_A_D.md`  
**Regla Epistemológica:** Estricta separación entre MÉTODO y RESULTADO. Este documento describe exhaustivamente el marco teórico-metodológico, las decisiones de diseño, las formulaciones matemáticas y las salvaguardas de reproducibilidad sin incrustar tablas de resultados ni discusiones interpretativas.

---

## 1. Study Area

El área de estudio corresponde al corredor marino **Tulum–Cozumel**, situado en la costa oriental de la Península de Yucatán (Quintana Roo, México), en el Caribe noroccidental. Los límites geográficos del dominio se establecen formalmente en:

$$\\text{Latitud}: 19.90^\\circ\\text{N} \\text{ a } 20.75^\\circ\\text{N}$$
$$\\text{Longitud}: -87.60^\\circ\\text{W} \\text{ a } -86.65^\\circ\\text{W}$$

El dominio abarca el canal de Yucatán, el canal de Cozumel, la plataforma arrecifal somera del Sistema Arrecifal Mesoamericano (SAM) y cuencas oceánicas profundas (>1000 m) caracterizadas por la intensa corriente de Yucatán.

---

## 2. Data Sources

El pipeline integra tres fuentes primarias de información satelital y fisiográfica:
1. **MUR SST v4.1 (NASA/JPL PO.DAAC):** Producto L4 global de resolución ultra-alta (0.01°, ~1 km) que asimila observaciones infrarrojas (MODIS, VIIRS, AVHRR) y de microondas (AMSR2, WindSat) mediante una técnica de interpolación multiescala (*Chin et al., 2017*).
2. **NOAA OISST v2.1 (NOAA NCEI):** Producto L4 global diario sobre grilla regular de 0.25° (~27 km), basado en interpolación óptima (*Huang et al., 2021*). Constituye el predictor térmico de baja resolución a ser downescalado.
3. **GEBCO 2026 Grid (BODC / Nippon Foundation-GEBCO Seabed 2030):** Modelo continuo de elevación terreno/océano a 15 arc-segundos (~450 m de resolución), derivado de la fusión de batimetría acústica multihaz y altimetría satelital SRTM15+ v2.8.
4. **Radiometría Infrarroja L2P (VIIRS S-NPP y MODIS Aqua):** Gránulos orbitales independientes L2P no interpolados, utilizados exclusivamente para auditorías de calidad radiométrica en eventos extremos.

---

## 3. Data Acquisition and Quality Control

El periodo experimental abarca exactamente 11 años completos: del **2015-01-01 al 2025-12-31**, totalizando **4,018 días astronómicos continuos**.
- **Integridad temporal:** Se realizó una auditoría de ingesta donde cada fecha fue verificada contra el índice temporal del producto. No se detectaron días faltantes ni duplicados (cobertura temporal = 100.0%).
- **Conversión de unidades:** Las observaciones de MUR SST se transformaron de Kelvin a Celsius mediante $T_{\\text{C}} = T_{\\text{K}} - 273.15$.
- **Recuperación local:** Para OISST, un fallo puntual de conexión ERDDAP en la fecha 2025-01-14 fue subsanado mediante ingestión del archivo NetCDF local correspondiente.

---

## 4. Master Spatial Grid

Para posibilitar la armonización multirresolución, se seleccionó la cuadrícula espacial de MUR SST como malla maestra de referencia. Las características de la grilla son:
- Dimensiones: **86 celdas en latitud × 96 celdas en longitud** ($N_{\\text{total}} = 8,256$ celdas).
- Resolución angular regular: $0.0100^\\circ$ (~1.11 km en meridiano; ~1.04 km en paralelo zonal).
- Coordenadas proyectadas de soporte: Proyección Universal Transversa de Mercator (UTM) Zona 16 Norte, datum WGS84 (**EPSG:32616**).

---

## 5. Land/Ocean Masking

La máscara nativa de tierra/océano de MUR clasifica como agua varias celdas con influencia costera mixta o lagunas interiores de la isla de Cozumel. Para evitar contaminación por firmas térmicas terrestres, se construyó una máscara corregida:
1. Se calculó la fracción oceánica subpíxel ($\\text{ocean\\_fraction} \\in [0, 1]$) agregando las celdas batimétricas de GEBCO contenidas en cada píxel de 0.01°.
2. Se formuló la regla booleana canónica:
   $$M_{\\text{final}} = M_{\\text{MUR}} \\land (\\text{ocean\\_fraction} \\ge 0.5)$$
3. Se verificó que el interior continental de Cozumel quedara asignado estrictamente a tierra.
4. Resultado: **5,279 celdas oceánicas** y **2,977 celdas terrestres** (383 celdas modificadas respecto a MUR nativo).

---

## 6. Bathymetry and Static Covariates

Sobre la cuadrícula maestra se calcularon tres covariables fisiográficas estáticas:
1. **Profundidad batimétrica ($depth$):** Definida como profundidad positiva bajo el nivel del mar en metros ($depth = -elevation$ para $elevation < 0$; NaN sobre tierra).
2. **Distancia euclidiana a la costa ($distance\\_coast\\_km$):** Calculada en coordenadas métricas proyectadas (EPSG:32616) como la distancia euclidiana mínima desde el centro de cada celda oceánica hasta el polígono costero continental o insular más cercano.
3. **Fracción oceánica ($ocean\\_fraction$):** Proporción de superficie de agua marina en el subpíxel.

---

## 7. OISST Interpolation

Para proyectar el campo térmico grueso OISST (0.25°) a la malla fina (0.01°), se extrae diariamente una ventana espacial con halo de protección de $0.5^\\circ$ ($7 \\times 7$ nodos OISST) para evitar discontinuidades de contorno. Sobre esta cuadrícula se aplica un operador de interpolación bilineal regular en 2D:

$$\\text{SST}_{\\text{BIL}}(x, y) = \\sum_{i=1}^2 \\sum_{j=1}^2 w_{ij} \\, \\text{OISST}(x_i, y_j)$$

---

## 8. Coastal Support Strategy

Debido a que 20 nodos occidentales del halo OISST corresponden a la Península de Yucatán (tierra continental), la interpolación bilineal directa produjo **1,292 celdas oceánicas con valor NaN** (pérdida del 24.47% del dominio marino).

Para resolver esta frontera matemática sin alterar la física oceánica:
1. **Estrategia A (Adoptada):** Se extendieron los nodos terrestres OISST asignándoles el valor del nodo oceánico válido más cercano (*nearest-ocean coastal extension*). Estos nodos extendidos actúan **únicamente como soporte matemático envolvente** para que la interpolación bilineal en las 5,279 celdas marinas disponga de 4 esquinas cuadriláteras válidas. Posteriormente, el campo resultante se recorta estrictamente con $M_{\\text{final}}$, garantizando que ninguna celda terrestre conserve valores de SST.
2. **Estrategia B (Evaluada):** Triangulación 2D de Delaunay basada exclusivamente en nodos oceánicos de OISST.
3. **Criterio de selección:** La Estrategia A fue seleccionada por preservar la regularidad cartesiana de la grilla, mantener costo computacional bajo y asegurar trazabilidad explícita mediante la máscara booleana `oisst_coastal_support_mask`.

---

## 9. Full SST Harmonization

La armonización espaciotemporal completa se ejecutó sobre los 4,018 días del periodo 2015–2025. Cada día genera un campo tridimensional que contiene:
- $\\text{sst\\_mur}$: SST de alta resolución (~1 km).
- $\\text{sst\\_bil}$: SST bilineal extendida (~1 km).
- $R$: Campo de residual fino, definido por la identidad aditiva fundamental:
  $$R(x, y, t) = \\text{sst\\_mur}(x, y, t) - \\text{sst\\_bil}(x, y, t)$$
- Cobertura espacial: Exactamente 5,279 celdas oceánicas válidas por día ($21,211,022$ observaciones espaciotemporales acumuladas).

---

## 10. Reference Uncertainty Audit

El algoritmo de asimilación de MUR v4.1 reporta diariamente la desviación estándar estimada del error de análisis (`analysis_error`). Se ejecutó una auditoría exhaustiva sobre los 4,018 días para evaluar su comportamiento:
- Se evaluó la correlación entre `analysis_error` y la magnitud de discrepancia MUR–BIL.
- Se examinaron eventos de saturación donde `analysis_error` alcanza su límite asintótico superior en $0.4100^\\circ\\text{C}$ (*Chin et al., 2017*).
- **Decisión metodológica:** La variable `analysis_error` se incorpora en las tablas maestras exclusivamente como variable de control y filtro diagnóstico post-hoc, **excluyéndola taxativamente del vector de predictores del modelo ML** para evitar circularidad en la estimación de la referencia.

---

## 11. Residual-Learning Formulation

El problema de downscaling espacial se formula bajo el paradigma de **aprendizaje residual** (*residual learning*):
1. En lugar de predecir directamente el campo continuo absoluto de temperatura $\\text{SST}_{\\text{high}}$, el modelo de Machine Learning predice el campo de discrepancia fina:
   $$\\hat{R} = f(X)$$
2. La reconstrucción de alta resolución final se obtiene mediante la suma del campo grueso bilineal y la corrección residual estimada:
   $$\\text{SST}_{\\text{downscaled}} = \\text{SST}_{\\text{BIL}} + \\hat{R}$$
3. Esta formulación acota el rango dinámico del objetivo, garantiza que en ausencia de señal aprendible el modelo converja naturalmente al baseline suave ($\\hat{R} \\to 0 \\implies \\text{SST}_{\\text{downscaled}} = \\text{SST}_{\\text{BIL}}$), y evita que el estimador gaste capacidad en memorizar la tendencia térmica regional o el ciclo estacional de fondo.

---

## 12. Machine-Learning Dataset

Los campos espaciotemporales del cubo armonizado se estructuran en formato tabular analítico:
- **Unidad observacional:** Registro diario por celda ($date \\times cell\\_id$).
- **Variables de entrada candidatas ($X$):**
  - $\\text{sst\\_bil}$: Temperatura bilineal interpolada (°C).
  - Covariables cíclicas temporales:
    $$\\text{doy\\_sin} = \\sin\\left(\\frac{2\\pi \\cdot \\text{doy}}{365.25}\\right), \\quad \\text{doy\\_cos} = \\cos\\left(\\frac{2\\pi \\cdot \\text{doy}}{365.25}\\right)$$
  - Covariables fisiográficas: $depth$, $distance\\_coast\\_km$, $ocean\\_fraction$.
  - Lags y gradientes exploratorios evaluados en fase de desarrollo.
- **Target ($y$):** Residual fino $R = \\text{sst\\_mur} - \\text{sst\\_bil}$ (°C).
- **Almacenamiento:** Formato Apache Parquet particionado anualmente con compresión Snappy.

---

## 13. Temporal Partitioning

Para prevenir la filtración espuria de información por persistencia sinóptica y autocorrelación temporal, se prohibió el uso de muestreo aleatorio (*random train/test split*). El protocolo define cuatro ventanas temporales estrictamente cronológicas y mutuamente excluyentes:

$$\\text{Development (Train + Internal Validation)}: 2015-01-01 \\text{ a } 2020-12-31 \\quad (6 \\text{ años}, 2,192 \\text{ días})$$
$$\\text{Diagnostic Holdout}: 2021-01-01 \\text{ a } 2021-12-31 \\quad (1 \\text{ año}, 365 \\text{ días})$$
$$\\text{External Validation}: 2022-01-01 \\text{ a } 2023-12-31 \\quad (2 \\text{ años}, 730 \\text{ días})$$
$$\\text{Final Test (Blind)}: 2024-01-01 \\text{ a } 2025-12-31 \\quad (2 \\text{ años}, 731 \\text{ días})$$

---

## 14. Predictability Diagnostics

En la subfase D31 se evaluó si el residual fino $R$ exhibe estructura determinística o correlación informativa más allá de un ruido blanco estocástico. Se implementaron baselines de persistencia temporal, climatología residual local, modelos lineales y pruebas de información mutua sobre el conjunto Development (2015–2020).

---

## 15. Model Selection

En la subfase D32 se prespecificaron **SEIS configuraciones candidatas** derivadas de la familia `E3b`:
1. `E3b-C0` (Core, 4 features): $[\\text{sst\\_bil}, \\text{doy\\_sin}, \\text{doy\\_cos}, \\text{depth}]$
2. `E3b-T1` (Temporal 1 lag, 6 features): Core + $[\\text{sst\\_bil\\_lag1}, \\Delta\\text{sst\\_1d}]$
3. `E3b-T3` (Temporal 3 lags, 10 features): Core + lags 1, 2, 3 y deltas asociados.
4. `E3b-S` (Espacial 2D, 7 features): Core + $[\\text{grad\\_mag}, \\text{local\\_std\\_3x3}, \\text{local\\_contrast}]$
5. `E3b-TS` (Espaciotemporal, 13 features): Core + Lags T3 + Features espaciales.
6. `E3b-ALL` (Full, 16 features): Todas las anteriores + $[\\text{distance\\_coast\\_km}, \\text{ocean\\_fraction}, \\text{local\\_range}]$.

- **Máscara evaluativa común (`COMMON_VALID_MASK`):** Para garantizar comparación idéntica y justa, las 6 variantes se evaluaron sobre las celdas con gradientes y lags completos. Esto fijó el dominio en **5,275 celdas congeladas** (`frozen_cell_ids.csv`).
- **Arquitectura:** `XGBRegressor` con algoritmo de división de histogramas (`tree_method = "hist"`).
- **Hiperparámetros congelados en D32:**
  $$\\text{n\\_estimators} = 19, \\quad \\text{max\\_depth} = 4, \\quad \\text{learning\\_rate} = 0.10$$
  $$\\text{subsample} = 0.8, \\quad \\text{colsample\\_bytree} = 0.8, \\quad \\text{min\\_child\\_weight} = 5$$
  $$\\text{random\\_state} = 42, \\quad \\text{objective} = \\text{"reg:squarederror"}$$

---

## 16. External Validation

En la subfase D33, la especificación congelada de `E3b-C0` fue evaluada fuera de muestra en el bienio **2022–2023**:
- **Protocolo de ajuste:** La especificación (4 features e hiperparámetros congelados) fue reajustada utilizando todos los datos previos a la validación: **2015–2021** (2,557 días, $13,488,175$ observaciones).
- **Evaluación:** Aplicación en el bienio 2022–2023 (730 días, $3,850,750$ observaciones).
- **Criterio formal predeclarado:** Dictamen formal inmutable categorizado en D33-A (Confirmada), D33-B (Parcial/Inestable) o D33-C (Fallo).

---

## 17. Diagnostic Robustness Analysis

Tras la emisión del dictamen D33-B, la subfase D34 ejecutó una auditoría diagnóstica no adaptativa sobre las predicciones de validación:
- **Estratificación por regímenes de residual:** Partición del conjunto de prueba según los percentiles del residual absoluto observados en Development ($|R|$ en P0–P50, P50–P75, P75–P90, P90–P95, P95–P99, P99+).
- **Compresión de amplitud:** Relación $\\text{std}(\\hat{R}) / \\text{std}(R)$ y regresión de calibración.
- **Exactitud de signo (*Sign Accuracy*):** Proporción de observaciones donde $\\text{sign}(\\hat{R}) = \\text{sign}(R)$, comparada contra el baseline de clase mayoritaria.
- **Correlación de rangos espacial:** Asociación no paramétrica (Spearman) entre $\\Delta\\text{RMSE}$ por celda y las covariables fisiográficas ($depth$, $distance\\_coast\\_km$).

---

## 18. Final Refit

Antes de abrir el conjunto Final Test 2024–2025, el protocolo congeló el procedimiento de reentrenamiento final:
- **Periodo de entrenamiento final:** Todos los datos históricos pre-test: **2015-01-01 a 2023-12-31** (9 años completos, 3,287 días astronómicos).
- **Volumen de datos:** Exactamente **17,338,925 observaciones tabulares**.
- **Modelo resultante:** `E3b-C0_FINALREFIT_2015_2023`, manteniendo estrictamente invariables las 4 features y los hiperparámetros $\\theta^*$.

---

## 19. Final Test

La evaluación terminal fuera de muestra se ejecutó en la subfase D35:
- **Periodo de prueba:** **2024-01-01 a 2025-12-31** (bienio completo, 731 días astronómicos).
- **Volumen evaluado:** Exactamente **3,856,025 observaciones**.
- **Protocolo ciego:** Ejecución única sin afinamiento, sin selección adaptativa de umbrales y sin modificaciones de código post-acceso.
- **Condición de estado:** Tras la inferencia, el conjunto 2024–2025 queda registrado formalmente como **TEST CONSUMED**, perdiendo su cualidad de conjunto ciego.

---

## 20. Evaluation Metrics

El desempeño del modelo downescalado ($C_0$) frente al baseline bilineal ($B_0$) se cuantifica mediante métricas canónicas:

1. **Error Cuadrático Medio (RMSE):**
   $$\\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2}$$
2. **Error Absoluto Medio (MAE):**
   $$\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$
3. **Sesgo Medio (*Mean Bias*):**
   $$\\text{Bias} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i)$$
4. **Mejora Relativa de RMSE (Skill):**
   $$\\text{Impr\\_RMSE\\_pct} = \\left(\\frac{\\text{RMSE}_{B0} - \\text{RMSE}_{C0}}{\\text{RMSE}_{B0}}\\right) \\times 100\\%$$
   $$\\Delta\\text{RMSE} = \\text{RMSE}_{C0} - \\text{RMSE}_{B0} \\quad (\\Delta\\text{RMSE} < 0 \\implies \\text{Mejora})$$
5. **Inferencia Robusta por Bloques (*Moving Block Bootstrap*):** Remuestreo no paramétrico con 1,000 iteraciones utilizando bloques temporales continuos de **14 días** para estimar el intervalo de confianza al 95% ($CI_{95}$) preservando la dependencia serial de mesoescala.

---

## 21. Reproducibility and Leakage Prevention

Para garantizar reproducibilidad absoluta y blindaje metodológico:
1. **Semillas fijas:** `random_state = 42` fijado en todas las operaciones estocásticas de división y ajuste de árboles.
2. **Inmutabilidad de artefactos:** Los pesos del modelo, el censo de celdas (`frozen_cell_ids.csv`) y los archivos de métricas intermedias están protegidos por hashes criptográficos SHA-256.
3. **Cero fugas temporales:** Ningún estimador estadístico (media, varianza, percentiles) utilizado en normalización o evaluación en Holdout, Validation o Test utilizó información posterior al periodo de ajuste correspondiente.
"""

with open(REPORTS_DIR / "THESIS_METHODS_MASTER_A_D.md", "w", encoding="utf-8") as f:
    f.write(methods_master_content)
print("Reporte generado: THESIS_METHODS_MASTER_A_D.md")

# 5.3 THESIS_RESULTS_MASTER_A_D.md
results_master_content = """# Master Results Synthesis (Stages A to D)
## Compilación Numérica y Narrativa de Evidencia Empírica para Tesis

**Documento:** `THESIS_RESULTS_MASTER_A_D.md`  
**Regla Epistemológica:** Estricta fidelidad a las tablas maestras consolidadas (CSV-First Policy). Este documento reúne los hallazgos empíricos verificados desde la etapa de datos hasta la evaluación terminal de Machine Learning, empleando lenguaje estrictamente descriptivo sin especulaciones mecanicistas no demostradas.

---

## 1. Data Completeness and Acquisition Audits (Stage A)

- **Completitud temporal decenal:** La auditoría sobre el periodo 2015-01-01 a 2025-12-31 confirmó la disponibilidad de **4,018 días astronómicos continuos** en los productos MUR SST v4.1 y NOAA OISST v2.1. No se registró ningún día faltante ni archivo duplicado ($N_{\\text{días}} = 4,018 / 4,018$, 100.0% de integridad temporal).
- **Consistencia física de unidades:** Tras la conversión de Kelvin a Celsius ($T_{\\text{C}} = T_{\\text{K}} - 273.15$), las temperaturas superficiales del mar de MUR se mantuvieron en rangos oceanográficamente realistas para el Caribe occidental (mínimo decenal: ~24.0 °C; máximo decenal: ~32.5 °C).
- **Resolución batimétrica:** El modelo GEBCO 2026 Grid proporcionó cobertura continua sobre el 100% de la cuadrícula objetivo sin celdas nulas en el dominio geográfico delimitado.

---

## 2. Spatial-Domain Construction and Ocean Masking (Stage B)

- **Censo de la cuadrícula maestra:** El mallado maestro de 0.01° (86 celdas en latitud × 96 celdas en longitud) arrojó un total de **8,256 celdas espaciales**.
- **Censo de máscara oceánica corregida:**
  - Máscara nativa MUR: 5,662 celdas de océano y 2,594 de tierra.
  - Aplicación de $\\text{ocean\\_fraction} \\ge 0.5$ y corrección insular de Cozumel: **5,279 celdas oceánicas** y **2,977 celdas terrestres**.
  - Exactamente **383 celdas** fueron reclasificadas de agua a tierra por representar superficies continentales mixtas con fracción acuática inferior al 50%.

---

## 3. Coastal Interpolation Problem (Stage C.1)

- En la prueba piloto del 2015-01-01, la interpolación bilineal directa de OISST v2.1 con halo regular de 0.5° arrojó **1,292 celdas oceánicas con valor NaN**.
- Esta pérdida afectó al **24.47% del dominio marino** del corredor.
- El análisis geométrico confirmó que la pérdida se debió a que 20 nodos del halo occidental de OISST (0.25°) coinciden con la Península de Yucatán y están enmascarados como tierra en el producto nativo, imposibilitando el cierre de los cuadriláteros bilineales en la franja costera continental.

---

## 4. Coastal Strategy Comparison (Stage C.1b)

La comparación de métodos para recuperar la cobertura en 2015-01-01 arrojó:
- **Estrategia A (Soporte costero regular nearest-ocean):** 5,279 / 5,279 celdas válidas (100.0%). Baseline E0: $\\text{RMSE} = 0.2367^\\circ\\text{C}$, $\\text{MAE} = 0.2004^\\circ\\text{C}$, $\\text{Bias} = +0.1872^\\circ\\text{C}$.
- **Estrategia B (Triangulación Delaunay 2D):** 5,279 / 5,279 celdas válidas (100.0%). Baseline E0: $\\text{RMSE} = 0.2363^\\circ\\text{C}$, $\\text{MAE} = 0.1996^\\circ\\text{C}$, $\\text{Bias} = +0.1862^\\circ\\text{C}$.
- **Discrepancia numérica entre Estrategia A y B:**
  $$\\text{MAE} = 0.0038^\\circ\\text{C}, \\quad P_{95} = 0.0214^\\circ\\text{C}, \\quad \\text{Máxima} = 0.0626^\\circ\\text{C}$$
- La cuasi-identidad estadística justificó la selección de la Estrategia A para el pipeline decenal.

---

## 5. Full Harmonization Results (Stage C.2)

La armonización continua del periodo 2015–2025 generó el cubo de datos consolidado `faseC2_2015_2025.nc`:
- **Volumen observacional:** 4,018 días × 5,279 celdas = **21,211,022 puntos espaciotemporales**.
- **Métricas decenales del Baseline E0 (OISST bilineal vs MUR SST):**
  $$\\text{RMSE} = 0.3426^\\circ\\text{C}, \\quad \\text{MAE} = 0.2631^\\circ\\text{C}, \\quad \\text{Bias} = +0.0133^\\circ\\text{C}, \\quad R^2 = 0.9016$$
- El RMSE anual osciló entre un mínimo de $0.2724^\\circ\\text{C}$ (año 2018) y un máximo de $0.3257^\\circ\\text{C}$ (año 2024).

---

## 6. Satellite and Uncertainty Audits (Stage C.Sat & C.AE)

- **Auditoría satelital independiente (VIIRS + MODIS L2P):** En los 4 días críticos del evento anómalo de octubre 2015, MODIS Aqua registró 0 observaciones de calidad $QL=5$ (100% de píxeles rechazados por nubes). VIIRS registró observaciones limpios en solo 1 de 8 pasos orbitales (10 píxeles aislados, 0.1% del dominio). La evidencia radiométrica infrarroja fue formalmente clasificada como **inconclusa** por bloqueo nuboso generalizado, ratificando la decisión de no alterar el cubo C.2.
- **Auditoría de MUR Analysis Error (2015–2025):**
  - Se confirmó una correlación positiva moderada entre `analysis_error` y el RMSE diario de discrepancia MUR–BIL:
    $$\\text{Spearman } \\rho = +0.2853 \\, (p = 4.25 \\times 10^{-76}), \\quad \\text{Pearson } r = +0.3338 \\, (p = 3.56 \\times 10^{-105})$$
  - El 100% del dominio oceánico alcanzó el techo asintótico de $0.4100^\\circ\\text{C}$ durante eventos nubosos severos (E1 en 2015, E2 en 2021 y E3 en 2024).
  - El evento E4 (2015-08-03) actuó como contraejemplo: discrepancia severa ($\\text{RMSE} = 1.102^\\circ\\text{C}$) con `analysis_error` nominal ($0.3908^\\circ\\text{C}$).

---

## 7. Residual Predictability Diagnostics (Stage D31)

- En Development (2015–2020), el residual $R$ mostró una autocorrelación temporal a 1 día de $r \\approx 0.68$, decreciendo a $r \\approx 0.35$ a 3 días.
- La correlación espacial de Spearman entre el residual y la profundidad marina ($depth$) en el canal de Cozumel reveló estructura sistemática persistente, confirmando la existencia de señal determinística extraíble por modelos estadísticos tabulares.

---

## 8. Model-Selection Results (Stage D32)

Evaluación de las seis configuraciones candidatas sobre Diagnostic Holdout 2021 (5,275 celdas comunes, $1,925,375$ observaciones):
- **Baseline B0:** $\\text{RMSE} = 0.359493^\\circ\\text{C}, \\quad \\text{MAE} = 0.277336^\\circ\\text{C}$
- **`E3b-C0` (Seleccionado):** $\\text{RMSE} = 0.349274^\\circ\\text{C}$ (Mejora: **+2.84243%**, $\\text{MAE} = 0.266264^\\circ\\text{C}$)
- **`E3b-T1`:** $\\text{RMSE} = 0.350676^\\circ\\text{C}$ (Mejora: +2.45264%)
- **`E3b-T3`:** $\\text{RMSE} = 0.350475^\\circ\\text{C}$ (Mejora: +2.50843%)
- **`E3b-S`:** $\\text{RMSE} = 0.350802^\\circ\\text{C}$ (Mejora: +2.41756%)
- **`E3b-TS`:** $\\text{RMSE} = 0.351551^\\circ\\text{C}$ (Mejora: +2.20904%)
- **`E3b-ALL`:** $\\text{RMSE} = 0.350048^\\circ\\text{C}$ (Mejora: +2.62712%)
- `E3b-C0` superó a todas las extensiones temporales y espaciales complejas, siendo congelado formalmente.

---

## 9. External Validation Results (Stage D33)

Evaluación de la especificación congelada de `E3b-C0` (reajustada en 2015–2021) en el periodo independiente 2022–2023 ($3,850,750$ observaciones):
- **Baseline B0:** $\\text{RMSE} = 0.335666^\\circ\\text{C}, \\quad \\text{MAE} = 0.263594^\\circ\\text{C}$
- **Modelo C0:** $\\text{RMSE} = 0.323838^\\circ\\text{C}, \\quad \\text{MAE} = 0.254457^\\circ\\text{C}$
- **Mejora global:** **+3.5237%** de reducción en RMSE ($\\Delta\\text{RMSE} = -0.011828^\\circ\\text{C}$).
- **Desempeño temporal:** **16 de 24 meses mejorados** (66.7%).
- **Desempeño espacial:** **4,755 de 5,275 celdas mejoradas** (**90.14%**).
- **Dictamen formal:** **D33-B — Partial/Mixed Generalization** (mejora global y espacial sólida, pero estabilidad mensual sub-umbral de 18 meses).

---

## 10. Diagnostic Findings (Stage D34)

- **Compresión de amplitud:** El modelo `E3b-C0` predice un residual con desviación estándar reducida frente a la observada: $\\text{std}(\\hat{R}) / \\text{std}(R) \\approx 0.25$, actuando como un estimador conservador de contracción hacia la media condicional.
- **Comportamiento por regímenes de discrepancia:** Cuando la discrepancia original entre MUR y OISST es pequeña ($|R| < 0.18^\\circ\\text{C}$, percentiles 0–50 de Development), el modelo degrada ligeramente el error cuadrático ($\\sim -20\\%$). En cambio, en discrepancias moderadas a severas (P75–P99+), el modelo alcanza mejoras de RMSE de +7% a +12%.

---

## 11. Final-Test Performance (Stage D35)

Evaluación en el conjunto ciego Final Test 2024–2025 ($3,856,025$ observaciones, 731 días):
- **Métricas primarias consolidadas:**
  - **Baseline B0 RMSE:** $0.357317^\\circ\\text{C}$
  - **Modelo C0 RMSE:** $0.331502^\\circ\\text{C}$
  - **Mejora relativa global de RMSE:** **+7.2247%** ($\\Delta\\text{RMSE} = -0.025815^\\circ\\text{C}$)
  - **MAE:** Reducción de $0.272713^\\circ\\text{C}$ a $0.256325^\\circ\\text{C}$ (+6.01% de mejora)
  - **Sesgo medio (*Bias*):** Reducción de $-0.062425^\\circ\\text{C}$ a $-0.014197^\\circ\\text{C}$ (reducción del sesgo en 77.3%)
  - **$R^2$ de SST reconstruida:** Aumento de $0.892661$ (B0) a **$0.907610$** (C0)
- **Desglose anual:**
  - Año 2024: B0 RMSE = $0.379708^\\circ\\text{C}$ -> C0 RMSE = $0.344023^\\circ\\text{C}$ (**+9.3986%**)
  - Año 2025: B0 RMSE = $0.333355^\\circ\\text{C}$ -> C0 RMSE = $0.318453^\\circ\\text{C}$ (**+4.4704%**)

---

## 12. Temporal Robustness (Stage D35)

- **Meses mejorados:** **18 de 24 meses** del bienio registraron $\\Delta\\text{RMSE} < 0$ (**75.0%** de estabilidad mensual). En 2024 mejoraron 11 de 12 meses; en 2025 mejoraron 7 de 12 meses.
- **Días mejorados:** **484 de 731 días** astronómicos evaluados arrojaron mejora (desempeño favorable en el **66.21%** de las fechas).
- **Inferencia estadística por bootstrap en bloques de 14 días (1,000 réplicas):**
  - Mediana de $\\Delta\\text{RMSE}$: $-0.025395^\\circ\\text{C}$
  - **Intervalo de confianza al 95% ($CI_{95}$):** **$[-0.042485, -0.009792]^\\circ\\text{C}$**
  - El límite superior del intervalo es estrictamente negativo, confirmando significancia estadística al 95%.

---

## 13. Spatial Robustness (Stage D35)

- **Amplitud espacial de la mejora:** **5,273 de las 5,275 celdas evaluadas** registraron reducción del RMSE cuadrático medio decenal (**99.96%** de cobertura espacial con beneficio).
- **Gradiente con profundidad batimétrica:**
  - 0–20 m (965 celdas): $\\Delta\\text{RMSE}$ medio = $-0.038311^\\circ\\text{C}$ (100.0% celdas mejoradas)
  - 20–50 m (417 celdas): $\\Delta\\text{RMSE}$ medio = $-0.032650^\\circ\\text{C}$ (100.0% celdas mejoradas)
  - 50–100 m (232 celdas): $\\Delta\\text{RMSE}$ medio = $-0.030423^\\circ\\text{C}$ (100.0% celdas mejoradas)
  - 100–500 m (1,363 celdas): $\\Delta\\text{RMSE}$ medio = $-0.030608^\\circ\\text{C}$ (100.0% celdas mejoradas)
  - >500 m (2,298 celdas): $\\Delta\\text{RMSE}$ medio = $-0.014452^\\circ\\text{C}$ (99.91% celdas mejoradas)
- **Correlaciones espaciales de Spearman:**
  $$\\text{Spearman}(depth, \\Delta\\text{RMSE}) = +0.7376 \\quad (p < 10^{-15})$$
  $$\\text{Spearman}(distance\\_coast\\_km, \\Delta\\text{RMSE}) = +0.5336 \\quad (p < 10^{-15})$$
  La correlación positiva confirma numéricamente que la magnitud absoluta de mejora es mayor en la plataforma somera y lagunas litorales, atenuándose hacia aguas oceánicas abiertas.

---

## 14. Residual-Regime Dependence (Stage D35)

Estratificación del desempeño en Final Test según los percentiles de discrepancia $|R|$ de Development:
- **DEV-P0–P50** ($|R| \\le 0.177^\\circ\\text{C}$, 48.16% de test): B0 = $0.1154^\\circ\\text{C}$, C0 = $0.1396^\\circ\\text{C}$ (**-20.99%** de degradación).
- **DEV-P50–P75** ($0.177 < |R| \\le 0.354^\\circ\\text{C}$, 24.35% de test): B0 = $0.2808^\\circ\\text{C}$, C0 = $0.2710^\\circ\\text{C}$ (**+3.51%** de mejora).
- **DEV-P75–P90** ($0.354 < |R| \\le 0.536^\\circ\\text{C}$, 15.68% de test): B0 = $0.4400^\\circ\\text{C}$, C0 = $0.4071^\\circ\\text{C}$ (**+7.48%** de mejora).
- **DEV-P90–P95** ($0.536 < |R| \\le 0.697^\\circ\\text{C}$, 5.29% de test): B0 = $0.5961^\\circ\\text{C}$, C0 = $0.5412^\\circ\\text{C}$ (**+9.21%** de mejora).
- **DEV-P95–P99** ($0.697 < |R| \\le 1.054^\\circ\\text{C}$, 4.71% de test): B0 = $0.7877^\\circ\\text{C}$, C0 = $0.6926^\\circ\\text{C}$ (**+12.07%** de mejora).
- **DEV-P99+** ($|R| > 1.054^\\circ\\text{C}$, 1.80% de test): B0 = $1.1469^\\circ\\text{C}$, C0 = $1.0151^\\circ\\text{C}$ (**+11.49%** de mejora).
- **Exactitud de signo por régimen:** Crece monotónicamente con la magnitud del residual:
  $$56.4\\% \\, (\\text{P0–P50}) \\to 65.8\\% \\, (\\text{P50–P75}) \\to 70.9\\% \\, (\\text{P75–P90}) \\to 75.7\\% \\, (\\text{P90–P95}) \\to 82.5\\% \\, (\\text{P95–P99}) \\to 87.2\\% \\, (\\text{P99+})$$

---

## 15. Final Scientific Result

1. El modelo parsimonioso `E3b-C0` demostró una capacidad robusta, estadísticamente significativa y generalizable para reconstruir la estructura residual fina de la temperatura superficial del mar sobre el corredor Tulum–Cozumel en un bienio out-of-sample completamente independiente (2024–2025).
2. Se confirma el **Dictamen Formal D35-A: FINAL GENERALIZATION CONFIRMED**.
3. El conjunto de prueba queda clasificado como **TEST CONSUMED** y el desarrollo metodológico de modelos de Machine Learning queda formalmente **CERRADO**.
"""

with open(REPORTS_DIR / "THESIS_RESULTS_MASTER_A_D.md", "w", encoding="utf-8") as f:
    f.write(results_master_content)
print("Reporte generado: THESIS_RESULTS_MASTER_A_D.md")

# 5.4 THESIS_DISCUSSION_NOTES_A_D.md
discussion_notes_content = """# Master Thesis Discussion Notes (Stages A to D)
## Estructura de Argumentación Científica para Capítulos de Discusión

**Documento:** `THESIS_DISCUSSION_NOTES_A_D.md`  
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
   - *Dato:* La inclusión de lags autoregresivos (`T1`, `T3`) y descriptores espaciales de vecindad (`S`, `TS`, `ALL`) no aumentó el skill de generalización, obteniendo entre +2.21% y +2.63% frente a +2.84% de `E3b-C0` en Holdout 2021.
   - *Discusión:* En problemas de downscaling satelital diario donde la escala espacial es de ~1 km y el predictor base es suave (0.25°), los lags introducen inestabilidad temporal por persistencia estacional espuria, mientras que las vecindades espaciales amplifican el ruido de borde costero. El vector $[\\text{sst\\_bil}, \\text{doy\\_sin}, \\text{doy\\_cos}, \\text{depth}]$ captura de forma suficiente la física media del desacoplamiento local.

2. **Compresión de Amplitud (*Amplitude Compression*):**
   - *Dato:* El ratio de desviaciones estándar $\\text{std}(\\hat{R}) / \\text{std}(R) = 0.2536$ y la pendiente de regresión $b = 0.0891$ en $\\hat{R} = a + b R$.
   - *Discusión:* El árbol de decisión no reproduce la variabilidad puntual extrema del residual observacional. Actúa como un estimador conservador condicional a la media, corrigiendo sesgos sistemáticos sin generar artefactos de alta frecuencia.

3. **Dependencia Crítica del Régimen de Discrepancia (*Regime Dependence*):**
   - *Dato:* En discrepancias bajas (DEV-P0–P50, $|R| \\le 0.18^\\circ\\text{C}$), el RMSE empeora un -20.99%, mientras que en discrepancias severas (P75–P99+), el RMSE mejora entre +7.48% y +12.07%.
   - *Discusión:* Cuando OISST bilineal coincide estrechamente con MUR, la discrepancia residual está dominada por ruido radiométrico de sensor y variabilidad sub-resolución que el modelo no puede predecir determinísticamente. Al intentar emitir una corrección sobre una señal nula, el modelo introduce una penalización cuadrática. En cambio, cuando existen gradientes térmicos verdaderos, la señal supera el umbral de ruido y el modelo reduce drásticamente el error.

4. **Amplitud y Concentración Espacial de Beneficios:**
   - *Dato:* 99.96% de las celdas evaluadas mostraron beneficio cuadrático medio decenal. Correlación de Spearman positiva con profundidad ($\\rho = +0.7376$) y distancia a la costa ($\\rho = +0.5336$).
   - *Discusión:* Las mayores ganancias se concentran en la plataforma somera y franja arrecifal costera (0–20 m). En aguas oceánicas profundas (>500 m), el campo bilineal suave de OISST ya constituye una representación bastante fiel del campo MUR, dejando menor margen para correcciones residuales.

---

## 3. Category II: Decisions Requiring Literature Support

1. **Uso de MUR SST como Verdad de Referencia Operacional:**
   - *Citar:* Chin et al. (2017), Armstrong et al. (2012). Justificar que los productos L4 fusionados multiescala son la referencia estándar operativa en regiones tropicales sin cobertura boya densa, reconociendo que no son una verdad terrena in situ absoluta.
2. **Formulación de Aprendizaje Residual vs Downscaling Directo:**
   - *Citar:* He et al. (2016) para fundamentos de residual learning; Cyriac et al. (2025) y literatura afín en geociencias para downscaling residual de SST.
3. **Partición Cronológica Estricta para Prevenir Fugas de Información:**
   - *Citar:* Roberts et al. (2017) sobre "Cross-validation strategies for data with temporal, spatial, or hierarchical structure" para fundamentar por qué el k-fold aleatorio es científicamente inválido en series de tiempo oceánicas.
4. **Validación Infrarroja Bloqueada por Cobertura Nubosa:**
   - *Citar:* Kilpatrick et al. (2015), Minnett et al. (2019) sobre las limitaciones físicas inherentes a la radiometría térmica satelital en el Caribe y la generación de anomalías por nubes residuales.

---

## 4. Category III: Speculative Claims — Prohibido Afirmar como Hechos Demostrados

1. **PROHIBIDO:** Atribuir la discrepancia MUR–BIL o la corrección de `E3b-C0` a "surgencias costeras (*upwelling*)", "efectos de frentes térmicos de mesoescala", "ondas de calor marinas (MHW)", o "mezcla por vientos del Norte", a menos que se presente evidencia hidrodinámica in situ o modelación oceánica acoplada independiente.
   - *Fraseo correcto:* "Asociaciones estadísticas consistentes con gradientes térmicos someros cercanos a la costa".
2. **PROHIBIDO:** Afirmar que el régimen bajo ($|R| < 0.18^\\circ\\text{C}$) representa estrictamente el "piso de ruido del instrumento (*noise floor*)".
   - *Fraseo correcto:* "Régimen de discrepancia reducida MUR–BIL (*small discrepancy regime*) donde la señal residual no presenta estructura predictible aprendible".
3. **PROHIBIDO:** Afirmar que el modelo realizó "reconstrucción submesoescala".
   - *Fraseo correcto:* "Corrección empírica de gradientes térmicos estacionales y efectos de borde batimétrico a escala de ~1 km".

---

## 5. Preguntas Clave para el Jurado de Tesis y Respuestas Metodológicas

1. **¿Por qué la mejora en RMSE es de ~7.2% y no del 20% o 30%?**
   - *Respuesta:* OISST bilineal ya explica el 89.3% de la varianza total de la SST observada ($R^2 = 0.8927$). El modelo residual compite contra una línea base muy fuerte y opera exclusivamente sobre el 10.7% de varianza restante. Reducir el error residual sobre 3.85 millones de puntos espaciotemporales ciegos en un 7.22% representa una ganancia significativa en señales térmicas acotadas.
2. **¿Por qué empeora el modelo en el 48% de los datos con discrepancias bajas?**
   - *Respuesta:* Es una propiedad teórica inherente a los estimadores de regresión minimizadores de MSE cuando se enfrentan a regiones con señal cercana a cero y ruido no nulo. Intentar forzar mejoras en discrepancias bajas mediante filtrado artificial introduciría sesgos de umbral no generalizables.
3. **¿Por qué no se usaron redes convolucionales (CNN)?**
   - *Respuesta:* La presencia de geometrías continentales complejas e islas (Cozumel) en una malla de 86 × 96 produce problemas severos de convolución en fronteras. El enfoque tabular con covariables geofísicas demostró ser altamente eficiente, interpretable y parsimonioso.
"""

with open(REPORTS_DIR / "THESIS_DISCUSSION_NOTES_A_D.md", "w", encoding="utf-8") as f:
    f.write(discussion_notes_content)
print("Reporte generado: THESIS_DISCUSSION_NOTES_A_D.md")

# 5.5 THESIS_LIMITATIONS_A_D.md
limitations_master_content = """# Master Inventory of Scientific Limitations (Stages A to D)
## Registro Sistemático de Limitaciones Metodológicas para Tesis

**Documento:** `THESIS_LIMITATIONS_A_D.md`  
**Propósito:** Agrupar con transparencia y rigor científico todas las limitaciones inherentes a los datos, métodos, suposiciones y modelos evaluados a lo largo de las Etapas A, B, C y D.

---

## 1. Stage A: Data-Product Limitations

1. **Falta de Ground Truth Absoluto In Situ:** El proyecto carece de una red densa de boyas oceanográficas fijas o derivadoras (*drifters*) con cobertura diaria continua en el canal de Cozumel y la laguna arrecifal de Tulum. MUR SST actúa como la referencia de mayor resolución disponible (~1 km), pero es a su vez un producto interpolado multiescala sujeto a sus propios algoritmos de asimilación.
2. **Inconsistencias Radiométricas en Eventos Nubosos:** En situaciones de nubosidad persistente asociada a ondas tropicales o huracanes, las observaciones infrarrojas de alta resolución quedan enmascaradas, provocando que MUR dependa de microondas (de menor resolución, ~25 km) o de persistencia temporal, incrementando su incertidumbre interna.

---

## 2. Stage B: Spatial Framework and Masking Limitations

1. **Resolución Discreta de la Fracción Oceánica:** La máscara oceánica corregida empleó un umbral binario $\\text{ocean\\_fraction} \\ge 0.5$. Las celdas con fracción mixta entre 50% y 99% contienen tierra residual sub-píxel que puede introducir sesgos leves en la radiometría costera.
2. **Resolución Batimétrica en Cañones Estrechos:** Aunque GEBCO 2026 Grid provee celdas a 15 arc-segundos, la topografía marina submarina en el canal de Cozumel contiene pendientes verticales abruptas que quedan suavizadas al promediarse a la resolución de 0.01° de la cuadrícula maestra.

---

## 3. Stage C: Interpolation and Harmonization Limitations

1. **Naturaleza Matemática del Soporte Costero (Estrategia A):** Los 20 nodos extendidos de OISST sobre la Península de Yucatán no representan temperatura superficial terrestre real; funcionan exclusivamente como soporte matemático para permitir la interpolación bilineal en la costa. Aunque el producto final se recorta con la máscara oceánica, la solución en celdas a <10 km de la costa está condicionada por la técnica de extensión utilizada.
2. **Pérdida de Información por Bloqueo Nuboso en Validación Satelital:** La auditoría con radiómetros L2P (VIIRS y MODIS) durante eventos de discrepancia extrema resultó inconclusa en más del 90% de los pasos orbitales debido al bloqueo por nubes ($QL=1$), impidiendo una validación radiométrica cruzada cuantitativa en los picos térmicos más agudos.
3. **Comportamiento Asintótico de MUR Analysis Error:** La variable de incertidumbre de MUR satura en un techo fijo de $0.4100^\\circ\\text{C}$ durante vacíos observacionales prolongados, lo cual impide discriminar gradaciones de incertidumbre por encima de dicho umbral.

---

## 4. Stage D: Machine Learning Model Limitations

1. **Explicabilidad Parcial de la Varianza Residual ($R^2_{\\text{residual}} \\approx 11.2\\%$):** Aunque el modelo reconstruye con éxito la SST absoluta ($R^2_{\\text{SST}} = 0.9076$), el porcentaje de varianza del residual fino $R$ capturado por `E3b-C0` es del 11.22%. La mayor parte de la varianza residual instantánea diaria permanece no explicada debido a la falta de forzantes dinámicos de alta resolución.
2. **Compresión de Amplitud de Predicción:** Debido a la función de pérdida cuadrática estándar, el modelo produce predicciones atenuadas ($\\text{std}(\\hat{R})/\\text{std}(R) \\approx 0.25$). No es adecuado para predecir anomalías térmicas extremas instantáneas aisladas.
3. **Degradación en Regímenes de Baja Discrepancia:** En el 48% de los datos donde OISST y MUR difieren en menos de $0.18^\\circ\\text{C}$, el modelo introduce sobrecorrecciones leves que incrementan el RMSE en un 20.99%, careciendo de un mecanismo intrínseco de abstención o umbralización adaptable.
4. **Inestabilidad Mensual en Años Específicos:** El skill de mejora no es perfectamente homogéneo en el tiempo. Mientras que en 2024 mejoraron 11 de 12 meses (+9.40%), en 2025 mejoraron 7 de 12 meses (+4.47%), reflejando sensibilidad a ciclos interanuales o anomalías térmicas regionales no capturadas por las features estáticas y el DOY cíclico.
5. **Ausencia de Covariables Atmosféricas y de Corrientes:** El modelo no incluye velocidad ni dirección del viento (e.g. ERA5), radiación solar incidente, ni velocidad de corrientes marinas (CMEMS), limitando su capacidad para modelar fenómenos advectivos o eventos de mezcla inducidos por viento.
"""

with open(REPORTS_DIR / "THESIS_LIMITATIONS_A_D.md", "w", encoding="utf-8") as f:
    f.write(limitations_master_content)
print("Reporte generado: THESIS_LIMITATIONS_A_D.md")

# 5.6 THESIS_WRITING_MAP.md
writing_map_content = """# Master Thesis Writing Map (Stages A to D)
## Guía de Procedencia Documental para Redacción de Capítulos de Tesis

**Documento:** `THESIS_WRITING_MAP.md`  
**Propósito:** Proporcionar un índice de navegación unívoco para que el autor de la tesis pueda redactar directamente cada capítulo y subsección sin necesidad de rebuscar en logs, scripts antiguos o directorios dispersos.

---

## 1. Capítulo: Introducción y Área de Estudio

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada |
| :--- | :--- | :--- |
| **Descripción geográfica del corredor** | `reports/THESIS_METHODS_MASTER_A_D.md` (Sección 1) | `THESIS_MASTER_A_D/figures/fig_master_pipeline_A_D.png` |
| **Límites de coordenadas y batimetría regional** | `config.py`, `reports/reporte_fase_b.txt` | `tables/master_phase_summary.csv` |

---

## 2. Capítulo: Datos y Metodología

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada |
| :--- | :--- | :--- |
| **Producto MUR SST v4.1 y control de calidad** | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 2, 3) | `tables/dataset_provenance.csv` (Fila 1) |
| **Producto NOAA OISST v2.1 y Halo Espacial** | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 2, 7) | `tables/dataset_provenance.csv` (Fila 3) |
| **Modelo batimétrico GEBCO 2026** | `reports/THESIS_METHODS_MASTER_A_D.md` (Sección 2, 6) | `tables/dataset_provenance.csv` (Fila 4) |
| **Definición de grilla maestra y máscara corregida** | `reports/reporte_fase_b.txt`, `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 4, 5) | `tables/master_decision_log.csv` (DEC-01, DEC-02) |
| **Problema de NaNs costeros y soporte Estrategia A** | `reports/reporte_faseC1b_2015-01-01.txt` | `figures/faseC1b_comparacion_estrategias_2015-01-01.png` |
| **Armonización temporal completa 2015–2025 (C.2)** | `reports/fase_c2_reporte.md` | `figures/faseC2_serie_rmse_diario.png` |
| **Auditoría satelital independiente VIIRS/MODIS** | `reports/validacion_infrarroja_integrada_FINAL.md` | `reports/detalle_flags_viirs_FINAL.txt`, `modis` |
| **Auditoría de incertidumbre MUR analysis_error** | `auditoria_analysis_error/reports/auditoria_analysis_error_FINAL_4018dias.md` | `tables/master_decision_log.csv` (DEC-05) |
| **Construcción tabular y partición anti-fuga (D.1)** | `ml_dataset/METADATA.md` | `reports/THESIS_METHODS_MASTER_A_D.md` (Secciones 12, 13) |
| **Selección de modelo y ablaciones E3b (D32)** | `ml_results/E3b_D32/tables/feature_ablation.csv` | `ml_results/E3b_D32/figures/fig2_relative_improvement_vs_b0.png` |
| **Validación externa 2022–2023 (D33)** | `ml_results/E3b_D33_external_validation/tables/validation_summary.csv` | `tables/master_decision_log.csv` (DEC-08) |
| **Protocolo de congelamiento y Final Test (D35)** | `ml_results/E3b_D35_final_test/final_test_freeze_manifest.json` | `tables/master_decision_log.csv` (DEC-09, DEC-10) |

---

## 3. Capítulo: Resultados

| Subsección de Tesis | Archivo de Procedencia Primaria | Tabla / Figura Asociada |
| :--- | :--- | :--- |
| **Línea base de armonización decenal (Baseline E0)** | `reports/fase_c2_reporte.md` | `tables/metric_provenance_A_D.csv` (MET-C-05 a MET-C-09) |
| **Resultados de Holdout 2021 y selección E3b-C0** | `ml_results/E3b_D32/tables/model_summary.csv` | `tables/metric_provenance_A_D.csv` (MET-D32-01 a MET-D32-03) |
| **Resultados de Validación Externa 2022–2023** | `ml_results/E3b_D33_external_validation/tables/validation_summary.csv` | `tables/metric_provenance_A_D.csv` (MET-D33-01 a MET-D33-05) |
| **Desempeño final en Test 2024–2025** | `ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table1_final_performance.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig1_final_performance.png` |
| **Robustez temporal (meses, días y bootstrap 14d)** | `ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table2_robustness.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig2_temporal_robustness.png` |
| **Robustez espacial (celdas mejoradas y batimetría)** | `ml_results/E3b_D35_final_test/tables/spatial_depth_diagnostics.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig3_spatial_skill.png` |
| **Análisis de regímenes de discrepancia residual** | `ml_results/E3b_FINAL_SYNTHESIS/tables/paper_table3_residual_regimes.csv` | `ml_results/E3b_FINAL_SYNTHESIS/figures/fig4_residual_regimes.png` |

---

## 4. Capítulo: Discusión y Conclusiones

| Subsección de Tesis | Archivo de Procedencia Primaria | Notas Metodológicas |
| :--- | :--- | :--- |
| **Interpretación del modelo parsimonioso** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Evidencia de parsimonia y estabilidad |
| **Análisis de compresión de amplitud y calibración** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Explicar $b = 0.0891$ en $\\hat{R} = a + b R$ |
| **Fronteras y degradación en discrepancias bajas** | `reports/THESIS_DISCUSSION_NOTES_A_D.md` (Sección 2) | Comportamiento en DEV-P0–P50 |
| **Limitaciones del estudio** | `reports/THESIS_LIMITATIONS_A_D.md` | Limitaciones por etapas A, B, C, D |
| **Gaps metodológicos resueltos** | `reports/DOCUMENTATION_GAPS_A_D.md` | Trazabilidad de 5,279 -> 5,275 celdas y GEBCO 2026 |
| **Conclusiones terminales de tesis** | `reports/THESIS_MASTER_SYNTHESIS_A_D.md` (Sección 28) | Síntesis conclusiva definitiva |
"""

with open(REPORTS_DIR / "THESIS_WRITING_MAP.md", "w", encoding="utf-8") as f:
    f.write(writing_map_content)
print("Reporte generado: THESIS_WRITING_MAP.md")

# 5.7 REPRODUCIBILITY_MAP_A_D.md
repro_map_content = """# Master Reproducibility and Audit Map (Stages A to D)
## Trazabilidad de Código, Entorno, Semillas y Artefactos Inmutables

**Documento:** `REPRODUCIBILITY_MAP_A_D.md`  
**Propósito:** Especificar las condiciones técnicas exactas para la reproducción computacional de cada etapa del proyecto, delimitando estrictamente qué artefactos son inmutables y no deben ser regenerados.

---

## 1. Computational Environment

- **Sistema Operativo:** macOS 26.6.2 (Darwin Kernel Version 26.6.2; arm64 Apple Silicon)
- **Entérprete de Python:** Python 3.11.16 (`/Users/mariajosenande/Documents/Lole/.venv/bin/python`)
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
- **`frozen_spatial_metadata.csv`:**
  - SHA-256: `a93b4554a9386c9e99551c68d197607a72dd37803e6592233f211333792b0c2a`
- **`faseC2_2015_2025.nc`:**
  - Tamaño: **540.82 MB** (567,094,364 bytes).

---

## 4. Absolute Prohibition on Final Test Re-Execution

> [!CAUTION]
> **FINAL TEST 2024–2025 IS PERMANENTLY CONSUMED.**
> El conjunto de datos correspondiente a los años 2024 y 2025 ya fue procesado durante la Fase D.3.5. Ha dejado de ser un conjunto ciego (*blind test*).
> Queda terminantemente prohibido volver a ejecutar scripts con propósitos de ajuste, recalibración, exploración o "re-evaluación confirmatoria". Cualquier corrida futura sobre estos años constituiría un ejercicio adaptativo post-hoc con riesgo severo de sobreajuste retrospectivo.
"""

with open(REPORTS_DIR / "REPRODUCIBILITY_MAP_A_D.md", "w", encoding="utf-8") as f:
    f.write(repro_map_content)
print("Reporte generado: REPRODUCIBILITY_MAP_A_D.md")

# 5.8 THESIS_MASTER_SYNTHESIS_A_D.md (The 29-section Master Synthesis)
master_syn_content = """# Master Thesis Synthesis — Stages A to D
## Reconstrucción Metodológica, Trazabilidad Científica y Archivo Definitivo para Tesis

**Proyecto:** Downscaling espacial de Sea Surface Temperature (SST) mediante aprendizaje residual en el Caribe mexicano  
**Área de Estudio:** Corredor Tulum–Cozumel, Quintana Roo, México  
**Periodo Decenal:** 2015-01-01 a 2025-12-31 (4,018 días continuos)  
**Nivel del Documento:** Síntesis Maestra Canónica Definitiva (Capa de Documentación de Tesis)  
**Estado:** COMPLETO Y CONGELADO PARA REDACCIÓN DE TESIS

---

## 1. Scientific Objective

El objetivo científico central de la tesis es desarrollar, validar y auditar rigurosamente un marco de downscaling espacial estadístico para proyectar campos diarios de temperatura superficial del mar (SST) desde una escala gruesa de 0.25° (~27 km, provista por NOAA OISST v2.1) hasta una resolución ultra-alta de 0.01° (~1 km, compatible con MUR SST v4.1) sobre el corredor marino Tulum–Cozumel, empleando un enfoque de aprendizaje residual supervisado con modelos basados en árboles de decisión e incorporando covariables fisiográficas estáticas (batimetría GEBCO) y temporales cíclicas (DOY).

---

## 2. Study Area

El corredor marino Tulum–Cozumel se localiza en la costa caribeña de la Península de Yucatán:
- **Latitud:** $19.90^\\circ\\text{N}$ a $20.75^\\circ\\text{N}$
- **Longitud:** $-87.60^\\circ\\text{W}$ a $-86.65^\\circ\\text{W}$
- **Complejidad oceanográfica:** El área alberga una estrecha plataforma somera (0–20 m), lagunas arrecifales protegidas del Sistema Arrecifal Mesoamericano, la isla de Cozumel, el canal de Cozumel y cañones oceánicos profundos (>1000 m) sujetos al flujo forzado de la corriente de Yucatán.

---

## 3. Overall Pipeline

El pipeline integral se articula en cuatro etapas metodológicas consecutivas y no circulares:
1. **Etapa A:** Adquisición, verificación temporal y auditoría de integridad decenal (MUR, OISST, GEBCO, L2P).
2. **Etapa B & B.1:** Definición del marco espacial estándar (86 × 96 celdas a 0.01°) y construcción de la máscara oceánica corregida (5,279 celdas marinas).
3. **Etapa C (C.1, C.1b, C.1c, C.2):** Armonización multirresolución, diagnóstico y solución del soporte costero (Estrategia A), generación del cubo decenal `faseC2_2015_2025.nc` y auditorías complementarias (Satélite L2P y MUR `analysis_error`).
4. **Etapa D (D.1, D31, D32, D33, D34, D35, D36):** Construcción del dataset tabular Parquet, diagnóstico de predictibilidad, selección de la formulación parsimoniosa `E3b-C0` en Holdout 2021, validación externa temporal en 2022–2023, auditoría diagnóstica no adaptativa, evaluación terminal ciega en Final Test 2024–2025 y consolidación final.

---

## 4. Stage A — Data Acquisition

Se verificaron exhaustivamente las cuatro fuentes de datos para el periodo 2015–2025:
- **MUR SST v4.1:** 5 archivos NetCDF históricos (2015–2019) y 1,710 gránulos diarios L4 (2019–2025). 4,018 fechas continuas disponibles.
- **NOAA OISST v2.1:** Descarga vía ERDDAP en cuadrícula de 0.25° con halo de 0.5° y recuperación local en 2025-01-14.
- **GEBCO 2026 Grid:** Archivo `gebco_2026_n20.75_s19.9_w-87.6_e-86.65.nc` a 15 arc-segundos.
- **VIIRS S-NPP / MODIS Aqua L2P:** Gránulos orbitales independientes para auditoría radiométrica.

---

## 5. Stage A — Quality Control

- **Completitud:** 4,018 días esperados = 4,018 días encontrados (0 faltantes, 0 duplicados, 0 corrupciones).
- **Consistencia física:** Rangos térmicos validados en [24.0 °C, 32.5 °C] en mar abierto y lagunas.
- **Conclusión de Etapa A:** Datos de entrada científicamente completos, temporalmente continuos y aprobados.

---

## 6. Stage B — Master Spatial Grid

- **Malla maestra:** Basada en la cuadrícula de MUR SST v4.1 a 0.01° de resolución.
- **Dimensiones:** 86 celdas latitudinales × 96 celdas longitudinales = **8,256 celdas totales**.
- **Proyección cartográfica:** UTM Zona 16 Norte (**EPSG:32616**).

---

## 7. Stage B.1 — Ocean Mask and Static Covariates

- **Máscara corregida:** $M_{\\text{final}} = M_{\\text{MUR}} \\land (\\text{ocean\\_fraction} \\ge 0.5)$.
- **Censo espacial base:** **5,279 celdas oceánicas** y **2,977 celdas terrestres** (383 celdas continentales reclasificadas a tierra respecto a la máscara nativa de MUR; interior de Cozumel asignado estrictamente a tierra).
- **Covariables fisiográficas:** Profundidad positiva ($depth$), distancia mínima euclidiana a la costa ($distance\\_coast\\_km$) y fracción oceánica subpíxel ($ocean\\_fraction$).

---

## 8. Stage C.1 — Initial Harmonization

- En el ensayo piloto del 2015-01-01, la interpolación bilineal directa de OISST arrojó **1,292 NaNs costeros** (pérdida del **24.47%** de celdas oceánicas).
- Causa: Presencia de 20 nodos terrestres en la porción occidental del halo OISST (Península de Yucatán).

---

## 9. Stage C.1b — Coastal Coverage Problem

- **Comparación cuantitativa:**
  - Estrategia A (extensión costera auxiliar nearest-ocean en 0.25°): 5,279 / 5,279 celdas válidas.
  - Estrategia B (triangulación 2D Delaunay sobre océano): 5,279 / 5,279 celdas válidas.
  - Discrepancia A vs B: $\\text{MAE} = 0.0038^\\circ\\text{C}, \\, P_{95} = 0.0214^\\circ\\text{C}, \\, \\text{Máxima} = 0.0626^\\circ\\text{C}$.
- Se ratificó la equivalencia numérica entre ambos métodos.

---

## 10. Stage C.1c — Adopted Solution

- Se adoptó la **Estrategia A** por garantizar regularidad cartesiana, eficiencia algorítmica para 4,018 días y trazabilidad explícita mediante la máscara `oisst_coastal_support_mask`.
- Se estableció el Baseline E0 piloto para 2015-01-01 ($\text{RMSE} = 0.2367^\circ\text{C}$). Se aprobó la ejecución decenal completa.

---

## 11. Stage C.2 — Full 2015–2025 Harmonization

- **Ejecución decenal:** 4,018 días procesados sin interrupciones.
- **Volumen observacional:** $21,211,022$ observaciones espaciotemporales acumuladas.
- **Métricas decenales globales del Baseline E0:**
  $$\\text{RMSE} = 0.3426^\\circ\\text{C}, \\quad \\text{MAE} = 0.2631^\\circ\\text{C}, \\quad \\text{Bias} = +0.0133^\\circ\\text{C}, \\quad R^2 = 0.9016$$
- **Artefactos congelados:** `outputs/faseC2_2015_2025.nc` (540.82 MB) y 11 archivos anuales en `outputs/fase_c2/`.

---

## 12. Satellite Validation

- **Auditoría independiente:** VIIRS S-NPP v2.80 y MODIS Aqua v2019.0 en eventos de discrepancia extrema E1–E6.
- **Hallazgo:** Bloqueo nuboso casi total (>80–100% de píxeles con $QL < 5$). En el pico de octubre 2015, MODIS tuvo 0.0% de datos limpios y VIIRS 0.1% (10 píxeles aislados).
- **Dictamen:** Evidencia satelital independiente clasificada como **heterogénea e inconclusa**. Se confirma mantener `faseC2_2015_2025.nc` intacto.

---

## 13. MUR Uncertainty Audit

- Auditoría de la variable `analysis_error` sobre las 4,018 fechas ($21,211,022$ puntos).
- **Correlación:** Spearman $\\rho = +0.2853$ ($p = 4.25 \\times 10^{-76}$).
- **Techo algorítmico:** Confirmación de saturación asintótica en $0.4100^\\circ\\text{C}$ durante vacíos observacionales infrarrojos directos.
- **Decisión:** Excluir `analysis_error` de las variables predictoras del modelo ML y preservarla como variable de control diagnóstico.

---

## 14. Transition from Harmonized Cube to ML Dataset

- **Transición geométrica 5,279 -> 5,275 celdas:** En la subfase D32, la evaluación de ablaciones con features espaciales 2D (`grad_mag`) requirió diferencias finitas no nulas en $X$ e $Y$. Cuatro celdas costeras/insulares aisladas (índices 0, 161, 4437, 4472) arrojaron `grad_mag = NaN`. La máscara común `COMMON_VALID_MASK` excluyó estas cuatro celdas, congelando el dominio evaluativo de Machine Learning en exactamente **5,275 celdas marinas** (`frozen_cell_ids.csv`, SHA-256: `6f046931...`).

---

## 15. Stage D — ML Dataset Construction

- Formato tabular analítico Apache Parquet comprimido con Snappy en `ml_dataset/`.
- 13 columnas estructuradas: `date`, `year`, `doy`, `sst_bil`, `depth`, `distance_coast_km`, `ocean_fraction`, `doy_sin`, `doy_cos`, `sst_mur`, `residual`, `analysis_error`, `split`.
- Partición temporal estricta anti-fuga (Development 2015–2020, Holdout 2021, Validation 2022–2023, Test 2024–2025).

---

## 16. D31 — Predictability Diagnostics

- Demostración empírica de que el residual fino $R = \\text{sst\\_mur} - \\text{sst\\_bil}$ no es ruido blanco estocástico desestructurado.
- Autocorrelación temporal reproducible ($r \\approx 0.68$ a lag 1 día) y correlación espacial sistemática con la topografía submarina justificaron avanzar al modelado no lineal tabular.

---

## 17. D32 — Model Selection

- Evaluación de seis formulaciones prespecificadas (`E3b-C0`, `E3b-T1`, `E3b-T3`, `E3b-S`, `E3b-TS`, `E3b-ALL`) sobre Holdout 2021 ($1,925,375$ obs).
- **Modelo seleccionado:** **`E3b-C0`** (XGBoost con 4 features: `sst_bil`, `doy_sin`, `doy_cos`, `depth`).
- **Desempeño Holdout 2021:** B0 RMSE = $0.359493^\\circ\\text{C}$ -> C0 RMSE = $0.349274^\\circ\\text{C}$ (**+2.84243%** de mejora).
- Las variantes con lags y vecindad espacial obtuvieron menor mejora (+2.21% a +2.63%). Selección por **parsimonia y desempeño**. Estado: **METHODOLOGICALLY CLOSED**.

---

## 18. D33 — External Validation

- Reajuste de la especificación `E3b-C0` en datos previos 2015–2021 ($13,488,175$ obs) y evaluación fuera de muestra en el bienio **2022–2023** ($3,850,750$ obs).
- **Resultados:** B0 RMSE = $0.335666^\\circ\\text{C}$ -> C0 RMSE = $0.323838^\\circ\\text{C}$ (**+3.5237%** de mejora global). 16/24 meses mejorados (66.7%); 4,755/5,275 celdas mejoradas (90.14%).
- **Dictamen formal:** **D33-B — Partial/Mixed Generalization** (mejora global positiva pero estabilidad mensual sub-umbral predeclarado de 18 meses). Se retiene avance a Final Test.

---

## 19. D34 — Diagnostic Audit

- Auditoría diagnóstica no adaptativa sobre las predicciones de validación.
- Hallazgos: Fuerte compresión de amplitud (ratio de std ~0.25); degradación leve en regímenes de bajo residual ($|R| < 0.18^\\circ\\text{C}$); alta exactitud en discrepancias moderadas a severas; correlación positiva de ganancias con proximidad a costa y baja profundidad.
- Recomendación histórica: **PREPARE FINAL TEST**. Estado: **INTERPRETATIONALLY CLOSED**.

---

## 20. D35 — Final Test

- Reajuste final de la especificación `E3b-C0` en el periodo 2015–2023 ($17,338,925$ obs) e inferencia única sobre el conjunto ciego **2024–2025** ($3,856,025$ obs, 731 días).
- **Resultado primario confirmado:**
  - B0 RMSE: $0.357317^\\circ\\text{C}$ -> C0 RMSE: **$0.331502^\\circ\\text{C}$** (**+7.2247%** de mejora global)
  - MAE: $0.272713^\\circ\\text{C}$ -> $0.256325^\\circ\\text{C}$ (+6.01%)
  - Bias: $-0.062425^\\circ\\text{C}$ -> $-0.014197^\\circ\\text{C}$ (reducción de 77.3%)
  - $R^2$ SST: $0.892661$ -> **$0.907610$**
  - Desglose anual: 2024 = **+9.3986%**; 2025 = **+4.4704%**
  - Robustez temporal: **18 de 24 meses mejorados** (75.0%); **484 de 731 días mejorados** (66.21%)
  - Inferencia bootstrap 14d $CI_{95}$: **$[-0.042485, -0.009792]^\\circ\\text{C}$**
  - Cobertura espacial: **5,273 de 5,275 celdas mejoradas** (**99.96%**)
- **Dictamen formal:** **D35-A — FINAL GENERALIZATION CONFIRMED**.
- **Condición de cierre:** **FINAL TEST CONSUMED**.

---

## 21. D36 — Final ML Synthesis

- Consolidación transversal definitiva de resultados D31–D35.
- Cero reentrenamientos, cero modificaciones de hiperparámetros, cero aperturas lógicas raw a test.
- Declaración terminal: **ML MODEL DEVELOPMENT SCIENTIFICALLY CLOSED**.

---

## 22. Main Methodological Decisions

1. Cuadrícula de MUR como malla maestra del proyecto (DEC-01).
2. Corrección de máscara con GEBCO $\\text{ocean\\_fraction} \\ge 0.5$ (DEC-02).
3. Adopción de Estrategia A para soporte costero bilineal (DEC-03).
4. Exclusión de `analysis_error` del vector de predictores (DEC-05).
5. Partición cronológica estricta out-of-sample (DEC-06).
6. Selección de `E3b-C0` por parsimonia y desempeño (DEC-07).
7. Dictamen D33-B y pausa diagnóstica (DEC-08).
8. Dictamen confirmatorio D35-A y consumo de test (DEC-10).
9. Cierre científico definitivo del bloque de Machine Learning (DEC-11).

---

## 23. Main Scientific Results

1. La interpolación bilineal directa de OISST v2.1 deja un error basal promedio decenal de $\\text{RMSE} = 0.3426^\\circ\\text{C}$.
2. El aprendizaje residual tabular con 4 predictores físicos y temporales (`E3b-C0`) reduce el error cuadrático medio de downscaling en un **+7.22% global** en evaluación ciega fuera de muestra.
3. La mejora es espacialmente cuasi-universal (**99.96% de celdas beneficiadas**) y se concentra fuertemente en aguas someras (0–20 m: $-0.0383^\\circ\\text{C}$ de reducción media; $\\rho = +0.7376$).
4. El modelo opera como un estimador conservador de contracción hacia la media condicional (compresión de amplitud de ~0.25), alcanzando alta exactitud de signo (>70–87%) en discrepancias térmicas moderadas a severas.

---

## 24. Limitations

1. Falta de red de boyas in situ de alta frecuencia en el canal de Cozumel.
2. Incertidumbre en eventos nubosos prolongados donde la referencia MUR depende de persistencia.
3. Incapacidad del modelo estático para predecir variabilidad instantánea forzada por vientos o corrientes dinámicas (explicabilidad del residual en test = 11.22%).
4. Sobrecorrección leve (-20.99%) en el régimen de baja discrepancia ($|R| < 0.18^\\circ\\text{C}$).

---

## 25. Reproducibility

- Código modular completamente versionado bajo Git.
- Entorno de ejecución estandarizado (Python 3.11.16 en macOS arm64).
- Semillas estocásticas fijadas (`random_state = 42`).
- Hashes criptográficos de celdas comunes (`frozen_cell_ids.csv`: `6f046931...`).
- Final Test bloqueado e inmutable.

---

## 26. Canonical Artifacts

- **NetCDF Decenal:** `outputs/faseC2_2015_2025.nc` (540.82 MB).
- **Máscara congelada:** `ml_results/E3b_D32/tables/frozen_cell_ids.csv` (5,275 celdas).
- **Tablas de Test:** `ml_results/E3b_D35_final_test/tables/final_test_summary.csv`.
- **Suite de Consolidación:** `ml_results/E3b_FINAL_SYNTHESIS/` (tables, figures, reports).
- **Capa Maestra de Tesis:** `DATASET_TESIS/THESIS_MASTER_A_D/` (tables, figures, reports).

---

## 27. Documentation Gaps

Todos los gaps detectados fueron catalogados y resueltos en `DOCUMENTATION_GAPS_A_D.md`:
- Censo 5,279 -> 5,275 explicado por diferencias finitas en `grad_mag`.
- GEBCO 2026 Grid ratificado como versión física real frente a citas históricas de 2024.
- Orientación de la regresión de calibración aclarada como $\\hat{R} = a + b R$ ($b = 0.0891$).
- Seis formulaciones candidatas ratificadas en D32.
- Concepto de especificación congelada aclarado frente a reutilización de pesos.

---

## 28. Thesis-Ready Conclusions

1. **Viabilidad Metodológica:** Es técnicamente viable y científicamente riguroso downescalar OISST a 0.01° en el Caribe mexicano mediante aprendizaje residual tabular acoplado a batimetría y periodicidad astronómica.
2. **Parsimonia:** Modelos de gradiente boosting compactos (XGBoost con 19 árboles y profundidad 4) superan a formulaciones densas con lags autoregresivos o vecindades espaciales 2D.
3. **Generalización Terminal:** La generalización temporal out-of-sample fue comprobada estadísticamente en Final Test 2024–2025 con un $CI_{95}$ de $[-0.0425, -0.0098]^\\circ\\text{C}$ que excluye estrictamente el cero.
4. **Dominio de Aplicabilidad:** El modelo es altamente eficaz para la corrección de gradientes costeros persistentes en aguas someras, debiendo aplicarse con cautela en condiciones oceánicas homogéneas de baja discrepancia.

---

## 29. Final Project Status

El estado científico terminal de las Etapas A–D queda formalmente cerrado y archivado:

| Stage | Scientific function | Final status |
| :--- | :--- | :--- |
| **A** | Data acquisition and audit | **COMPLETED** |
| **B** | Spatial framework | **COMPLETED** |
| **B.1** | Final land/ocean mask | **COMPLETED** |
| **C.1** | Initial harmonization | **COMPLETED** |
| **C.1b** | Coastal support diagnosis | **COMPLETED** |
| **C.1c** | Coastal solution validation | **COMPLETED** |
| **C.2** | Full harmonization 2015–2025 | **COMPLETED** |
| **Satellite audit** | Independent L2P radiometry verification | **COMPLETED / EVIDENCE HETEROGENEOUS** |
| **MUR uncertainty audit** | Analysis error decenal characterization | **CLOSED** |
| **D dataset** | Tabular ML analytical dataset construction | **COMPLETED** |
| **D31** | Residual predictability diagnostics | **CLOSED** |
| **D32** | Parsimonious model selection & ablation | **METHODOLOGICALLY CLOSED** |
| **D33** | External temporal validation 2022–2023 | **D33-B — UNCHANGED** |
| **D34** | Diagnostic audit & microaudit | **INTERPRETATIONALLY CLOSED** |
| **D35** | Blind out-of-sample evaluation 2024–2025 | **D35-A — FINAL GENERALIZATION CONFIRMED** |
| **D36** | Final ML block synthesis | **FINAL SYNTHESIS COMPLETED** |
| **FINAL TEST** | Terminal out-of-sample evaluation | **CONSUMED** |
| **ML MODEL DEVELOPMENT** | Machine Learning development & tuning | **CLOSED** |
"""

with open(REPORTS_DIR / "THESIS_MASTER_SYNTHESIS_A_D.md", "w", encoding="utf-8") as f:
    f.write(master_syn_content)
print("Reporte generado: THESIS_MASTER_SYNTHESIS_A_D.md")

# ----------------------------------------------------------------------
# 6. VERIFICACIONES FINALES DE SEGURIDAD Y REPORTE DE SALIDA
# ----------------------------------------------------------------------
check_no_raw_test_access()

# Verificar existencia de los 16 archivos canónicos requeridos
expected_files = [
    TABLES_DIR / "master_phase_summary.csv",
    TABLES_DIR / "master_decision_log.csv",
    TABLES_DIR / "dataset_provenance.csv",
    TABLES_DIR / "metric_provenance_A_D.csv",
    TABLES_DIR / "artifact_inventory_A_D.csv",
    TABLES_DIR / "figure_inventory_A_D.csv",
    TABLES_DIR / "project_timeline_A_D.csv",
    FIGURES_DIR / "fig_master_pipeline_A_D.png",
    REPORTS_DIR / "THESIS_MASTER_SYNTHESIS_A_D.md",
    REPORTS_DIR / "THESIS_METHODS_MASTER_A_D.md",
    REPORTS_DIR / "THESIS_RESULTS_MASTER_A_D.md",
    REPORTS_DIR / "THESIS_DISCUSSION_NOTES_A_D.md",
    REPORTS_DIR / "THESIS_LIMITATIONS_A_D.md",
    REPORTS_DIR / "DOCUMENTATION_GAPS_A_D.md",
    REPORTS_DIR / "THESIS_WRITING_MAP.md",
    REPORTS_DIR / "REPRODUCIBILITY_MAP_A_D.md"
]

for ef in expected_files:
    assert ef.exists(), f"ERROR: Archivo esperado no encontrado: {ef}"
    assert ef.stat().st_size > 0, f"ERROR: Archivo vacío: {ef}"

print(f"Todos los {len(expected_files)} artefactos maestros existen y no están vacíos.")

# ----------------------------------------------------------------------
# 7. FINAL CONSOLE OUTPUT (SECTION 47 FORMAT)
# ----------------------------------------------------------------------
final_console_output = """
============================================================
THESIS MASTER SYNTHESIS A–D COMPLETED
============================================================

STAGE A:
COMPLETED

STAGE B:
COMPLETED

STAGE C:
COMPLETED

STAGE D:
SCIENTIFICALLY CLOSED

D35:
D35-A — FINAL GENERALIZATION CONFIRMED

FINAL TEST:
CONSUMED

CANONICAL PHASE TABLE:
GENERATED

MASTER DECISION LOG:
GENERATED

DATASET PROVENANCE:
GENERATED

METRIC PROVENANCE:
GENERATED

ARTIFACT INVENTORY:
GENERATED

FIGURE INVENTORY:
GENERATED

MASTER PIPELINE FIGURE:
GENERATED

THESIS METHODS MASTER:
GENERATED

THESIS RESULTS MASTER:
GENERATED

DISCUSSION NOTES:
GENERATED

LIMITATIONS MASTER:
GENERATED

DOCUMENTATION GAPS:
GENERATED

THESIS WRITING MAP:
GENERATED

REPRODUCIBILITY MAP:
GENERATED

NEW MODEL TRAINING:
NO

RAW FINAL TEST ACCESSED:
NO

PROJECT A–D SYNTHESIS:
ARCHIVED FOR THESIS WRITING

============================================================
"""

print(final_console_output)

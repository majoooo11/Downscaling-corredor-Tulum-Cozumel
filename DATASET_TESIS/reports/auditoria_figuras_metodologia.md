# Auditoría de Figuras Metodológicas del Pipeline

**Fecha de Auditoría:** 2026-08-25  
**Dominio Geográfico:** Corredor Tulum–Cozumel (Lat: 19.90°N a 20.75°N, Lon: -87.60°W a -86.65°W)  
**Cuadrícula Maestra:** $86 \times 96$ celdas (8256 celdas totales, resolución $\sim 0.01^\circ$)  
**Estado:** **FIGURAS ACTIVAS OFICIALES VALIDADAS / OBSOLETAS ARCHIVADAS EN `figures/archive/`**

---

## 1. Catálogo de Figuras Activas Oficiales (`DATASET_TESIS/figures/`)

Solo las 11 figuras siguientes están presentes en el directorio raíz de figuras para su uso directo en el manuscrito y reportes:

| Figura | Fuente / Script Generador | Versión | Celdas Válidas | Máscara | Cozumel | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mapa_ocean_mask.png`** | `armonizar_datos_tesis.py` (`plot_all_control_figures`) | Fase B.1 Definitiva | 8256 (5279 océano / 2977 tierra) | `ocean_mask_final` | Tierra ($0$) | **ACTIVA / OK** |
| **`mapa_ocean_fraction.png`** | `armonizar_datos_tesis.py` (`plot_all_control_figures`) | Fase B.1 Definitiva | 8256 (rango $[0, 1]$ continuo) | `ocean_fraction` | Tierra ($0.0$) | **ACTIVA / OK** |
| **`mapa_depth_mur.png`** | `armonizar_datos_tesis.py` (`plot_all_control_figures`) | Fase B.1 Definitiva | 5279 oceánicas ($\text{Depth} > 0$) | `ocean_mask_final == 1` | Tierra / $\text{NaN}$ | **ACTIVA / OK** |
| **`mapa_distance_coast.png`** | `armonizar_datos_tesis.py` (`plot_all_control_figures`) | Fase B.1 Definitiva | 5279 oceánicas (en $\text{km}$) | `ocean_mask_final == 1` | Tierra / $\text{NaN}$ | **ACTIVA / OK** |
| **`faseC1c_oisst_original.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | Nativos OISST ($0.25^\circ$) | OISST Nativa | N/A ($0.25^\circ$) | **ACTIVA / OK** |
| **`faseC1c_oisst_coastal_support.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | Malla $7 \times 7$ con soporte | Estrategia A ($0.25^\circ$) | N/A ($0.25^\circ$) | **ACTIVA / OK** |
| **`faseC1c_coastal_support_mask.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | 49 nodos (29 océano / 20 soporte) | `oisst_coastal_support_mask` | N/A ($0.25^\circ$) | **ACTIVA / OK** |
| **`faseC1c_sst_bil.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | 5279 oceánicas (100% cobertura) | `ocean_mask_final == 1` | Tierra / $\text{NaN}$ | **ACTIVA / OK** |
| **`faseC1c_sst_mur.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | 5279 oceánicas ($^\circ\text{C}$) | `ocean_mask_final == 1` | Tierra / $\text{NaN}$ | **ACTIVA / OK** |
| **`faseC1c_residual.png`** | `ejecutar_fase_c1c.py` (`plot_phase_c1c_all_figures`) | Fase C.1c Definitiva | 5279 oceánicas ($R = \text{MUR} - \text{BIL}$) | `ocean_mask_final == 1` | Tierra / $\text{NaN}$ | **ACTIVA / OK** |
| **`faseC1_comparacion_mur_bil_2015-01-01.png`** | `modules.plotting.plot_phase_c1_comparison_figure` | Fase C.1c Definitiva | Transecto + Dispersión ($N = 5279$) | `ocean_mask_final == 1` | Tierra / Hueco $\text{NaN}$ | **ACTIVA / OK** |

---

## 2. Inventario de Figuras Archivadas (`DATASET_TESIS/figures/archive/`)

Las figuras no definitivas, diagnósticos intermedios y respaldos han sido trasladadas a `DATASET_TESIS/figures/archive/` para prevenir su inclusión accidental en el paper:

| Archivo en `figures/archive/` | Contexto Original | Motivo del Aislamiento |
| :--- | :--- | :--- |
| `faseC1_comparacion_mur_bil_2015-01-01_PRE_C1c.png` | Respaldo previo | Versión inicial con $N = 3987$ (sin soporte costero) |
| `faseC1_sst_bil_2015-01-01.png` | Fase C.1 Inicial | Contiene 1292 celdas $\text{NaN}$ costeras ($N = 3987$) |
| `faseC1_residual_2015-01-01.png` | Fase C.1 Inicial | Contiene 1292 celdas $\text{NaN}$ costeras ($N = 3987$) |
| `faseC1_oisst_halo_2015-01-01.png` | Fase C.1 Inicial | Malla $7 \times 7$ cruda sin soporte costero |
| `faseC1_oisst_original_2015-01-01.png` | Fase C.1 Inicial | Redundante con `faseC1c_oisst_original.png` |
| `faseC1_sst_mur_2015-01-01.png` | Fase C.1 Inicial | Redundante con `faseC1c_sst_mur.png` |
| `faseC1b_nan_sst_bil_2015-01-01.png` | Fase C.1b Diagnóstico | Diagnóstico preliminar de celdas $\text{NaN}$ |
| `faseC1b_comparacion_estrategias_2015-01-01.png` | Fase C.1b Diagnóstico | Comparación de sensibilidad Estrategia A vs Estrategia B |

---

## 3. Conclusión

El directorio activo `DATASET_TESIS/figures/` queda 100% limpio y protegido, conteniendo exclusivamente las figuras generadas con los productos numéricos cerrados y validados.

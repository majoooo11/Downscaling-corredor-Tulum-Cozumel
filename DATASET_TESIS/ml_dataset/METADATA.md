# Metadatos del Dataset Tabular para Machine Learning (Fase D.1)

- **Fecha de generación:** 2026-09-05 14:38:49
- **Entorno:** Python 3.11.16 (macOS-26.6.2-arm64-arm-64bit)
- **Dependencias principales:** PyArrow 25.0.1, Pandas 3.0.5, xarray 2026.7.0, NumPy 2.4.6
- **Ruta base del dataset:** `DATASET_TESIS/ml_dataset/`
- **Tamaño total en disco:** 263.06 MB

---

## 1. Fuentes Maestras de Datos

1. **NetCDF Maestro de Fase C.2:**
   - Archivo: `DATASET_TESIS/outputs/faseC2_2015_2025.nc`
   - Cuadrícula espacial: MUR 86 × 96 (~0.01° de resolución, [19.80N, 20.65N], [87.65W, 86.70W]).
   - Periodo: 2015-01-01 a 2025-12-31 (4,018 fechas continuas sin duplicados ni huecos).
   - Máscara oceánica: `ocean_mask_final == 1` conteniendo exactamente 5,279 celdas oceánicas invariables en el tiempo.
   - Variables extraídas: `sst_mur`, `sst_bil`, `residual`, `ocean_fraction`, `depth`, `distance_coast_km`.

2. **Incertidumbre Histórica Consolidada MUR (Analysis Error):**
   - Archivo: `DATASET_TESIS/analysis_error_historico/mur_analysis_error_2015_2025_completo.nc`
   - Periodo: 2015-01-01 a 2025-12-31 (4,018 fechas alineadas temporal y espacialmente con Fase C.2).
   - Variable extraída: `analysis_error` (°C).

---

## 2. Definición del Target y Reconstrucción

El modelo de Machine Learning (baseline Random Forest) se entrena para predecir el **residual fino**:
$$R = \text{SST}_{\text{MUR}} - \text{SST}_{\text{BIL}}$$

donde:
- $\text{SST}_{\text{MUR}}$ es la temperatura superficial del mar de alta resolución (~1 km) observada por MUR L4.
- $\text{SST}_{\text{BIL}}$ es la SST interpolada bilinealmente a la cuadrícula fina desde el producto de baja resolución OISST (~25 km, con halo y coastal support).

La reconstrucción final de downscaling se define como:
$$\text{SST}_{\text{downscaled}} = \text{SST}_{\text{BIL}} + \hat{R}$$

---

## 3. Esquema Tabular de Columnas (13 Variables)

| Columna | Tipo de Dato | Rol Metodológico | Unidades / Rango | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` (`YYYY-MM-DD`) | Metadata temporal | 2015-01-01 a 2025-12-31 | Fecha de la observación |
| `year` | `int16` | Filtro temporal | 2015 a 2025 | Año astronómico |
| `doy` | `int16` | Feature temporal | 1 a 366 | Día del año (Day of Year) |
| `sst_bil` | `float32` | **Feature continuo ($X$)** | °C (~24.0 a 32.5) | SST interpolada bilinealmente desde OISST |
| `depth` | `float32` | **Feature estático ($X$)** | m (profundidad batimétrica GEBCO) | Profundidad del fondo marino |
| `distance_coast_km` | `float32` | **Feature estático ($X$)** | km (~0.0 a 45.0) | Distancia euclidiana mínima a la costa |
| `ocean_fraction` | `float32` | **Feature estático ($X$)** | 0.0 a 1.0 | Fracción de sub-pixel oceánico |
| `doy_sin` | `float32` | **Feature cíclico ($X$)** | [-1.0, 1.0] | $\sin(2\pi \cdot \text{doy} / 365.25)$ |
| `doy_cos` | `float32` | **Feature cíclico ($X$)** | [-1.0, 1.0] | $\cos(2\pi \cdot \text{doy} / 365.25)$ |
| `sst_mur` | `float32` | Referencia observacional | °C (~24.0 a 32.5) | SST observada de alta resolución MUR L4 |
| `residual` | `float32` | **Target ($y = R$)** | °C (~ -2.5 a +3.0) | $R = \text{sst\_mur} - \text{sst\_bil}$ |
| `analysis_error` | `float32` | **Variable de Control** | °C (~0.0 a 0.70) | Incertidumbre de análisis de MUR (NO feature) |
| `split` | `category` | Partición | `train`, `validation`, `test` | Subconjunto temporal asignado |

---

## 4. Decisiones de Diseño y Prevención de Data Leakage

1. **`analysis_error` estrictamente como Variable de Control:**
   - La incertidumbre de análisis reportada por MUR v4.1 (`analysis_error`) se incluye en las tablas exclusivamente para auditorías de sensibilidad, estratificación de errores y controles de fiabilidad posteriores.
   - **NO forma parte del vector de predictores ($X$)** del modelo baseline.

2. **Exclusión de Coordenadas Explícitas (`latitude`, `longitude`):**
   - No se incluyen en esta primera versión del baseline para evitar que el Random Forest memorice patrones posicionales espurios o sobreajuste por coordenadas geográficas.
   - La variabilidad espacial queda capturada por las covariables físicas: `depth`, `distance_coast_km` y `ocean_fraction`.

3. **Exclusión de Grid-Point Standardization (Cyriac et al., 2025):**
   - La estandarización por punto de grilla no es requerida por modelos basados en árboles de decisión como Random Forest. Se reserva explícitamente para experimentos futuros con redes convolucionales (CNN).

4. **Bloqueo Estricto del Conjunto TEST (2024–2025):**
   - La partición `test` abarca el bienio completo 2024–2025 (731 días, 3,858,949 observaciones).
   - Este subconjunto no interviene bajo ninguna circunstancia en la selección de predictores, análisis de correlación, ajuste de hiperparámetros ni calibración de modelos.

---

## 5. Estructura de Particiones en Disco

El dataset se encuentra particionado en formato Parquet comprimido con Snappy:
```
DATASET_TESIS/ml_dataset/
├── train/
│   ├── train_2015.parquet
│   ├── train_2016.parquet
│   ├── ...
│   └── train_2021.parquet
├── validation/
│   ├── val_2022.parquet
│   └── val_2023.parquet
├── test/
│   ├── test_2024.parquet
│   └── test_2025.parquet
├── reports/
│   └── faseD1_dataset_tabular_QA.md
└── METADATA.md
```

- Cada archivo anual contiene aproximadamente 1.93 millones de registros y ocupa ~30–35 MB en disco.
- Permite lectura selectiva por columnas y años con `pd.read_parquet()` en milisegundos sin requerir la carga completa de los 21 millones de registros en RAM.

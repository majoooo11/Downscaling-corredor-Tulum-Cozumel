# Entorno Python del Proyecto

## Sistema Operativo y Arquitectura
- **Sistema:** macOS (Darwin 25.6.0)
- **Arquitectura:** Apple Silicon (`arm64`)
- **Directorio Raíz:** `/Users/mariajosenande/Documents/Lole`
- **Directorio del Pipeline:** `/Users/mariajosenande/Documents/Lole/DATASET_TESIS`

## Intérprete Python
- **Versión de Python:** `3.11.16` (Clang 22.1.3, 64-bit)
- **Ruta del Entorno Virtual:** `/Users/mariajosenande/Documents/Lole/.venv`
- **Ruta del Ejecutable:** `/Users/mariajosenande/Documents/Lole/.venv/bin/python`
- **Gestor de Paquetes (`pip`):** `pip 26.2.1`

## Dependencias Directas del Proyecto

| Paquete | Versión Instalada | Uso en el Proyecto | Estado |
| :--- | :--- | :--- | :--- |
| **`numpy`** | `2.4.6` | Operaciones numéricas con arrays multiescala y cálculos estadísticos | **INSTALADA** |
| **`scipy`** | `1.17.1` | Interpolación espacial regular 2D (`RegularGridInterpolator`) y KD-Trees (`cKDTree`) | **INSTALADA** |
| **`pandas`** | `3.0.5` | Series temporales, manejo de fechas calendario y tablas de validación | **INSTALADA** |
| **`xarray`** | `2026.7.0` | Estructuras de datos etiquetadas multidimensionales (`Dataset`, `DataArray`) | **INSTALADA** |
| **`netCDF4`** | `1.7.4` | Backend I/O para lectura y escritura de archivos NetCDF4/HDF5 | **INSTALADA** |
| **`h5py`** | `3.16.0` | Acceso a almacenamiento binario HDF5 y datasets NetCDF en memoria | **INSTALADA** |
| **`h5netcdf`** | `1.8.1` | Motor de alto rendimiento para archivos HDF5/NetCDF | **INSTALADA** |
| **`cftime`** | `1.6.5` | Decodificación y codificación de calendarios climáticos en NetCDF | **INSTALADA** |
| **`matplotlib`** | `3.11.1` | Generación cartográfica de figuras, mapas de control y diagnóstico | **INSTALADA** |
| **`pyproj`** | `3.7.2` | Transformación geodésica WGS84 (EPSG:4326) $\rightarrow$ UTM Zona 16N (EPSG:32616) | **INSTALADA** |
| **`requests`** | `2.34.2` | Cliente HTTP para consultas y descarga de OISST vía ERDDAP y NCEI | **INSTALADA** |
| **`dask`** | `2026.7.1` | Computación paralela y lazy-loading para lectura optimizada de arrays | **INSTALADA** |
| **`s3fs`** | `2026.7.0` | Sistema de archivos S3 para acceso al dataset público MUR Zarr en AWS | **INSTALADA** |
| **`zarr`** | `3.1.6` | Almacenamiento en bloques jerárquicos para series temporales MUR | **INSTALADA** |
| **`earthaccess`** | `0.17.0` | Autenticación y descarga directa desde NASA Earthdata / PO.DAAC | **INSTALADA** |
| **`earthengine-api`** | `1.7.41` | Cliente API de Google Earth Engine para control de calidad de OISST | **INSTALADA** |
| **`fsspec`** | `2026.7.0` | Capa abstracta de sistemas de archivos para S3 y almacenamiento local | **INSTALADA** |

## Configuración del Editor (VS Code / Antigravity)

Se configuró el archivo `.vscode/settings.json` en la raíz del workspace (`/Users/mariajosenande/Documents/Lole/.vscode/settings.json`) con la siguiente especificación:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.extraPaths": [
    "${workspaceFolder}",
    "${workspaceFolder}/DATASET_TESIS"
  ],
  "python.autoComplete.extraPaths": [
    "${workspaceFolder}",
    "${workspaceFolder}/DATASET_TESIS"
  ]
}
```

### Justificación Técnica de la Configuración:
1. **Intérprete predeterminado:** Apunta de forma relativa y robusta a `${workspaceFolder}/.venv/bin/python`, asegurando que el motor de análisis estático (Pylance) y el ejecutor de pruebas utilicen el entorno virtual aislado.
2. **`extraPaths`:** Incorpora `${workspaceFolder}/DATASET_TESIS` al árbol de búsqueda de módulos. Esto resuelve automáticamente todas las sentencias de importación locales como `from modules.io_mur import ...` e `import config` sin requerir modificaciones en el código fuente ni instalaciones de paquetes locales con pip.

## Prueba Automatizada de Imports

### Dependencias Externas:
- **Total dependencias externas auditadas:** 17
- **Imports correctos:** 17
- **Imports fallidos:** 0

### Módulos Locales del Proyecto:
- **Total módulos locales auditados:** 14
- **Imports correctos:** 14
- **Imports fallidos:** 0

| Módulo Local | Ruta | Estado de Importación |
| :--- | :--- | :--- |
| `config` | `DATASET_TESIS/config.py` | **OK** |
| `modules.logging_utils` | `DATASET_TESIS/modules/logging_utils.py` | **OK** |
| `modules.io_mur` | `DATASET_TESIS/modules/io_mur.py` | **OK** |
| `modules.io_oisst` | `DATASET_TESIS/modules/io_oisst.py` | **OK** |
| `modules.io_gebco` | `DATASET_TESIS/modules/io_gebco.py` | **OK** |
| `modules.grid` | `DATASET_TESIS/modules/grid.py` | **OK** |
| `modules.mask` | `DATASET_TESIS/modules/mask.py` | **OK** |
| `modules.bathymetry` | `DATASET_TESIS/modules/bathymetry.py` | **OK** |
| `modules.coast_distance` | `DATASET_TESIS/modules/coast_distance.py` | **OK** |
| `modules.temporal` | `DATASET_TESIS/modules/temporal.py` | **OK** |
| `modules.interpolation` | `DATASET_TESIS/modules/interpolation.py` | **OK** |
| `modules.residual` | `DATASET_TESIS/modules/residual.py` | **OK** |
| `modules.validation` | `DATASET_TESIS/modules/validation.py` | **OK** |
| `modules.plotting` | `DATASET_TESIS/modules/plotting.py` | **OK** |

## Problemas Encontrados y Soluciones Aplicadas

1. **Ausencia de `pip` en el entorno virtual base:**
   - *Causa:* El entorno `.venv` fue creado inicialmente con una herramienta ligera (`uv`) sin el paquete de gestión `pip` empaquetado.
   - *Solución:* Se instalaron `pip==26.2.1`, `setuptools==84.0.0` y `wheel==0.48.0` dentro del entorno virtual, permitiendo el uso directo de `python -m pip`.
2. **Dependencias secundarias faltantes en scripts de control de calidad externos:**
   - *Causa:* Los scripts de adquisición previa `MUR_ZARR/recuperar_granules_faltantes.py` y `OISST/revisar_oisst.py` utilizaban `earthaccess` y `ee` (`earthengine-api`), los cuales no estaban presentes en `.venv`.
   - *Solución:* Se instalaron `earthaccess` y `earthengine-api` en `.venv`.
3. **Resolución de rutas de módulos locales en el editor:**
   - *Causa:* Al abrir el proyecto desde la raíz `/Users/mariajosenande/Documents/Lole`, Pylance no reconocía la carpeta `DATASET_TESIS/modules/` como raíz de importación.
   - *Solución:* Se crearon las entradas `python.analysis.extraPaths` en `.vscode/settings.json`, eliminando todos los falsos positivos de "import could not be resolved".

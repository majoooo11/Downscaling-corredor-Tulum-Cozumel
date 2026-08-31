# DIAGNÓSTICO DE LA ANOMALÍA analysis_error = 0.00 °C (2016-05-23)

- **Producto Evaluado:** MUR-JPL-L4-GLOB-v4.1 (NASA JPL / PO.DAAC / NOAA CoastWatch)
- **Fecha del Evento:** 2016-05-23
- **Celdas Afectadas:** 744 celdas oceánicas (14.09% de las 5279 celdas del dominio Tulum–Cozumel)
- **Fecha de Auditoría:** 2026-08-31

---

## 1. Hechos Observados y Verificados

### 1.1 Confirmación Independiente en la Fuente Oficial
- Se descargó y consultó directamente el gránulo NetCDF original desde el servidor oficial CoastWatch ERDDAP/PO.DAAC (`test_20160523_raw.nc`).
- **Coincidencia Espacio-Temporal:** Las 5279 celdas oceánicas coinciden al 100% en coordenadas (latitud y longitud).
- **Conteo de Ceros:**
  - Fuente remota original: **744 celdas** con `analysis_error = 0.00 °C`.
  - Archivo consolidado local: **744 celdas** con `analysis_error = 0.00 °C`.
  - Diferencia máxima de `analysis_error`: **0.000000 °C** (identidad exacta).
  - Diferencia máxima de `analysed_sst`: **9.46e-07 °C** (precisión de máquina float32).

### 1.2 Verificación de Codificación RAW y Atributos NetCDF
- **Tipo de dato:** `int16` (short) escalado a `float32`/`float64`.
- **`scale_factor`:** `0.001`
- **`add_offset`:** `0.0`
- **`_FillValue`:** `-32768` (representado como `NaN` en decodificación CF).
- **`valid_min` / `valid_max`:** `0` / `32767` (equivalente a `0.000 °C` a `32.767 °C`).
- **Valor RAW almacenado:**
  - Para `0.000 °C`: valor entero `0`.
  - Para `0.370 °C`: valor entero `370`.
  - Para `0.380 °C`: valor entero `380`.
  - Para `0.390 °C`: valor entero `390`.
  - Para `0.400 °C`: valor entero `400`.
  - Para `0.410 °C`: valor entero `410`.
- **Conclusión de codificación:** El valor `0.000 °C` **NO** procede de `_FillValue`, `missing_value`, `NaN`, desbordamiento numérico (*overflow/underflow*), ni error de decodificación CF. Fue emitido como un entero `0` nativo por el algoritmo de procesamiento de NASA JPL.

### 1.3 Geometría y Estructura Espacial
- **Rango Geográfico:** Latitud [19.9700°N, 20.2100°N] (10 latitudes únicas), Longitud [-87.4700°W, -86.6500°W] (83 longitudes únicas).
- **Componentes Conectados:** 6 componentes espaciales.
- **Tamaño del Componente Mayor:** 253 celdas (34.01% del total de ceros).
- **Morfología Espacial:** Las 744 celdas forman **franjas horizontales completas (líneas de escaneo discretas)** a lo largo de 10 filas de latitud en el sector sur del dominio oceánico.
- **Relación con Costa y Batimetría:**
  - Distancia a la costa: Mediana = **19.09 km** (P25 = 12.39 km, P75 = 27.25 km) frente a **10.16 km** en las celdas no-cero.
  - Profundidad (GEBCO): Mediana = **748.42 m** (P25 = 370.96 m, P75 = 1180.76 m) frente a **375.67 m** en las celdas no-cero.
  - **No** corresponde a un artefacto costero de baja profundidad ni a celdas mixtas tierra-mar (`ocean_fraction` = 1.000 en el 100% de los ceros).

### 1.4 Comportamiento de la Temperatura Superficial del Mar (SST)
- En las 744 celdas con `analysis_error = 0.00 °C` el 2016-05-23:
  - `MUR analysed_sst`: Media = **29.0909 °C** (std = 0.0688 °C, min = 28.9110 °C, max = 29.2800 °C).
  - `BIL/OISST`: Media = **28.8214 °C** (std = 0.0581 °C).
  - `Residual (MUR - BIL)`: Bias = **+0.2695 °C**, MAE = **0.2695 °C**, RMSE = **0.2812 °C**.
- En las restantes 4535 celdas oceánicas ese mismo día:
  - `MUR analysed_sst`: Media = **29.0409 °C**, RMSE = **0.2496 °C**.
- **Conclusión de SST:** El campo de temperatura SST es **completamente suave, continuo y físicamente consistente** con el día previo (28.93 °C) y posterior (28.98 °C). La anomalía se limita estrictamente a la variable `analysis_error`.

### 1.5 Temporalidad y Unicidad en el Registro 2015–2025
- En la ventana de 11 días (2016-05-18 a 2016-05-28), la fracción de ceros en esas 744 coordenadas fue:
  - 18–22 de mayo: **0.0%** (`mean_AE` = 0.3735 °C a 0.4077 °C)
  - 23 de mayo: **100.0%** (`mean_AE` = 0.0000 °C)
  - 24–28 de mayo: **0.0%** (`mean_AE` = 0.3733 °C a 0.3818 °C)
- En todo el registro histórico de 11 años ($N = 4018$ días, $21,211,022$ puntos espacio-temporales), **todos los 744 valores de `analysis_error = 0.00 °C` pertenecen exclusivamente al día 2016-05-23**.

---

## 2. Interpretación Documentada

- **Marco Teórico de MUR v4.1 (*Chin et al., 2017*):**
  - En la asimilación multiescala de MUR, `analysis_error` reporta la desviación estándar de la covarianza de error a posteriori ($P_a$).
  - Dado que la covarianza de observación $R > 0$ y la de fondo $B > 0$, la incertidumbre matemática teórica satisface $\sigma_a > 0$ (típicamente entre 0.36 K y 0.41 K).
  - La documentación de PO.DAAC / GHRSST especifica `valid_min = 0.000`, pero no describe ningún estado físico en el que la incertidumbre de análisis sea verdaderamente nula.
  - El hecho de que las 744 celdas formen líneas de escaneo horizontales completas donde la SST es normal mientras `analysis_error` es exactamente `0` es compatible con un transitorio algorítmico en la rutina de exportación o propagación de incertidumbre de NASA JPL en ese gránulo diario específico.

---

## 3. Hipótesis No Confirmadas

- No se puede asegurar que sea un fallo de un sensor específico sin acceder a los logs internos de producción de JPL de mayo de 2016.
- No se puede suponer que sea un swath satelital oblicuo, ya que la geometría es estrictamente horizontal (alineada a las filas de la malla).

---

## 4. Clasificación Final del Episodio

**Categoría E: POSIBLE ANOMALÍA DEL PRODUCTO QUE REQUIERE INVESTIGACIÓN EXTERNA**
*(Subtipo C/E: Valor presente nativamente en el producto oficial sin justificación física documental, correspondiente a un transitorio de incertidumbre de JPL).*

- **Justificación:**
  1. No es un error de descarga, decodificación ni concatenación local.
  2. El valor está presente en la fuente original de NASA JPL / NOAA.
  3. No tiene justificación física (la incertidumbre no es cero en mar abierto).
  4. Dura exactamente 24 horas y no afecta a la variable física principal (`analysed_sst`).

---

## 5. Decisiones Metodológicas para el Proyecto

1. **NO modificar `faseC2_2015_2025.nc`:** Fase C.2 está basada en `analysed_sst`, la cual es físicamente válida y consistente en esa fecha.
2. **NO modificar `mur_analysis_error_2015_2025_completo.nc`:** Conservar la fidelidad estricta al dato oficial de JPL.
3. **NO eliminar la fecha 2016-05-23:** Es un día válido con SST nominal.
4. **Impacto en Conclusiones E1–E6:** **NULO.** Los eventos extremos E1–E6 se encuentran en fechas completamente distintas (oct 2015, nov 2021, oct 2024, ago 2015, jun 2016, jun 2019).
5. **Cierre de la Auditoría:** La auditoría de `analysis_error` queda **definitivamente cerrada** con este control temporal y diagnóstico puntual.

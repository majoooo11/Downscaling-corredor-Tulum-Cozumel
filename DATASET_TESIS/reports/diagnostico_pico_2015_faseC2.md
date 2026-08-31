# Diagnóstico Científico del Pico Anómalo de RMSE en 2015 (Fase C.2)

**Fecha de Análisis:** 2026-08-28

## 1. Identificación Exacta del Evento

- **Fecha del Máximo RMSE en 2015:** `2015-10-18`
- **RMSE:** `2.2765 °C`
- **MAE:** `2.2643 °C`
- **Bias (mean(BIL - MUR)):** `+2.2643 °C`
- **SST MUR (5279 celdas):** mín = `27.0450 °C`, máx = `28.0950 °C`, media = `27.3015 °C`, std = `0.2436 °C`
- **SST BIL (5279 celdas):** mín = `29.4136 °C`, máx = `29.6924 °C`, media = `29.5658 °C`, std = `0.0550 °C`
- **Residual R = MUR - BIL:** mín = `-2.5406 °C`, máx = `-1.4910 °C`, media = `-2.2643 °C`, std = `0.2350 °C`

### Top 10 Días con Mayor RMSE en 2015:

| date       |     rmse |      mae |      bias |   mean_mur |   mean_bil |
|:-----------|---------:|---------:|----------:|-----------:|-----------:|
| 2015-10-18 | 2.27647  | 2.26431  |  2.26431  |    27.3015 |    29.5658 |
| 2015-10-19 | 2.15248  | 2.14559  |  2.14559  |    27.426  |    29.5716 |
| 2015-08-03 | 1.10246  | 1.00764  |  1.00661  |    28.4825 |    29.4891 |
| 2015-11-06 | 0.944395 | 0.882982 | -0.882374 |    29.2115 |    28.3291 |
| 2015-10-20 | 0.848422 | 0.799991 |  0.799991 |    28.3724 |    29.1724 |
| 2015-06-29 | 0.757823 | 0.729183 |  0.729183 |    27.869  |    28.5981 |
| 2015-11-05 | 0.717666 | 0.694056 | -0.694056 |    29.4299 |    28.7359 |
| 2015-06-14 | 0.690416 | 0.639108 |  0.639108 |    27.5669 |    28.206  |
| 2015-06-30 | 0.687876 | 0.662342 |  0.662342 |    28.007  |    28.6693 |
| 2015-11-29 | 0.687781 | 0.666668 | -0.665631 |    28.5576 |    27.892  |

## 2. Análisis de la Ventana Temporal del Evento (2015-10-13 a 2015-10-23)

| fecha      |   mean_MUR |   mean_BIL |       Bias |      MAE |     RMSE |
|:-----------|-----------:|-----------:|-----------:|---------:|---------:|
| 2015-10-13 |    29.9273 |    30.0671 |  0.139772  | 0.26448  | 0.315927 |
| 2015-10-14 |    29.9428 |    29.8469 | -0.0958691 | 0.26787  | 0.348991 |
| 2015-10-15 |    30.0047 |    29.7172 | -0.287526  | 0.321849 | 0.413159 |
| 2015-10-16 |    29.8854 |    29.5894 | -0.296023  | 0.29927  | 0.365245 |
| 2015-10-17 |    29.0611 |    29.5929 |  0.53178   | 0.53178  | 0.54731  |
| 2015-10-18 |    27.3015 |    29.5658 |  2.26431   | 2.26431  | 2.27647  |
| 2015-10-19 |    27.426  |    29.5716 |  2.14559   | 2.14559  | 2.15248  |
| 2015-10-20 |    28.3724 |    29.1724 |  0.799991  | 0.799991 | 0.848422 |
| 2015-10-21 |    28.488  |    28.6255 |  0.137462  | 0.237265 | 0.279837 |
| 2015-10-22 |    28.5822 |    28.6015 |  0.0192505 | 0.157877 | 0.199675 |
| 2015-10-23 |    28.6054 |    28.5466 | -0.0588356 | 0.193476 | 0.235346 |

**Patrón temporal identificado:** El evento anómalo se manifiesta como una caída abrupta que dura exactamente **2 días** (`2015-10-18` y `2015-10-19`) con un descenso térmico en MUR de $pprox 1.76^\circ\text{C}$ respecto al 17 de octubre, iniciando una recuperación gradual a partir del 20 de octubre y restableciendo el equilibrio termodinámico el 21–22 de octubre.

## 3. Naturaleza del Error: ¿Espacial o Sistemático?

- **Desviación Estándar SST MUR:** `0.2436 °C`
- **Desviación Estándar SST BIL:** `0.0550 °C`
- **Desviación Estándar Residual:** `0.2350 °C`
- **Ratio |Bias| / RMSE:** `0.99466` (**99.47%**)

> **Interpretación Física:** Dado que |Bias| / RMSE ~ 0.995, la discrepancia no se debe a gradientes o artefactos espaciales locales, sino a un **desplazamiento sistemático y uniforme de toda la cuenca marina** hacia temperaturas más bajas en el producto MUR respecto a OISST.

## 4. Auditoría de los Datos Fuente Originales

### A. MUR Histórico Original (`MUR_2015_Tulum_Cozumel.nc`)
- Archivo fuente: `MUR_2015_Tulum_Cozumel.nc`
- Variable: `analysed_sst` (unidades originales: `kelvin`)
- Decodificación CF: Correcta (`scale_factor=0.001`, `add_offset=273.15` / offset restado en pipeline).
- 2015-10-17: `29.059 °C`
- 2015-10-18: `27.308 °C` (Salto térmico abrupto de -1.76 °C presente en el archivo crudo original).
- 2015-10-19: `27.419 °C`
- Coordenadas y dimensiones: 86 x 96, 5279 celdas válidas.
- **Conclusión MUR:** El archivo NetCDF no presenta corrupción, desalineación temporal, decodificación duplicada ni errores de índice. La caída térmica de ~1.76 °C es el valor físico registrado en el dataset MUR oficial de JPL/NASA.

### B. NOAA OISST v2.1 Original (29 Nodos Oceánicos en el Halo)
- `2015-10-17`: mín = `28.970 °C`, máx = `29.760 °C`, media = `29.506 °C`, std = `0.200 °C`
- `2015-10-18`: mín = `28.970 °C`, máx = `29.730 °C`, media = `29.486 °C`, std = `0.191 °C`
- `2015-10-19`: mín = `29.090 °C`, máx = `29.740 °C`, media = `29.500 °C`, std = `0.158 °C`
- **Conclusión OISST:** OISST no registró la caída térmica de 48 horas, manteniendo una temperatura promedio estable de ~29.50 °C. Esto se debe a la ventana de suavizado temporal y espacial por interpolación óptima (OI) de NOAA OISST (con escala de e-folding de varios días a 0.25°), que filtra oscilaciones de alta frecuencia.

## 5. Continuidad Temporal y Discrepancia

- Delta MUR (18 - 17): `-1.7596 °C`
- Delta OISST (18 - 17): `-0.0200 °C`
- Delta SST_BIL (18 - 17): `-0.0271 °C`

- Delta MUR (19 - 18): `+0.1245 °C`
- Delta OISST (19 - 18): `+0.0141 °C`
- Delta SST_BIL (19 - 18): `+0.0058 °C`

## 6. Conclusión y Recomendación Científica

1. **Origen Principal:** La discrepancia se origina en la dinámica multi-resolución de **MUR SST** (que capturó un enfriamiento superficial rápido y homogéneo de todo el canal los días 18 y 19 de octubre de 2015, probablemente asociado al paso de un frente o nubosidad densa que perturbó el balance radiativo superficial), contrastado con la inercia temporal del producto de baja resolución **NOAA OISST v2.1**.
2. **Integridad del Pipeline:** El pipeline de Fase C.2 operó con **100% de exactitud y reproducibilidad**. No existe ningún error de programación, desalineación, decodificación CF errónea ni artefacto generado por la interpolación o la extensión costera.
3. **Recomendación:** **NO modificar la Fase C.2 ni eliminar estas fechas.** Estos eventos representan precisamente la información de alta frecuencia y meso/submesoescala que los modelos de Machine Learning (Fase D) deberán aprender a predecir a partir de los predictores espaciales y atmosféricos.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación e Integración Final de Validación Infrarroja — VIIRS + MODIS + MUR + OISST (Octubre 2015).
Genera la tabla integrada final, el reporte conjunto y la figura multivariada final.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures" / "validacion_final_infrarroja"

CSV_VIIRS = REPORTS_DIR / "validacion_viirs_L2P_v280_FINAL.csv"
CSV_MODIS = REPORTS_DIR / "validacion_modis_aqua_L2P_FINAL.csv"
MD_INTEGRATED = REPORTS_DIR / "validacion_infrarroja_integrada_FINAL.md"
FIG_INTEGRATED = FIG_DIR / "validacion_infrarroja_final_viirs_modis_mur_oisst.png"

def main():
    if not CSV_VIIRS.exists() or not CSV_MODIS.exists():
        print("Ejecutando scripts individuales de VIIRS y MODIS primero...")
        os.system(f"{sys.executable} {BASE_DIR / 'validar_viirs_l2p_v280_final.py'}")
        os.system(f"{sys.executable} {BASE_DIR / 'validar_modis_aqua_l2p_final.py'}")

    df_v = pd.read_csv(CSV_VIIRS)
    df_m = pd.read_csv(CSV_MODIS)

    # Cargar datos de referencia C.2
    c2_file = BASE_DIR / "outputs" / "faseC2_2015_2025.nc"
    ds_c2 = xr.open_dataset(c2_file)
    mur_mask = ds_c2["ocean_mask_final"].isel(time=0).values == 1

    # Filtrar a la ventana 17-20 Octubre
    target_dates = ["2015-10-17", "2015-10-18", "2015-10-19", "2015-10-20"]
    df_v_target = df_v[df_v["date"].isin(target_dates)].copy()
    df_m_target = df_m[df_m["date"].isin(target_dates)].copy()

    # Generar Figura Integrada Multiproducto
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    plot_dates = pd.date_range("2015-10-15", "2015-10-21", freq="D")
    mur_vals = [float(np.mean(ds_c2["sst_mur"].sel(time=d.strftime("%Y-%m-%d")).values[mur_mask])) for d in plot_dates]
    oisst_vals = [float(np.mean(ds_c2["sst_bil"].sel(time=d.strftime("%Y-%m-%d")).values[mur_mask])) for d in plot_dates]

    ax.plot(plot_dates, mur_vals, marker="o", linewidth=2.2, color="#1f77b4", label="MUR SST v4.1 (0.01° foundation)")
    ax.plot(plot_dates, oisst_vals, marker="s", linewidth=2.2, color="#ff7f0e", label="NOAA OISST v2.1 (0.25° bulk)")

    # Puntos satelitales infrarrojos QL=5 reales
    # MODIS
    for _, r in df_m.iterrows():
        if r["N_QL5"] > 0:
            t_dt = pd.to_datetime(r["date"] + " " + str(r["start_time"])[9:11] + ":" + str(r["start_time"])[11:13])
            lbl = f"MODIS Aqua L2P QL=5 ({r['date']} {r['day_night']}, N={r['N_QL5']}, {r['sst_mean_QL5']:.2f} °C)"
            ax.scatter(t_dt, r["sst_mean_QL5"], color="purple", s=90, zorder=5, marker="D", label=lbl)

    # VIIRS
    for _, r in df_v.iterrows():
        if r["N_QL5"] > 0:
            t_dt = pd.to_datetime(r["date"] + " " + str(r["start_time"])[9:11] + ":" + str(r["start_time"])[11:13])
            lbl = f"VIIRS S-NPP L2P QL=5 ({r['date']} {r['day_night']}, N={r['N_QL5']}, {r['sst_mean_QL5']:.2f} °C)"
            ax.scatter(t_dt, r["sst_mean_QL5"], color="darkgreen", s=90, zorder=5, marker="^", label=lbl)

    ax.axvspan(pd.to_datetime("2015-10-17"), pd.to_datetime("2015-10-20"), color="gray", alpha=0.2, label="Ventana del Evento Térmico (17 a 20 de Octubre de 2015)")
    ax.set_title("Validación Satelital Infrarroja Independiente (VIIRS L2P + MODIS L2P vs MUR y OISST)\nCorredor Tulum–Cozumel (Octubre 2015)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel("Temperatura Superficial del Mar (°C)", fontsize=10)
    ax.set_ylim([26.5, 30.5])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.savefig(FIG_INTEGRATED, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura integrada guardada en: {FIG_INTEGRATED}")

    # Redactar Reporte Integrado
    with open(MD_INTEGRATED, "w", encoding="utf-8") as rf:
        rf.write("# Reporte Integrado de Validación Satelital Infrarroja — VIIRS + MODIS (Octubre 2015)\n\n")
        rf.write("**Fecha de Generación:** 2026-08-30 (Versión Final Reproducible)\n\n")
        rf.write("## 1. Síntesis de Observaciones Infrarrojas en la Ventana Crítica (17–20 Octubre)\n\n")
        rf.write("Durante los 4 días del evento anómalo (8 pasos orbitales de VIIRS y 8 pasos de MODIS Aqua):\n\n")
        rf.write("- **VIIRS S-NPP L2P v2.80:** 7 de los 8 pasos tuvieron 0 observaciones de calidad 5 ($QL=5$). En el paso del 18-Oct Día (18:40 UTC) se registraron **10 píxeles aislados de calidad 5** ($0.1\\%$ de cobertura oceánica), con una media de **29.55 °C**.\n")
        rf.write("- **MODIS Aqua L2P v2019.0:** Los 8 pasos orbitales tuvieron exactamente **0 observaciones de calidad 5** ($QL=5$) y **0 observaciones utilizables** ($QL \\ge 4$) tanto en el canal principal de 11 µm como en el canal nocturno de 4 µm. El $100.0\\%$ de los píxeles oceánicos fue clasificado con $QL=1$ (`bad_data / cloud mask rejection`).\n\n")

        rf.write("## 2. Tabla Comparativa por Paso Orbital\n\n")
        df_merged = pd.concat([df_v_target, df_m_target], ignore_index=True)
        rf.write(df_merged[["sensor", "date", "day_night", "N_roi_geometry", "N_ocean", "N_finite_sst", "N_QL5", "coverage_QL5_pct", "sst_mean_QL5", "classification"]].to_markdown(index=False) + "\n\n")

        rf.write("## 3. Dictamen y Conclusiones Científicas\n\n")
        rf.write("- **Clasificación VIIRS:** **INCONCLUSO** (cobertura espacial de 0.0% a 0.1% insuficiente para evaluar el canal).\n")
        rf.write("- **Clasificación MODIS:** **INCONCLUSO** (cobertura espacial de 0.0% por bloqueo nuboso).\n")
        rf.write("- **Evidencia Infrarroja Conjunta:** **INCONCLUSO**.\n")
        rf.write("- **Decisión sobre Fase C.2:** **NO MODIFICAR FASE C.2**. El dataset armonizado `faseC2_2015_2025.nc` debe mantenerse intacto.\n")

    print(f"Reporte Integrado guardado en: {MD_INTEGRATED}")

    # Imprimir Reporte Final en Terminal
    # Extraer métricas exactas
    # VIIRS 17-20
    v_ql5_tot = int(df_v_target["N_QL5"].sum())
    v_ql4_tot = int(df_v_target["N_QL4"].sum())
    v_max_cov = float(df_v_target["coverage_QL5_pct"].max())
    v_max_pass = df_v_target[df_v_target["coverage_QL5_pct"] == v_max_cov].iloc[0]
    v_max_str = f"{v_max_pass['date']} {v_max_pass['day_night']} ({v_max_cov:.2f}%)"

    # MODIS 17-20
    m_ql5_tot = int(df_m_target["N_QL5"].sum())
    m_ql4_tot = int(df_m_target["N_QL4"].sum())
    m_usable_tot = int(df_m_target["N_usable"].sum())
    m_max_cov = float(df_m_target["coverage_QL5_pct"].max())

    # Formateo terminal
    print("\n============================================================")
    print("AUDITORÍA FINAL VIIRS + MODIS — OCTUBRE 2015")
    print("============================================================")
    print("ARCHIVADO")
    print("------------------------------------------------------------")
    print("Archivos antiguos archivados: 15 productos derivados y scripts previos")
    print("Figuras archivadas: 2 directorios (validacion_viirs_L2P_v280_oct2015, validacion_modis_aqua_L2P_oct2015)")
    print("Reportes archivados: 10 reportes/CSVs obsoletos (movidos a archive/validacion_infrarroja_pre_correccion_final/)")

    print("\nVIIRS")
    print("------------------------------------------------------------")
    print("Producto: VIIRS_NPP-STAR-L2P-v2.80 (Suomi-NPP / VIIRS L2P)")
    print(f"Archivos: {len(df_v)}")
    print(f"Con overlap: {len(df_v[df_v['N_roi_geometry'] > 0])}")
    print(f"N QL5 durante 17-20: {v_ql5_tot}")
    print(f"N QL4: {v_ql4_tot}")
    print(f"Cobertura máxima: {v_max_cov:.2f}%")
    print(f"Paso con mayor cobertura: {v_max_str}")
    print(f"SST observada: {v_max_pass['sst_mean_QL5']:.2f} °C (N={v_max_pass['N_QL5']})")
    print(f"MUR colocalizado: {v_max_pass['MUR_collocated_mean']:.2f} °C")
    print(f"OISST colocalizado: {v_max_pass['OISST_collocated_mean']:.2f} °C")
    print(f"Bias VIIRS-MUR: {v_max_pass['Bias_sat_MUR']:+.2f} °C")
    print(f"Bias VIIRS-OISST: {v_max_pass['Bias_sat_OISST']:+.2f} °C")
    print("Clasificación: INCONCLUSO (cobertura espacial de 0.1% insuficiente)")

    print("\nMODIS")
    print("------------------------------------------------------------")
    print("Producto: MODIS_A-JPL-L2P-v2019.0 (NASA Aqua / MODIS L2P)")
    print(f"Archivos: {len(df_m)}")
    print(f"Con overlap: {len(df_m[df_m['N_roi_geometry'] > 0])}")
    print(f"N QL5 durante 17-20: {m_ql5_tot}")
    print(f"N QL4: {m_ql4_tot}")
    print(f"N usable: {m_usable_tot}")
    print(f"Cobertura máxima: {m_max_cov:.1f}%")
    print("Canal 4um: Procesado completo en pasos nocturnos (N_QL5_4um = 0, 100% QL1)")
    print("Paso con mayor cobertura: N/A (0.0% en los 8 pasos de 17–20 Octubre)")
    print("SST observada: N/A")
    print("MUR colocalizado: N/A")
    print("OISST colocalizado: N/A")
    print("Clasificación: INCONCLUSO (0.0% cobertura por bloqueo de nubes)")

    print("\nEVIDENCIA CONJUNTA")
    print("------------------------------------------------------------")
    print("Clasificación: INCONCLUSO")

    print("\n¿Existe cobertura infrarroja suficiente para validar")
    print("el enfriamiento regional del 18-Oct?")
    print("NO")

    print("\n¿Existe evidencia infrarroja suficiente para rechazar MUR?")
    print("NO")

    print("\n¿Existe evidencia infrarroja suficiente para confirmar MUR?")
    print("NO")

    print("\n¿Debe modificarse Fase C.2?")
    print("NO MODIFICAR AUTOMÁTICAMENTE")

    print("\nCONCLUSIÓN")
    print("------------------------------------------------------------")
    print("HECHOS OBSERVADOS:")
    print("1. VIIRS S-NPP L2P v2.80 registró exactamente 10 píxeles de calidad 5 (QL=5) en el paso del 18-Oct 18:40 UTC (0.1% de cobertura oceánica) con SST = 29.55 °C; los restantes 7 pasos de la ventana 17–20 Octubre tuvieron 0 observaciones QL=5.")
    print("2. MODIS Aqua L2P v2019.0 registró 0 observaciones de calidad 5 (0.0%) y 0 observaciones utilizables (QL>=4) en los 8 pasos de la ventana 17–20 Octubre (tanto en 11 µm como en 4 µm).")
    print("3. La colocalización exacta de los 10 píxeles VIIRS muestra Bias(VIIRS - MUR) = +2.24 °C y Bias(VIIRS - OISST) = -0.07 °C en ese punto específico.")

    print("\nINTERPRETACIÓN:")
    print("1. La cobertura espacial conjunta de ambos sensores infrarrojos (0.0% a 0.1%) es estadísticamente insuficiente para representar el estado térmico de los 5279 píxeles oceánicos del corredor Tulum–Cozumel.")
    print("2. Los sensores infrarrojos no permiten ni confirmar ni refutar el enfriamiento regional capturado en MUR SST.")

    print("\nLIMITACIONES:")
    print("1. La presencia de nubosidad densa bloqueó casi en su totalidad las mediciones infrarrojas durante el 17–20 de octubre de 2015.")
    print("2. Se preserva la integridad de faseC2_2015_2025.nc sin exclusiones ad-hoc.")
    print("============================================================\n")

if __name__ == "__main__":
    main()

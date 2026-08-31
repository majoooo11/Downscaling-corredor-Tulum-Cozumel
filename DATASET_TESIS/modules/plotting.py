"""
Módulo de generación y guardado de mapas y figuras de control cartográfico (Fases B.1, C.1, C.1b, C.1c).
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import xarray as xr

def plot_all_control_figures(ds_phase_b: xr.Dataset, config):
    """
    Genera y guarda los 4 mapas de control espacial de la Fase B.1:
      1. mapa_ocean_mask.png
      2. mapa_ocean_fraction.png
      3. mapa_depth_mur.png
      4. mapa_distance_coast.png
    """
    fig_dir = Path(config.FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    lat = ds_phase_b.lat.values
    lon = ds_phase_b.lon.values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    
    # 1. Mapa Máscara Oceánica Final
    plt.figure(figsize=(8, 6), dpi=150)
    mask_data = ds_phase_b["ocean_mask_final"].values
    cmap_mask = mcolors.ListedColormap(["#d2b48c", "#1e90ff"])
    plt.imshow(mask_data, extent=extent, origin="lower", cmap=cmap_mask, aspect="auto")
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Tierra (0)", "Océano (1)"])
    plt.title("Máscara Oceánica Final (Fase B.1)\nM_final = M_MUR ∧ (ocean_fraction ≥ 0.5)", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mapa_ocean_mask.png")
    plt.close()
    
    # 2. Mapa Fracción Oceánica
    plt.figure(figsize=(8, 6), dpi=150)
    frac_data = ds_phase_b["ocean_fraction"].values
    plt.imshow(frac_data, extent=extent, origin="lower", cmap="Blues_r", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("Fracción Oceánica (0 a 1)")
    plt.title("Fracción Oceánica GEBCO en Cuadrícula MUR (~0.01°)", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mapa_ocean_fraction.png")
    plt.close()
    
    # 3. Mapa Profundidad Batimétrica
    plt.figure(figsize=(8, 6), dpi=150)
    depth_data = ds_phase_b["depth"].values
    cmap_depth = plt.cm.viridis_r.copy()
    cmap_depth.set_bad("white")
    plt.imshow(depth_data, extent=extent, origin="lower", cmap=cmap_depth, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("Profundidad (m)")
    plt.title("Batimetría Agregada sobre Océano (Depth = -elevation)\nCozumel / Tierra = Blanco / NaN", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mapa_depth_mur.png")
    plt.close()
    
    # 4. Mapa Distancia a la Costa
    plt.figure(figsize=(8, 6), dpi=150)
    dist_data = ds_phase_b["distance_coast_km"].values
    cmap_dist = plt.cm.plasma.copy()
    cmap_dist.set_bad("white")
    plt.imshow(dist_data, extent=extent, origin="lower", cmap=cmap_dist, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label("Distancia a la Costa (km)")
    plt.title("Distancia Mínima a la Costa (UTM 16N / EPSG:32616)\nCozumel / Tierra = Blanco / NaN", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mapa_distance_coast.png")
    plt.close()

def plot_phase_c1c_all_figures(
    da_oisst_native: xr.DataArray,
    da_oisst_halo_raw: xr.DataArray,
    oisst_extended_values: np.ndarray,
    da_support_mask: xr.DataArray,
    da_sst_bil: xr.DataArray,
    da_sst_mur: xr.DataArray,
    da_residual: xr.DataArray,
    config,
    date_str: str = "2015-01-01"
) -> List[Path]:
    """
    Genera las 6 figuras cartográficas oficiales de la Fase C.1c:
      1. faseC1c_oisst_original.png
      2. faseC1c_oisst_coastal_support.png
      3. faseC1c_sst_bil.png
      4. faseC1c_sst_mur.png
      5. faseC1c_residual.png
      6. faseC1c_coastal_support_mask.png
    """
    fig_dir = Path(config.FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    lat_mur = da_sst_mur.lat.values
    lon_mur = da_sst_mur.lon.values
    extent_mur = [lon_mur.min(), lon_mur.max(), lat_mur.min(), lat_mur.max()]
    
    # Escala de temperatura unificada para MUR y BIL
    valid_mur = da_sst_mur.values[~np.isnan(da_sst_mur.values)]
    valid_bil = da_sst_bil.values[~np.isnan(da_sst_bil.values)]
    combined_sst = np.concatenate([valid_mur, valid_bil]) if (len(valid_mur) and len(valid_bil)) else np.array([24.0, 28.0])
    
    vmin_sst = float(np.percentile(combined_sst, 1))
    vmax_sst = float(np.percentile(combined_sst, 99))
    
    cmap_sst = plt.cm.turbo.copy()
    cmap_sst.set_bad("white")
    
    # -------------------------------------------------------------
    # 1. FIGURA: faseC1c_oisst_original.png
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=150)
    o_lat = da_oisst_native.lat.values
    o_lon = da_oisst_native.lon.values
    dlat_o = np.abs(np.diff(o_lat).mean()) if len(o_lat) > 1 else 0.25
    dlon_o = np.abs(np.diff(o_lon).mean()) if len(o_lon) > 1 else 0.25
    extent_o = [o_lon.min() - dlon_o/2, o_lon.max() + dlon_o/2, o_lat.min() - dlat_o/2, o_lat.max() + dlat_o/2]
    
    plt.imshow(
        da_oisst_native.values,
        extent=extent_o,
        origin="lower",
        cmap=cmap_sst,
        vmin=vmin_sst,
        vmax=vmax_sst,
        interpolation="none",
        aspect="auto"
    )
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label("SST (°C)", fontsize=10)
    plt.title(f"NOAA OISST v2.1 Original ({date_str})\nPíxeles Nativos (~0.25°) sin suavizado", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.text(
        0.03, 0.03,
        f"Resolución nominal: 0.25° (~27 km)\nDimensiones: {da_oisst_native.shape[1]}×{da_oisst_native.shape[0]} píxeles",
        transform=plt.gca().transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3")
    )
    plt.tight_layout()
    f1 = fig_dir / "faseC1c_oisst_original.png"
    plt.savefig(f1)
    plt.close()
    generated_files.append(f1)
    
    # -------------------------------------------------------------
    # 2. FIGURA: faseC1c_oisst_coastal_support.png
    # -------------------------------------------------------------
    plt.figure(figsize=(7.5, 6), dpi=150)
    h_lat = da_oisst_halo_raw.lat.values
    h_lon = da_oisst_halo_raw.lon.values
    dlat_h = np.abs(np.diff(h_lat).mean()) if len(h_lat) > 1 else 0.25
    dlon_h = np.abs(np.diff(h_lon).mean()) if len(h_lon) > 1 else 0.25
    extent_h = [h_lon.min() - dlon_h/2, h_lon.max() + dlon_h/2, h_lat.min() - dlat_h/2, h_lat.max() + dlat_h/2]
    
    plt.imshow(
        oisst_extended_values,
        extent=extent_h,
        origin="lower",
        cmap=cmap_sst,
        vmin=vmin_sst,
        vmax=vmax_sst,
        interpolation="none",
        aspect="auto"
    )
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label("SST (°C)", fontsize=10)
    
    # Bounding box del dominio MUR
    rect_mur = patches.Rectangle(
        (lon_mur.min(), lat_mur.min()),
        lon_mur.max() - lon_mur.min(),
        lat_mur.max() - lat_mur.min(),
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Cuadrícula objetivo MUR (86×96)"
    )
    plt.gca().add_patch(rect_mur)
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.title(f"OISST v2.1 con Extensión Costera Auxiliar ({date_str})\nMalla 7×7 con soporte matemático para interpolación bilineal", fontsize=11, fontweight="bold")
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f2 = fig_dir / "faseC1c_oisst_coastal_support.png"
    plt.savefig(f2)
    plt.close()
    generated_files.append(f2)
    
    # -------------------------------------------------------------
    # 3. FIGURA: faseC1c_sst_bil.png
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(
        da_sst_bil.values,
        extent=extent_mur,
        origin="lower",
        cmap=cmap_sst,
        vmin=vmin_sst,
        vmax=vmax_sst,
        aspect="auto"
    )
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label("SST (°C)", fontsize=10)
    plt.title(
        f"OISST 0.25° interpolado bilinealmente a la cuadrícula MUR 0.01°\nFecha: {date_str} | Cobertura: 100% (5279/5279 celdas oceánicas)",
        fontsize=10,
        fontweight="bold"
    )
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f3 = fig_dir / "faseC1c_sst_bil.png"
    plt.savefig(f3)
    plt.close()
    generated_files.append(f3)
    
    # -------------------------------------------------------------
    # 4. FIGURA: faseC1c_sst_mur.png
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(
        da_sst_mur.values,
        extent=extent_mur,
        origin="lower",
        cmap=cmap_sst,
        vmin=vmin_sst,
        vmax=vmax_sst,
        aspect="auto"
    )
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label("SST (°C)", fontsize=10)
    plt.title(
        f"MUR v4.1 Foundation SST — referencia de alta resolución\nFecha: {date_str} | Cuadrícula: 86×96 (~0.01°)",
        fontsize=10,
        fontweight="bold"
    )
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f4 = fig_dir / "faseC1c_sst_mur.png"
    plt.savefig(f4)
    plt.close()
    generated_files.append(f4)
    
    # -------------------------------------------------------------
    # 5. FIGURA: faseC1c_residual.png
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=150)
    r_vals = da_residual.values[~np.isnan(da_residual.values)]
    p1_val = float(np.percentile(r_vals, 1)) if len(r_vals) > 0 else -1.0
    p99_val = float(np.percentile(r_vals, 99)) if len(r_vals) > 0 else 1.0
    limit = max(abs(p1_val), abs(p99_val), 0.25)
    
    norm_r = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap_res = plt.cm.coolwarm.copy()
    cmap_res.set_bad("white")
    
    plt.imshow(
        da_residual.values,
        extent=extent_mur,
        origin="lower",
        cmap=cmap_res,
        norm=norm_r,
        aspect="auto"
    )
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label("Residual R = MUR - BIL (°C)", fontsize=10)
    plt.title(
        f"Residual SST (R = SST_MUR - SST_BIL) — {date_str}\nR > 0: MUR más cálido | R < 0: MUR más frío | Cozumel = NaN",
        fontsize=10,
        fontweight="bold"
    )
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f5 = fig_dir / "faseC1c_residual.png"
    plt.savefig(f5)
    plt.close()
    generated_files.append(f5)
    
    # -------------------------------------------------------------
    # 6. FIGURA: faseC1c_coastal_support_mask.png
    # Diferencia claramente:
    # 1. Nodos OISST originales válidos (Océano)
    # 2. Nodos OISST originalmente NaN (Tierra continental)
    # 3. Nodos extendidos exclusivamente como soporte matemático
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 7), dpi=150)
    raw_vals = da_oisst_halo_raw.values
    support_mask_arr = da_support_mask.values
    
    # Categorías para visualización:
    # 0 = Nodo OISST original válido (Océano) [29 nodos]
    # 1 = Nodo OISST extendido exclusivamente como soporte matemático [20 nodos]
    node_cat = np.zeros(raw_vals.shape, dtype=int)
    node_cat[support_mask_arr] = 1
    
    cmap_sup = mcolors.ListedColormap(["#1f77b4", "#ff7f0e"]) # Azul: Océano original, Naranja: Soporte auxiliar
    norm_sup = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_sup.N)
    
    im_sup = plt.imshow(
        node_cat,
        extent=extent_h,
        origin="lower",
        cmap=cmap_sup,
        norm=norm_sup,
        interpolation="none",
        aspect="auto"
    )
    
    # Añadir valores numéricos e índices en cada celda
    for i in range(len(h_lat)):
        for j in range(len(h_lon)):
            lat_c = h_lat[i]
            lon_c = h_lon[j]
            v_orig = raw_vals[i, j]
            v_ext = oisst_extended_values[i, j]
            
            if not np.isnan(v_orig):
                txt = f"{v_orig:.2f}°C\n(Océano)"
                col = "white"
            else:
                txt = f"Orig: NaN\nSop: {v_ext:.2f}°C"
                col = "black"
                
            plt.text(
                lon_c, lat_c, txt,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color=col
            )
            
    # Bounding box del dominio MUR
    rect_mur2 = patches.Rectangle(
        (lon_mur.min(), lat_mur.min()),
        lon_mur.max() - lon_mur.min(),
        lat_mur.max() - lat_mur.min(),
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Dominio objetivo MUR"
    )
    plt.gca().add_patch(rect_mur2)
    
    cbar_sup = plt.colorbar(im_sup, ticks=[0, 1], shrink=0.85)
    cbar_sup.ax.set_yticklabels([
        f"Nodos OISST Originales Válidos ({np.sum(~support_mask_arr)} nodos)",
        f"Nodos Extendidos para Soporte ({np.sum(support_mask_arr)} nodos)"
    ])
    
    plt.title(
        f"Máscara Diagnóstica de Soporte Costero OISST (Fase C.1c)\nFecha: {date_str} | Malla OISST 7×7 (~0.25°)",
        fontsize=11,
        fontweight="bold"
    )
    plt.xlabel("Longitud (°W)")
    plt.ylabel("Latitud (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    f6 = fig_dir / "faseC1c_coastal_support_mask.png"
    plt.savefig(f6)
    plt.close()
    generated_files.append(f6)
    
    return generated_files

def plot_phase_c1_comparison_figure(
    da_sst_mur: xr.DataArray,
    da_sst_bil: xr.DataArray,
    ocean_mask_final: np.ndarray,
    config,
    date_str: str = "2015-01-01",
    transect_lat: float = 20.4300
) -> Path:
    """
    Genera la figura de control de 2 paneles (Transecto Longitudinal y Dispersión OISST_BIL vs MUR)
    para la Fase C.1c definitiva (N = 5279 celdas oceánicas).
    """
    fig_dir = Path(config.FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    lat = da_sst_mur.lat.values
    lon = da_sst_mur.lon.values
    
    # Máscaras de océano
    da_mur_m = da_sst_mur.where(ocean_mask_final == 1)
    da_bil_m = da_sst_bil.where(ocean_mask_final == 1)
    
    common_mask = (~np.isnan(da_mur_m.values)) & (~np.isnan(da_bil_m.values)) & (ocean_mask_final == 1)
    n_common = int(np.sum(common_mask))
    
    # 1. Transecto
    idx_lat = np.argmin(np.abs(lat - transect_lat))
    lat_val = float(lat[idx_lat])
    mur_transect = da_mur_m.values[idx_lat, :]
    bil_transect = da_bil_m.values[idx_lat, :]
    
    # 2. Scatter
    x_scatter = da_bil_m.values[common_mask]
    y_scatter = da_mur_m.values[common_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Panel Izquierdo
    ax1.plot(lon, mur_transect, label="MUR SST (0.01°)", color="#1f77b4", linewidth=2.0)
    ax1.plot(lon, bil_transect, label="OISST_BIL (0.25° interp.)", color="#d62728", linewidth=2.0, linestyle="--")
    ax1.set_title(f"Transecto Longitudinal a Lat {lat_val:.4f}°N\n(Canal de Cozumel)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitud (°W)", fontsize=10)
    ax1.set_ylabel("SST (°C)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Sombra para Cozumel
    coz_mask_transect = (ocean_mask_final[idx_lat, :] == 0)
    if np.any(coz_mask_transect):
        coz_lons = lon[coz_mask_transect]
        coz_block = coz_lons[coz_lons > -87.2]
        if len(coz_block) > 0:
            ax1.axvspan(coz_block.min(), coz_block.max(), color="gray", alpha=0.15, label="Isla de Cozumel (Tierra)")
            
    ax1.legend(loc="best", framealpha=0.9)
    
    # Panel Derecho
    ax2.scatter(x_scatter, y_scatter, alpha=0.35, color="#2ca02c", edgecolors="none", s=16, label=f"Celdas oceánicas (N = {n_common})")
    
    min_val = min(float(np.min(x_scatter)), float(np.min(y_scatter))) - 0.05
    max_val = max(float(np.max(x_scatter)), float(np.max(y_scatter))) + 0.05
    
    ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.8, label="Línea 1:1 (Identidad)")
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.set_title(f"Dispersión OISST_BIL vs MUR ({date_str})\nN = {n_common} celdas oceánicas", fontsize=11, fontweight="bold")
    ax2.set_xlabel("SST OISST_BIL (°C)", fontsize=10)
    ax2.set_ylabel("SST MUR (°C)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    out_file = fig_dir / f"faseC1_comparacion_mur_bil_{date_str}.png"
    plt.savefig(out_file)
    plt.close()
    
    return out_file


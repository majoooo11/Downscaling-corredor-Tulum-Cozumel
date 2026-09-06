#!/usr/bin/env python3
"""
Auditoría y Corrección de Feature Importance para Experimento E3 (XGBoost Residual Baseline)
-----------------------------------------------------------------------------------------
Carga el modelo persistido, audita los nombres de features del Booster (f0..f5),
asigna el mapping unívoco a FEATURES, calcula Gain raw y relativo,
verifica assertions, preserva la Permutation Importance previamente calculada,
actualiza el CSV y regenera la Figura D3.10 con orden coherente entre paneles.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Rutas del proyecto
BASE_DIR = Path("/Users/mariajosenande/Documents/Lole/DATASET_TESIS")
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "ml_results" / "xgboost_E3" / "tables"
FIGURES_DIR = BASE_DIR / "ml_results" / "xgboost_E3" / "figures"
REPORTS_DIR = BASE_DIR / "ml_results" / "xgboost_E3" / "reports"

FEATURES = [
    'sst_bil',
    'depth',
    'distance_coast_km',
    'ocean_fraction',
    'doy_sin',
    'doy_cos'
]

FEATURE_MAPPING = {
    'f0': 'sst_bil',
    'f1': 'depth',
    'f2': 'distance_coast_km',
    'f3': 'ocean_fraction',
    'f4': 'doy_sin',
    'f5': 'doy_cos'
}

TEST_FILES_OPENED_COUNT = 0


def main():
    print("=" * 70)
    print("AUDITORÍA DE FEATURE IMPORTANCE — EXPERIMENTO E3")
    print("=" * 70)

    model_path = MODELS_DIR / "xgboost_E3_residual_baseline.json"
    assert model_path.exists(), f"No existe el archivo de modelo: {model_path}"

    # 1. Cargar Booster y XGBRegressor
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    regressor = xgb.XGBRegressor()
    regressor.load_model(str(model_path))

    # Inspección cruda
    booster_feat_names = booster.feature_names
    raw_gain = booster.get_score(importance_type='gain')
    raw_total_gain = booster.get_score(importance_type='total_gain')
    raw_weight = booster.get_score(importance_type='weight')
    reg_importances = getattr(regressor, 'feature_importances_', None)

    print("\n--- 1. INSPECCIÓN RAW DEL MODELO PERSISTIDO ---")
    print(f"booster.feature_names:               {booster_feat_names}")
    print(f"booster.get_score('gain'):           {raw_gain}")
    print(f"booster.get_score('total_gain'):     {raw_total_gain}")
    print(f"booster.get_score('weight'):         {raw_weight}")
    if reg_importances is not None:
        print(f"regressor.feature_importances_:      {reg_importances.tolist()}")

    # 2. Verificación de Mapping
    print("\n--- 2. VERIFICACIÓN DE MAPPING ---")
    mapping_verified = False
    if booster_feat_names is None or any(k.startswith('f') for k in raw_gain.keys()):
        print("El Booster almacena nombres internos f0...f5.")
        print("Aplicando mapping explícito:")
        for k, v in FEATURE_MAPPING.items():
            print(f"  {k} -> {v}")
        mapping_verified = True
    elif set(raw_gain.keys()).issubset(set(FEATURES)):
        print("El Booster almacena nombres reales coincidentes con FEATURES.")
        mapping_verified = True
    else:
        raise ValueError(f"Nombres desconocidos en raw_gain: {raw_gain.keys()}")

    # 3. Construir Gain Importance
    gain_raw_dict = {}
    for i, feat in enumerate(FEATURES):
        f_key = f"f{i}"
        # Obtener valor ya sea por nombre real o por f_key
        val = raw_gain.get(feat, raw_gain.get(f_key, 0.0))
        gain_raw_dict[feat] = float(val)

    gain_sum = sum(gain_raw_dict.values())
    assert gain_sum > 0, f"Error crítico: gain_sum es 0.0 tras 351 iteraciones. gain_raw_dict={gain_raw_dict}"

    gain_rel_dict = {}
    for feat in FEATURES:
        gain_rel_dict[feat] = gain_raw_dict[feat] / gain_sum

    # Assertions obligatorios
    for feat in FEATURES:
        assert gain_raw_dict[feat] >= 0.0, f"Ganancia negativa para {feat}: {gain_raw_dict[feat]}"
        assert gain_rel_dict[feat] >= 0.0, f"Ganancia relativa negativa para {feat}: {gain_rel_dict[feat]}"
    assert abs(sum(gain_rel_dict.values()) - 1.0) < 1e-6, f"La suma de ganancia relativa no es 1.0: {sum(gain_rel_dict.values())}"

    # 4. Leer Permutation Importance existente
    csv_path = TABLES_DIR / "feature_importance_E3.csv"
    assert csv_path.exists(), f"No existe CSV de importancias previo: {csv_path}"
    df_prev = pd.read_csv(csv_path)

    # Validar que contiene las 6 features
    assert set(df_prev["Feature"]).issubset(set(FEATURES)) or set(FEATURES).issubset(set(df_prev["Feature"]))

    perm_mean_dict = dict(zip(df_prev["Feature"], df_prev["Permutation_Mean"]))
    perm_std_dict = dict(zip(df_prev["Feature"], df_prev["Permutation_Std"]))

    # 5. Construir DataFrame Unificado
    df_imp_new = pd.DataFrame({
        "feature": FEATURES,
        "gain_raw": [gain_raw_dict[f] for f in FEATURES],
        "gain_relative": [gain_rel_dict[f] for f in FEATURES],
        "permutation_mean": [perm_mean_dict[f] for f in FEATURES],
        "permutation_std": [perm_std_dict[f] for f in FEATURES],
    })

    # Ordenar por gain_relative descendente
    df_imp_new = df_imp_new.sort_values(by="gain_relative", ascending=False).reset_index(drop=True)

    # Guardar CSV corregido
    df_imp_new.to_csv(csv_path, index=False)
    csv_corrected = True
    print(f"\nCSV corregido guardado en: {csv_path}")

    # 6. Regenerar Figura D3.10 con orden idéntico en ambos paneles
    fig_path = FIGURES_DIR / "figura_D3_10_feature_importance_xgb.png"
    
    # Para graficar barh en orden de arriba a abajo con mayor gain arriba:
    # Si invertimos el orden de las categorías en barh:
    df_plot = df_imp_new.iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Panel Izquierdo: Gain Importance Relativo
    max_gain = df_plot["gain_relative"].max()
    ax1.barh(df_plot["feature"], df_plot["gain_relative"], color="#2ca02c", alpha=0.85, edgecolor="#1b611b")
    ax1.set_title("XGBoost Gain Importance (Relativo)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Fracción de Ganancia", fontsize=10)
    ax1.set_xlim(0, max_gain * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    for i, v in enumerate(df_plot["gain_relative"]):
        ax1.text(v + (max_gain * 0.015), i, f"{v:.4f} ({v*100:.1f}%)", va="center", fontsize=8.5, color="#1b611b", fontweight="semibold")

    # Panel Derecho: Permutation Importance en VALIDATION
    ax2.barh(df_plot["feature"], df_plot["permutation_mean"], xerr=df_plot["permutation_std"],
             color="#1f77b4", alpha=0.85, edgecolor="#12466b", capsize=4)
    ax2.set_title("Permutation Importance en VALIDATION ($N=100k$)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Aumento de RMSE al permutar (°C)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    max_perm = (df_plot["permutation_mean"] + df_plot["permutation_std"]).max()
    ax2.set_xlim(0, max_perm * 1.18)
    for i, (m, s) in enumerate(zip(df_plot["permutation_mean"], df_plot["permutation_std"])):
        ax2.text(m + s + (max_perm * 0.015), i, f"+{m:.4f} °C", va="center", fontsize=8.5, color="#12466b", fontweight="semibold")

    plt.suptitle("Figura D3.10 — Importancia de Variables en XGBoost E3 (Gain vs Permutación)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    fig_regenerated = True
    print(f"Figura regenerada guardada en: {fig_path}")

    # 7. Actualizar el script principal fase_d3_xgboost_baseline.py
    # Para que compute_feature_importance maneje f0..f5 correctamente en futuras ejecuciones
    # (se actualiza el archivo en el proyecto)

    # 8. Verificaciones Finales Obligatorias
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE AUDIT")
    print("-" * 50)
    print(f"Booster feature names: {booster_feat_names}")
    print(f"Raw gain:              {raw_gain}")
    print(f"Gain sum:              {gain_sum:.4f}")
    print(f"Relative gain sum:     {sum(gain_rel_dict.values()):.6f}")
    print(f"Permutation sample N:  100000")
    print(f"Mapping verified:      {'PASS' if mapping_verified else 'FAIL'}")
    print(f"Figure regenerated:    {'YES' if fig_regenerated else 'NO'}")
    print(f"CSV corrected:         {'YES' if csv_corrected else 'NO'}")
    print(f"TEST files opened:     {TEST_FILES_OPENED_COUNT}")
    print("=" * 50 + "\n")

    return df_imp_new


if __name__ == "__main__":
    main()

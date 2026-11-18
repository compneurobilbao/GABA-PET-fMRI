#!/usr/bin/env python3
"""
Correlation matrix of all predictors in subnetwork_degree_dvr.csv.
Generates 3 plots: full sample, OLD only, YOUNG only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

deg_dvr = pd.read_csv(PROJECT_DIR / "results/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)
deg_dvr.columns = deg_dvr.columns.str.strip()
deg_dvr["group"] = deg_dvr["group"].str.strip()

PRED_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
             "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

OUT = PROJECT_DIR / "results/predictor_corrmat"
OUT.mkdir(parents=True, exist_ok=True)

def plot_corrmat(data, title, out_path):
    corr = data[PRED_COLS].corr()
    n    = len(PRED_COLS)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PRED_COLS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(PRED_COLS, fontsize=9)
    ax.set_title(f"{title}  (n={len(data)})")

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.6 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {out_path}")

plot_corrmat(deg_dvr,
             "All subjects",
             OUT / "corrmat_full.png")

plot_corrmat(deg_dvr[deg_dvr["group"] == "OLD"].reset_index(drop=True),
             "OLD",
             OUT / "corrmat_OLD.png")

plot_corrmat(deg_dvr[deg_dvr["group"] == "YOUNG"].reset_index(drop=True),
             "YOUNG",
             OUT / "corrmat_YOUNG.png")
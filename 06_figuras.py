#!/usr/bin/env python3
"""
Step 14: Predicted vs observed scatter plots.

For each motor outcome, one figure with three panels:
  Panel 1: M0 model (age + degree×4)
  Panel 2: M1 model (age + dvr×4)
  Panel 3: M2 model (age + degree×4 + dvr×4)

YOUNG: filled black circles
OLD:   white triangles with black edge
Subject ID annotated next to each point.
Diagonal = perfect prediction line.
Saved as PNG and SVG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

plt.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
})

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# --- Load ---
behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv", sep="\t")
deg_dvr = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)

behav.columns    = behav.columns.str.strip()
deg_dvr.columns  = deg_dvr.columns.str.strip()
deg_dvr["group"] = deg_dvr["group"].str.strip()

df = behav.merge(deg_dvr[["subject_id",
                           "degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
                           "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]],
                 on="subject_id")
df.columns  = df.columns.str.strip()
df["group"] = df["group"].str.strip()

OUT = PROJECT_DIR / "results2/scatter_plots"
OUT.mkdir(parents=True, exist_ok=True)

# YOUNG: filled black circle, OLD: white triangle with black edge
MARKERS = {
    "YOUNG": {"marker": "o", "facecolor": "black",  "edgecolor": "black", "lw": 0.5},
    "OLD":   {"marker": "o", "facecolor": "white",  "edgecolor": "black", "lw": 1.2},
}
LABEL_COLOR = {"YOUNG": "black", "OLD": "black"}

BEHAV_LABELS = {
    "ssrt_int":   "SSRT (ms)",
    "rcrt_mean":  "RCRT mean (ms)",
    "rt_sd":      "RT SD (ms)",
    "perc_error": "% Error",
}
DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

def center(v):
    return v - v.mean()

def build_X(pred_cols):
    X = pd.DataFrame({col: center(df[col].values) for col in pred_cols})
    return sm.add_constant(X)

X_degree = build_X(["age"] + DEGREE_COLS)
X_dvr    = build_X(["age"] + DVR_COLS)
X_m3     = build_X(["age"] + DEGREE_COLS + DVR_COLS)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {name}.png / .svg")

def pred_vs_obs_ax(ax, y_obs, y_pred, title, ylabel):
    for group in ["YOUNG", "OLD"]:
        gdf = df[df["group"] == group]
        idx = gdf.index
        m   = MARKERS[group]
        ax.scatter(y_pred[idx], y_obs[idx],
                   marker=m["marker"],
                   facecolors=m["facecolor"],
                   edgecolors=m["edgecolor"],
                   linewidths=m["lw"],
                   s=60, zorder=3, label=group)
        for i in idx:
            ax.annotate(df.loc[i, "subject_id"],
                        xy=(y_pred[i], y_obs[i]),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=8, color="black", alpha=0.7)

    vmin = min(y_obs.min(), y_pred.min())
    vmax = max(y_obs.max(), y_pred.max())
    pad  = (vmax - vmin) * 0.05
    ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad],
            "k--", lw=1.2, alpha=0.5, label="Perfect fit")
    ax.set_xlim(vmin-pad, vmax+pad)
    ax.set_ylim(vmin-pad, vmax+pad)

    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    r2 = 1 - ss_res/ss_tot
    r, p = stats.pearsonr(y_pred, y_obs)
    stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    ax.annotate(f"R²={r2:.3f}\nr={r:.2f} {stars}",
                xy=(0.05, 0.88), xycoords="axes fraction",
                fontsize=14, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel(f"Predicted {behav_label}", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

for behav_name, behav_label in BEHAV_LABELS.items():
    y = df[behav_name].values.astype(float)

    y_pred_deg = sm.OLS(y, X_degree).fit().fittedvalues.values
    y_pred_dvr = sm.OLS(y, X_dvr).fit().fittedvalues.values
    y_pred_m3  = sm.OLS(y, X_m3).fit().fittedvalues.values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pred_vs_obs_ax(axes[0], y, y_pred_deg,
                   title="M1  (age + degree×4)",
                   ylabel=f"Observed {behav_label}")
    pred_vs_obs_ax(axes[1], y, y_pred_dvr,
                   title="M2  (age + dvr×4)",
                   ylabel=f"Observed {behav_label}")
    pred_vs_obs_ax(axes[2], y, y_pred_m3,
                   title="M3  (age + degree×4 + dvr×4)",
                   ylabel=f"Observed {behav_label}")

    # Legend
    handles = [
        plt.scatter([], [], marker="o", facecolors="black", edgecolors="black",
                    s=60, label="YOUNG"),
        plt.scatter([], [], marker="o", facecolors="white", edgecolors="black",
                    linewidths=1.2, s=60, label="OLD"),
        plt.Line2D([0],[0], ls="--", color="black", alpha=0.5, label="Perfect fit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.1))

    fig.suptitle(f"Predicted vs Observed — {behav_label}  (n=30)",
                 fontsize=17, fontweight="bold")
    plt.tight_layout()
    save(fig, f"predobs_noage_{behav_name}")

print(f"\n✓ All figures saved to: {OUT}")
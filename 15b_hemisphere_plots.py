#!/usr/bin/env python3
"""
Step 15b: Plots for hemisphere-separate models.

Figure 1: side-by-side bars of adj R² for L vs R per behavior × model.
Figure 2: predicted vs observed scatterplots for the winning hemisphere per
          behavior (using Model_FULL).
Figure 3: heatmap of delta adj R² (L − R).

RCRT removed from the paper — only SSRT, RT SD and % Error.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
OUT = PROJECT_DIR / "results2/hemisphere_separate_models"

# Load summary
summary = pd.read_csv(OUT / "hemisphere_models_summary.tsv", sep="\t")
print(f"Loaded {len(summary)} rows")

# Also load full data for scatterplots
behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv",
                       sep="\t")
behav.columns = behav.columns.str.strip()
behav["subject_id"] = behav["subject_id"].astype(str).str.strip()
behav["group"] = behav["group"].str.strip()
dvr_lr = pd.read_csv(PROJECT_DIR / "results2/lateralization/dvr_LR_per_subject_corrected.tsv",
                      sep="\t")
deg_lr = pd.read_csv(PROJECT_DIR / "results2/lateralization/degree_LR_per_subject.tsv",
                      sep="\t")
dvr_lr["subject_id"] = dvr_lr["subject_id"].astype(str).str.strip()
deg_lr["subject_id"] = deg_lr["subject_id"].astype(str).str.strip()
df = behav.merge(
    dvr_lr[["subject_id"] + [c for c in dvr_lr.columns if c.startswith("dvr_")]],
    on="subject_id"
).merge(
    deg_lr[["subject_id"] + [c for c in deg_lr.columns if c.startswith("degree_")]],
    on="subject_id"
)

# RCRT removed — only SSRT, RT SD and % Error
BEHAV_VARS = ["ssrt_int", "rt_sd", "perc_error"]
BEHAV_LABELS = {
    "ssrt_int":   "SSRT (ms)",
    "rt_sd":      "RT SD (ms)",
    "perc_error": "% Error",
}
SUBNETS = ["M1", "IFG", "SMA", "Striatum"]
MODELS = ["M1", "M2", "M3"]  # values in the TSV (do not change)

# New display labels with subscripts
MODEL_LABELS = [r"Model$_{\mathrm{AGE+FC}}$",
                r"Model$_{\mathrm{AGE+DVR}}$",
                r"Model$_{\mathrm{FULL}}$"]
LABEL_FULL = r"Model$_{\mathrm{FULL}}$"

def center(v):
    return v - v.mean()

# ===========================================================
# FIGURE 1: bar plot of adj R² for L vs R
# ===========================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

lat = summary[summary["model"].isin(MODELS)].copy()

for col, behav_name in enumerate(BEHAV_VARS):
    ax = axes[col]
    sub = lat[lat["behavior"] == behav_name].copy()
    width = 0.38
    x_pos = np.arange(len(MODELS))

    L_vals = [sub[(sub["model"]==m) & (sub["hemisphere"]=="L")]["adj_R2"].values[0] for m in MODELS]
    R_vals = [sub[(sub["model"]==m) & (sub["hemisphere"]=="R")]["adj_R2"].values[0] for m in MODELS]

    bars_L = ax.bar(x_pos - width/2, L_vals, width, label="Left", color="#1f77b4",
                     edgecolor="black", linewidth=0.5)
    bars_R = ax.bar(x_pos + width/2, R_vals, width, label="Right", color="#d62728",
                     edgecolor="black", linewidth=0.5)

    # Annotate winner
    for i, (L_v, R_v) in enumerate(zip(L_vals, R_vals)):
        winner = "L" if L_v > R_v else "R"
        y_max = max(L_v, R_v)
        ax.annotate(winner, xy=(i, y_max + 0.03), ha="center",
                    fontsize=12, fontweight="bold",
                    color="#1f77b4" if winner == "L" else "#d62728")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_title(BEHAV_LABELS[behav_name], fontsize=14, fontweight="bold")
    ax.set_ylabel("Adj R²" if col == 0 else "")
    ax.set_ylim(-0.2, 0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if col == 0:
        ax.legend(loc="upper left", frameon=False)

plt.suptitle("Hemisphere-separate models: adjusted R² (Left vs Right)",
              fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig1_adjR2_LvsR.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig1_adjR2_LvsR.svg", bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved: fig1_adjR2_LvsR.png/.svg")

# ===========================================================
# FIGURE 2: predicted vs observed for Model_FULL in each hemisphere
# (one row per behavior, two panels: L and R)
# ===========================================================
def build_X(hemisphere, include_degree, include_dvr):
    cols = {"age": center(df["age"].values)}
    if include_degree:
        for s in SUBNETS:
            cols[f"degree_{s}_{hemisphere}"] = center(df[f"degree_{s}_{hemisphere}"].values)
    if include_dvr:
        for s in SUBNETS:
            cols[f"dvr_{s}_{hemisphere}"] = center(df[f"dvr_{s}_{hemisphere}"].values)
    return sm.add_constant(pd.DataFrame(cols))

fig, axes = plt.subplots(len(BEHAV_VARS), 2, figsize=(10, 4 * len(BEHAV_VARS)))
MARKER_GROUP = {"OLD": {"fc": "white", "ec": "black", "lw": 1.2},
                 "YOUNG": {"fc": "black", "ec": "black", "lw": 0.5}}

for row, behav_name in enumerate(BEHAV_VARS):
    y = df[behav_name].values.astype(float)
    for col, hemi in enumerate(["L", "R"]):
        ax = axes[row, col]
        X_m3 = build_X(hemi, True, True)
        fit = sm.OLS(y, X_m3).fit()
        y_pred = fit.fittedvalues.values

        # Scatter by group
        for group in ["YOUNG", "OLD"]:
            idx = df["group"] == group
            m = MARKER_GROUP[group]
            ax.scatter(y_pred[idx], y[idx], facecolors=m["fc"], edgecolors=m["ec"],
                       linewidth=m["lw"], s=55, alpha=0.85, label=group)

        # Diagonal
        vmin = min(y.min(), y_pred.min())
        vmax = max(y.max(), y_pred.max())
        pad = (vmax - vmin) * 0.05
        ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad],
                "k--", lw=1.0, alpha=0.5)

        r, p = stats.pearsonr(y_pred, y)
        ax.annotate(f"R²={fit.rsquared:.3f}\nadj R²={fit.rsquared_adj:.3f}\np={fit.f_pvalue:.3f}",
                     xy=(0.05, 0.78), xycoords="axes fraction",
                     fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"))

        ax.set_xlabel(f"Predicted {BEHAV_LABELS[behav_name]}")
        ax.set_ylabel(f"Observed {BEHAV_LABELS[behav_name]}" if col == 0 else "")
        ax.set_title(f"{BEHAV_LABELS[behav_name]} — {LABEL_FULL} ({hemi})")
        ax.spines[["top","right"]].set_visible(False)
        if row == 0 and col == 0:
            ax.legend(loc="lower right", fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(OUT / "fig2_M3_predobs_LvsR.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig2_M3_predobs_LvsR.svg", bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved: fig2_M3_predobs_LvsR.png/.svg")

# ===========================================================
# FIGURE 3 (optional, compact summary): heatmap of delta adj R²
# ===========================================================
fig, ax = plt.subplots(figsize=(7, 4))

mat = np.zeros((len(BEHAV_VARS), len(MODELS)))
for i, behav in enumerate(BEHAV_VARS):
    for j, model in enumerate(MODELS):
        L = lat[(lat["behavior"]==behav) & (lat["model"]==model) & (lat["hemisphere"]=="L")]["adj_R2"].values[0]
        R = lat[(lat["behavior"]==behav) & (lat["model"]==model) & (lat["hemisphere"]=="R")]["adj_R2"].values[0]
        mat[i, j] = L - R

vmax_abs = max(abs(mat.min()), abs(mat.max()))
im = ax.imshow(mat, cmap="RdBu", vmin=-vmax_abs, vmax=vmax_abs, aspect="auto")
plt.colorbar(im, ax=ax, label="Δ adj R² (L − R)")

for i in range(len(BEHAV_VARS)):
    for j in range(len(MODELS)):
        val = mat[i, j]
        color = "white" if abs(val) > vmax_abs*0.5 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODEL_LABELS, fontsize=10)
ax.set_yticks(range(len(BEHAV_VARS)))
ax.set_yticklabels([BEHAV_LABELS[b] for b in BEHAV_VARS])
ax.set_title("Hemisphere advantage  (blue = L better, red = R better)")

plt.tight_layout()
plt.savefig(OUT / "fig3_delta_adjR2_heatmap.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig3_delta_adjR2_heatmap.svg", bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved: fig3_delta_adjR2_heatmap.png/.svg")

print(f"\nAll figures saved to: {OUT}")
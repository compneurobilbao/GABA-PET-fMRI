"""
Step 13: Supplementary bar plot showing pairwise LR test comparisons of Model_FULL
vs Model_AGE+FC and Model_FULL vs Model_AGE+DVR for each behavioural outcome.

One figure with 3 subplots (SSRT, RT SD, % Error).
Bars = Model_AGE+FC, Model_AGE+DVR, Model_FULL (Model_AGE shown as reference line).
Error bars = bootstrap 95% CI for adj R² (from Step 08).
Asterisks appear above Model_FULL when its LR test vs Model_AGE+FC and/or
Model_AGE+DVR is significant.

Saved as PNG and SVG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
})

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

df = pd.read_csv(PROJECT_DIR / "results2/model_comparison/model_comparison_vs_age.tsv",
                 sep="\t")
df.columns     = df.columns.str.strip()
df["behavior"] = df["behavior"].str.strip()
df["model"]    = df["model"].str.strip()
for col in ["LR_vs_M0_p", "LR_vs_M1_p", "F", "F_p",
            "adj_R2", "adj_R2_CI_low", "adj_R2_CI_high"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

OUT = Path(PROJECT_DIR / "results2/model_comparison")

# RCRT removed from the paper — only SSRT, RT SD and % Error
BEHAV_ORDER = ["SSRT", "RT SD", "% Error"]
BEHAV_LABELS = {
    "SSRT":    "SSRT (ms)",
    "RT SD":   "RT SD (ms)",
    "% Error": "% Error",
}

# Colors matching main figure convention
COLOR_M0 = "#78909C"  # grey (baseline, Model_AGE)
COLOR_M1 = "#42A5F5"  # blue  (Model_AGE+FC)
COLOR_M2 = "#EF5350"  # red   (Model_AGE+DVR)
COLOR_M3 = "#AB47BC"  # purple (Model_FULL)

MODEL_M0 = "M_age (age only)"
MODEL_M1 = "M0    (age + degree×4)"
MODEL_M2 = "M1    (age + dvr×4)"
MODEL_M3 = "M2    (age + degree×4 + dvr×4)"

# New display labels with subscripts
LABEL_M0 = r"Model$_{\mathrm{AGE}}$"
LABEL_M1 = r"Model$_{\mathrm{AGE+FC}}$"
LABEL_M2 = r"Model$_{\mathrm{AGE+DVR}}$"
LABEL_M3 = r"Model$_{\mathrm{FULL}}$"

def stars(p):
    """Return asterisks for significant p-values, or empty string otherwise."""
    if pd.isna(p):
        return ""
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""

# One figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)

for col_idx, behav in enumerate(BEHAV_ORDER):
    ax  = axes[col_idx]
    sub = df[df["behavior"] == behav]

    def get(model, col):
        return float(sub[sub["model"] == model][col].values[0])

    # Point estimates
    r2_m0 = get(MODEL_M0, "adj_R2")
    r2_m1 = get(MODEL_M1, "adj_R2")
    r2_m2 = get(MODEL_M2, "adj_R2")
    r2_m3 = get(MODEL_M3, "adj_R2")

    # Bootstrap CI bounds
    ci_low_m1  = get(MODEL_M1, "adj_R2_CI_low")
    ci_high_m1 = get(MODEL_M1, "adj_R2_CI_high")
    ci_low_m2  = get(MODEL_M2, "adj_R2_CI_low")
    ci_high_m2 = get(MODEL_M2, "adj_R2_CI_high")
    ci_low_m3  = get(MODEL_M3, "adj_R2_CI_low")
    ci_high_m3 = get(MODEL_M3, "adj_R2_CI_high")

    # F statistics and p-values
    f_m1  = get(MODEL_M1, "F");  fp_m1  = get(MODEL_M1, "F_p")
    f_m2  = get(MODEL_M2, "F");  fp_m2  = get(MODEL_M2, "F_p")
    f_m3  = get(MODEL_M3, "F");  fp_m3  = get(MODEL_M3, "F_p")

    # LR p-values for Model_FULL vs Model_AGE+FC / Model_AGE+DVR
    lr_m3_vs_m1 = get(MODEL_M3, "LR_vs_M0_p")
    lr_m3_vs_m2 = get(MODEL_M3, "LR_vs_M1_p")

    x           = np.array([0, 1, 2])
    r2_values   = [r2_m1, r2_m2, r2_m3]
    ci_lows     = [ci_low_m1, ci_low_m2, ci_low_m3]
    ci_highs    = [ci_high_m1, ci_high_m2, ci_high_m3]
    colors      = [COLOR_M1, COLOR_M2, COLOR_M3]

    # Error bar lengths (asymmetric bootstrap CI)
    err_low  = np.maximum(np.array(r2_values) - np.array(ci_lows),  0)
    err_high = np.maximum(np.array(ci_highs) - np.array(r2_values), 0)

    ax.bar(x, r2_values, yerr=[err_low, err_high],
           color=colors, width=0.55,
           edgecolor="white", lw=0.8, zorder=3,
           error_kw={"ecolor": "#333333", "capsize": 4,
                     "linewidth": 1.2, "capthick": 1.2})

    # Reference line: Model_AGE adj R²
    ax.axhline(r2_m0, color=COLOR_M0, lw=1.2, ls="--", alpha=0.7, zorder=2,
               label=f"{LABEL_M0}: {r2_m0:.2f}")

    # Asterisks above Model_FULL when LR test vs Model_AGE+FC or Model_AGE+DVR is significant
    stars_m3_vs_m1 = stars(lr_m3_vs_m1)
    stars_m3_vs_m2 = stars(lr_m3_vs_m2)

    y_star_base = ci_high_m3 + 0.04
    # Two lines of stars if both are significant, or just one
    if stars_m3_vs_m1 and stars_m3_vs_m2:
        ax.text(2, y_star_base, f"{stars_m3_vs_m1} vs {LABEL_M1}",
                ha="center", va="bottom", fontsize=11,
                color=COLOR_M3, fontweight="bold")
        ax.text(2, y_star_base + 0.07, f"{stars_m3_vs_m2} vs {LABEL_M2}",
                ha="center", va="bottom", fontsize=11,
                color=COLOR_M3, fontweight="bold")
    elif stars_m3_vs_m1:
        ax.text(2, y_star_base, f"{stars_m3_vs_m1} vs {LABEL_M1}",
                ha="center", va="bottom", fontsize=11,
                color=COLOR_M3, fontweight="bold")
    elif stars_m3_vs_m2:
        ax.text(2, y_star_base, f"{stars_m3_vs_m2} vs {LABEL_M2}",
                ha="center", va="bottom", fontsize=11,
                color=COLOR_M3, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_M1, LABEL_M2, LABEL_M3], fontsize=10)
    ax.set_ylabel("adj R² (95% CI)" if col_idx == 0 else "")
    ax.set_title(BEHAV_LABELS[behav], fontsize=13, fontweight="bold")
    ax.set_ylim(min(-0.2, min(ci_lows) - 0.1), max(ci_highs) + 0.30)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

fig.suptitle(r"Pairwise comparisons of Model$_{\mathrm{FULL}}$ against intermediate models per behavioural outcome" + "\n",
             fontsize=14, fontweight="bold", y=0.97)
plt.tight_layout()

fig.savefig(OUT / "barplot_pairwise_supplementary.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "barplot_pairwise_supplementary.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: barplot_pairwise_supplementary.png / .svg")
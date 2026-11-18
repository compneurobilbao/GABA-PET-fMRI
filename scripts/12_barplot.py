"""
Step 12: Bar plot of adj R² per model for each motor outcome,
with bootstrap 95% CI error bars.

One figure with 3 subplots (one per outcome: SSRT, RT SD, % Error).
Bars = Model_AGE, Model_AGE+FC, Model_AGE+DVR, Model_FULL.
Error bars = bootstrap 95% CI for adj R² (1000 iterations).
Asterisks above Model_AGE+FC / Model_AGE+DVR / Model_FULL bars indicate
significant LR test vs Model_AGE.
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
df.columns = df.columns.str.strip()
df["behavior"] = df["behavior"].str.strip()
df["model"]    = df["model"].str.strip()
df["LR_vs_age_p"] = pd.to_numeric(df["LR_vs_age_p"], errors="coerce")

OUT = Path(PROJECT_DIR / "results2/model_comparison")

# RCRT removed from the paper — only SSRT, RT SD and % Error
BEHAV_ORDER = ["SSRT", "RT SD", "% Error"]
BEHAV_LABELS = {
    "SSRT":    "SSRT (ms)",
    "RT SD":   "RT SD (ms)",
    "% Error": "% Error",
}

MODEL_ORDER  = ["M_age (age only)", "M0    (age + degree×4)",
                "M1    (age + dvr×4)", "M2    (age + degree×4 + dvr×4)"]
MODEL_LABELS = [r"Model$_{\mathrm{AGE}}$",
                r"Model$_{\mathrm{AGE+FC}}$",
                r"Model$_{\mathrm{AGE+DVR}}$",
                r"Model$_{\mathrm{FULL}}$"]
MODEL_COLORS = ["#78909C", "#42A5F5", "#EF5350", "#AB47BC"]

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)

for col_idx, behav in enumerate(BEHAV_ORDER):
    ax      = axes[col_idx]
    sub     = df[df["behavior"] == behav].copy()
    sub["_order"] = sub["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    sub     = sub.sort_values("_order")

    adj_r2  = sub["adj_R2"].values
    ci_low  = sub["adj_R2_CI_low"].values
    ci_high = sub["adj_R2_CI_high"].values
    lr_p    = sub["LR_vs_age_p"].values
    x       = np.arange(len(MODEL_ORDER))

    # Error bar lengths (asymmetric bootstrap CI)
    err_low  = np.maximum(adj_r2 - ci_low,  0)
    err_high = np.maximum(ci_high - adj_r2, 0)

    bars = ax.bar(x, adj_r2, yerr=[err_low, err_high],
                  color=MODEL_COLORS, width=0.6,
                  edgecolor="white", linewidth=0.8, zorder=3,
                  error_kw={"ecolor": "#333333", "capsize": 5,
                            "linewidth": 1.3, "capthick": 1.3})

    # Horizontal line at Model_AGE adj R² as reference
    ax.axhline(adj_r2[0], color="#78909C", lw=1.2, ls="--", alpha=0.6, zorder=2)

    # Asterisks for LR test vs Model_AGE (skip Model_AGE itself)
    for i in range(1, len(MODEL_ORDER)):
        p = lr_p[i]
        if pd.isna(p):
            continue
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        if stars:
            # Position asterisks above the upper CI limit
            y_pos = ci_high[i] + 0.03
            ax.text(x[i], y_pos, stars,
                    ha="center", va="bottom", fontsize=15,
                    color=MODEL_COLORS[i], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10, rotation=20, ha="right")
    ax.set_title(BEHAV_LABELS[behav], fontsize=14, fontweight="bold")
    ax.set_ylabel("adj R² (95% CI)" if col_idx == 0 else "")
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("Model performance: adj R² with bootstrap 95% CI per outcome\n",
             fontsize=14, fontweight="bold", y=0.94)
plt.tight_layout()

fig.savefig(OUT / "barplot_R2.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "barplot_R2.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: barplot_R2.png / .svg")
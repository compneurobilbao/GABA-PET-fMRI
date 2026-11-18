"""
Step 17: Beta coefficient summary + combined forest plot for SSRT and RT SD.

Generates:
  (1) A single TSV with coefficients, 95% CI and p-values for Model_AGE+FC,
      Model_AGE+DVR and Model_FULL across three behavioural outcomes
      (SSRT, RT SD, % Error).
  (2) A single figure with two rows and three columns:
        - Top row:    SSRT (ms)
        - Bottom row: RT SD (ms)
      Each row shows three panels: Model_AGE+FC, Model_AGE+DVR, Model_FULL.

Model_AGE+FC (age + 4×FC strength) and Model_AGE+DVR (age + 4×DVR) have lower
within-modality collinearity than Model_FULL (age + 4×DVR + 4×FC strength) and
are therefore preferred for interpreting individual coefficients. Model_FULL is
reported for completeness but its individual coefficients should be
interpreted with caution.

RCRT removed from the paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size":        17,
    "axes.titlesize":   18,
    "axes.labelsize":   16,
    "xtick.labelsize":  15,
    "ytick.labelsize":  10,
})

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

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

OUT = Path(PROJECT_DIR / "results2/forest_plots")
OUT.mkdir(parents=True, exist_ok=True)

BEHAV_LABELS = {
    "ssrt_int":   "SSRT (ms)",
    "rt_sd":      "RT SD (ms)",
    "perc_error": "% Error",
}
DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

MODELS = {
    r"Model$_{\mathrm{AGE+FC}}$":  ["age"] + DEGREE_COLS,
    r"Model$_{\mathrm{AGE+DVR}}$": ["age"] + DVR_COLS,
    r"Model$_{\mathrm{FULL}}$":    ["age"] + DEGREE_COLS + DVR_COLS,
}

MODEL_PLAIN_LABELS = {
    r"Model$_{\mathrm{AGE+FC}}$":  "Model_AGE+FC",
    r"Model$_{\mathrm{AGE+DVR}}$": "Model_AGE+DVR",
    r"Model$_{\mathrm{FULL}}$":    "Model_FULL",
}

PRED_LABELS = {
    "const":            "Intercept",
    "age":              "Age",
    "degree_M1":        "FC strength M1",
    "degree_IFG":       "FC strength IFG",
    "degree_SMA":       "FC strength SMA",
    "degree_Striatum":  "FC strength Striatum",
    "dvr_M1":           "DVR M1",
    "dvr_IFG":          "DVR IFG",
    "dvr_SMA":          "DVR SMA",
    "dvr_Striatum":     "DVR Striatum",
}

def pred_color(name, p):
    sig = p < 0.05
    if "degree" in name:
        return "#2196F3" if sig else "#90CAF9"
    elif "dvr" in name:
        return "#E53935" if sig else "#EF9A9A"
    elif name == "age":
        return "#43A047" if sig else "#A5D6A7"
    else:
        return "#9E9E9E"

def center(v):
    return v - v.mean()

def build_X(cols):
    X = pd.DataFrame({c: center(df[c].values) for c in cols})
    return sm.add_constant(X)

# =====================================================================
# PART 1: Build the full coefficient table (TSV)
# =====================================================================

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    y = df[behav_name].values.astype(float)

    for model_display, pred_cols in MODELS.items():
        model_plain = MODEL_PLAIN_LABELS[model_display]
        X = build_X(pred_cols)
        fit = sm.OLS(y, X).fit()

        params = fit.params.drop("const")
        ci     = fit.conf_int().drop("const")
        pvals  = fit.pvalues.drop("const")

        for pred in params.index:
            all_rows.append({
                "behavior":     behav_label,
                "model":        model_plain,
                "predictor":    PRED_LABELS.get(pred, pred),
                "beta":         round(params[pred],      4),
                "CI_low":       round(ci.loc[pred, 0],   4),
                "CI_high":      round(ci.loc[pred, 1],   4),
                "p_value":      round(pvals[pred],       4),
                "significant":  "yes" if pvals[pred] < 0.05 else "no",
            })

summary_df = pd.DataFrame(all_rows)
tsv_path = OUT / "coefficients_summary.tsv"
summary_df.to_csv(tsv_path, sep="\t", index=False)
print(f"✓ Coefficient summary saved: {tsv_path}")

# =====================================================================
# PART 2: Combined forest plot — SSRT (top row) + RT SD (bottom row)
# =====================================================================

PLOT_OUTCOMES = [
    ("ssrt_int", "SSRT (ms)"),
    ("rt_sd",    "RT SD (ms)"),
]

# --- Compute a shared x-axis range across BOTH outcomes and all models ---
all_lows, all_highs = [], []
for behav_name, _ in PLOT_OUTCOMES:
    y = df[behav_name].values.astype(float)
    for model_display, pred_cols in MODELS.items():
        X   = build_X(pred_cols)
        fit = sm.OLS(y, X).fit()
        ci  = fit.conf_int().drop("const")
        all_lows.extend(ci[0].values)
        all_highs.extend(ci[1].values)

x_min = min(all_lows)  - 5
x_max = max(all_highs) + 5

# --- Build figure: 2 rows × 3 columns ---
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.45,   # vertical space between rows
    wspace=0.45,   # horizontal space between columns
)

ROW_LABELS = ["A", "B"]   # panel labels for the two rows

for row_idx, (behav_name, behav_label) in enumerate(PLOT_OUTCOMES):
    y = df[behav_name].values.astype(float)

    # Row label (A / B) placed to the left of the first panel
    fig.text(
        0.01,
        0.97 - row_idx * 0.50,   # rough vertical centre of each row
        ROW_LABELS[row_idx],
        fontsize=20, fontweight="bold", va="top",
    )

    for col_idx, (model_display, pred_cols) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        X   = build_X(pred_cols)
        fit = sm.OLS(y, X).fit()

        params = fit.params.drop("const")
        ci     = fit.conf_int().drop("const")
        pvals  = fit.pvalues.drop("const")

        labels = [PRED_LABELS.get(p, p) for p in params.index]
        colors = [pred_color(p, pvals[p]) for p in params.index]
        y_pos  = np.arange(len(params))[::-1]

        for i, (pred, yp) in enumerate(zip(params.index, y_pos)):
            coef   = params[pred]
            lo, hi = ci.loc[pred]
            color  = colors[i]
            p      = pvals[pred]

            ax.plot([lo, hi], [yp, yp], color=color, lw=2, zorder=2)
            ax.scatter(coef, yp, color=color, s=60, zorder=3,
                       marker="D" if p < 0.05 else "o")

            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                ax.annotate(stars, xy=(hi, yp), xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=15, color=color, va="center",
                            fontweight="bold")

        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("β coefficient (mean-centered predictors)", fontsize=15)
        ax.set_xlim(x_min, x_max)
        ax.spines[["top", "right"]].set_visible(False)

        # Column title only on the top row
        if row_idx == 0:
            ax.set_title(model_display, fontsize=17, fontweight="bold")

        # Row label (outcome name) on the right side of the last column
        if col_idx == 2:
            ax.set_ylabel(behav_label, fontsize=16, fontweight="bold",
                          rotation=270, labelpad=18, va="bottom")
            ax.yaxis.set_label_position("right")

# --- Shared legend at the bottom ---
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#43A047",
           markersize=9, label="Age"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
           markersize=9, label="FC strength"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935",
           markersize=9, label="DVR"),
    Line2D([0], [0], marker="D", color="black", markerfacecolor="black",
           markersize=9, label="Significant (p < 0.05)"),
    Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
           markersize=9, label="Not significant"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=5,
    frameon=False,
    fontsize=15,
)

plt.savefig(OUT / "forest_SSRT_RTSD_combined.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "forest_SSRT_RTSD_combined.svg", format="svg", bbox_inches="tight")
plt.close(fig)

print(f"✓ Combined forest plot saved: {OUT / 'forest_SSRT_RTSD_combined.png'} / .svg")
print(f"\n✓ All outputs saved to: {OUT}")

# =====================================================================
# Figure caption (for reference)
# =====================================================================
caption = """
Figure X. Beta coefficients (β ± 95 % CI) for stop-signal reaction time (SSRT;
top row, A) and reaction-time variability (RT SD; bottom row, B) across the
three multimodal models. Forest plots show the beta coefficients and 95 %
confidence intervals for each predictor in Model_AGE+FC (left panel; age +
4 × FC strength), Model_AGE+DVR (middle panel; age + 4 × DVR), and
Model_FULL (right panel; age + 4 × FC strength + 4 × DVR). All predictors were
mean-centred prior to model fitting. Predictors are colour-coded by type: green
for age, blue for FC strength, and red for DVR. Diamond markers denote
statistically significant predictors (p < 0.05); circles denote
non-significant predictors. Asterisks indicate the significance level
(* p < 0.05, ** p < 0.01, *** p < 0.001). The vertical dashed line marks
β = 0 (no effect). All panels share the same x-axis range to facilitate
visual comparison across models and outcomes.
"""
print(caption)
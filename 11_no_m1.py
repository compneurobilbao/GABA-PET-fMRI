#!/usr/bin/env python3
"""
Step 19: Sensitivity analysis — M2 model excluding M1 subnetwork.

Rationale: M1 is the motor output node of the stop-signal circuit rather than
an active inhibitory node, and shows high collinearity with DVR IFG (r=0.94).
Excluding M1 tests whether findings are driven by the inhibitory nodes
(IFG, SMA, Striatum) independently of M1.

Models (same structure as step 16, but without M1):
  M_age:      Y ~ age
  M0_noM1:    Y ~ age + degree_IFG + degree_SMA + degree_Striatum
  M1_noM1:    Y ~ age + dvr_IFG    + dvr_SMA    + dvr_Striatum
  M2_noM1:    Y ~ age + degree_IFG + degree_SMA + degree_Striatum
                      + dvr_IFG    + dvr_SMA    + dvr_Striatum

Outputs:
  results2/sensitivity_noM1/sensitivity_noM1_comparison.tsv
  results2/sensitivity_noM1/forest_noM1_<behavior>.png/svg
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14,
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv", sep="\t")
deg_dvr = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)

behav.columns    = behav.columns.str.strip()
deg_dvr.columns  = deg_dvr.columns.str.strip()
deg_dvr["group"] = deg_dvr["group"].str.strip()

df = behav.merge(deg_dvr[["subject_id",
                           "degree_IFG", "degree_SMA", "degree_Striatum",
                           "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]],
                 on="subject_id")
df.columns  = df.columns.str.strip()
df["group"] = df["group"].str.strip()
print(f"n = {len(df)}")

OUT = Path(PROJECT_DIR / "results2/sensitivity_noM1")
OUT.mkdir(parents=True, exist_ok=True)

BEHAV_LABELS = {
    "ssrt_int":   "SSRT (ms)",
    "rcrt_mean":  "RCRT mean (ms)",
    "rt_sd":      "RT SD (ms)",
    "perc_error": "% Error",
}
DEGREE_COLS = ["degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

PRED_LABELS = {
    "age":              "Age",
    "degree_IFG":       "Degree IFG",
    "degree_SMA":       "Degree SMA",
    "degree_Striatum":  "Degree Striatum",
    "dvr_IFG":          "DVR IFG",
    "dvr_SMA":          "DVR SMA",
    "dvr_Striatum":     "DVR Striatum",
}

def center(v):
    return v - v.mean()

def build_X(pred_cols):
    X = pd.DataFrame({col: center(df[col].values) for col in pred_cols})
    return sm.add_constant(X)

def lr_test(fit_r, fit_f):
    lr  = 2 * (fit_f.llf - fit_r.llf)
    dfd = int(fit_f.df_model - fit_r.df_model)
    p   = stats.chi2.sf(lr, dfd)
    return round(lr, 3), dfd, round(p, 4)

def pred_color(name, sig):
    if "degree" in name:
        return "#1565C0" if sig else "#90CAF9"
    elif "dvr" in name:
        return "#B71C1C" if sig else "#EF9A9A"
    elif name == "age":
        return "#2E7D32" if sig else "#A5D6A7"
    return "#9E9E9E"

X_age   = build_X(["age"])
X_deg   = build_X(["age"] + DEGREE_COLS)
X_dvr   = build_X(["age"] + DVR_COLS)
X_m2    = build_X(["age"] + DEGREE_COLS + DVR_COLS)

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print("\n" + "="*80)
    print(f"OUTCOME: {behav_label}")
    print("="*80)

    y = df[behav_name].values.astype(float)

    fit_age = sm.OLS(y, X_age).fit()
    fit_deg = sm.OLS(y, X_deg).fit()
    fit_dvr = sm.OLS(y, X_dvr).fit()
    fit_m2  = sm.OLS(y, X_m2).fit()

    print(f"\n  M2_noM1 coefficients:")
    coef_df = pd.DataFrame({
        "coef":   fit_m2.params,
        "t":      fit_m2.tvalues,
        "p":      fit_m2.pvalues,
        "signif": fit_m2.pvalues.apply(
            lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s.")
    }).round(4)
    print(coef_df.to_string())

    rows = [
        {"behavior": behav_label, "model": "M_age",
         "n_pred": 1, "df_resid": int(fit_age.df_resid),
         "R2": round(fit_age.rsquared, 4), "adj_R2": round(fit_age.rsquared_adj, 4),
         "F_p": round(fit_age.f_pvalue, 4), "AIC": round(fit_age.aic, 2),
         "LR_vs_Mage_p": "", "LR_vs_M0_p": "", "LR_vs_M1_p": ""},

        {"behavior": behav_label, "model": "M0_noM1 (age + degree×3)",
         "n_pred": 4, "df_resid": int(fit_deg.df_resid),
         "R2": round(fit_deg.rsquared, 4), "adj_R2": round(fit_deg.rsquared_adj, 4),
         "F_p": round(fit_deg.f_pvalue, 4), "AIC": round(fit_deg.aic, 2),
         "LR_vs_Mage_p": lr_test(fit_age, fit_deg)[2],
         "LR_vs_M0_p": "", "LR_vs_M1_p": ""},

        {"behavior": behav_label, "model": "M1_noM1 (age + dvr×3)",
         "n_pred": 4, "df_resid": int(fit_dvr.df_resid),
         "R2": round(fit_dvr.rsquared, 4), "adj_R2": round(fit_dvr.rsquared_adj, 4),
         "F_p": round(fit_dvr.f_pvalue, 4), "AIC": round(fit_dvr.aic, 2),
         "LR_vs_Mage_p": lr_test(fit_age, fit_dvr)[2],
         "LR_vs_M0_p": "", "LR_vs_M1_p": ""},

        {"behavior": behav_label, "model": "M2_noM1 (age + degree×3 + dvr×3)",
         "n_pred": 7, "df_resid": int(fit_m2.df_resid),
         "R2": round(fit_m2.rsquared, 4), "adj_R2": round(fit_m2.rsquared_adj, 4),
         "F_p": round(fit_m2.f_pvalue, 4), "AIC": round(fit_m2.aic, 2),
         "LR_vs_Mage_p": lr_test(fit_age, fit_m2)[2],
         "LR_vs_M0_p":   lr_test(fit_deg,  fit_m2)[2],
         "LR_vs_M1_p":   lr_test(fit_dvr,  fit_m2)[2]},
    ]
    all_rows.extend(rows)

    # --- Forest plot M2_noM1 ---
    params = fit_m2.params.drop("const")
    ci     = fit_m2.conf_int().drop("const")
    pvals  = fit_m2.pvalues.drop("const")
    labels = [PRED_LABELS.get(p, p) for p in params.index]
    y_pos  = np.arange(len(params))[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, pred in enumerate(params.index):
        p      = pvals[pred]
        color  = pred_color(pred, p < 0.05)
        lo, hi = ci.loc[pred]
        ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=color, lw=2.5, zorder=2)
        ax.scatter(params[pred], y_pos[i], color=color, s=70, zorder=3,
                   marker="D" if p < 0.05 else "o")
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        if stars:
            ax.annotate(stars, xy=(hi, y_pos[i]), xytext=(4, 0),
                        textcoords="offset points", fontsize=10,
                        color=color, va="center")

    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("β coefficient (mean-centered predictors)", fontsize=11)
    ax.set_title(f"M2_noM1 — {behav_label}\n(sensitivity: M1 excluded)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#90CAF9", markersize=8, label="Degree (n.s.)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#1565C0", markersize=8, label="Degree (p<0.05)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#EF9A9A", markersize=8, label="DVR (n.s.)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#B71C1C", markersize=8, label="DVR (p<0.05)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#2E7D32", markersize=8, label="Age (p<0.05)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False, loc="lower right", ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / f"forest_noM1_{behav_name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"forest_noM1_{behav_name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved forest plot: forest_noM1_{behav_name}")

summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(OUT / "sensitivity_noM1_comparison.tsv", sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {OUT}/sensitivity_noM1_comparison.tsv")
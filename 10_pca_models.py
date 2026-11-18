#!/usr/bin/env python3
"""
Step 18: PCA-based OLS models — PC_degree and PC_DVR as predictors.

Reduces the 4 degree and 4 DVR predictors to their first principal component each,
eliminating multicollinearity by construction.

Models:
  M_age:    Y ~ age                        (baseline)
  M_pcdeg:  Y ~ age + PC_degree
  M_pcdvr:  Y ~ age + PC_dvr
  M_pc:     Y ~ age + PC_degree + PC_dvr   (combined)

Outputs:
  results2/pca_models/pca_variance_explained.tsv  — PCA summary
  results2/pca_models/pca_model_comparison.tsv    — model comparison
  results2/pca_models/pca_forest_<behavior>.png/svg — forest plots
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
                           "degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
                           "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]],
                 on="subject_id")
df.columns  = df.columns.str.strip()
df["group"] = df["group"].str.strip()
print(f"n = {len(df)}")

OUT = Path(PROJECT_DIR / "results2/pca_models")
OUT.mkdir(parents=True, exist_ok=True)

DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]
BEHAV_LABELS = {
    "ssrt_int":   "SSRT (ms)",
    "rcrt_mean":  "RCRT mean (ms)",
    "rt_sd":      "RT SD (ms)",
    "perc_error": "% Error",
}

# --- PCA (z-score required before PCA) ---
def fit_pca(cols, label):
    X     = df[cols].values
    scaler = StandardScaler()
    X_sc  = scaler.fit_transform(X)
    pca   = PCA(n_components=1)
    pc    = pca.fit_transform(X_sc).flatten()
    var_exp = pca.explained_variance_ratio_[0]
    loadings = pd.Series(pca.components_[0], index=cols)
    print(f"\nPC_{label}: explains {var_exp*100:.1f}% of variance")
    print("  Loadings:", loadings.round(3).to_dict())
    return pc, var_exp, loadings

pc_degree, var_deg, load_deg = fit_pca(DEGREE_COLS, "degree")
pc_dvr,    var_dvr, load_dvr = fit_pca(DVR_COLS,    "dvr")

df["PC_degree"] = pc_degree
df["PC_dvr"]    = pc_dvr

# Save PCA summary
pca_summary = pd.DataFrame({
    "component":         ["PC_degree", "PC_dvr"],
    "variance_explained": [round(var_deg, 4), round(var_dvr, 4)],
    "loadings_M1":       [round(load_deg["degree_M1"], 3), round(load_dvr["dvr_M1"], 3)],
    "loadings_IFG":      [round(load_deg["degree_IFG"], 3), round(load_dvr["dvr_IFG"], 3)],
    "loadings_SMA":      [round(load_deg["degree_SMA"], 3), round(load_dvr["dvr_SMA"], 3)],
    "loadings_Striatum": [round(load_deg["degree_Striatum"], 3), round(load_dvr["dvr_Striatum"], 3)],
})
pca_summary.to_csv(OUT / "pca_variance_explained.tsv", sep="\t", index=False)
print(f"\n✓ Saved PCA summary")

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

X_age   = build_X(["age"])
X_pcdeg = build_X(["age", "PC_degree"])
X_pcdvr = build_X(["age", "PC_dvr"])
X_pc    = build_X(["age", "PC_degree", "PC_dvr"])

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print("\n" + "="*80)
    print(f"OUTCOME: {behav_label}")
    print("="*80)

    y = df[behav_name].values.astype(float)

    fit_age   = sm.OLS(y, X_age).fit()
    fit_pcdeg = sm.OLS(y, X_pcdeg).fit()
    fit_pcdvr = sm.OLS(y, X_pcdvr).fit()
    fit_pc    = sm.OLS(y, X_pc).fit()

    print(f"\n  M_pc coefficients:")
    coef_df = pd.DataFrame({
        "coef":   fit_pc.params,
        "t":      fit_pc.tvalues,
        "p":      fit_pc.pvalues,
        "signif": fit_pc.pvalues.apply(
            lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s.")
    }).round(4)
    print(coef_df.to_string())

    rows = [
        {"behavior": behav_label, "model": "M_age",
         "n_pred": 1,  "df_resid": int(fit_age.df_resid),
         "R2": round(fit_age.rsquared, 4),   "adj_R2": round(fit_age.rsquared_adj, 4),
         "F_p": round(fit_age.f_pvalue, 4),  "AIC": round(fit_age.aic, 2),
         "LR_vs_Mage_p": "", "LR_vs_pcdeg_p": "", "LR_vs_pcdvr_p": ""},

        {"behavior": behav_label, "model": "M_pcdeg (age + PC_degree)",
         "n_pred": 2,  "df_resid": int(fit_pcdeg.df_resid),
         "R2": round(fit_pcdeg.rsquared, 4), "adj_R2": round(fit_pcdeg.rsquared_adj, 4),
         "F_p": round(fit_pcdeg.f_pvalue, 4), "AIC": round(fit_pcdeg.aic, 2),
         "LR_vs_Mage_p": lr_test(fit_age, fit_pcdeg)[2],
         "LR_vs_pcdeg_p": "", "LR_vs_pcdvr_p": ""},

        {"behavior": behav_label, "model": "M_pcdvr (age + PC_dvr)",
         "n_pred": 2,  "df_resid": int(fit_pcdvr.df_resid),
         "R2": round(fit_pcdvr.rsquared, 4), "adj_R2": round(fit_pcdvr.rsquared_adj, 4),
         "F_p": round(fit_pcdvr.f_pvalue, 4), "AIC": round(fit_pcdvr.aic, 2),
         "LR_vs_Mage_p": lr_test(fit_age, fit_pcdvr)[2],
         "LR_vs_pcdeg_p": "", "LR_vs_pcdvr_p": ""},

        {"behavior": behav_label, "model": "M_pc (age + PC_degree + PC_dvr)",
         "n_pred": 3,  "df_resid": int(fit_pc.df_resid),
         "R2": round(fit_pc.rsquared, 4),    "adj_R2": round(fit_pc.rsquared_adj, 4),
         "F_p": round(fit_pc.f_pvalue, 4),   "AIC": round(fit_pc.aic, 2),
         "LR_vs_Mage_p":   lr_test(fit_age,   fit_pc)[2],
         "LR_vs_pcdeg_p":  lr_test(fit_pcdeg, fit_pc)[2],
         "LR_vs_pcdvr_p":  lr_test(fit_pcdvr, fit_pc)[2]},
    ]
    all_rows.extend(rows)

    # --- Forest plot for M_pc ---
    params  = fit_pc.params.drop("const")
    ci      = fit_pc.conf_int().drop("const")
    pvals   = fit_pc.pvalues.drop("const")
    labels  = ["Age", "PC degree", "PC DVR"]
    colors_map = {"age": "#43A047", "PC_degree": "#2196F3", "PC_dvr": "#E53935"}
    sig_map    = {"age": "#2E7D32", "PC_degree": "#1565C0", "PC_dvr": "#B71C1C"}
    y_pos   = np.arange(len(params))[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, pred in enumerate(params.index):
        p      = pvals[pred]
        color  = sig_map[pred] if p < 0.05 else colors_map[pred]
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
    ax.set_title(f"M_pc coefficients — {behav_label}", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / f"pca_forest_{behav_name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"pca_forest_{behav_name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved forest plot: pca_forest_{behav_name}")

summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(OUT / "pca_model_comparison.tsv", sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved model comparison: {OUT}/pca_model_comparison.tsv")
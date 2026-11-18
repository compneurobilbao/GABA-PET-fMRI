#!/usr/bin/env python3
"""
Step 9: Model comparison per motor score.

Three nested models (all include age as covariate):
  M_degree : age + degree_M1 + degree_IFG + degree_SMA + degree_Striatum        (5 pred)
  M_dvr    : age + dvr_M1    + dvr_IFG    + dvr_SMA    + dvr_Striatum           (5 pred)
  M3       : age + degree×4  + dvr×4                                            (9 pred)

Comparison table per outcome:
  R², adj R², AIC, BIC, F, p
  + likelihood-ratio test M3 vs M_degree and M3 vs M_dvr (chi2, p)

Outputs:
  - results/M3_comparison/model_comparison_<behavior>.tsv  — per behavior
  - results/M3_comparison/model_comparison_summary.tsv     — all behaviors combined
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

behav = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv", sep="\t")
brain = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv")

behav.columns = behav.columns.str.strip()
brain.columns = brain.columns.str.strip()

BRAIN_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
              "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

df = behav.merge(brain[["subject_id"] + BRAIN_COLS], on="subject_id")
print(f"n = {len(df)}")

OUT = PROJECT_DIR / "results2/degree_dvr_n30/M3_comparison_n30"
OUT.mkdir(exist_ok=True)

BEHAV_VARS  = ["ssrt_int", "rcrt_mean", "rt_sd", "perc_error"]
DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

def center(v):
    return v - v.mean()

# Center all predictors once
age_c    = center(df["age"].values)
degree_c = {c: center(df[c].values) for c in DEGREE_COLS}
dvr_c    = {c: center(df[c].values) for c in DVR_COLS}

def make_X(include_degree=True, include_dvr=True):
    cols = {"age": age_c}
    if include_degree:
        cols.update(degree_c)
    if include_dvr:
        cols.update(dvr_c)
    return sm.add_constant(pd.DataFrame(cols))

X_degree = make_X(include_degree=True,  include_dvr=False)
X_dvr    = make_X(include_degree=False, include_dvr=True)
X_full   = make_X(include_degree=True,  include_dvr=True)

def lr_test(fit_restricted, fit_full):
    """Likelihood-ratio test: full vs restricted model."""
    lr_stat = 2 * (fit_full.llf - fit_restricted.llf)
    df_diff = fit_full.df_model - fit_restricted.df_model
    p_val   = stats.chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 3), int(df_diff), round(p_val, 4)

def model_row(label, fit, lr_vs_full=None):
    row = {
        "model":    label,
        "n_pred":   int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "R2":       round(fit.rsquared,     4),
        "adj_R2":   round(fit.rsquared_adj, 4),
        "F":        round(fit.fvalue,       3),
        "F_p":      round(fit.f_pvalue,     4),
        "AIC":      round(fit.aic,          2),
        "BIC":      round(fit.bic,          2),
        "LR_chi2":  "",
        "LR_df":    "",
        "LR_p":     "",
    }
    if lr_vs_full is not None:
        row["LR_chi2"] = lr_vs_full[0]
        row["LR_df"]   = lr_vs_full[1]
        row["LR_p"]    = lr_vs_full[2]
    return row

all_summary = []

for behav_name in BEHAV_VARS:
    print("\n" + "="*80)
    print(f"OUTCOME: {behav_name}")
    print("="*80)

    y = df[behav_name].values

    fit_degree = sm.OLS(y, X_degree).fit()
    fit_dvr    = sm.OLS(y, X_dvr).fit()
    fit_full   = sm.OLS(y, X_full).fit()

    lr_degree = lr_test(fit_degree, fit_full)   # M3 vs degree-only
    lr_dvr    = lr_test(fit_dvr,    fit_full)   # M3 vs DVR-only

    rows = [
        model_row("M_degree (age + degree×4)",      fit_degree, lr_vs_full=lr_degree),
        model_row("M_dvr    (age + dvr×4)",          fit_dvr,    lr_vs_full=lr_dvr),
        model_row("M3       (age + degree×4 + dvr×4)", fit_full),
    ]

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    comp_df.to_csv(OUT / f"model_comparison_{behav_name}.tsv",
                   sep="\t", index=False, float_format="%.4f")

    for r in rows:
        r["behavior"] = behav_name
    all_summary.extend(rows)

# Combined summary
summary_df = pd.DataFrame(all_summary)[
    ["behavior", "model", "n_pred", "df_resid",
     "R2", "adj_R2", "F", "F_p", "AIC", "BIC", "LR_chi2", "LR_df", "LR_p"]
]
summary_df.to_csv(OUT / "model_comparison_summary.tsv",
                  sep="\t", index=False, float_format="%.4f")

print("\n\n" + "="*80)
print("FULL SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))

print(f"\n✓ Saved per-behavior : {OUT}/model_comparison_<behavior>.tsv")
print(f"✓ Saved summary      : {OUT}/model_comparison_summary.tsv")
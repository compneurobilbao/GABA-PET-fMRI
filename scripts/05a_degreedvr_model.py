#!/usr/bin/env python3
"""
Step 5a: OLS model predicting motor-inhibition scores from subnetwork degree + DVR.

Model (main effects only, centered predictors):
  Y = b0 + b1·age
        + b2·degree_M1 + b3·degree_IFG + b4·degree_SMA + b5·degree_Striatum
        + b6·dvr_M1    + b7·dvr_IFG    + b8·dvr_SMA    + b9·dvr_Striatum

Outcomes: ssrt_int, rcrt_mean, rt_sd, perc_error

"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

behav = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv", sep="\t")
brain = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv")

# Strip whitespace from column names (behav tsv has leading spaces)
behav.columns = behav.columns.str.strip()
brain.columns = brain.columns.str.strip()

df = behav.merge(brain[["subject_id",
                         "degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
                         "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]],
                 on="subject_id")
print(f"n = {len(df)}")

OUT = PROJECT_DIR / "results2/degree_dvr_n30"
OUT.mkdir(exist_ok=True)

BEHAV_VARS   = ["ssrt_int", "rcrt_mean", "rt_sd", "perc_error"]
DEGREE_COLS  = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS     = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]
PRED_COLS    = ["age"] + DEGREE_COLS + DVR_COLS

# --- Center all predictors ---
def center(v):
    return v - v.mean()

X = pd.DataFrame({col: center(df[col].values) for col in PRED_COLS},
                 columns=PRED_COLS)
X = sm.add_constant(X)

# --- Fit one model per outcome ---
summary_rows = []
for behav_name in BEHAV_VARS:
    print("\n" + "="*80)
    print(f"OUTCOME: {behav_name}")
    print("="*80)

    y = df[behav_name].values
    fit = sm.OLS(y, X).fit()

    print(f"R²     = {fit.rsquared:.4f}")
    print(f"adj R² = {fit.rsquared_adj:.4f}")
    print(f"F({fit.df_model:.0f}, {fit.df_resid:.0f}) = {fit.fvalue:.3f},  p = {fit.f_pvalue:.4f}")
    print(f"AIC = {fit.aic:.2f},  BIC = {fit.bic:.2f}")

    coef_df = pd.DataFrame({
        "coef":    fit.params,
        "std_err": fit.bse,
        "t":       fit.tvalues,
        "p":       fit.pvalues,
        "ci_lo":   fit.conf_int()[0],
        "ci_hi":   fit.conf_int()[1],
    }).round(4)

    print("\nCoefficient table:")
    print(coef_df.to_string())

    coef_df.to_csv(OUT / f"M3_coefs_{behav_name}.tsv",
                   sep="\t", float_format="%.4f")

    summary_rows.append({
        "behavior": behav_name,
        "R2":       round(fit.rsquared,     4),
        "adj_R2":   round(fit.rsquared_adj, 4),
        "F":        round(fit.fvalue,       3),
        "F_p":      round(fit.f_pvalue,     4),
        "AIC":      round(fit.aic,          2),
        "BIC":      round(fit.bic,          2),
        "n":        int(fit.nobs),
        "df_resid": int(fit.df_resid),
    })

# --- Summary table ---
print("\n\n" + "="*80)
print("SUMMARY: M3 model (degree + DVR + age) per behavior")
print("="*80)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT / "M3_summary.tsv", sep="\t", index=False, float_format="%.4f")

print(f"\n✓ Saved per-behavior coefs : {OUT}/coefs_<behavior>.tsv")
print(f"✓ Saved summary            : {OUT}/summary.tsv")
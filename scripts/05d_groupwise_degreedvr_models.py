#!/usr/bin/env python3
"""
Step 13: OLS models per group — degree-only vs DVR-only predictors (no age).

Models (OLD and YOUNG separately):
  M_degree: Y ~ degree_M1 + degree_IFG + degree_SMA + degree_Striatum
  M_dvr:    Y ~ dvr_M1    + dvr_IFG    + dvr_SMA    + dvr_Striatum

Outcomes: ssrt_int, rcrt_mean, rt_sd, perc_error

Output rows ordered as: OLD/behavior, YOUNG/behavior (alternating per behavior)
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n28.tsv", sep="\t")
deg_dvr = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)

behav.columns   = behav.columns.str.strip()
deg_dvr.columns = deg_dvr.columns.str.strip()
deg_dvr["group"] = deg_dvr["group"].str.strip()

df = behav.merge(deg_dvr[["subject_id",
                           "degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum",
                           "dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]],
                 on="subject_id")
df.columns  = df.columns.str.strip()
df["group"] = df["group"].str.strip()

print(f"Total n={len(df)}  |  OLD: {(df['group']=='OLD').sum()}  YOUNG: {(df['group']=='YOUNG').sum()}")

OUT = PROJECT_DIR / "results2/group_degree_dvr_models"
OUT.mkdir(parents=True, exist_ok=True)

BEHAV_VARS  = ["ssrt_int", "rcrt_mean", "rt_sd", "perc_error"]
DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

def center(v):
    return v - v.mean()

def build_X(sub_df, pred_cols):
    X = pd.DataFrame({col: center(sub_df[col].values) for col in pred_cols})
    return sm.add_constant(X)

def model_row(fit, group, model_name, behav_name):
    return {
        "group":    group,
        "behavior": behav_name,
        "model":    model_name,
        "n":        int(fit.nobs),
        "n_pred":   int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "R2":       round(fit.rsquared,     4),
        "adj_R2":   round(fit.rsquared_adj, 4),
        "F":        round(fit.fvalue,       3),
        "F_p":      round(fit.f_pvalue,     4),
        "AIC":      round(fit.aic,          2),
        "BIC":      round(fit.bic,          2),
    }

# Collect rows per behavior, then interleave OLD/YOUNG
all_rows = []
for behav_name in BEHAV_VARS:
    for group_name in ["OLD", "YOUNG"]:
        sub_df = df[df["group"] == group_name].reset_index(drop=True)
        X_deg  = build_X(sub_df, DEGREE_COLS)
        X_dvr  = build_X(sub_df, DVR_COLS)
        y      = sub_df[behav_name].values

        fit_deg = sm.OLS(y, X_deg).fit()
        fit_dvr = sm.OLS(y, X_dvr).fit()

        all_rows.append(model_row(fit_deg, group_name, "M_degree", behav_name))
        all_rows.append(model_row(fit_dvr, group_name, "M_dvr",    behav_name))

summary_df = pd.DataFrame(all_rows)[
    ["group", "behavior", "model", "n", "n_pred", "df_resid",
     "R2", "adj_R2", "F", "F_p", "AIC", "BIC"]
]

out_path = OUT / "group_models_summary.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")

print("\n" + "="*80)
print(summary_df.to_string(index=False))
print(f"\n✓ Saved: {out_path}")
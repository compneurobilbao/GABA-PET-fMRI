#!/usr/bin/env python3
"""
Step 05c: FC-only vs Coupling-only model comparison (full sample, n=28).

Models:
  FC_only      : Y ~ age + 6 pairwise FC (full 4x4 upper triangle)
                 predictors: M1_vs_IFG, M1_vs_SMA, M1_vs_Striatum,
                              IFG_vs_SMA, IFG_vs_Striatum, SMA_vs_Striatum

  Coupling_only: Y ~ age + 12 DVR×FC terms (DVR of each node × their shared FC)
                 predictors: DVR_M1×M1_vs_IFG,      DVR_IFG×M1_vs_IFG,
                              DVR_M1×M1_vs_SMA,      DVR_SMA×M1_vs_SMA,
                              DVR_M1×M1_vs_Striatum, DVR_Striatum×M1_vs_Striatum,
                              DVR_IFG×IFG_vs_SMA,    DVR_SMA×IFG_vs_SMA,
                              DVR_IFG×IFG_vs_Striatum, DVR_Striatum×IFG_vs_Striatum,
                              DVR_SMA×SMA_vs_Striatum, DVR_Striatum×SMA_vs_Striatum

Output:
  results/subnetwork_FC_vs_coupling/FC_vs_coupling_fullsample.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# --- Load data ---
behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n28.tsv", sep="\t")
fc_mat  = pd.read_csv(PROJECT_DIR / "results2/FC_subnetworks/subnetwork_fc_corrmat.tsv", sep="\t")
deg_dvr = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)

behav.columns   = behav.columns.str.strip()
fc_mat.columns  = fc_mat.columns.str.strip()
deg_dvr.columns = deg_dvr.columns.str.strip()

FC_COLS = ["M1_vs_IFG", "M1_vs_SMA", "M1_vs_Striatum",
           "IFG_vs_SMA", "IFG_vs_Striatum", "SMA_vs_Striatum"]

df = behav.merge(fc_mat[["subject_id"] + FC_COLS], on="subject_id")
df = df.merge(deg_dvr[["subject_id",
                        "dvr_M1", "dvr_IFG", "dvr_SMA", "dvr_Striatum"]],
              on="subject_id")
df.columns = df.columns.str.strip()
df['group'] = df['group'].str.strip()
print(f"n = {len(df)}")

OUT = PROJECT_DIR / "results2/subnetwork_FC_vs_coupling"
OUT.mkdir(parents=True, exist_ok=True)

BEHAV_VARS = ["ssrt_int", "rcrt_mean", "rt_sd", "perc_error"]
SUB_NAMES  = ["M1", "IFG", "SMA", "Striatum"]
PAIRS      = [(a, b) for i, a in enumerate(SUB_NAMES) for b in SUB_NAMES[i+1:]]

def center(v):
    return v - v.mean()

def build_FC_model(data):
    """age + 6 pairwise FC"""
    X = pd.DataFrame({"age": center(data["age"].values)})
    for fc_col in FC_COLS:
        X[fc_col] = center(data[fc_col].values)
    return sm.add_constant(X)

def build_Coupling_model(data):
    """age + 12 DVR×FC terms"""
    X = pd.DataFrame({"age": center(data["age"].values)})
    for a, b in PAIRS:
        fc_col  = f"{a}_vs_{b}"
        fc_c    = center(data[fc_col].values)
        dvr_a_c = center(data[f"dvr_{a}"].values)
        dvr_b_c = center(data[f"dvr_{b}"].values)
        X[f"DVR_{a}x{a}_{b}"] = dvr_a_c * fc_c
        X[f"DVR_{b}x{a}_{b}"] = dvr_b_c * fc_c
    return sm.add_constant(X)

def model_summary(fit, label, behav_name):
    return {
        "behavior": behav_name,
        "model":    label,
        "n":        int(fit.nobs),
        "n_pred":   int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "R2":       round(fit.rsquared,     4),
        "adj_R2":   round(fit.rsquared_adj, 4),
        "AIC":      round(fit.aic,          2),
        "BIC":      round(fit.bic,          2),
        "F":        round(fit.fvalue,       3),
        "F_p":      round(fit.f_pvalue,     4),
    }

X_fc   = build_FC_model(df)
X_coup = build_Coupling_model(df)

all_rows = []
print("\n" + "="*80)
for behav_name in BEHAV_VARS:
    y = df[behav_name].values

    fit_fc   = sm.OLS(y, X_fc).fit()
    fit_coup = sm.OLS(y, X_coup).fit()

    s_fc   = model_summary(fit_fc,   "FC_only",       behav_name)
    s_coup = model_summary(fit_coup, "Coupling_only", behav_name)

    delta_aic = s_coup["AIC"] - s_fc["AIC"]
    delta_bic = s_coup["BIC"] - s_fc["BIC"]
    delta_adj = s_coup["adj_R2"] - s_fc["adj_R2"]

    print(f"\n{behav_name}")
    print(f"  FC_only      n_pred={s_fc['n_pred']}  df_resid={s_fc['df_resid']}"
          f"  R²={s_fc['R2']:.3f}  adjR²={s_fc['adj_R2']:.3f}"
          f"  AIC={s_fc['AIC']:.1f}  F_p={s_fc['F_p']:.3f}")
    print(f"  Coupling     n_pred={s_coup['n_pred']}  df_resid={s_coup['df_resid']}"
          f"  R²={s_coup['R2']:.3f}  adjR²={s_coup['adj_R2']:.3f}"
          f"  AIC={s_coup['AIC']:.1f}  F_p={s_coup['F_p']:.3f}")
    print(f"  ΔAIC={delta_aic:+.2f}  ΔBIC={delta_bic:+.2f}  ΔadjR²={delta_adj:+.3f}"
          f"  → AIC favors {'Coupling' if delta_aic < 0 else 'FC'}")

    all_rows.extend([s_fc, s_coup])

# --- Save ---
result_df = pd.DataFrame(all_rows)[
    ["behavior", "model", "n", "n_pred", "df_resid",
     "R2", "adj_R2", "AIC", "BIC", "F", "F_p"]
]
out_path = OUT / "FC_vs_coupling_fullsample.tsv"
result_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {out_path}")
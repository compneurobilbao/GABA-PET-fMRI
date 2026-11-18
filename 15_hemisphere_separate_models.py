#!/usr/bin/env python3
"""
Step 15: Hemisphere-separate models (Option A).

Models:
  M0: age only                                  (NOT lateralized — same for L and R)
  M1: age + 4 degrees (per hemisphere)          (lateralized)
  M2: age + 4 DVRs    (per hemisphere)          (lateralized)
  M3: age + 4 degrees + 4 DVRs (per hemisphere) (lateralized)

For each outcome, fit M0 once, then M1/M2/M3 separately for L and R.
Compare L vs R predictive performance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# --- Load behavioral ---
behav = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv",
                     sep="\t")
behav.columns = behav.columns.str.strip()
behav["subject_id"] = behav["subject_id"].astype(str).str.strip()
behav["group"] = behav["group"].str.strip()

# --- Load lateralized DVR and degree ---
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
print(f"n = {len(df)}")

OUT = PROJECT_DIR / "results2/hemisphere_separate_models"
OUT.mkdir(parents=True, exist_ok=True)

BEHAV_VARS = ["ssrt_int", "rcrt_mean", "rt_sd", "perc_error"]
SUBNETS = ["M1", "IFG", "SMA", "Striatum"]

def center(v):
    return v - v.mean()

age_c = center(df["age"].values)

def build_X(hemisphere=None, include_degree=False, include_dvr=False):
    cols = {"age": age_c}
    if include_degree:
        for s in SUBNETS:
            cols[f"degree_{s}_{hemisphere}"] = center(df[f"degree_{s}_{hemisphere}"].values)
    if include_dvr:
        for s in SUBNETS:
            cols[f"dvr_{s}_{hemisphere}"] = center(df[f"dvr_{s}_{hemisphere}"].values)
    return sm.add_constant(pd.DataFrame(cols))

def lr_test(fit_restricted, fit_full):
    lr_stat = 2 * (fit_full.llf - fit_restricted.llf)
    df_diff = int(fit_full.df_model - fit_restricted.df_model)
    p_val   = stats.chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 3), df_diff, round(p_val, 4)

def model_row(fit, model_name, behav_name, hemisphere, lr_vs_m0=None,
              lr_vs_m1=None, lr_vs_m2=None):
    return {
        "hemisphere":    hemisphere,
        "behavior":      behav_name,
        "model":         model_name,
        "n_pred":        int(fit.df_model),
        "df_resid":      int(fit.df_resid),
        "R2":            round(fit.rsquared,     4),
        "adj_R2":        round(fit.rsquared_adj, 4),
        "F":             round(fit.fvalue,       3),
        "F_p":           round(fit.f_pvalue,     4),
        "AIC":           round(fit.aic,          2),
        "BIC":           round(fit.bic,          2),
        "LR_vs_M0_p":    lr_vs_m0[2] if lr_vs_m0 else "",
        "LR_vs_M1_p":    lr_vs_m1[2] if lr_vs_m1 else "",
        "LR_vs_M2_p":    lr_vs_m2[2] if lr_vs_m2 else "",
    }

all_rows = []

# M0 is computed ONCE (not lateralized)
print("\n" + "="*90)
print("M0 (age only) — same for both hemispheres")
print("="*90)
X_M0 = build_X(include_degree=False, include_dvr=False)
for behav_name in BEHAV_VARS:
    y = df[behav_name].values.astype(float)
    fit_m0 = sm.OLS(y, X_M0).fit()
    print(f"  {behav_name:<12s}  R²={fit_m0.rsquared:.3f}  adj R²={fit_m0.rsquared_adj:.3f}  "
          f"AIC={fit_m0.aic:.1f}  F_p={fit_m0.f_pvalue:.4f}")
    all_rows.append(model_row(fit_m0, "M0", behav_name, "both"))

# M1, M2, M3 are lateralized
for hemisphere in ["L", "R"]:
    print("\n" + "="*90)
    print(f"HEMISPHERE: {hemisphere}")
    print("="*90)
    X_M1 = build_X(hemisphere, include_degree=True,  include_dvr=False)  # age + degree
    X_M2 = build_X(hemisphere, include_degree=False, include_dvr=True)   # age + DVR
    X_M3 = build_X(hemisphere, include_degree=True,  include_dvr=True)   # age + degree + DVR

    for behav_name in BEHAV_VARS:
        print(f"\n--- Outcome: {behav_name} ---")
        y = df[behav_name].values.astype(float)
        fit_m0 = sm.OLS(y, X_M0).fit()  # for LR comparisons
        fit_m1 = sm.OLS(y, X_M1).fit()
        fit_m2 = sm.OLS(y, X_M2).fit()
        fit_m3 = sm.OLS(y, X_M3).fit()

        rows = [
            model_row(fit_m1, "M1", behav_name, hemisphere,
                      lr_vs_m0=lr_test(fit_m0, fit_m1)),
            model_row(fit_m2, "M2", behav_name, hemisphere,
                      lr_vs_m0=lr_test(fit_m0, fit_m2)),
            model_row(fit_m3, "M3", behav_name, hemisphere,
                      lr_vs_m0=lr_test(fit_m0, fit_m3),
                      lr_vs_m1=lr_test(fit_m1, fit_m3),
                      lr_vs_m2=lr_test(fit_m2, fit_m3)),
        ]
        for r in rows:
            print(f"  {r['model']:<4}  R²={r['R2']:.3f}  adj R²={r['adj_R2']:.3f}  "
                  f"AIC={r['AIC']:.1f}  F_p={r['F_p']:.4f}")
        all_rows.extend(rows)

# Save full summary
summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(OUT / "hemisphere_models_summary.tsv",
                  sep="\t", index=False, float_format="%.4f")

# Pivoted L vs R comparison — only M1/M2/M3 (M0 not lateralized)
lat_df = summary_df[summary_df["model"].isin(["M1", "M2", "M3"])]

print("\n\n" + "="*100)
print("L vs R COMPARISON — adjusted R² per behavior and model")
print("="*100)
pivoted = lat_df.pivot_table(
    index=["behavior", "model"], columns="hemisphere", values="adj_R2"
).round(3)
pivoted["winner_adjR2"] = np.where(pivoted["L"] > pivoted["R"], "L",
                                     np.where(pivoted["R"] > pivoted["L"], "R", "tie"))
pivoted["delta_adjR2_LvsR"] = (pivoted["L"] - pivoted["R"]).round(3)
print(pivoted)

print("\n" + "="*100)
print("L vs R COMPARISON — AIC (lower is better)")
print("="*100)
pivoted_aic = lat_df.pivot_table(
    index=["behavior", "model"], columns="hemisphere", values="AIC"
).round(2)
pivoted_aic["winner_AIC"] = np.where(pivoted_aic["L"] < pivoted_aic["R"], "L",
                                       np.where(pivoted_aic["R"] < pivoted_aic["L"], "R", "tie"))
pivoted_aic["delta_AIC_LvsR"] = (pivoted_aic["L"] - pivoted_aic["R"]).round(2)
print(pivoted_aic)

print(f"\n✓ Saved: {OUT}/hemisphere_models_summary.tsv")
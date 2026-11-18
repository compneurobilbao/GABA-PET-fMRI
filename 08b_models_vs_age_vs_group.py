#!/usr/bin/env python3
"""
Step 16b: Model comparison table — age (continuous) VS age-group (categorical).

For each motor outcome, we fit the same four models twice:
  Once using age (continuous):
    M_age: Y ~ age
    M0:    Y ~ age + degree×4
    M1:    Y ~ age + dvr×4
    M2:    Y ~ age + degree×4 + dvr×4
  Once using age-group (categorical, YOUNG=0 / OLD=1):
    M_grp: Y ~ group
    M0:    Y ~ group + degree×4
    M1:    Y ~ group + dvr×4
    M2:    Y ~ group + degree×4 + dvr×4

LR tests: M0 vs baseline, M1 vs baseline, M2 vs baseline, M2 vs M0, M2 vs M1

Output:
  results2/model_comparison/model_comparison_age_vs_group.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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

# Create categorical age-group variable: YOUNG = 0, OLD = 1
df["group_dummy"] = (df["group"] == "OLD").astype(int)

print(f"n = {len(df)}")
print(f"Group counts: {df['group'].value_counts().to_dict()}")
print(f"Age range YOUNG: {df[df['group']=='YOUNG']['age'].min():.1f}–{df[df['group']=='YOUNG']['age'].max():.1f} y")
print(f"Age range OLD:   {df[df['group']=='OLD']['age'].min():.1f}–{df[df['group']=='OLD']['age'].max():.1f} y")

OUT = PROJECT_DIR / "results2/model_comparison"
Path(OUT).mkdir(parents=True, exist_ok=True)

BEHAV_LABELS = {
    "ssrt_int":   "SSRT",
    "rcrt_mean":  "RCRT mean",
    "rt_sd":      "RT SD",
    "perc_error": "% Error",
}
DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

def center(v):
    return v - v.mean()

def build_X(pred_cols):
    """Build design matrix with mean-centered predictors."""
    X = pd.DataFrame({col: center(df[col].values) for col in pred_cols})
    return sm.add_constant(X)

def lr_test(fit_restricted, fit_full):
    lr_stat = 2 * (fit_full.llf - fit_restricted.llf)
    df_diff = int(fit_full.df_model - fit_restricted.df_model)
    p_val   = stats.chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 3), df_diff, round(p_val, 4)

def model_row(fit, model_name, behav_name, age_type,
              lr_vs_base=None, lr_vs_m0=None, lr_vs_m1=None):
    row = {
        "behavior":        behav_name,
        "age_type":        age_type,
        "model":           model_name,
        "n_pred":          int(fit.df_model),
        "df_resid":        int(fit.df_resid),
        "R2":              round(fit.rsquared,     4),
        "adj_R2":          round(fit.rsquared_adj, 4),
        "F":               round(fit.fvalue,       3),
        "F_p":             round(fit.f_pvalue,     4),
        "AIC":             round(fit.aic,          2),
        "BIC":             round(fit.bic,          2),
        "LR_vs_base_chi2": lr_vs_base[0] if lr_vs_base else "",
        "LR_vs_base_df":   lr_vs_base[1] if lr_vs_base else "",
        "LR_vs_base_p":    lr_vs_base[2] if lr_vs_base else "",
        "LR_vs_M0_chi2":   lr_vs_m0[0]  if lr_vs_m0  else "",
        "LR_vs_M0_df":     lr_vs_m0[1]  if lr_vs_m0  else "",
        "LR_vs_M0_p":      lr_vs_m0[2]  if lr_vs_m0  else "",
        "LR_vs_M1_chi2":   lr_vs_m1[0]  if lr_vs_m1  else "",
        "LR_vs_M1_df":     lr_vs_m1[1]  if lr_vs_m1  else "",
        "LR_vs_M1_p":      lr_vs_m1[2]  if lr_vs_m1  else "",
    }
    return row

# Build design matrices for BOTH age formulations
# --- Age continuous ---
X_age_cont    = build_X(["age"])
X_deg_cont    = build_X(["age"] + DEGREE_COLS)
X_dvr_cont    = build_X(["age"] + DVR_COLS)
X_m3_cont     = build_X(["age"] + DEGREE_COLS + DVR_COLS)

# --- Age categorical (group_dummy) ---
X_age_cat     = build_X(["group_dummy"])
X_deg_cat     = build_X(["group_dummy"] + DEGREE_COLS)
X_dvr_cat     = build_X(["group_dummy"] + DVR_COLS)
X_m3_cat      = build_X(["group_dummy"] + DEGREE_COLS + DVR_COLS)

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print("\n" + "="*80)
    print(f"OUTCOME: {behav_label}")
    print("="*80)

    y = df[behav_name].values.astype(float)

    # ============ AGE CONTINUOUS ============
    fit_age_cont = sm.OLS(y, X_age_cont).fit()
    fit_deg_cont = sm.OLS(y, X_deg_cont).fit()
    fit_dvr_cont = sm.OLS(y, X_dvr_cont).fit()
    fit_m3_cont  = sm.OLS(y, X_m3_cont).fit()

    rows_cont = [
        model_row(fit_age_cont, "M_age (age only)",            behav_label, "continuous"),
        model_row(fit_deg_cont, "M0    (age + degree×4)",      behav_label, "continuous",
                  lr_vs_base=lr_test(fit_age_cont, fit_deg_cont)),
        model_row(fit_dvr_cont, "M1    (age + dvr×4)",         behav_label, "continuous",
                  lr_vs_base=lr_test(fit_age_cont, fit_dvr_cont)),
        model_row(fit_m3_cont,  "M2    (age + degree×4 + dvr×4)", behav_label, "continuous",
                  lr_vs_base=lr_test(fit_age_cont, fit_m3_cont),
                  lr_vs_m0=lr_test(fit_deg_cont,   fit_m3_cont),
                  lr_vs_m1=lr_test(fit_dvr_cont,   fit_m3_cont)),
    ]

    # ============ AGE CATEGORICAL (group) ============
    fit_age_cat  = sm.OLS(y, X_age_cat).fit()
    fit_deg_cat  = sm.OLS(y, X_deg_cat).fit()
    fit_dvr_cat  = sm.OLS(y, X_dvr_cat).fit()
    fit_m3_cat   = sm.OLS(y, X_m3_cat).fit()

    rows_cat = [
        model_row(fit_age_cat, "M_grp (group only)",            behav_label, "categorical"),
        model_row(fit_deg_cat, "M0    (group + degree×4)",      behav_label, "categorical",
                  lr_vs_base=lr_test(fit_age_cat, fit_deg_cat)),
        model_row(fit_dvr_cat, "M1    (group + dvr×4)",         behav_label, "categorical",
                  lr_vs_base=lr_test(fit_age_cat, fit_dvr_cat)),
        model_row(fit_m3_cat,  "M2    (group + degree×4 + dvr×4)", behav_label, "categorical",
                  lr_vs_base=lr_test(fit_age_cat, fit_m3_cat),
                  lr_vs_m0=lr_test(fit_deg_cat,   fit_m3_cat),
                  lr_vs_m1=lr_test(fit_dvr_cat,   fit_m3_cat)),
    ]

    # Print comparison side by side
    print("\n--- AGE CONTINUOUS ---")
    print(pd.DataFrame(rows_cont)[["model","n_pred","df_resid","R2","adj_R2","F_p","AIC","BIC",
                                    "LR_vs_base_p","LR_vs_M0_p","LR_vs_M1_p"]].to_string(index=False))

    print("\n--- AGE CATEGORICAL (group) ---")
    print(pd.DataFrame(rows_cat)[["model","n_pred","df_resid","R2","adj_R2","F_p","AIC","BIC",
                                   "LR_vs_base_p","LR_vs_M0_p","LR_vs_M1_p"]].to_string(index=False))

    all_rows.extend(rows_cont)
    all_rows.extend(rows_cat)

# Save full summary
summary_df = pd.DataFrame(all_rows)
out_path = Path(OUT) / "model_comparison_age_vs_group.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n{'='*80}")
print(f"✓ Saved: {out_path}")
print(f"{'='*80}")

# Print a compact summary table at the end
print("\n\n" + "="*90)
print("QUICK COMPARISON: adj R² by outcome × model × age formulation")
print("="*90)

compact = pd.DataFrame(all_rows)[["behavior","model","age_type","adj_R2","AIC","LR_vs_base_p"]]
for outcome in [v for v in BEHAV_LABELS.values()]:
    print(f"\n{outcome}:")
    sub = compact[compact["behavior"] == outcome].copy()
    # Reshape to compare side-by-side
    pivot = sub.pivot_table(index="model", columns="age_type",
                             values=["adj_R2","AIC","LR_vs_base_p"],
                             aggfunc="first")
    print(pivot.to_string())
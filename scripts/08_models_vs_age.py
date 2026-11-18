"""
Step 08: Model comparison table — Y ~ age  VS  Y ~ age + brain (degree/dvr).

For each motor outcome, we fit four nested models:
  M_age: Y ~ age                        (age-only baseline)
  M0:    Y ~ age + degree×4
  M1:    Y ~ age + dvr×4
  M2:    Y ~ age + degree×4 + dvr×4     (age + full brain set)

LR tests:
  M0 vs M_age   → does degree improve on age alone?
  M1 vs M_age   → does dvr    improve on age alone?
  M2 vs M_age   → does full brain set improve on age alone?
  M2 vs M0      → does dvr    improve on age + degree?
  M2 vs M1      → does degree improve on age + dvr?

Bootstrap 95% CI for adj R² (1000 iterations) are also reported per model.

Output:
  results2/model_comparison/model_comparison_vs_age.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# Bootstrap settings
N_BOOT = 1000
RNG_SEED = 42

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

def build_X(pred_cols, data=None):
    """Build design matrix with mean-centered predictors."""
    src = data if data is not None else df
    X = pd.DataFrame({col: center(src[col].values) for col in pred_cols})
    return sm.add_constant(X)

def lr_test(fit_restricted, fit_full):
    lr_stat = 2 * (fit_full.llf - fit_restricted.llf)
    df_diff = int(fit_full.df_model - fit_restricted.df_model)
    p_val   = stats.chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 3), df_diff, round(p_val, 4)

def bootstrap_adj_r2(y_col, pred_cols, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Bootstrap 95% CI for adjusted R² via case resampling.

    Draws n_boot samples with replacement from the rows of df, refits the OLS
    model with the given predictors, and stores adj R² each time. Returns the
    2.5th and 97.5th percentiles (percentile CI).

    Predictors are re-centered *within each bootstrap sample* to match the
    original modelling choice (mean-centering).
    """
    rng   = np.random.default_rng(seed)
    n     = len(df)
    adj_r2_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        boot_df  = df.iloc[idx].copy()
        X_boot   = build_X(pred_cols, data=boot_df)
        y_boot   = boot_df[y_col].values.astype(float)
        try:
            fit_boot = sm.OLS(y_boot, X_boot).fit()
            adj_r2_boot[b] = fit_boot.rsquared_adj
        except Exception:
            adj_r2_boot[b] = np.nan
    ci_lo, ci_hi = np.nanpercentile(adj_r2_boot, [2.5, 97.5])
    return round(float(ci_lo), 4), round(float(ci_hi), 4)

def model_row(fit, model_name, behav_name, pred_cols,
              lr_vs_age=None, lr_vs_m0=None, lr_vs_m1=None):
    """One row of the summary table, including bootstrap CI for adj R²."""
    ci_lo, ci_hi = bootstrap_adj_r2(behav_name, pred_cols)
    row = {
        "behavior":         BEHAV_LABELS[behav_name],
        "model":            model_name,
        "n_pred":           int(fit.df_model),
        "df_resid":         int(fit.df_resid),
        "R2":               round(fit.rsquared,     4),
        "adj_R2":           round(fit.rsquared_adj, 4),
        "adj_R2_CI_low":    ci_lo,
        "adj_R2_CI_high":   ci_hi,
        "F":                round(fit.fvalue,       3),
        "F_p":              round(fit.f_pvalue,     4),
        "AIC":              round(fit.aic,          2),
        "BIC":              round(fit.bic,          2),
        "LR_vs_age_chi2":   lr_vs_age[0] if lr_vs_age else "",
        "LR_vs_age_df":     lr_vs_age[1] if lr_vs_age else "",
        "LR_vs_age_p":      lr_vs_age[2] if lr_vs_age else "",
        "LR_vs_M0_chi2":    lr_vs_m0[0]  if lr_vs_m0  else "",
        "LR_vs_M0_df":      lr_vs_m0[1]  if lr_vs_m0  else "",
        "LR_vs_M0_p":       lr_vs_m0[2]  if lr_vs_m0  else "",
        "LR_vs_M1_chi2":    lr_vs_m1[0]  if lr_vs_m1  else "",
        "LR_vs_M1_df":      lr_vs_m1[1]  if lr_vs_m1  else "",
        "LR_vs_M1_p":       lr_vs_m1[2]  if lr_vs_m1  else "",
    }
    return row

# --- Design matrices ---
X_age = build_X(["age"])
X_deg = build_X(["age"] + DEGREE_COLS)
X_dvr = build_X(["age"] + DVR_COLS)
X_m3  = build_X(["age"] + DEGREE_COLS + DVR_COLS)

pred_cols_age = ["age"]
pred_cols_deg = ["age"] + DEGREE_COLS
pred_cols_dvr = ["age"] + DVR_COLS
pred_cols_m3  = ["age"] + DEGREE_COLS + DVR_COLS

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print(f"\n=== {behav_label} ===")

    y = df[behav_name].values.astype(float)

    fit_age = sm.OLS(y, X_age).fit()
    fit_deg = sm.OLS(y, X_deg).fit()
    fit_dvr = sm.OLS(y, X_dvr).fit()
    fit_m3  = sm.OLS(y, X_m3 ).fit()

    print(f"  Bootstrap CI computation (n_boot={N_BOOT}) ...")

    rows = [
        model_row(fit_age, "M_age (age only)",            behav_name, pred_cols_age),
        model_row(fit_deg, "M0    (age + degree×4)",      behav_name, pred_cols_deg,
                  lr_vs_age=lr_test(fit_age, fit_deg)),
        model_row(fit_dvr, "M1    (age + dvr×4)",         behav_name, pred_cols_dvr,
                  lr_vs_age=lr_test(fit_age, fit_dvr)),
        model_row(fit_m3,  "M2    (age + degree×4 + dvr×4)", behav_name, pred_cols_m3,
                  lr_vs_age=lr_test(fit_age, fit_m3),
                  lr_vs_m0 =lr_test(fit_deg, fit_m3),
                  lr_vs_m1 =lr_test(fit_dvr, fit_m3)),
    ]

    # Print rows for this behavior
    for r in rows:
        ci_str = f"[{r['adj_R2_CI_low']:.3f}, {r['adj_R2_CI_high']:.3f}]"
        line = (f"  {r['model']:<32s} "
                f"adjR² = {r['adj_R2']:.3f} {ci_str:<20s} "
                f"F = {r['F']:.2f}  Fp = {r['F_p']}  "
                f"AIC = {r['AIC']}")
        print(line)
        if r["LR_vs_age_p"] != "":
            print(f"    LR vs age: chi²={r['LR_vs_age_chi2']}, df={r['LR_vs_age_df']}, p={r['LR_vs_age_p']}")
        if r["LR_vs_M0_p"] != "":
            print(f"    LR vs M0:  chi²={r['LR_vs_M0_chi2']}, df={r['LR_vs_M0_df']}, p={r['LR_vs_M0_p']}")
        if r["LR_vs_M1_p"] != "":
            print(f"    LR vs M1:  chi²={r['LR_vs_M1_chi2']}, df={r['LR_vs_M1_df']}, p={r['LR_vs_M1_p']}")

    all_rows.extend(rows)

# --- Save summary TSV ---
summary_df = pd.DataFrame(all_rows)
out_path = Path(OUT) / "model_comparison_vs_age.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {out_path}")
print(f"  Includes bootstrap 95% CI for adj R² (n_boot={N_BOOT}, seed={RNG_SEED})")
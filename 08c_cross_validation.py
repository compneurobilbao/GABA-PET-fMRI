"""
Step 08c: Leave-one-out cross-validation for the four nested models per outcome.

For each behavioural outcome (SSRT, RCRT, RT SD, % Error), we compute the
out-of-sample R² (R²_CV) for each model using leave-one-out cross-validation:

  M_age: Y ~ age
  M0:    Y ~ age + degree×4
  M1:    Y ~ age + dvr×4
  M2:    Y ~ age + degree×4 + dvr×4

For each fold, one subject is held out, the model is refit on the remaining 29,
and the prediction for the held-out subject is stored. R²_CV is then computed
across the 30 held-out predictions:

  R²_CV = 1 - SS_res_CV / SS_tot

Where SS_tot uses the mean of the full sample as reference. If R²_CV drops
substantially compared to the in-sample adj R², the model may be overfitting.

Output:
  results2/model_comparison/model_cross_validation.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

MODEL_PREDS = {
    "M_age (age only)":               ["age"],
    "M0    (age + degree×4)":         ["age"] + DEGREE_COLS,
    "M1    (age + dvr×4)":            ["age"] + DVR_COLS,
    "M2    (age + degree×4 + dvr×4)": ["age"] + DEGREE_COLS + DVR_COLS,
}

def loo_cv_r2(y, X):
    """
    Leave-one-out CV R² for a linear model.

    IMPORTANT: X is centered on the training fold (n-1 subjects), not on the
    full sample. This avoids leakage of the held-out subject's information
    into the training predictors. The intercept is always kept.
    """
    n              = len(y)
    y              = np.asarray(y, dtype=float)
    X_raw          = np.asarray(X, dtype=float)          # (n, p) — un-centered predictors, no intercept
    y_pred_holdout = np.empty(n)

    for i in range(n):
        train_idx           = np.arange(n) != i
        X_train_raw         = X_raw[train_idx]
        X_test_raw          = X_raw[i:i+1]

        # Center predictors using training-fold means only
        col_means           = X_train_raw.mean(axis=0)
        X_train_c           = X_train_raw - col_means
        X_test_c            = X_test_raw  - col_means

        # Add intercept
        X_train_final       = sm.add_constant(X_train_c, has_constant="add")
        X_test_final        = sm.add_constant(X_test_c,  has_constant="add")

        fit                 = sm.OLS(y[train_idx], X_train_final).fit()
        y_pred_holdout[i]   = fit.predict(X_test_final)[0]

    ss_res  = np.sum((y - y_pred_holdout) ** 2)
    ss_tot  = np.sum((y - y.mean())        ** 2)
    r2_cv   = 1.0 - ss_res / ss_tot
    return round(float(r2_cv), 4)

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print(f"\n=== {behav_label} ===")

    y = df[behav_name].values.astype(float)

    for model_name, pred_cols in MODEL_PREDS.items():
        X_raw = df[pred_cols].values     # un-centered — LOO will center per fold

        # In-sample fit (for reference, using paper's centering on full sample)
        col_means_full = X_raw.mean(axis=0)
        X_c_full       = X_raw - col_means_full
        X_final_full   = sm.add_constant(X_c_full, has_constant="add")
        fit_full       = sm.OLS(y, X_final_full).fit()

        # LOO-CV R²
        r2_cv = loo_cv_r2(y, X_raw)

        row = {
            "behavior":       behav_label,
            "model":          model_name,
            "n_pred":         int(fit_full.df_model),
            "adj_R2_insample": round(fit_full.rsquared_adj, 4),
            "R2_CV_LOO":      r2_cv,
            "delta_R2":       round(fit_full.rsquared_adj - r2_cv, 4),
        }
        all_rows.append(row)

        print(f"  {model_name:<32s} "
              f"adj R² (in-sample) = {row['adj_R2_insample']:>+.3f}   "
              f"R²_CV (LOO)        = {row['R2_CV_LOO']:>+.3f}   "
              f"Δ = {row['delta_R2']:>+.3f}")

# --- Save table ---
summary_df = pd.DataFrame(all_rows)
out_path = Path(OUT) / "model_cross_validation.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {out_path}")
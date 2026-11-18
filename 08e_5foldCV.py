"""
Step 09b: Repeated 5-fold cross-validation for the four nested models per outcome.

Same nested models as in Steps 08 and 09:
  M_age: Y ~ age
  M0:    Y ~ age + degree×4
  M1:    Y ~ age + dvr×4
  M2:    Y ~ age + degree×4 + dvr×4

For each behavioural outcome, we compute out-of-sample R² using repeated 5-fold
cross-validation:
  - 5 folds per repetition (24 subjects train, 6 subjects test per fold)
  - 10 repetitions with different random fold assignments
  - Total: 50 fold evaluations per model
  - Predictors are mean-centered within each training fold to avoid leakage

The R²_CV reported is computed by pooling all out-of-fold predictions across
the 50 evaluations and comparing them against the observed values (referenced
to the pooled mean).

Compared to LOO-CV, repeated 5-fold CV is less sensitive to the removal of
individual subjects and typically produces more stable estimates.

Output:
  results2/model_comparison/model_repeated_kfold_cv.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# CV settings
N_SPLITS  = 5
N_REPEATS = 10
RNG_SEED  = 42

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

print(f"n = {len(df)}, {N_SPLITS}-fold CV × {N_REPEATS} repetitions")

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

def repeated_kfold_r2(y, X_raw, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=RNG_SEED):
    """
    Repeated k-fold CV R² with in-fold mean-centering.

    For each of the n_repeats × n_splits folds:
      1. Split subjects into train and test
      2. Center predictors on training-fold means only (no leakage from test)
      3. Fit model on training fold
      4. Predict test fold subjects
      5. Store test predictions

    R²_CV is then computed by pooling all test predictions across all folds
    and comparing them to observed values.

    Returns:
      r2_cv         : pooled R² across all fold predictions
      r2_cv_per_rep : mean R² per repetition (10 values), then averaged
    """
    y     = np.asarray(y,     dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)
    n     = len(y)

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    r2_per_repetition = []

    # Track per-repetition R²
    fold_counter = 0
    y_pred_repetition = np.full(n, np.nan)
    obs_index_covered = set()

    for train_idx, test_idx in rkf.split(X_raw):
        X_train_raw = X_raw[train_idx]
        X_test_raw  = X_raw[test_idx]
        y_train     = y[train_idx]

        # Center on training fold means only
        col_means   = X_train_raw.mean(axis=0)
        X_train_c   = X_train_raw - col_means
        X_test_c    = X_test_raw  - col_means

        X_train_final = sm.add_constant(X_train_c, has_constant="add")
        X_test_final  = sm.add_constant(X_test_c,  has_constant="add")

        try:
            fit_train = sm.OLS(y_train, X_train_final).fit()
            y_pred    = fit_train.predict(X_test_final)
            for j, i in enumerate(test_idx):
                y_pred_repetition[i] = y_pred[j]
                obs_index_covered.add(i)
        except Exception:
            continue

        fold_counter += 1

        # Every n_splits folds = one full repetition completed
        if fold_counter % n_splits == 0:
            # All subjects should have exactly one prediction in this repetition
            if len(obs_index_covered) == n:
                ss_res_rep = np.sum((y - y_pred_repetition) ** 2)
                ss_tot_rep = np.sum((y - y.mean())        ** 2)
                r2_rep     = 1.0 - ss_res_rep / ss_tot_rep
                r2_per_repetition.append(r2_rep)
            # Reset for next repetition
            y_pred_repetition  = np.full(n, np.nan)
            obs_index_covered  = set()

    r2_cv_mean = float(np.mean(r2_per_repetition))
    r2_cv_std  = float(np.std (r2_per_repetition, ddof=1))

    return round(r2_cv_mean, 4), round(r2_cv_std, 4)

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print(f"\n=== {behav_label} ===")

    y = df[behav_name].values.astype(float)

    for model_name, pred_cols in MODEL_PREDS.items():
        X_raw = df[pred_cols].values

        r2_cv_mean, r2_cv_std = repeated_kfold_r2(y, X_raw)

        # In-sample fit for reference
        col_means_full = X_raw.mean(axis=0)
        X_c_full       = X_raw - col_means_full
        X_final_full   = sm.add_constant(X_c_full, has_constant="add")
        fit_full       = sm.OLS(y, X_final_full).fit()

        row = {
            "behavior":         behav_label,
            "model":            model_name,
            "n_pred":           int(fit_full.df_model),
            "adj_R2_insample":  round(fit_full.rsquared_adj, 4),
            "R2_CV_mean":       r2_cv_mean,
            "R2_CV_std":        r2_cv_std,
            "delta_R2":         round(fit_full.rsquared_adj - r2_cv_mean, 4),
        }
        all_rows.append(row)

        print(f"  {model_name:<32s} "
              f"adj R² = {row['adj_R2_insample']:>+.3f}   "
              f"R²_CV = {row['R2_CV_mean']:>+.3f} ± {row['R2_CV_std']:.3f}   "
              f"Δ = {row['delta_R2']:>+.3f}")

# --- Save table ---
summary_df = pd.DataFrame(all_rows)
out_path = Path(OUT) / "model_repeated_kfold_cv.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {out_path}")
print(f"  Settings: {N_SPLITS}-fold × {N_REPEATS} repetitions (seed = {RNG_SEED})")
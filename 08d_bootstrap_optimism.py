"""
Step 10: Bootstrap optimism correction for adjusted R² per model × outcome.

For each behavioural outcome and each nested model, we estimate the "optimism"
of the in-sample adjusted R² using the bootstrap optimism-correction procedure
(Efron & Tibshirani, 1993; Harrell, 2015).

Algorithm (per model × outcome):
  1. Fit the model on the original sample (n = 30) → adj_R2_apparent
  2. For b = 1 to B:
     a. Draw a bootstrap sample (n = 30, with replacement)
     b. Fit the model on the bootstrap sample → adj_R2_boot_train
     c. Apply the bootstrap-fitted model to the ORIGINAL sample
        → compute adj_R2 on original data using boot coefficients
        → adj_R2_boot_test
     d. optimism_b = adj_R2_boot_train − adj_R2_boot_test
  3. optimism = mean of optimism_b across all B iterations
  4. adj_R2_corrected = adj_R2_apparent − optimism

The corrected adj R² is an estimate of the model's performance after removing
the sample-specific overfitting bias.

Output:
  results2/model_comparison/model_optimism_correction.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

# Bootstrap settings
N_BOOT   = 1000
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

MODEL_PREDS = {
    "M_age (age only)":               ["age"],
    "M0    (age + degree×4)":         ["age"] + DEGREE_COLS,
    "M1    (age + dvr×4)":            ["age"] + DVR_COLS,
    "M2    (age + degree×4 + dvr×4)": ["age"] + DEGREE_COLS + DVR_COLS,
}

def compute_adj_r2_from_predictions(y_true, y_pred, n, k):
    """
    Compute adjusted R² from arbitrary predictions (not from a fitted model).
    y_true : observed outcome
    y_pred : predicted outcome (from any model)
    n      : sample size
    k      : number of predictors (excluding intercept)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    # Adjusted R² formula
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    return adj_r2

def bootstrap_optimism(y_col, pred_cols, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Bootstrap optimism correction for adjusted R².

    Returns:
      adj_r2_apparent   : original in-sample adj R²
      optimism          : mean bootstrap optimism
      adj_r2_corrected  : apparent − optimism
    """
    rng = np.random.default_rng(seed)
    n   = len(df)
    k   = len(pred_cols)  # excluding intercept

    y_orig = df[y_col].values.astype(float)

    # Step 1: fit on original sample → apparent adj R²
    X_orig_raw   = df[pred_cols].values.astype(float)
    col_means_o  = X_orig_raw.mean(axis=0)
    X_orig_c     = X_orig_raw - col_means_o
    X_orig_final = sm.add_constant(X_orig_c, has_constant="add")
    fit_orig     = sm.OLS(y_orig, X_orig_final).fit()
    adj_r2_apparent = fit_orig.rsquared_adj

    # Step 2-4: bootstrap optimism
    optimism_all = np.empty(n_boot)
    for b in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].copy()
        y_boot  = boot_df[y_col].values.astype(float)

        # Fit on bootstrap sample
        X_boot_raw   = boot_df[pred_cols].values.astype(float)
        col_means_b  = X_boot_raw.mean(axis=0)
        X_boot_c     = X_boot_raw - col_means_b
        X_boot_final = sm.add_constant(X_boot_c, has_constant="add")

        try:
            fit_boot = sm.OLS(y_boot, X_boot_final).fit()

            # adj R² on the bootstrap sample itself (train)
            adj_r2_boot_train = fit_boot.rsquared_adj

            # Apply boot-fitted model to the ORIGINAL sample (test)
            # Important: center original data using the BOOT means
            # so the design matrix matches the boot model's coefficients
            X_orig_using_boot_means   = X_orig_raw - col_means_b
            X_orig_using_boot_final   = sm.add_constant(X_orig_using_boot_means,
                                                        has_constant="add")
            y_pred_on_orig = fit_boot.predict(X_orig_using_boot_final)
            adj_r2_boot_test = compute_adj_r2_from_predictions(
                y_true=y_orig,
                y_pred=y_pred_on_orig,
                n=n,
                k=k
            )

            optimism_all[b] = adj_r2_boot_train - adj_r2_boot_test
        except Exception:
            optimism_all[b] = np.nan

    optimism         = float(np.nanmean(optimism_all))
    adj_r2_corrected = adj_r2_apparent - optimism

    return (round(float(adj_r2_apparent),   4),
            round(optimism,                  4),
            round(float(adj_r2_corrected),   4))

all_rows = []

for behav_name, behav_label in BEHAV_LABELS.items():
    print(f"\n=== {behav_label} ===")

    for model_name, pred_cols in MODEL_PREDS.items():
        adj_r2_app, opt, adj_r2_corr = bootstrap_optimism(behav_name, pred_cols)

        row = {
            "behavior":         behav_label,
            "model":            model_name,
            "n_pred":           len(pred_cols),
            "adj_R2_apparent":  adj_r2_app,
            "optimism":         opt,
            "adj_R2_corrected": adj_r2_corr,
        }
        all_rows.append(row)

        print(f"  {model_name:<32s} "
              f"apparent = {adj_r2_app:>+.3f}   "
              f"optimism = {opt:>+.3f}   "
              f"corrected = {adj_r2_corr:>+.3f}")

# --- Save table ---
summary_df = pd.DataFrame(all_rows)
out_path = Path(OUT) / "model_optimism_correction.tsv"
summary_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {out_path}")
print(f"  Bootstrap iterations: {N_BOOT} (seed = {RNG_SEED})")
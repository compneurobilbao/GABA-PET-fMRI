"""
Block-wise VIF: evaluates whether each predictor from one modality
can be predicted by the OTHER modality (plus age).

This specifically tests whether DVR and functional connectivity carry
non-redundant information, independent of within-modality collinearity
(which is biologically expected).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

behav   = pd.read_csv(PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv", sep="\t")
deg_dvr = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                      sep=",", skipinitialspace=True)

behav.columns   = behav.columns.str.strip()
deg_dvr.columns = deg_dvr.columns.str.strip()

df = behav.merge(deg_dvr, on="subject_id")

DEGREE_COLS = ["degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]
DVR_COLS    = ["dvr_M1",    "dvr_IFG",    "dvr_SMA",    "dvr_Striatum"]

def center(cols):
    X = df[cols].copy()
    return X - X.mean()

def block_vif(target_col, other_block_cols, include_age=True):
    """
    Compute VIF-like metric for target_col using only other_block_cols
    (and optionally age) as predictors in the auxiliary regression.
    """
    predictors = (["age"] if include_age else []) + other_block_cols
    y = center([target_col]).values.flatten()
    X = center(predictors).values
    X = sm.add_constant(X)
    fit = sm.OLS(y, X).fit()
    r2 = fit.rsquared
    vif = 1 / (1 - r2) if r2 < 1 else np.inf
    return r2, vif

print("=" * 70)
print("BLOCK-WISE VIF: can DVR predictors be explained by degree + age?")
print("=" * 70)
print(f"{'Predictor':<20s} {'R² (vs other block)':<22s} {'block-VIF':<10s}")
print("-" * 70)
for dvr_col in DVR_COLS:
    r2, vif = block_vif(dvr_col, DEGREE_COLS, include_age=True)
    print(f"{dvr_col:<20s} {r2:<22.3f} {vif:<10.3f}")

print()
print("=" * 70)
print("BLOCK-WISE VIF: can degree predictors be explained by DVR + age?")
print("=" * 70)
print(f"{'Predictor':<20s} {'R² (vs other block)':<22s} {'block-VIF':<10s}")
print("-" * 70)
for deg_col in DEGREE_COLS:
    r2, vif = block_vif(deg_col, DVR_COLS, include_age=True)
    print(f"{deg_col:<20s} {r2:<22.3f} {vif:<10.3f}")

print()
print("Interpretation:")
print("  Block-VIF < 2 : the target predictor is NOT explained by the other block")
print("                  → DVR and degree carry non-redundant information")
print("  Block-VIF > 5 : the target predictor IS largely explained by the other block")
print("                  → the two modalities overlap substantially")
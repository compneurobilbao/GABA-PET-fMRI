#!/usr/bin/env python3
"""
Sex effect analysis: independent t-tests for DVR and FC degree by sex.
Reports t, p, Cohen's d, and group means for each variable.
"""

from pathlib import Path
import pandas as pd
import numpy as np
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
df["Sex"]   = df["Sex"].str.strip()

print(f"n = {len(df)}  |  M: {(df['Sex']=='M').sum()}  F: {(df['Sex']=='F').sum()}")
print(f"Sex balance by group:")
print(df.groupby(["group","Sex"]).size(), "\n")

TEST_VARS = ["dvr_M1", "dvr_IFG", "dvr_SMA", "dvr_Striatum",
             "degree_M1", "degree_IFG", "degree_SMA", "degree_Striatum"]

def cohens_d(a, b):
    pooled_std = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) /
                         (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_std

rows = []
for var in TEST_VARS:
    m = df[df["Sex"]=="M"][var].dropna()
    f = df[df["Sex"]=="F"][var].dropna()
    t, p = stats.ttest_ind(m, f)
    d    = cohens_d(m, f)
    rows.append({
        "variable": var,
        "M_mean":   round(m.mean(), 4),
        "M_sd":     round(m.std(),  4),
        "F_mean":   round(f.mean(), 4),
        "F_sd":     round(f.std(),  4),
        "t":        round(t, 3),
        "p":        round(p, 4),
        "cohens_d": round(d, 3),
        "significant": "YES" if p < 0.05 else "no",
    })

result_df = pd.DataFrame(rows)

print("="*80)
print("SEX EFFECTS ON DVR AND FC DEGREE (independent t-test)")
print("="*80)
print(result_df.to_string(index=False))

OUT = PROJECT_DIR / "results2/sex_effects"
Path(OUT).mkdir(parents=True, exist_ok=True)
result_df.to_csv(Path(OUT) / "sex_effects_dvr_degree.tsv",
                 sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {OUT}/sex_effects_dvr_degree.tsv")
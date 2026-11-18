"""
Generate Table 2 (descriptive brain measures): mean DVR and FC strength
per sub-network × group (YOUNG vs OLD), with t-test and Cohen's d.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"

df = pd.read_csv(PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv",
                 sep=",", skipinitialspace=True)
df.columns = df.columns.str.strip()
df["group"] = df["group"].str.strip()

OUT = PROJECT_DIR / "results2/tables"
OUT.mkdir(parents=True, exist_ok=True)

SUBNETS = ["M1", "IFG", "SMA", "Striatum"]

def summarize(col_name, display_name, modality_label):
    """Compute mean ± SD per group, t-test, and Cohen's d for a given predictor."""
    young = df[df["group"] == "YOUNG"][col_name].values
    old   = df[df["group"] == "OLD"][col_name].values

    mean_y, sd_y = young.mean(), young.std(ddof=1)
    mean_o, sd_o = old.mean(),   old.std(ddof=1)

    # Independent samples t-test
    t, p = stats.ttest_ind(young, old, equal_var=False)   # Welch's t-test

    # Cohen's d (pooled SD)
    pooled_sd = np.sqrt(((len(young)-1)*sd_y**2 + (len(old)-1)*sd_o**2)
                        / (len(young) + len(old) - 2))
    d = (mean_o - mean_y) / pooled_sd

    return {
        "Modality":   modality_label,
        "Sub-network": display_name,
        "YOUNG (mean ± SD)": f"{mean_y:.2f} ± {sd_y:.2f}",
        "OLD (mean ± SD)":   f"{mean_o:.2f} ± {sd_o:.2f}",
        "t":          round(t, 2),
        "p":          round(p, 3),
        "Cohen's d":  round(d, 2),
    }

rows = []

# DVR per sub-network
for sn in SUBNETS:
    rows.append(summarize(f"dvr_{sn}", sn, "DVR"))

# FC strength per sub-network
for sn in SUBNETS:
    rows.append(summarize(f"degree_{sn}", sn, "FC strength"))

table = pd.DataFrame(rows)

# Save as TSV
tsv_path = OUT / "table2_descriptive_brain.tsv"
table.to_csv(tsv_path, sep="\t", index=False)


print(f"✓ Table 2 saved: {tsv_path}")
print()
print(table.to_string(index=False))
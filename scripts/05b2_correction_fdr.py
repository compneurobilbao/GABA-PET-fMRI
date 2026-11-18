"""
Apply FDR Benjamini-Hochberg correction to the LR tests from the model
comparison, separating the single confirmatory test (Model_FULL vs Model_AGE
for SSRT) from the exploratory tests.

Camino C: confirmatory test NOT corrected; exploratory tests FDR-corrected.
"""

from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
TSV = PROJECT_DIR / "results2/model_comparison/model_comparison_vs_age.tsv"

df = pd.read_csv(TSV, sep="\t")
df.columns = df.columns.str.strip()
df["behavior"] = df["behavior"].str.strip()
df["model"]    = df["model"].str.strip()

# RCRT removed from the paper
df = df[df["behavior"].isin(["SSRT", "RT SD", "% Error"])].copy()

# Define model name mapping (TSV internal names → display names)
model_rename = {
    "M_age (age only)":                "Model_AGE",
    "M0    (age + degree×4)":          "Model_AGE+FC",
    "M1    (age + dvr×4)":             "Model_AGE+DVR",
    "M2    (age + degree×4 + dvr×4)":  "Model_FULL",
}
df["model_display"] = df["model"].map(model_rename)

# Keep only rows with LR tests (Model_AGE row has no LR_vs_age_p)
df["LR_vs_age_p"] = pd.to_numeric(df["LR_vs_age_p"], errors="coerce")
df["LR_vs_M0_p"]  = pd.to_numeric(df["LR_vs_M0_p"],  errors="coerce")
df["LR_vs_M1_p"]  = pd.to_numeric(df["LR_vs_M1_p"],  errors="coerce")

# Build a long table of all LR tests
rows = []
for _, r in df.iterrows():
    if r["model_display"] == "Model_AGE":
        continue  # baseline, no LR vs itself

    # Test vs Model_AGE
    if not pd.isna(r["LR_vs_age_p"]):
        rows.append({
            "behavior":   r["behavior"],
            "comparison": f"{r['model_display']} vs Model_AGE",
            "p_raw":      r["LR_vs_age_p"],
        })

    # For Model_FULL only: also tests vs Model_AGE+FC and Model_AGE+DVR
    if r["model_display"] == "Model_FULL":
        if not pd.isna(r["LR_vs_M0_p"]):
            rows.append({
                "behavior":   r["behavior"],
                "comparison": "Model_FULL vs Model_AGE+FC",
                "p_raw":      r["LR_vs_M0_p"],
            })
        if not pd.isna(r["LR_vs_M1_p"]):
            rows.append({
                "behavior":   r["behavior"],
                "comparison": "Model_FULL vs Model_AGE+DVR",
                "p_raw":      r["LR_vs_M1_p"],
            })

lr_tests = pd.DataFrame(rows)
lr_tests["sig_raw"] = (lr_tests["p_raw"] < 0.05).map({True: "yes", False: "no"})

# Identify the confirmatory test
confirmatory_mask = (lr_tests["behavior"] == "SSRT") & \
                    (lr_tests["comparison"] == "Model_FULL vs Model_AGE")

confirmatory = lr_tests[confirmatory_mask].copy()
confirmatory["p_fdr"] = confirmatory["p_raw"]  # not corrected
confirmatory["sig_fdr"] = confirmatory["sig_raw"]
confirmatory["test_type"] = "CONFIRMATORY"

# Apply FDR to the exploratory tests
exploratory = lr_tests[~confirmatory_mask].copy()
reject, p_fdr, _, _ = multipletests(exploratory["p_raw"].values,
                                     alpha=0.05, method="fdr_bh")
exploratory["p_fdr"] = p_fdr
exploratory["sig_fdr"] = pd.Series(reject).map({True: "yes", False: "no"}).values
exploratory["test_type"] = "exploratory"

# Combine and report
all_tests = pd.concat([confirmatory, exploratory], ignore_index=True)
all_tests = all_tests.sort_values(["test_type", "behavior", "comparison"],
                                   ascending=[True, True, True])

# Print
print("=" * 90)
print("CAMINO C: 1 confirmatory test + FDR correction on 14 exploratory tests")
print("=" * 90)
print(f"\n{'Behavior':<10s} {'Comparison':<35s} {'p_raw':>8s} {'p_fdr':>8s} {'sig_raw':>8s} {'sig_fdr':>8s} {'type':<13s}")
print("-" * 90)
for _, r in all_tests.iterrows():
    print(f"{r['behavior']:<10s} {r['comparison']:<35s} "
          f"{r['p_raw']:>8.4f} {r['p_fdr']:>8.4f} "
          f"{r['sig_raw']:>8s} {r['sig_fdr']:>8s} {r['test_type']:<13s}")

# Save output
OUT = PROJECT_DIR / "results2/model_comparison"
all_tests.to_csv(OUT / "lr_tests_fdr_corrected.tsv", sep="\t",
                 index=False, float_format="%.4f")
print(f"\n✓ Saved: {OUT / 'lr_tests_fdr_corrected.tsv'}")

# Summary
n_lost = ((all_tests["sig_raw"] == "yes") & (all_tests["sig_fdr"] == "no")).sum()
n_gained = ((all_tests["sig_raw"] == "no") & (all_tests["sig_fdr"] == "yes")).sum()
print(f"\nSummary:")
print(f"  Tests significant before correction: {(all_tests['sig_raw']=='yes').sum()}")
print(f"  Tests significant after correction : {(all_tests['sig_fdr']=='yes').sum()}")
print(f"  Tests that lost significance       : {n_lost}")
print(f"  Tests that gained significance     : {n_gained}")
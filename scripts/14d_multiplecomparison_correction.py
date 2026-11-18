"""
Apply FDR correction (Benjamini-Hochberg) to lateralization paired t-test
p-values. Correction is applied separately per subset (ALL, OLD, YOUNG)
across the 4 sub-networks.
"""

from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
OUT = PROJECT_DIR / "results2/lateralization"

for modality_file in ["degree_lateralization_ttest.tsv",
                      "dvr_lateralization_corrected.tsv"]:
    f = OUT / modality_file
    if not f.exists():
        print(f"[SKIP] {f} not found")
        continue

    results = pd.read_csv(f, sep="\t")

    corrected = []
    for subset in ["ALL", "OLD", "YOUNG"]:
        sub = results[results["subset"] == subset].copy()
        if len(sub) == 0:
            continue
        reject, p_fdr, _, _ = multipletests(sub["p"].values, alpha=0.05,
                                             method="fdr_bh")
        sub["p_fdr"] = p_fdr
        sub["significant_fdr"] = reject
        corrected.append(sub)

    results_corrected = pd.concat(corrected, ignore_index=True)
    out_name = f.stem + "_fdr.tsv"
    results_corrected.to_csv(OUT / out_name,
                             sep="\t", index=False, float_format="%.4f")

    print(f"\n{'='*90}")
    print(f"FDR-corrected results: {modality_file}")
    print('='*90)
    for subset in ["ALL", "OLD", "YOUNG"]:
        sub = results_corrected[results_corrected["subset"] == subset]
        if len(sub) == 0:
            continue
        print(f"\n[{subset}]")
        cols = ["subnetwork", "mean_L", "mean_R", "mean_diff_L-R",
                "t", "p", "p_fdr", "cohens_d", "significant_fdr"]
        print(sub[[c for c in cols if c in sub.columns]
                  ].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print(f"\n✓ Saved: {OUT / out_name}")
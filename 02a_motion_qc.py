#!/usr/bin/env python3
"""
Step 2a: Motion QC for resting-state fMRI.
Extracts framewise displacement from DeepPrep confound files,
computes standard motion metrics, flags potential exclusions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
FMRI_DIR = PROJECT_DIR / "data/fmri_data"
OUT_DIR = PROJECT_DIR / "results/QC"
OUT_DIR.mkdir(exist_ok=True)

QC_TSV = OUT_DIR / "motion_qc.tsv"
QC_PLOT = OUT_DIR / "motion_qc.png"

# --- Thresholds (standard in the FC literature) ---
FD_VOL_THRESH = 0.5   # volumes above this are "high-motion"
MEAN_FD_WARN = 0.2    # subjects above this are flagged (warn)
MEAN_FD_EXCL = 0.3    # subjects above this are flagged (recommend exclusion)
PCT_HIGH_FD_EXCL = 20 # % volumes with FD > 0.5 above this → exclusion

# --- Build subject list ---
subjects = []
for code in ["O", "Y"]:
    group = "OLD" if code == "O" else "YOUNG"
    for i in range(1, 16):
        sid = f"{code}{i:02d}"
        subjects.append((sid, group))

# --- Extract motion per subject ---
rows = []
for sid, group in subjects:
    conf_file = FMRI_DIR / f"sub-{sid}_task-rest_desc-confounds_timeseries.tsv"
    if not conf_file.exists():
        print(f"WARNING: {conf_file} not found, skipping")
        continue

    df = pd.read_csv(conf_file, sep="\t")
    if "framewise_displacement" not in df.columns:
        print(f"WARNING: no framewise_displacement in {sid}")
        continue

    fd = df["framewise_displacement"].values
    # First volume's FD is NaN by definition (no previous volume to compare)
    fd_valid = fd[~np.isnan(fd)]

    n_vols = len(fd)
    mean_fd = np.mean(fd_valid)
    max_fd = np.max(fd_valid)
    median_fd = np.median(fd_valid)
    n_high = np.sum(fd_valid > FD_VOL_THRESH)
    pct_high = 100 * n_high / len(fd_valid)

    # Flag status
    flags = []
    if mean_fd > MEAN_FD_EXCL:
        flags.append("HIGH_MEAN_FD")
    elif mean_fd > MEAN_FD_WARN:
        flags.append("warn_mean_fd")
    if pct_high > PCT_HIGH_FD_EXCL:
        flags.append("HIGH_PCT_SCRUB")
    if max_fd > 5.0:
        flags.append("EXTREME_SPIKE")
    flag_str = ",".join(flags) if flags else "ok"

    rows.append({
        "subject_id": sid,
        "group": group,
        "n_vols": n_vols,
        "mean_fd": mean_fd,
        "median_fd": median_fd,
        "max_fd": max_fd,
        "n_vols_fd_gt_0p5": n_high,
        "pct_vols_fd_gt_0p5": pct_high,
        "flag": flag_str,
    })

qc = pd.DataFrame(rows)
qc.to_csv(QC_TSV, sep="\t", index=False, float_format="%.4f")

# --- Print summary ---
print("\n" + "="*70)
print("MOTION QC SUMMARY")
print("="*70)
print(qc.to_string(index=False))

print("\n" + "="*70)
print("GROUP COMPARISON (mean FD)")
print("="*70)
grp_summary = qc.groupby("group")[["mean_fd", "median_fd", "max_fd", "pct_vols_fd_gt_0p5"]].describe()
print(grp_summary)

# Test for group difference in mean FD
from scipy import stats
old_fd = qc[qc["group"] == "OLD"]["mean_fd"].values
yng_fd = qc[qc["group"] == "YOUNG"]["mean_fd"].values
t, p = stats.ttest_ind(old_fd, yng_fd)
print(f"\nOLD vs YOUNG mean FD: t={t:.2f}, p={p:.3f}")
print(f"  OLD:   mean={old_fd.mean():.3f} (SD={old_fd.std(ddof=1):.3f})")
print(f"  YOUNG: mean={yng_fd.mean():.3f} (SD={yng_fd.std(ddof=1):.3f})")

# --- Flag subjects ---
flagged = qc[qc["flag"] != "ok"]
if len(flagged) > 0:
    print("\n" + "="*70)
    print(f"⚠️  {len(flagged)} SUBJECTS FLAGGED")
    print("="*70)
    print(flagged[["subject_id", "group", "mean_fd",
                   "pct_vols_fd_gt_0p5", "flag"]].to_string(index=False))
else:
    print("\n✓ All subjects within motion thresholds")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: boxplot mean FD by group with individual points
sns.boxplot(data=qc, x="group", y="mean_fd", ax=axes[0],
            order=["YOUNG", "OLD"], showfliers=False)
sns.stripplot(data=qc, x="group", y="mean_fd", ax=axes[0],
              order=["YOUNG", "OLD"], color="black", alpha=0.7, size=7)
axes[0].axhline(MEAN_FD_WARN, color="orange", linestyle="--",
                label=f"warn ({MEAN_FD_WARN}mm)")
axes[0].axhline(MEAN_FD_EXCL, color="red", linestyle="--",
                label=f"exclude ({MEAN_FD_EXCL}mm)")
axes[0].set_ylabel("Mean framewise displacement (mm)")
axes[0].set_title(f"Mean FD by group (p={p:.3f})")
axes[0].legend(fontsize=8)

# Panel 2: per-subject mean FD
qc_sorted = qc.sort_values(["group", "subject_id"])
colors = ["#d62728" if g == "OLD" else "#1f77b4" for g in qc_sorted["group"]]
axes[1].bar(range(len(qc_sorted)), qc_sorted["mean_fd"], color=colors)
axes[1].axhline(MEAN_FD_WARN, color="orange", linestyle="--", alpha=0.6)
axes[1].axhline(MEAN_FD_EXCL, color="red", linestyle="--", alpha=0.6)
axes[1].set_xticks(range(len(qc_sorted)))
axes[1].set_xticklabels(qc_sorted["subject_id"], rotation=90, fontsize=8)
axes[1].set_ylabel("Mean FD (mm)")
axes[1].set_title("Per-subject mean FD")

# Panel 3: % volumes scrubbed
axes[2].bar(range(len(qc_sorted)), qc_sorted["pct_vols_fd_gt_0p5"], color=colors)
axes[2].axhline(PCT_HIGH_FD_EXCL, color="red", linestyle="--", alpha=0.6,
                label=f"exclude ({PCT_HIGH_FD_EXCL}%)")
axes[2].set_xticks(range(len(qc_sorted)))
axes[2].set_xticklabels(qc_sorted["subject_id"], rotation=90, fontsize=8)
axes[2].set_ylabel("% volumes with FD > 0.5mm")
axes[2].set_title("High-motion volumes")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(QC_PLOT, dpi=150)
print(f"\nPlot saved: {QC_PLOT}")
print(f"QC table saved: {QC_TSV}")
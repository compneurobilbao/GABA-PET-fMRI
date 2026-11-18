#!/usr/bin/env python3
"""
Step 04a: Inter-subnetwork FC correlation matrix.

For each subject, extracts the mean time-series of each of the 4 sub-networks
of the motor-inhibition network and computes pairwise Pearson correlations
(Fisher-z transformed) between them:
  1 = fc_M1       (Precentral L/R)
  2 = fc_IFG      (Frontal_Inf_Oper L/R + Frontal_Inf_Tri L/R)
  3 = fc_SMA      (Supp_Motor_Area L/R + ACC_pre L/R)
  4 = fc_Striatum (Caudate L/R + Putamen L/R)

Outputs:
  - TSV with per-subject Fisher-z FC values (upper triangle, flattened)
  - PNG of the mean correlation matrix across all subjects
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import time
import gc

PROJECT_DIR    = Path.home() / "projects/14koen_cuypers/final_analysis2"
FMRI_CLEAN_DIR = PROJECT_DIR / "derivatives/fmri_cleaned"
NET_ATLAS      = PROJECT_DIR / "derivatives/network_masks/motor_inhibition_subnetwork_atlas2.nii.gz"

OUT_TSV        = PROJECT_DIR / "results2/subnetwork_fc_corrmat.tsv"
OUT_PNG        = PROJECT_DIR / "results2/subnetwork_fc_corrmat.png"

# --- Sub-network definitions (atlas values 1-4) ---
SUBNETWORKS = {
    1: "M1",
    2: "IFG",
    3: "SMA",
    4: "Striatum",
}
SUB_LABELS = list(SUBNETWORKS.keys())
SUB_NAMES  = list(SUBNETWORKS.values())

# --- Load atlas (once) ---
net_atlas = nib.load(str(NET_ATLAS)).get_fdata().astype(np.int16)
print("Sub-network voxel counts:")
for label, name in SUBNETWORKS.items():
    print(f"  {name:10s}: {int((net_atlas == label).sum())} voxels")

# --- Subject list ---
subjects = []
for code in ["O", "Y"]:
    group = "OLD" if code == "O" else "YOUNG"
    for i in range(1, 16):
        subjects.append((f"{code}{i:02d}", group))

# --- Process each subject ---
rows = []
for sid, group in subjects:
    start = time.time()
    bold_file = FMRI_CLEAN_DIR / f"sub-{sid}_cleaned_bold.nii.gz"
    if not bold_file.exists():
        print(f"[{sid}] MISSING, skipping")
        continue

    bold_img  = nib.load(str(bold_file))
    bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)
    del bold_img

    # Extract mean time-series per sub-network
    ts = {}
    for label, name in SUBNETWORKS.items():
        mask = net_atlas == label
        if mask.sum() < 10: # sanity check for very small/noisy networks
            ts[name] = None
            continue
        ts[name] = bold_data[mask].mean(axis=0)

    del bold_data
    gc.collect()

    # Pairwise FC (Fisher-z) between sub-networks — upper triangle only
    row = {"subject_id": sid, "group": group}
    for i, n1 in enumerate(SUB_NAMES):
        for j, n2 in enumerate(SUB_NAMES):
            if j <= i:
                continue
            col = f"{n1}_vs_{n2}"
            if ts[n1] is None or ts[n2] is None:
                row[col] = np.nan
                continue
            r = float(np.corrcoef(ts[n1], ts[n2])[0, 1])
            r = max(-0.9999, min(0.9999, r))
            row[col] = float(np.arctanh(r)) # Fisher-z transform: la correlación de Pearson r no tiene distribución normal, y eso importa cuando quieres comparar o promediar valores entre sujetos.

    rows.append(row)
    elapsed = time.time() - start
    print(f"[{sid}] ({group})  {elapsed:.1f}s")

df = pd.DataFrame(rows)
df.to_csv(OUT_TSV, sep="\t", index=False, float_format="%.4f")
print(f"\n✓ Saved: {OUT_TSV}")

# --- Build mean 4x4 matrix across subjects ---
n = len(SUB_NAMES)
mean_mat = np.zeros((n, n))

for i, n1 in enumerate(SUB_NAMES):
    for j, n2 in enumerate(SUB_NAMES):
        if i == j:
            mean_mat[i, j] = 1.0
        elif j > i:
            col = f"{n1}_vs_{n2}"
            mean_mat[i, j] = df[col].mean()
            mean_mat[j, i] = mean_mat[i, j]   # symmetric

print("\n" + "="*60)
print("MEAN INTER-SUBNETWORK FC MATRIX (Fisher-z)")
print("="*60)
print(pd.DataFrame(mean_mat, index=SUB_NAMES, columns=SUB_NAMES).round(3))

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mean_mat, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax, label="Fisher-z (mean across subjects)")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(SUB_NAMES)
ax.set_yticklabels(SUB_NAMES)
ax.set_title("Inter-subnetwork FC (motor-inhibition network)")

for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{mean_mat[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(mean_mat[i, j]) > 0.5 else "black")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"\n✓ Saved plot: {OUT_PNG}")
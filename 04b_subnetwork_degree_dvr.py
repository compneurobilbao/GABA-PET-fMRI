#!/usr/bin/env python3
"""
Step 7: Weighted degree + mean DVR per subnetwork.

For each subject:
  - Loads the per-subject FC matrix (Fisher-z, upper triangle)
  - Computes weighted degree per subnetwork = sum of Fisher-z with the other 3 nodes
  - Resamples the subnetwork atlas to PET space and extracts mean DVR per subnetwork

Output CSV columns:
  subject_id, group,
  degree_M1, degree_IFG, degree_SMA, degree_Striatum,
  dvr_M1,    dvr_IFG,    dvr_SMA,    dvr_Striatum
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

PROJECT_DIR   = Path.home() / "projects/14koen_cuypers/final_analysis2"
FC_TSV        = PROJECT_DIR / "results2/FC_subnetworks/subnetwork_fc_corrmat.tsv"
NET_ATLAS     = PROJECT_DIR / "derivatives/network_masks/motor_inhibition_subnetwork_atlas.nii.gz"
PET_DIR       = PROJECT_DIR / "derivatives/pet_gm_masked"

OUT_CSV       = PROJECT_DIR / "results2/degree_dvr/subnetwork_degree_dvr.csv"
mkdir = OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
# --- Sub-network definitions (must match step 6b) ---
SUBNETWORKS = {
    1: "M1",
    2: "IFG",
    3: "SMA",
    4: "Striatum",
}
SUB_NAMES = list(SUBNETWORKS.values())   # ["M1", "IFG", "SMA", "Striatum"]

# --- Load FC TSV ---
fc_df = pd.read_csv(FC_TSV, sep="\t")
print(f"Loaded FC data: {len(fc_df)} subjects")

# --- Compute weighted degree per subject ---
# degree_X = sum of Fisher-z of X with every other node
degree_rows = []
for _, row in fc_df.iterrows():
    d = {"subject_id": row["subject_id"], "group": row["group"]}
    for n1 in SUB_NAMES:
        total = 0.0
        n_valid = 0
        for n2 in SUB_NAMES:
            if n1 == n2:
                continue
            # column is always stored as alphabetically first_vs_second
            col = f"{n1}_vs_{n2}" if f"{n1}_vs_{n2}" in fc_df.columns else f"{n2}_vs_{n1}"
            val = row[col]
            if not np.isnan(val):
                total += val
                n_valid += 1
        d[f"degree_{n1}"] = total if n_valid > 0 else np.nan
    degree_rows.append(d)

degree_df = pd.DataFrame(degree_rows)

# --- Resample atlas to PET space (once, using first available subject as ref) ---
atlas_img = nib.load(str(NET_ATLAS))

first_pet = sorted(PET_DIR.glob("sub-*_DVR_GM.nii.gz"))[0]
pet_ref   = nib.load(str(first_pet))
print(f"\nPET reference space: {pet_ref.shape} | affine diag: {np.diag(pet_ref.affine)[:3]}")

atlas_pet_space = resample_to_img(
    atlas_img, pet_ref,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
atlas_pet_data = atlas_pet_space.get_fdata().astype(np.int16)

print("Atlas resampled to PET space. Voxel counts:")
for label, name in SUBNETWORKS.items():
    print(f"  {name:10s}: {int((atlas_pet_data == label).sum())} voxels")

# --- Extract mean DVR per subnetwork per subject ---
dvr_rows = []
for _, row in degree_df.iterrows():
    sid   = row["subject_id"]
    pet_file = PET_DIR / f"sub-{sid}_DVR_GM.nii.gz"
    if not pet_file.exists():
        print(f"[{sid}] PET MISSING, skipping DVR")
        dvr_rows.append({"subject_id": sid})
        continue

    pet_data = nib.load(str(pet_file)).get_fdata(dtype=np.float32)

    d = {"subject_id": sid}
    for label, name in SUBNETWORKS.items():
        mask = atlas_pet_data == label
        if mask.sum() < 10:
            d[f"dvr_{name}"] = np.nan
            continue
        d[f"dvr_{name}"] = float(pet_data[mask].mean())
    dvr_rows.append(d)

dvr_df = pd.DataFrame(dvr_rows)

# --- Merge degree + DVR ---
out_df = degree_df.merge(dvr_df, on="subject_id")
out_df.to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"\n✓ Saved: {OUT_CSV}")

# --- Summary ---
degree_cols = [f"degree_{n}" for n in SUB_NAMES]
dvr_cols    = [f"dvr_{n}"    for n in SUB_NAMES]

print("\n" + "="*60)
print("WEIGHTED DEGREE SUMMARY")
print("="*60)
print(out_df[degree_cols].agg(["mean", "std"]).round(3))

print("\n" + "="*60)
print("MEAN DVR PER SUBNETWORK")
print("="*60)
print(out_df[dvr_cols].agg(["mean", "std"]).round(3))

print("\n" + "="*60)
print("GROUP MEANS")
print("="*60)
print(out_df.groupby("group")[degree_cols + dvr_cols].mean().round(3))
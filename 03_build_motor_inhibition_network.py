#!/usr/bin/env python3
"""
Step 3: Build a motor-inhibition network mask from AAL3.
Combines bilateral M1, SMA, pre-SMA, IFG, and striatum into one atlas image
where each SUB-NETWORK gets its own integer label (1-4):
  1 = fc_M1       (Precentral L/R)
  2 = fc_IFG      (Frontal_Inf_Oper L/R + Frontal_Inf_Tri L/R)
  3 = fc_SMA      (Supp_Motor_Area L/R + ACC_pre L/R)
  4 = fc_Striatum (Caudate L/R + Putamen L/R)
"""

from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
import csv

PROJECT_DIR  = Path.home() / "projects/14koen_cuypers/final_analysis2"
AAL3_PATH    = Path.home() / "projects/atlases/AAL3/AAL3_MNI_V7_2mm.nii"
BOLD_REF     = PROJECT_DIR / "derivatives/fmri_cleaned/sub-O01_cleaned_bold.nii.gz"
GM_MASK      = PROJECT_DIR / "derivatives/masks/gm_mask_fmri_space.nii.gz"

OUT_DIR      = PROJECT_DIR / "derivatives/network_masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Individual AAL3 labels (for verification) ---
NETWORK = {
    "Precentral_L":       1,
    "Precentral_R":       2,
    "Frontal_Inf_Oper_L": 7,
    "Frontal_Inf_Oper_R": 8,
    "Frontal_Inf_Tri_L":  9,
    "Frontal_Inf_Tri_R":  10,
    "Supp_Motor_Area_L":  15,
    "Supp_Motor_Area_R":  16,
    "ACC_pre_L":          153,
    "ACC_pre_R":          154,
    "Caudate_L":          75,
    "Caudate_R":          76,
    "Putamen_L":          77,
    "Putamen_R":          78,
}

# --- Sub-network definitions: output label -> AAL3 labels ---
SUBNETWORKS = {
    1: {"name": "fc_M1",       "aal3_labels": [1, 2]},
    2: {"name": "fc_IFG",      "aal3_labels": [7, 8, 9, 10]},
    3: {"name": "fc_SMA",      "aal3_labels": [15, 16]},
    4: {"name": "fc_Striatum", "aal3_labels": [75, 76, 77, 78]},
}

# --- Load AAL3 ---
aal = nib.load(str(AAL3_PATH))
aal_data = aal.get_fdata()
print(f"AAL3 shape: {aal_data.shape}, affine diag: {np.diag(aal.affine)[:3]}")
print(f"AAL3 unique labels: {len(np.unique(aal_data))}")

# --- Verify all AAL3 labels exist ---
print("\nVerifying network labels:")
missing = []
for name, label in NETWORK.items():
    n = int((aal_data == label).sum())
    status = "✓" if n > 0 else "✗ MISSING"
    print(f"  {label:4d} {name:28s} {n:6d} voxels  {status}")
    if n == 0:
        missing.append((name, label))

if missing:
    print(f"\n⚠️ {len(missing)} labels missing — check AAL3 label file")
    raise SystemExit(1)

# --- Resample to BOLD space ---
ref_bold = nib.load(str(BOLD_REF))
ref_3d = nib.Nifti1Image(
    ref_bold.get_fdata()[..., 0].astype(np.float32),
    ref_bold.affine, ref_bold.header
)

aal_resampled = resample_to_img(
    aal, ref_3d,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
aal_res_data = aal_resampled.get_fdata()

# --- Build sub-network atlas: assign label 1-4 per sub-network ---
net_data = np.zeros_like(aal_res_data, dtype=np.int16)
for subnet_label, info in SUBNETWORKS.items():
    for aal3_label in info["aal3_labels"]:
        net_data[aal_res_data == aal3_label] = subnet_label

# --- Intersect with GM mask ---
gm = nib.load(str(GM_MASK)).get_fdata().astype(bool)
net_data[~gm] = 0

# --- Report per-subnetwork voxel counts ---
print("\nFinal sub-network voxel counts (BOLD space, GM-masked):")
for subnet_label, info in SUBNETWORKS.items():
    n = int((net_data == subnet_label).sum())
    print(f"  {subnet_label} {info['name']:15s} {n:6d} voxels")

total_vox = int((net_data > 0).sum())
print(f"\nTotal network voxels: {total_vox}")

# --- Save atlas (4 values: 1-4) ---
net_img = nib.Nifti1Image(net_data, ref_3d.affine, ref_3d.header)
out_atlas = OUT_DIR / "motor_inhibition_subnetwork_atlas2.nii.gz"
net_img.to_filename(str(out_atlas))

# Binary mask for convenience
net_bin = nib.Nifti1Image((net_data > 0).astype(np.uint8),
                          ref_3d.affine, ref_3d.header)
out_bin = OUT_DIR / "motor_inhibition_network_mask2.nii.gz"
net_bin.to_filename(str(out_bin))

# Save label mapping as TSV
map_tsv = OUT_DIR / "motor_inhibition_subnetwork_labels2.tsv"
with open(map_tsv, "w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["label", "name", "aal3_labels"])
    for subnet_label, info in SUBNETWORKS.items():
        w.writerow([subnet_label, info["name"], str(info["aal3_labels"])])

print(f"\n✓ Saved atlas:  {out_atlas}")
print(f"✓ Saved mask:   {out_bin}")
print(f"✓ Saved labels: {map_tsv}")
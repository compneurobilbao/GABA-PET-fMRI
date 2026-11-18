#!/usr/bin/env python3
"""
Check if L>R asymmetry is global (artifact) or specific to motor network.
Tests several AAL3 regions including visual/auditory areas that should be
relatively symmetric.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from scipy import stats

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
AAL3_PATH   = Path.home() / "projects/atlases/AAL3/AAL3_MNI_V7_2mm.nii"
PET_DIR     = Path.home() / "projects/14koen_cuypers/final_analysis/derivatives/pet_gm_masked"

# Control regions that should be approximately symmetric in healthy adults
CONTROL_REGIONS = {
    "Calcarine":   {"L": 45, "R": 46},   # primary visual
    "Lingual":     {"L": 47, "R": 48},   # visual
    "Cuneus":      {"L": 49, "R": 50},   # visual
    "Heschl":      {"L": 87, "R": 88},   # primary auditory
    "Hippocampus": {"L": 41, "R": 42},   # memory
    "Thalamus":    {"L": 121, "R": 122}, # thalamus (check AAL3 manual if labels different)
    # Plus the motor network for comparison
    "M1":          {"L": 1,  "R": 2},
    "Precuneus":   {"L": 71, "R": 72},   # default mode
}

aal_img = nib.load(str(AAL3_PATH))
first_pet = sorted(PET_DIR.glob("sub-*_DVR_GM.nii.gz"))[0]
pet_ref = nib.load(str(first_pet))
aal_pet = resample_to_img(aal_img, pet_ref, interpolation="nearest",
                          force_resample=True, copy_header=True)
aal_data = aal_pet.get_fdata().astype(np.int16)

pet_files = sorted(PET_DIR.glob("sub-*_DVR_GM.nii.gz"))
print(f"Found {len(pet_files)} PET files")

rows = []
for pet_file in pet_files:
    sid = pet_file.stem.replace("sub-", "").replace("_DVR_GM.nii", "")
    pet_data = nib.load(str(pet_file)).get_fdata(dtype=np.float32)
    r = {"subject_id": sid}
    for region_name, labels in CONTROL_REGIONS.items():
        L_mask = aal_data == labels["L"]
        R_mask = aal_data == labels["R"]
        L_vals = pet_data[L_mask]
        R_vals = pet_data[R_mask]
        # Use only nonzero voxels (within GM mask)
        L_vals = L_vals[L_vals > 0]
        R_vals = R_vals[R_vals > 0]
        if len(L_vals) > 10 and len(R_vals) > 10:
            r[f"{region_name}_L"] = float(L_vals.mean())
            r[f"{region_name}_R"] = float(R_vals.mean())
            r[f"{region_name}_n_L"] = len(L_vals)
            r[f"{region_name}_n_R"] = len(R_vals)
        else:
            r[f"{region_name}_L"] = np.nan
            r[f"{region_name}_R"] = np.nan
    rows.append(r)

df = pd.DataFrame(rows)

print("\n" + "="*90)
print("LATERALIZATION TEST: motor network vs control regions")
print("="*90)
print(f"{'Region':<14s} {'mean_L':>8s} {'mean_R':>8s} {'L-R':>8s} {'t':>7s} {'p':>9s} {'d':>7s}")
print("-"*90)

for region_name in CONTROL_REGIONS.keys():
    L = df[f"{region_name}_L"].values
    R = df[f"{region_name}_R"].values
    L = L[~np.isnan(L)]
    R = R[~np.isnan(R)]
    if len(L) < 3: continue
    diff = L - R
    t, p = stats.ttest_rel(L, R)
    d = diff.mean() / diff.std(ddof=1)
    nL = df[f"{region_name}_n_L"].mean() if f"{region_name}_n_L" in df.columns else np.nan
    nR = df[f"{region_name}_n_R"].mean() if f"{region_name}_n_R" in df.columns else np.nan
    pct_diff = 100 * (L.mean() - R.mean()) / ((L.mean() + R.mean())/2)
    print(f"{region_name:<14s} {L.mean():>8.3f} {R.mean():>8.3f} {diff.mean():>+8.3f} "
          f"{t:>+7.2f} {p:>9.2e} {d:>+7.2f}  ({pct_diff:+.1f}%)")

print("\nVoxel counts (mean across subjects):")
for region_name in CONTROL_REGIONS.keys():
    if f"{region_name}_n_L" in df.columns:
        nL = df[f"{region_name}_n_L"].mean()
        nR = df[f"{region_name}_n_R"].mean()
        print(f"  {region_name:<14s}  L={nL:.0f}  R={nR:.0f}  ({100*(nL-nR)/((nL+nR)/2):+.1f}%)")
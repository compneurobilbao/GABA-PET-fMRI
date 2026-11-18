#!/usr/bin/env python3
"""
Lateralization DVR (corrected): exclude zero-DVR voxels before averaging.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_DIR  = Path.home() / "projects/14koen_cuypers/final_analysis2"
AAL3_PATH    = Path.home() / "projects/atlases/AAL3/AAL3_MNI_V7_2mm.nii"

PET_CANDIDATES = [
    PROJECT_DIR / "derivatives/pet_gm_masked",
    Path.home() / "projects/14koen_cuypers/final_analysis/derivatives/pet_gm_masked",
]
PET_DIR = next((p for p in PET_CANDIDATES if p.exists() and any(p.glob("sub-*_DVR_GM.nii.gz"))), None)
if PET_DIR is None:
    raise SystemExit("No PET directory found")
print(f"Using PET dir: {PET_DIR}")

BEHAV_FILE   = PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv"
OUT = PROJECT_DIR / "results2/lateralization"
OUT.mkdir(parents=True, exist_ok=True)

SUBNETS_LR = {
    "M1":       {"L": [1],       "R": [2]},
    "IFG":      {"L": [7, 9],    "R": [8, 10]},
    "SMA":      {"L": [15, 153], "R": [16, 154]},
    "Striatum": {"L": [75, 77],  "R": [76, 78]},
}

behav = pd.read_csv(BEHAV_FILE, sep="\t")
behav.columns = behav.columns.str.strip()
behav["subject_id"] = behav["subject_id"].astype(str).str.strip()
behav["group"] = behav["group"].str.strip()
groups_df = behav[["subject_id", "group"]].copy()
subj_list = groups_df["subject_id"].tolist()
print(f"n = {len(subj_list)}")

aal_img = nib.load(str(AAL3_PATH))
first_pet = sorted(PET_DIR.glob("sub-*_DVR_GM.nii.gz"))[0]
pet_ref = nib.load(str(first_pet))
aal_pet = resample_to_img(aal_img, pet_ref, interpolation="nearest",
                          force_resample=True, copy_header=True)
aal_data = aal_pet.get_fdata().astype(np.int16)
print(f"AAL3 resampled to PET space ({pet_ref.shape})")

rows = []
for sid in subj_list:
    pet_file = PET_DIR / f"sub-{sid}_DVR_GM.nii.gz"
    if not pet_file.exists():
        print(f"[{sid}] PET MISSING, skipping")
        continue
    pet_data = nib.load(str(pet_file)).get_fdata(dtype=np.float32)
    r = {"subject_id": sid}
    for sname, hemi in SUBNETS_LR.items():
        for side, labels in hemi.items():
            mask = np.isin(aal_data, labels)
            vals = pet_data[mask]
            # IMPORTANT: exclude zero-DVR voxels (outside GM mask)
            vals = vals[vals > 0]
            r[f"dvr_{sname}_{side}"] = float(vals.mean()) if len(vals) > 10 else np.nan
            r[f"nvox_{sname}_{side}"] = int(len(vals))
    rows.append(r)

if not rows:
    raise SystemExit("No subjects processed")
print(f"\nProcessed {len(rows)} subjects")

dvr_lr = pd.DataFrame(rows).merge(groups_df, on="subject_id")
dvr_lr.to_csv(OUT / "dvr_LR_per_subject_corrected.tsv",
              sep="\t", index=False, float_format="%.4f")

print("\nVoxel counts (mean per region across subjects):")
for sname in SUBNETS_LR.keys():
    nL = dvr_lr[f"nvox_{sname}_L"].mean()
    nR = dvr_lr[f"nvox_{sname}_R"].mean()
    print(f"  {sname:10s}  L={nL:.0f}  R={nR:.0f}  ({100*(nL-nR)/((nL+nR)/2):+.1f}%)")

def paired_test(sub_df, label):
    res = []
    for sname in SUBNETS_LR.keys():
        L = sub_df[f"dvr_{sname}_L"].values
        R = sub_df[f"dvr_{sname}_R"].values
        diff = L - R
        LI = (L - R) / (L + R)
        t, p = stats.ttest_rel(L, R)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        res.append({
            "subset":         label,
            "n":              len(sub_df),
            "subnetwork":     sname,
            "mean_L":         L.mean(),
            "mean_R":         R.mean(),
            "mean_diff_L-R":  diff.mean(),
            "pct_diff":       100 * diff.mean() / ((L.mean() + R.mean()) / 2),
            "mean_LI":        LI.mean(),
            "t":              t,
            "p":              p,
            "cohens_d":       d,
        })
    return res

all_rows = []
all_rows.extend(paired_test(dvr_lr,                              "ALL"))
all_rows.extend(paired_test(dvr_lr[dvr_lr["group"] == "OLD"],   "OLD"))
all_rows.extend(paired_test(dvr_lr[dvr_lr["group"] == "YOUNG"], "YOUNG"))

results = pd.DataFrame(all_rows)
results.to_csv(OUT / "dvr_lateralization_corrected.tsv",
               sep="\t", index=False, float_format="%.4f")

print("\n" + "="*100)
print("CORRECTED PAIRED T-TEST DVR — L vs R per subnetwork (zero voxels excluded)")
print("="*100)
for subset in ["ALL", "OLD", "YOUNG"]:
    sub = results[results["subset"] == subset]
    print(f"\n[{subset}, n={sub.iloc[0]['n']}]")
    print(sub[["subnetwork", "mean_L", "mean_R", "mean_diff_L-R",
               "pct_diff", "t", "p", "cohens_d"]
              ].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# --- Visualization (same as before) ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")
subnets = list(SUBNETS_LR.keys())

for col, sname in enumerate(subnets):
    ax = axes[0, col]
    positions, data, labels = [], [], []
    for i, (label, sub) in enumerate([("ALL", dvr_lr),
                                       ("OLD", dvr_lr[dvr_lr["group"]=="OLD"]),
                                       ("YOUNG", dvr_lr[dvr_lr["group"]=="YOUNG"])]):
        L = sub[f"dvr_{sname}_L"].values
        R = sub[f"dvr_{sname}_R"].values
        positions.extend([i*3 + 0.4, i*3 + 1.2])
        data.extend([L, R])
        labels.extend([f"{label}\nL", f"{label}\nR"])
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for j, box in enumerate(bp["boxes"]):
        box.set_facecolor("#1f77b4" if j%2==0 else "#d62728")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{sname}: DVR L vs R", fontsize=13, fontweight="bold")
    if col == 0:
        ax.set_ylabel("DVR")

for col, sname in enumerate(subnets):
    ax = axes[1, col]
    for label, sub_df, color in [("OLD", dvr_lr[dvr_lr["group"]=="OLD"], "#620d5d"),
                                  ("YOUNG", dvr_lr[dvr_lr["group"]=="YOUNG"], "#1fb453")]:
        L = sub_df[f"dvr_{sname}_L"].values
        R = sub_df[f"dvr_{sname}_R"].values
        ax.scatter(L, R, c=color, label=label, s=60, alpha=0.8,
                   edgecolor="white", linewidth=1)
    vmin = min(dvr_lr[f"dvr_{sname}_L"].min(), dvr_lr[f"dvr_{sname}_R"].min())
    vmax = max(dvr_lr[f"dvr_{sname}_L"].max(), dvr_lr[f"dvr_{sname}_R"].max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, label="L=R")
    ax.set_xlabel(f"DVR L")
    if col == 0:
        ax.set_ylabel(f"DVR R")
    ax.set_title(f"{sname}: paired L vs R", fontsize=13, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "dvr_lateralization.png", dpi=150)
print(f"\n✓ Saved: {OUT}/dvr_LR_per_subject.tsv")
print(f"✓ Saved: {OUT}/dvr_lateralization_ttest.tsv")
print(f"✓ Saved: {OUT}/dvr_lateralization.png")
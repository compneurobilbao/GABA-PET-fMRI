#!/usr/bin/env python3
"""
Lateralization check — degree per hemisphere per sub-network.

Recomputes node-to-node FC (14×14 matrix per subject) and derives:
  degree_<network>_L = sum of Fisher-z between left node(s) of <network>
                       and left nodes of all other 3 networks
  degree_<network>_R = same, right hemisphere

Then runs paired t-test L vs R within ALL / OLD / YOUNG per sub-network.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from scipy import stats
import matplotlib.pyplot as plt
import time, gc

PROJECT_DIR    = Path.home() / "projects/14koen_cuypers/final_analysis2"
AAL3_PATH      = Path.home() / "projects/atlases/AAL3/AAL3_MNI_V7_2mm.nii"
FMRI_CLEAN_DIR = PROJECT_DIR / "derivatives/fmri_cleaned"
BOLD_REF       = FMRI_CLEAN_DIR / "sub-O01_cleaned_bold.nii.gz"
GM_MASK        = PROJECT_DIR / "derivatives/masks/gm_mask_fmri_space.nii.gz"
BEHAV_FILE     = PROJECT_DIR / "data/behavioral_data/behav_brain_table_n30.tsv"

OUT = PROJECT_DIR / "results2/lateralization"
OUT.mkdir(parents=True, exist_ok=True)

# --- Individual AAL3 nodes ---
NODES = {
    "Precentral_L":       1,    "Precentral_R":       2,
    "Frontal_Inf_Oper_L": 7,    "Frontal_Inf_Oper_R": 8,
    "Frontal_Inf_Tri_L":  9,    "Frontal_Inf_Tri_R":  10,
    "Supp_Motor_Area_L":  15,   "Supp_Motor_Area_R":  16,
    "ACC_pre_L":          153,  "ACC_pre_R":          154,
    "Caudate_L":          75,   "Caudate_R":          76,
    "Putamen_L":          77,   "Putamen_R":          78,
}

# Sub-network membership per hemisphere
SUBNETS_NODES = {
    "M1":       {"L": ["Precentral_L"],          "R": ["Precentral_R"]},
    "IFG":      {"L": ["Frontal_Inf_Oper_L", "Frontal_Inf_Tri_L"],
                 "R": ["Frontal_Inf_Oper_R", "Frontal_Inf_Tri_R"]},
    "SMA":      {"L": ["Supp_Motor_Area_L", "ACC_pre_L"],
                 "R": ["Supp_Motor_Area_R", "ACC_pre_R"]},
    "Striatum": {"L": ["Caudate_L", "Putamen_L"],
                 "R": ["Caudate_R", "Putamen_R"]},
}

# --- Load behavioral (for group info) ---
behav = pd.read_csv(BEHAV_FILE, sep="\t")
behav.columns = behav.columns.str.strip()
behav["subject_id"] = behav["subject_id"].astype(str).str.strip()
behav["group"] = behav["group"].str.strip()
groups_df = behav[["subject_id", "group"]].copy()
subj_list = groups_df["subject_id"].tolist()
print(f"n = {len(subj_list)}")

# --- Resample AAL3 to BOLD space ---
ref = nib.load(str(BOLD_REF))
ref_3d = nib.Nifti1Image(ref.get_fdata()[..., 0].astype(np.float32),
                          ref.affine, ref.header)
aal = nib.load(str(AAL3_PATH))
aal_res = resample_to_img(aal, ref_3d, interpolation="nearest",
                          force_resample=True, copy_header=True).get_fdata().astype(np.int16)
gm = nib.load(str(GM_MASK)).get_fdata().astype(bool)

# Build per-node voxel masks (in BOLD space, GM-masked)
node_masks = {}
print("\nNode voxel counts:")
for nname, label in NODES.items():
    mask = (aal_res == label) & gm
    node_masks[nname] = mask
    print(f"  {nname:24s} ({label:4d}): {int(mask.sum())} voxels")

# --- Compute node-to-node FC matrix per subject ---
print("\nComputing FC matrices per subject...")
NODE_LIST = list(NODES.keys())
n_nodes = len(NODE_LIST)

fc_matrices = {}   # sid -> (14×14) Fisher-z matrix
for sid in subj_list:
    start = time.time()
    bold_file = FMRI_CLEAN_DIR / f"sub-{sid}_cleaned_bold.nii.gz"
    if not bold_file.exists():
        print(f"[{sid}] MISSING, skipping")
        continue

    bold_img  = nib.load(str(bold_file))
    bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)
    del bold_img

    # Extract mean time-series per node
    ts = np.zeros((n_nodes, bold_data.shape[-1]), dtype=np.float32)
    for i, nname in enumerate(NODE_LIST):
        m = node_masks[nname]
        if m.sum() < 10:
            ts[i, :] = np.nan
            continue
        ts[i, :] = bold_data[m].mean(axis=0)

    del bold_data
    gc.collect()

    # Pearson correlations
    R = np.corrcoef(ts)
    R = np.clip(R, -0.9999, 0.9999)
    Z = np.arctanh(R)                          # Fisher-z
    np.fill_diagonal(Z, 0.0)                   # zero diagonal
    fc_matrices[sid] = Z

    print(f"  [{sid}]  {time.time()-start:.1f}s")

# --- Compute degree_L and degree_R per sub-network per subject ---
def hemi_degree(Z, target_nodes, other_nodes_same_hemi):
    """Sum of Fisher-z from target nodes to all other-hemisphere nodes."""
    target_idx = [NODE_LIST.index(n) for n in target_nodes]
    other_idx  = [NODE_LIST.index(n) for n in other_nodes_same_hemi]
    if not target_idx or not other_idx:
        return np.nan
    # Mean over target nodes × sum across other-network same-hemi nodes
    vals = Z[np.ix_(target_idx, other_idx)]
    return float(vals.mean(axis=1).sum())   # sum across other networks, mean within multi-node subnet

# Build the per-hemisphere "other-network" node lists
def others_same_hemi(sname, side):
    nodes = []
    for other_name, hemi in SUBNETS_NODES.items():
        if other_name == sname:
            continue
        nodes.extend(hemi[side])
    return nodes

rows = []
for sid in subj_list:
    if sid not in fc_matrices:
        continue
    Z = fc_matrices[sid]
    r = {"subject_id": sid}
    for sname, hemi in SUBNETS_NODES.items():
        for side in ["L", "R"]:
            tgt = hemi[side]
            oth = others_same_hemi(sname, side)
            r[f"degree_{sname}_{side}"] = hemi_degree(Z, tgt, oth)
    rows.append(r)

deg_lr = pd.DataFrame(rows).merge(groups_df, on="subject_id")
deg_lr.to_csv(OUT / "degree_LR_per_subject.tsv",
              sep="\t", index=False, float_format="%.4f")

# --- Paired t-tests ---
def paired_test(sub_df, label):
    res = []
    for sname in SUBNETS_NODES.keys():
        L = sub_df[f"degree_{sname}_L"].values
        R = sub_df[f"degree_{sname}_R"].values
        diff = L - R
        # LI may behave weirdly with Fisher-z degrees (can include 0 or negative)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = L + R
            LI = np.where(np.abs(denom) > 1e-6, (L - R) / denom, np.nan)
        t, p = stats.ttest_rel(L, R)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        res.append({
            "subset":        label,
            "n":             len(sub_df),
            "subnetwork":    sname,
            "mean_L":        L.mean(),
            "mean_R":        R.mean(),
            "mean_diff_L-R": diff.mean(),
            "mean_LI":       np.nanmean(LI),
            "t":             t,
            "p":             p,
            "cohens_d":      d,
            "interpretation": (
                "L > R (sig)" if (p < 0.05 and t > 0) else
                "R > L (sig)" if (p < 0.05 and t < 0) else
                "no asymmetry"
            ),
        })
    return res

all_rows = []
all_rows.extend(paired_test(deg_lr,                              "ALL"))
all_rows.extend(paired_test(deg_lr[deg_lr["group"] == "OLD"],   "OLD"))
all_rows.extend(paired_test(deg_lr[deg_lr["group"] == "YOUNG"], "YOUNG"))

results = pd.DataFrame(all_rows)
results.to_csv(OUT / "degree_lateralization_ttest.tsv",
               sep="\t", index=False, float_format="%.4f")

print("\n" + "="*100)
print("PAIRED T-TEST DEGREE — L vs R per subnetwork")
print("(Note: degree_L computed within left hemisphere only, same for R)")
print("="*100)
for subset in ["ALL", "OLD", "YOUNG"]:
    sub = results[results["subset"] == subset]
    print(f"\n[{subset}, n={sub.iloc[0]['n']}]")
    print(sub[["subnetwork", "mean_L", "mean_R", "mean_diff_L-R",
               "mean_LI", "t", "p", "cohens_d", "interpretation"]
              ].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# --- Visualization ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")
subnets = list(SUBNETS_NODES.keys())

for col, sname in enumerate(subnets):
    ax = axes[0, col]
    positions = []
    data = []
    labels = []
    for i, (label, sub) in enumerate([("ALL", deg_lr),
                                       ("OLD", deg_lr[deg_lr["group"]=="OLD"]),
                                       ("YOUNG", deg_lr[deg_lr["group"]=="YOUNG"])]):
        L = sub[f"degree_{sname}_L"].values
        R = sub[f"degree_{sname}_R"].values
        positions.extend([i*3 + 0.4, i*3 + 1.2])
        data.extend([L, R])
        labels.extend([f"{label}\nL", f"{label}\nR"])
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for j, box in enumerate(bp["boxes"]):
        box.set_facecolor("#1f77b4" if j%2==0 else "#d62728")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{sname}: FC strength L vs R", fontsize=13, fontweight="bold")
    if col == 0:
        ax.set_ylabel("FC strength (sum Fisher-z)")

for col, sname in enumerate(subnets):
    ax = axes[1, col]
    for label, sub_df, color in [("OLD", deg_lr[deg_lr["group"]=="OLD"], "#620d5d"),
                                  ("YOUNG", deg_lr[deg_lr["group"]=="YOUNG"], "#1fb453")]:
        L = sub_df[f"degree_{sname}_L"].values
        R = sub_df[f"degree_{sname}_R"].values
        ax.scatter(L, R, c=color, label=label, s=60, alpha=0.8,
                   edgecolor="white", linewidth=1)
    vmin = min(deg_lr[f"degree_{sname}_L"].min(), deg_lr[f"degree_{sname}_R"].min())
    vmax = max(deg_lr[f"degree_{sname}_L"].max(), deg_lr[f"degree_{sname}_R"].max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, label="L=R")
    ax.set_xlabel(f"FC strength L")
    if col == 0:
        ax.set_ylabel(f"FC strength R")
    ax.set_title(f"{sname}: paired L vs R", fontsize=13, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "degree_lateralization.png", dpi=150)
print(f"\n✓ Saved: {OUT}/degree_LR_per_subject.tsv")
print(f"✓ Saved: {OUT}/degree_lateralization_ttest.tsv")
print(f"✓ Saved: {OUT}/degree_lateralization.png")
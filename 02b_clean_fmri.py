#!/usr/bin/env python3
"""
Step 2b: Confound regression + temporal filtering + spatial smoothing
for resting-state fMRI data (DeepPrep output).

For each subject:
  - Build confound matrix: Friston-24 motion + 5 aCompCor + spike regressors
  - Apply nilearn.image.clean_img: detrend, confound regression, bandpass
  - Apply 6mm FWHM Gaussian smoothing
  - Save cleaned 4D BOLD to derivatives/fmri_cleaned/

Denoising strategy follows standard FC practice (Power 2012, Ciric 2017):
- 24 Friston motion parameters
- First 5 aCompCor components
- Spike regression for FD > 0.5mm
- Bandpass 0.01-0.08 Hz
- 6mm FWHM smoothing (post-cleaning)
- No GSR (sensitivity analysis can be added later)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import clean_img, smooth_img
from nilearn.masking import apply_mask, unmask
import time

# --- Paths ---
PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
FMRI_DIR = PROJECT_DIR / "data/fmri_data"
GM_MASK = PROJECT_DIR / "derivatives/masks/gm_mask_fmri_space.nii.gz"
OUT_DIR = PROJECT_DIR / "derivatives/fmri_cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = PROJECT_DIR / "results/QC/fmri_cleaning_log.tsv"

# --- Parameters ---
TR = 1.7                # seconds
LOW_PASS = 0.08         # Hz
HIGH_PASS = 0.01        # Hz
SMOOTH_FWHM = 6.0       # mm
N_ACOMPCOR = 5          # first 5 aCompCor components
FD_SPIKE_THRESH = 0.5   # mm, threshold for spike regressors

# --- Friston-24 column names (from DeepPrep confound TSV) ---
FRISTON24_COLS = []
for param in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]:
    FRISTON24_COLS.extend([
        param,
        f"{param}_derivative1",
        f"{param}_power2",
        f"{param}_derivative1_power2",
    ])
# 6 params × 4 versions = 24 regressors

ACOMPCOR_COLS = [f"a_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)]

# --- Build subject list ---
subjects = []
for code in ["O", "Y"]:
    group = "OLD" if code == "O" else "YOUNG"
    for i in range(1, 16):
        subjects.append((f"{code}{i:02d}", group))

# --- Process each subject ---
log_rows = []

for sid, group in subjects:
    start = time.time()
    bold_file = FMRI_DIR / f"sub-{sid}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz"
    conf_file = FMRI_DIR / f"sub-{sid}_task-rest_desc-confounds_timeseries.tsv"
    out_file = OUT_DIR / f"sub-{sid}_cleaned_bold.nii.gz"

    if not bold_file.exists() or not conf_file.exists():
        print(f"[{sid}] MISSING FILES, skipping")
        continue

    print(f"\n[{sid}] ({group}) cleaning...")

    # --- Load confounds ---
    conf = pd.read_csv(conf_file, sep="\t")

    # Friston-24
    missing_f24 = [c for c in FRISTON24_COLS if c not in conf.columns]
    if missing_f24:
        print(f"  WARNING: missing Friston-24 cols: {missing_f24}")
    f24 = conf[[c for c in FRISTON24_COLS if c in conf.columns]].copy()

    # aCompCor
    acc = conf[[c for c in ACOMPCOR_COLS if c in conf.columns]].copy()

    # Spike regressors for FD > threshold
    fd = conf["framewise_displacement"].values
    # First volume FD is NaN -> treat as 0 (no motion by definition)
    fd = np.nan_to_num(fd, nan=0.0)
    spike_idx = np.where(fd > FD_SPIKE_THRESH)[0]
    n_spikes = len(spike_idx)

    spike_regs = pd.DataFrame()
    for j, idx in enumerate(spike_idx):
        reg = np.zeros(len(conf))
        reg[idx] = 1
        spike_regs[f"spike_{j:03d}"] = reg

    # Concatenate all confounds
    confounds = pd.concat([f24, acc, spike_regs], axis=1)
    # Replace any remaining NaNs (first-volume derivatives) with 0
    confounds = confounds.fillna(0)

    n_confounds = confounds.shape[1]
    n_vols = len(conf)
    df_remaining = n_vols - n_confounds
    print(f"  n_vols={n_vols}, n_confounds={n_confounds} "
          f"(F24={f24.shape[1]}, aCC={acc.shape[1]}, spikes={n_spikes}), "
          f"df_remaining={df_remaining}")

    if df_remaining < 50:
        print(f"  WARNING: low remaining df ({df_remaining}) for {sid}")

    # --- Clean: detrend + regress confounds + bandpass ---
    # nilearn.image.clean_img applies per-voxel: detrend -> regress -> filter -> standardize
    cleaned = clean_img(
        str(bold_file),
        detrend=True,
        standardize=False,        # keep units for interpretability of FC
        confounds=confounds.values,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        t_r=TR,
        mask_img=str(GM_MASK),    # restrict to GM
    )

    # --- Smooth ---
    smoothed = smooth_img(cleaned, fwhm=SMOOTH_FWHM)

    # --- Save ---
    smoothed.to_filename(str(out_file))

    elapsed = time.time() - start
    print(f"  ✓ saved {out_file.name} ({elapsed:.1f}s)")

    log_rows.append({
        "subject_id": sid,
        "group": group,
        "n_vols": n_vols,
        "n_f24": f24.shape[1],
        "n_acompcor": acc.shape[1],
        "n_spikes": n_spikes,
        "n_confounds_total": n_confounds,
        "df_remaining": df_remaining,
        "elapsed_sec": elapsed,
    })

# --- Save log ---
log_df = pd.DataFrame(log_rows)
log_df.to_csv(LOG_FILE, sep="\t", index=False, float_format="%.2f")

print("\n" + "="*70)
print("CLEANING COMPLETE")
print("="*70)
print(log_df.to_string(index=False))
print(f"\nLog saved: {LOG_FILE}")
print(f"Cleaned BOLD files: {OUT_DIR}")
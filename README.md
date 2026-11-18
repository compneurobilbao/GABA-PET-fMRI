# Simultaneous GABA-PET and fMRI in motor inhibition network

![Cover](images/infografia.webp)

This repository contains the analysis code for the research paper "Simultaneous PET-fMRI reveals combined GABA-A and functional connectivity contributions to motor inhibition in aging."

## Overview

This project investigates whether GABA-A receptor availability (PET) and functional connectivity (fMRI) contribute independently to motor inhibition, explaining more than age alone.


## Repository Structure

```
GABA-PET-fMRI/
├── images/            # Figures used in this README
├── scripts/           # Analysis code
├── README.md
└── requirements.txt
```

The analysis scripts in `scripts/` should be executed in the following order:

- `01_mask_pet.sh` – Masks PET DVR maps with the gray-matter mask and extracts mean GM DVR values for QC.
- `01b_qc_plot.py` – Visualizes PET gray-matter DVR distributions and identifies outliers.
- `02a_motion_qc.py` – Quantifies fMRI framewise displacement, flags motion, and summarizes motion across groups.
- `02b_clean_fmri.py` – Regresses Friston-24 motion, aCompCor, and spike confounds; filters and smooths resting-state fMRI.
- `03_build_motor_inhibition_network.py` – Builds an AAL3 atlas containing M1, IFG, SMA, and striatum subnetwork masks.
- `04a_corr_between_networks.py` – Computes pairwise Fisher-z-transformed functional connectivity between motor-inhibition subnetworks.
- `04b_subnetwork_degree_dvr.py` – Calculates subnetwork weighted degree and mean DVR for each participant.
- `04c_predictor_corr_map.py` – Generates correlation matrices for the subnetwork degree and DVR predictors.
- `04d_subnetwork_degreedvr_group_diff.py` – Compares subnetwork DVR and functional connectivity between age groups.
- `05a_degreedvr_model.py` – Fits OLS models predicting motor-inhibition scores from age, degree, and DVR.
- `05b_model_comparison.py` – Compares age-plus-degree, age-plus-DVR, and combined degree–DVR models using likelihood-ratio tests.
- `05b2_correction_fdr.py` – Applies confirmatory-test and FDR correction strategies to model-comparison p-values.
- `05c_FC_vs_coupling.py` – Compares functional-connectivity-only and connectivity×DVR coupling-only models.
- `05d_groupwise_degreedvr_models.py` – Fits degree-only and DVR-only models separately within young and older groups.
- `05e_vif_check.py` – Performs block-wise VIF analysis to assess redundancy between degree and DVR predictors.
- `06_figuras.py` – Creates predicted-versus-observed plots for nested degree and DVR models.
- `06b_corrmatrix_fig.py` – Generates publication-quality predictor correlation matrices for the full sample and age groups.
- `07_sex_effect.py` – Tests sex differences in DVR and subnetwork degree measures.
- `08_models_vs_age.py` – Compares age-only and age-plus-brain models using adjusted R² and likelihood-ratio tests.
- `08b_models_vs_age_vs_group.py` – Compares models using continuous age versus categorical age group.
- `08c_cross_validation.py` – Estimates nested-model performance using leave-one-out cross-validation.
- `08d_bootstrap_optimism.py` – Applies bootstrap optimism correction to adjusted R² estimates.
- `08e_5foldCV.py` – Estimates nested-model performance using repeated five-fold cross-validation.
- `09_forest_plots.py` – Generates coefficient tables and forest plots for degree, DVR, and combined models.
- `10_pca_models.py` – Reduces degree and DVR predictors using PCA and compares PCA-based models.
- `11_no_m1.py` – Performs a sensitivity analysis excluding M1 from the degree and DVR models.
- `12_barplot.py` – Plots adjusted R² with bootstrap confidence intervals for the main models.
- `13_barplot_m2.py` – Compares full-model performance against age-plus-degree and age-plus-DVR models.
- `14a_lateralization_dvr.py` – Computes left-versus-right subnetwork DVR and performs lateralization tests.
- `14b_check_lat_artifact.py` – Checks whether DVR lateralization reflects a global imaging artifact or network-specific asymmetry.
- `14c_lateralization_degree.py` – Computes and compares left-versus-right subnetwork functional degrees.
- `14d_multiplecomparison_correction.py` – Applies FDR correction to lateralization test p-values.
- `15_hemisphere_separate_models.py` – Fits degree and DVR models separately for left and right hemispheres.
- `15b_hemisphere_plots.py` – Visualizes hemisphere-specific model performance and predicted-versus-observed relationships.

## Important Notes

**This is not an automated pipeline.** These scripts are provided as a reference implementation rather than a ready-to-run analysis pipeline.

* No data are included in this repository
* Acquisition protocols and preprocessing steps may differ between sites and scanners
* The scripts assume you have your own preprocessed PET and fMRI data in the appropriate format
* The code is commented to explain each analytical step
* You will need to adapt file paths and data structures to your own setup

## Requirements

The code requires [Python 3.x] and the following libraries:

* [NumPy]
* [Pandas]
* [SciPy]
* [Matplotlib]
* [Nibabel / Nilearn / ...]

Please see `requirements.txt` for the full list.


## Contact

For questions regarding the analysis methods or code implementation, please open an issue in this repository or contact [razkinmalen@gmail.com](mailto:razkinmalen@gmail.com).

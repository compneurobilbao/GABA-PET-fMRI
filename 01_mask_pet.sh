#!/bin/bash
# Step 1: Apply GM mask to all PET DVR maps and extract mean GM DVR per subject

set -e  # exit on error

# Paths
PROJECT_DIR=~/projects/14koen_cuypers/final_analysis2
PET_DIR=${PROJECT_DIR}/data/pet_data
MASK=${PROJECT_DIR}/derivates/masks/gm_mask_pet_space.nii.gz
OUT_DIR=${PROJECT_DIR}/derivates/pet_gm_masked
QC_FILE=${PROJECT_DIR}/results/QC/qc_mean_gm_dvr.tsv

mkdir -p ${OUT_DIR}
mkdir -p ${PROJECT_DIR}/results/QC

# Header for QC file
echo -e "subject_id\tgroup\tmean_gm_dvr" > ${QC_FILE}

# Loop over all subjects (OLD then YOUNG)
for group_code in O Y; do
    if [ "$group_code" = "O" ]; then
        group_label="OLD"
    else
        group_label="YOUNG"
    fi
    
    for i in $(seq -w 01 15); do
        subj="${group_code}${i}"
        in_pet=${PET_DIR}/sCOR_${subj}_DVR_PVC_MRTM2_pons.nii
        out_pet=${OUT_DIR}/sub-${subj}_DVR_GM.nii.gz
        
        if [ ! -f "$in_pet" ]; then
            echo "WARNING: ${in_pet} not found, skipping"
            continue
        fi
        
        # Apply GM mask
        fslmaths ${in_pet} -mas ${MASK} ${out_pet}
        
        # Extract mean of non-zero voxels (i.e., within GM mask)
        mean_dvr=$(fslstats ${out_pet} -M)
        
        echo -e "${subj}\t${group_label}\t${mean_dvr}" >> ${QC_FILE}
        echo "Processed ${subj} (${group_label}): mean GM DVR = ${mean_dvr}"
    done
done

echo ""
echo "Done. QC file saved to: ${QC_FILE}"
echo "Masked PET maps saved to: ${OUT_DIR}"
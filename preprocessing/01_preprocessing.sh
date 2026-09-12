#!/bin/bash
# 01_preprocessing.sh
# Step 2 and 3: Remove alpha bands and rescale to reflectance values
# Author: __REDACTED_USERNAME__
# Date: 2026-06-08

module load bear-apps/2023a/live
module load QGIS/3.40.2-foss-2023a

IN_DIR="__REDACTED_CLUSTER_PROJECT__/Imaging_data"
OUT_DIR="__REDACTED_CLUSTER_PROJECT__/__REDACTED_WORKSPACE__/data_processed"

echo "Step 2: Removing alpha bands..."
for band in Green_1 Red_1 NIR_1 RedEdge_1 Green_2 Red_2 NIR_2 RedEdge_2; do
    gdal_translate -b 1 "${IN_DIR}/${band}.tif" "${OUT_DIR}/${band}_noalpha.tif"
    echo "Alpha removed: ${band}"
done

echo "Step 3: Rescaling to reflectance..."
for band in Green_1 Red_1 NIR_1 RedEdge_1 Green_2 Red_2 NIR_2 RedEdge_2; do
    gdal_calc.py \
        -A "${OUT_DIR}/${band}_noalpha.tif" \
        --outfile="${OUT_DIR}/${band}_rescaled.tif" \
        --calc="A/32768.0" \
        --type=Float32 \
        --overwrite
    echo "Rescaled: ${band}"
done

echo "Preprocessing complete"

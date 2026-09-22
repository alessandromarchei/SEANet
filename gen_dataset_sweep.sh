#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# SEANet visual embedding FPS sweep
# ============================================================

VIDEO_ROOT="/scratch_nvme/VoxCeleb2-2Mix/orig/train"
OUTPUT_BASE="/home/ale/datasets"
VISUAL_FRONTEND="pretrain_networks/visual_frontend.pt"

SOURCE_FPS=25

FPS_VALUES=(
    20
    17.5
    15
    12.5
)

echo "============================================================"
echo " SEANet Visual Embedding FPS Sweep"
echo "============================================================"
echo "Video root:      ${VIDEO_ROOT}"
echo "Output base:     ${OUTPUT_BASE}"
echo "Visual frontend: ${VISUAL_FRONTEND}"
echo "Source FPS:      ${SOURCE_FPS}"
echo "Target FPS:      ${FPS_VALUES[*]}"
echo "============================================================"
echo

for FPS in "${FPS_VALUES[@]}"; do

    OUTPUT_ROOT="${OUTPUT_BASE}/VoxCeleb2-2Mix_SEANet_${FPS}fps"

    echo
    echo "============================================================"
    echo "Generating embeddings @ ${FPS} FPS"
    echo "Output: ${OUTPUT_ROOT}"
    echo "============================================================"

    python scripts/generate_visual_embeddings.py \
        --video_root "${VIDEO_ROOT}" \
        --output_root "${OUTPUT_ROOT}" \
        --visual_frontend "${VISUAL_FRONTEND}" \
        --fps "${FPS}" \
        --source_fps "${SOURCE_FPS}" \
        --data_list "configs/data_list.csv"

    echo
    echo "[OK] ${FPS} FPS completed."
done

echo
echo "============================================================"
echo " FPS sweep completed"
echo "============================================================"

for FPS in "${FPS_VALUES[@]}"; do
    echo "${OUTPUT_BASE}/VoxCeleb2-2Mix_SEANet_${FPS}fps"
done

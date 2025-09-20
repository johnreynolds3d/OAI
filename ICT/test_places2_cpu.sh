#!/bin/bash

# ICT test script for Places2 pre-trained model on OAI dataset (CPU VERSION)
# Uses CPU instead of GPU for systems with limited GPU memory
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR" && CUDA_VISIBLE_DEVICES="" python run_fast.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_Places2_cpu \
    --Places2_Nature \
    --visualize_all

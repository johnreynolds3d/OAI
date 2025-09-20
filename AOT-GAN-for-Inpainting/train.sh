#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR/src" && python train.py \
    --dir_image "../../OAI_dataset" \
    --dir_mask "../../OAI_dataset" \
    --data_train img \
    --mask_type pconv \
    --image_size 224 \
    --iterations 500 \
    --batch_size 4 \
    --num_workers 2 \
    --save_every 100 \
    --print_every 100

#!/bin/bash

# AOT-GAN test script for Places2 pre-trained model on OAI dataset
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR/src" && python test.py \
    --dir_image="../../OAI_dataset/test/img" \
    --dir_mask="../../OAI_dataset/test/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_places2" \
    --pre_train="../experiments/places2/G0000000.pt"

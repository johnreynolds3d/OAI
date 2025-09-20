#!/bin/bash

# ICT test script for single image testing (SUPER FAST)
# Tests only the first image for quick validation
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a temporary directory with just one image
TEMP_DIR="/tmp/ict_single_test"
mkdir -p "$TEMP_DIR/img" "$TEMP_DIR/mask"

# Copy only the first image and mask
FIRST_IMG=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/img/*.png | head -1)
FIRST_MASK=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/mask/*.png | head -1)

if [ -n "$FIRST_IMG" ] && [ -n "$FIRST_MASK" ]; then
    cp "$FIRST_IMG" "$TEMP_DIR/img/"
    cp "$FIRST_MASK" "$TEMP_DIR/mask/"
    
    echo "Testing with single image: $(basename "$FIRST_IMG")"
    
    # Change to the script directory and run from there
    cd "$SCRIPT_DIR" && python run_fast.py \
        --input_image "$TEMP_DIR/img" \
        --input_mask "$TEMP_DIR/mask" \
        --sample_num 1 \
        --save_place ../../OAI_dataset/output/ICT_single_test \
        --Places2_Nature \
        --visualize_all
    
    # Clean up
    rm -rf "$TEMP_DIR"
else
    echo "No test images found!"
    exit 1
fi

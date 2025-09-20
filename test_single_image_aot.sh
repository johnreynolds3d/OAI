#!/bin/bash

# AOT-GAN single image test script
# Tests one image to verify the model works
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create temporary directory with single image
TEMP_DIR="/tmp/aot_single_test"
mkdir -p "$TEMP_DIR/img" "$TEMP_DIR/mask"

# Copy only the first image and mask
FIRST_IMG=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/img/*.png | head -1)
FIRST_MASK=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/mask/*.png | head -1)

if [ -n "$FIRST_IMG" ] && [ -n "$FIRST_MASK" ]; then
    cp "$FIRST_IMG" "$TEMP_DIR/img/"
    cp "$FIRST_MASK" "$TEMP_DIR/mask/"
    
    echo "Testing AOT-GAN with single image: $(basename "$FIRST_IMG")"
    
    # Change to the script directory and run from there
    cd "$SCRIPT_DIR/AOT-GAN-for-Inpainting/src" && python test.py \
        --dir_image="$TEMP_DIR/img" \
        --dir_mask="$TEMP_DIR/mask" \
        --image_size=256 \
        --outputs="$TEMP_DIR/output" \
        --pre_train="../experiments/celebahq/G0000000.pt"
    
    # Copy result to main output directory
    mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/AOT_single_test
    cp -r "$TEMP_DIR/output"/* /home/john/Documents/git/OAI/OAI_dataset/output/AOT_single_test/ 2>/dev/null || true
    
    echo "✅ AOT-GAN single image test completed"
    echo "   Result saved to: OAI_dataset/output/AOT_single_test/"
    
    # Clean up
    rm -rf "$TEMP_DIR"
else
    echo "❌ No test images found!"
    exit 1
fi

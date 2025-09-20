#!/bin/bash

# AOT-GAN Comprehensive Test Script
# Tests all available pre-trained models on OAI dataset
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 AOT-GAN Comprehensive Testing"
echo "================================"

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/AOT_celebahq
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/AOT_places2

# Test CelebA-HQ model
echo "Testing CelebA-HQ model..."
cd "$SCRIPT_DIR/src" && python test.py \
    --dir_image="../../OAI_dataset/test/img" \
    --dir_mask="../../OAI_dataset/test/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_celebahq" \
    --pre_train="../experiments/celebahq/G0000000.pt"

if [ $? -eq 0 ]; then
    echo "✅ CelebA-HQ: SUCCESS"
else
    echo "❌ CelebA-HQ: FAILED"
fi

# Test Places2 model
echo "Testing Places2 model..."
cd "$SCRIPT_DIR/src" && python test.py \
    --dir_image="../../OAI_dataset/test/img" \
    --dir_mask="../../OAI_dataset/test/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_places2" \
    --pre_train="../experiments/places2/G0000000.pt"

if [ $? -eq 0 ]; then
    echo "✅ Places2: SUCCESS"
else
    echo "❌ Places2: FAILED"
fi

echo ""
echo "📊 AOT-GAN Test Results:"
echo "========================"
echo "CelebA-HQ: Check OAI_dataset/output/AOT_celebahq/"
echo "Places2:   Check OAI_dataset/output/AOT_places2/"
echo ""
echo "🎉 AOT-GAN testing completed!"

#!/bin/bash

# ICT Comprehensive Test Script
# Tests all available pre-trained models on OAI dataset
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 ICT Comprehensive Testing"
echo "============================"

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_FFHQ
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_ImageNet
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_Places2

# Test FFHQ model
echo "Testing FFHQ model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_FFHQ \
    --FFHQ \
    --visualize_all

if [ $? -eq 0 ]; then
    echo "✅ FFHQ: SUCCESS"
else
    echo "❌ FFHQ: FAILED"
fi

# Test ImageNet model
echo "Testing ImageNet model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_ImageNet \
    --ImageNet \
    --visualize_all

if [ $? -eq 0 ]; then
    echo "✅ ImageNet: SUCCESS"
else
    echo "❌ ImageNet: FAILED"
fi

# Test Places2 model
echo "Testing Places2 model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_Places2 \
    --Places2_Nature \
    --visualize_all

if [ $? -eq 0 ]; then
    echo "✅ Places2: SUCCESS"
else
    echo "❌ Places2: FAILED"
fi

echo ""
echo "📊 ICT Test Results:"
echo "===================="
echo "FFHQ:     Check OAI_dataset/output/ICT_FFHQ/"
echo "ImageNet: Check OAI_dataset/output/ICT_ImageNet/"
echo "Places2:  Check OAI_dataset/output/ICT_Places2/"
echo ""
echo "🎉 ICT testing completed!"

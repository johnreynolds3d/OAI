#!/bin/bash

# ICT Comprehensive Test Script (FIXED VERSION)
# Tests all available pre-trained models on OAI dataset with proper error handling
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 ICT Comprehensive Testing (Fixed)"
echo "===================================="

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_FFHQ
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_ImageNet
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/ICT_Places2

# Clear GPU memory before each test
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test FFHQ model
echo "Testing FFHQ model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_FFHQ \
    --FFHQ \
    --visualize_all 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ FFHQ: SUCCESS"
    FFHQ_STATUS="SUCCESS"
else
    echo "❌ FFHQ: FAILED (GPU memory issue)"
    FFHQ_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test ImageNet model
echo "Testing ImageNet model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_ImageNet \
    --ImageNet \
    --visualize_all 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ImageNet: SUCCESS"
    IMAGENET_STATUS="SUCCESS"
else
    echo "❌ ImageNet: FAILED (GPU memory issue)"
    IMAGENET_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test Places2 model
echo "Testing Places2 model..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image ../../OAI_dataset/test/img \
    --input_mask ../../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../../OAI_dataset/output/ICT_Places2 \
    --Places2_Nature \
    --visualize_all 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Places2: SUCCESS"
    PLACES2_STATUS="SUCCESS"
else
    echo "❌ Places2: FAILED (GPU memory issue)"
    PLACES2_STATUS="FAILED"
fi

echo ""
echo "📊 ICT Test Results:"
echo "===================="
echo "FFHQ:     $FFHQ_STATUS - Check OAI_dataset/output/ICT_FFHQ/"
echo "ImageNet: $IMAGENET_STATUS - Check OAI_dataset/output/ICT_ImageNet/"
echo "Places2:  $PLACES2_STATUS - Check OAI_dataset/output/ICT_Places2/"

# Count successful models
SUCCESS_COUNT=0
if [ "$FFHQ_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$IMAGENET_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$PLACES2_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi

echo ""
if [ $SUCCESS_COUNT -eq 3 ]; then
    echo "🎉 ICT: ALL MODELS SUCCESS"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  ICT: $SUCCESS_COUNT out of 3 models succeeded"
    exit 1
else
    echo "❌ ICT: ALL MODELS FAILED"
    exit 1
fi

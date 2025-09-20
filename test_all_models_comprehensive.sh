#!/bin/bash

# Master Comprehensive Test Script (FIXED VERSION)
# Tests ALL available pre-trained models across ALL architectures with proper error handling
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE (FIXED)"
echo "============================================="
echo "Testing ALL available pre-trained models on OAI dataset"
echo ""

# Create all output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/{AOT_celebahq,AOT_places2,ICT_FFHQ,ICT_ImageNet,ICT_Places2,RePaint_Places,RePaint_CelebA}

# Clear GPU memory at start
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test AOT-GAN models
echo ""
echo "🎯 Testing AOT-GAN Models"
echo "========================="
if AOT-GAN-for-Inpainting/test_all_models.sh; then
    echo "✅ AOT-GAN: ALL MODELS SUCCESS"
    AOT_STATUS="SUCCESS"
else
    echo "❌ AOT-GAN: SOME MODELS FAILED"
    AOT_STATUS="FAILED"
fi

# Clear GPU memory between tests
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test ICT models
echo ""
echo "🎯 Testing ICT Models"
echo "====================="
if ICT/test_all_models.sh; then
    echo "✅ ICT: ALL MODELS SUCCESS"
    ICT_STATUS="SUCCESS"
else
    echo "❌ ICT: SOME MODELS FAILED"
    ICT_STATUS="FAILED"
fi

# Clear GPU memory between tests
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test RePaint models
echo ""
echo "🎯 Testing RePaint Models"
echo "========================="
if RePaint/test_all_models.sh; then
    echo "✅ RePaint: ALL MODELS SUCCESS"
    REPAINT_STATUS="SUCCESS"
else
    echo "❌ RePaint: SOME MODELS FAILED"
    REPAINT_STATUS="FAILED"
fi

# Generate comprehensive report
echo ""
echo "📊 COMPREHENSIVE TEST RESULTS"
echo "============================="
echo ""
echo "🏗️  ARCHITECTURE BREAKDOWN:"
echo "---------------------------"
echo "AOT-GAN: $AOT_STATUS"
echo "  ├── CelebA-HQ: OAI_dataset/output/AOT_celebahq/"
echo "  └── Places2:   OAI_dataset/output/AOT_places2/"
echo ""
echo "ICT: $ICT_STATUS"
echo "  ├── FFHQ:      OAI_dataset/output/ICT_FFHQ/"
echo "  ├── ImageNet:  OAI_dataset/output/ICT_ImageNet/"
echo "  └── Places2:   OAI_dataset/output/ICT_Places2/"
echo ""
echo "RePaint: $REPAINT_STATUS"
echo "  ├── Places2:   OAI_dataset/output/RePaint_Places/"
echo "  └── CelebA:    OAI_dataset/output/RePaint_CelebA/"
echo ""

# Count successful architectures
SUCCESS_ARCHITECTURES=0
if [ "$AOT_STATUS" = "SUCCESS" ]; then ((SUCCESS_ARCHITECTURES++)); fi
if [ "$ICT_STATUS" = "SUCCESS" ]; then ((SUCCESS_ARCHITECTURES++)); fi
if [ "$REPAINT_STATUS" = "SUCCESS" ]; then ((SUCCESS_ARCHITECTURES++)); fi

echo "📈 SUMMARY:"
echo "----------"
echo "Total Architectures: 3"
echo "Successful Architectures: $SUCCESS_ARCHITECTURES"
echo "Pre-trained Variants: 7"
echo ""

if [ $SUCCESS_ARCHITECTURES -eq 3 ]; then
    echo "🎉 ALL ARCHITECTURES WORKING!"
    echo "Check the output directories above for results."
    exit 0
elif [ $SUCCESS_ARCHITECTURES -gt 0 ]; then
    echo "⚠️  SOME ARCHITECTURES WORKING"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO ARCHITECTURES WORKING"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

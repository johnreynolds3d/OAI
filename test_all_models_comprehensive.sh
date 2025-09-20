#!/bin/bash

# Master Comprehensive Test Script
# Tests ALL available pre-trained models across ALL architectures
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE"
echo "====================================="
echo "Testing ALL available pre-trained models on OAI dataset"
echo ""

# Create all output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/{AOT_celebahq,AOT_places2,ICT_FFHQ,ICT_ImageNet,ICT_Places2,RePaint_Places,RePaint_CelebA}

# Test AOT-GAN models
echo ""
echo "🎯 Testing AOT-GAN Models"
echo "========================="
if AOT-GAN-for-Inpainting/test_all_models.sh; then
    echo "✅ AOT-GAN: ALL MODELS SUCCESS"
else
    echo "❌ AOT-GAN: SOME MODELS FAILED"
fi

# Test ICT models
echo ""
echo "🎯 Testing ICT Models"
echo "====================="
if ICT/test_all_models.sh; then
    echo "✅ ICT: ALL MODELS SUCCESS"
else
    echo "❌ ICT: SOME MODELS FAILED"
fi

# Test RePaint models
echo ""
echo "🎯 Testing RePaint Models"
echo "========================="
if RePaint/test_all_models.sh; then
    echo "✅ RePaint: ALL MODELS SUCCESS"
else
    echo "❌ RePaint: SOME MODELS FAILED"
fi

# Generate comprehensive report
echo ""
echo "📊 COMPREHENSIVE TEST RESULTS"
echo "============================="
echo ""
echo "🏗️  ARCHITECTURE BREAKDOWN:"
echo "---------------------------"
echo "AOT-GAN:"
echo "  ├── CelebA-HQ: OAI_dataset/output/AOT_celebahq/"
echo "  └── Places2:   OAI_dataset/output/AOT_places2/"
echo ""
echo "ICT:"
echo "  ├── FFHQ:      OAI_dataset/output/ICT_FFHQ/"
echo "  ├── ImageNet:  OAI_dataset/output/ICT_ImageNet/"
echo "  └── Places2:   OAI_dataset/output/ICT_Places2/"
echo ""
echo "RePaint:"
echo "  ├── Places2:   OAI_dataset/output/RePaint_Places/"
echo "  └── CelebA:    OAI_dataset/output/RePaint_CelebA/"
echo ""

# Count total models tested
TOTAL_MODELS=7
echo "📈 SUMMARY:"
echo "----------"
echo "Total Models Tested: $TOTAL_MODELS"
echo "Architectures: 3 (AOT-GAN, ICT, RePaint)"
echo "Pre-trained Variants: 7"
echo ""
echo "🎉 Comprehensive testing completed!"
echo "Check the output directories above for results."

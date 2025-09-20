#!/bin/bash

# RePaint Test Script (FIXED VERSION)
# Based on the backup configuration that actually produced output successfully
# Addresses GPU memory issues with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 RePaint Testing (FIXED VERSION)"
echo "=================================="
echo "Testing RePaint using the restored working configuration"
echo "Based on backup evidence of successful 25-image generation"
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places/{inpainted,gt_masked,gt,gt_keep_mask}

# Set environment variables for better memory management
echo "🔧 Setting up environment for RePaint..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory before starting
echo "🧹 Clearing GPU memory..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
    print(f'GPU Available: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated')
else:
    print('No CUDA GPU available')
"

# Check if RePaint dependencies are available
echo "🔍 Checking RePaint dependencies..."
cd "$SCRIPT_DIR/RePaint"

if [ ! -f "test.py" ]; then
    echo "❌ RePaint test.py not found!"
    exit 1
fi

if [ ! -f "data/pretrained/places256_300000.pt" ]; then
    echo "❌ RePaint pre-trained model not found!"
    echo "   Expected: data/pretrained/places256_300000.pt"
    exit 1
fi

if [ ! -f "confs/oai_test_restored.yml" ]; then
    echo "❌ RePaint configuration not found!"
    exit 1
fi

echo "✅ All RePaint dependencies found"

# Test RePaint with restored configuration
echo ""
echo "🎯 Testing RePaint Places2 (Restored Working Configuration)"
echo "=========================================================="

# Run RePaint with the restored configuration
echo "Starting RePaint with restored configuration..."
echo "Configuration: confs/oai_test_restored.yml"
echo "Model: places256_300000.pt"
echo "Image size: 256x256"
echo "Timesteps: 25 (fast mode)"
echo ""

# Run the test with proper error handling
cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_restored.yml

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ RePaint Places2: SUCCESS"
    echo "📊 Checking output..."
    
    # Count generated images
    INPAINTED_COUNT=$(ls ../OAI_dataset/output/RePaint_Places/inpainted/*.png 2>/dev/null | wc -l)
    echo "   Generated images: $INPAINTED_COUNT"
    
    if [ $INPAINTED_COUNT -gt 0 ]; then
        echo "   ✅ Images successfully generated!"
        echo "   📁 Output directory: ../OAI_dataset/output/RePaint_Places/inpainted/"
        
        # Show file sizes
        echo "   📏 File sizes:"
        ls -lah ../OAI_dataset/output/RePaint_Places/inpainted/*.png | head -5 | while read line; do
            echo "      $line"
        done
        
        REPAINT_STATUS="SUCCESS"
    else
        echo "   ❌ No images generated"
        REPAINT_STATUS="FAILED"
    fi
else
    echo ""
    echo "❌ RePaint Places2: FAILED"
    echo "   Check GPU memory and model compatibility"
    REPAINT_STATUS="FAILED"
fi

# Generate report
echo ""
echo "📊 RePaint FIXED TEST RESULTS"
echo "============================="
echo ""
echo "🏗️  RePaint Results:"
echo "-------------------"
echo "Places2: $REPAINT_STATUS - OAI_dataset/output/RePaint_Places/"
echo ""

if [ "$REPAINT_STATUS" = "SUCCESS" ]; then
    echo "🎉 RePaint SUCCESS with restored configuration!"
    echo "✅ This matches the backup evidence of successful generation"
    echo "📁 Check the output directory for results"
    exit 0
else
    echo "❌ RePaint FAILED with restored configuration"
    echo "⚠️  This suggests a deeper environment or dependency issue"
    echo "💡 Try running in Google Colab with higher GPU memory"
    exit 1
fi

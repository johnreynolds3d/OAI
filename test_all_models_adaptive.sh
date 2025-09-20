#!/bin/bash

# Adaptive Testing Script for All Models
# Automatically detects hardware capabilities and runs appropriate tests
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting Adaptive Model Testing"
echo "=================================="

# Detect hardware capabilities
echo "Step 1: Detecting hardware capabilities..."
cd "$SCRIPT_DIR"
python test_hardware_capabilities.py
HARDWARE_LEVEL=$?

# Set hardware level based on detection
if [ $HARDWARE_LEVEL -eq 0 ]; then
    HARDWARE_LEVEL="cpu_only"
elif [ $HARDWARE_LEVEL -eq 1 ]; then
    HARDWARE_LEVEL="low"
elif [ $HARDWARE_LEVEL -eq 2 ]; then
    HARDWARE_LEVEL="medium"
else
    HARDWARE_LEVEL="high_end"
fi

echo "Detected hardware level: $HARDWARE_LEVEL"
echo ""

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/{AOT_single_test,RePaint_single_test,ICT_single_test}

# Test AOT-GAN (most reliable)
echo "Step 2: Testing AOT-GAN..."
echo "-------------------------"
if ./test_single_image_aot.sh; then
    echo "✅ AOT-GAN: SUCCESS"
    AOT_STATUS="SUCCESS"
else
    echo "❌ AOT-GAN: FAILED"
    AOT_STATUS="FAILED"
fi
echo ""

# Test RePaint based on hardware level
echo "Step 3: Testing RePaint..."
echo "-------------------------"
if [ "$HARDWARE_LEVEL" = "high_end" ] || [ "$HARDWARE_LEVEL" = "medium" ]; then
    # Try GPU version first
    echo "Attempting GPU version..."
    if cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_memory_optimized.yml 2>/dev/null; then
        echo "✅ RePaint (GPU): SUCCESS"
        REPAINT_STATUS="SUCCESS (GPU)"
    else
        echo "GPU failed, trying CPU version..."
        if ./test_single_image_repaint.sh; then
            echo "✅ RePaint (CPU): SUCCESS"
            REPAINT_STATUS="SUCCESS (CPU)"
        else
            echo "❌ RePaint: FAILED"
            REPAINT_STATUS="FAILED"
        fi
    fi
else
    # Use CPU version directly
    if ./test_single_image_repaint.sh; then
        echo "✅ RePaint (CPU): SUCCESS"
        REPAINT_STATUS="SUCCESS (CPU)"
    else
        echo "❌ RePaint: FAILED"
        REPAINT_STATUS="FAILED"
    fi
fi
echo ""

# Test ICT based on hardware level
echo "Step 4: Testing ICT..."
echo "---------------------"
if [ "$HARDWARE_LEVEL" = "high_end" ]; then
    # Try full ICT test
    echo "Attempting full ICT test..."
    if cd "$SCRIPT_DIR" && ICT/test_places2_fast.sh 2>/dev/null; then
        echo "✅ ICT (Fast): SUCCESS"
        ICT_STATUS="SUCCESS (Fast)"
    else
        echo "Full test failed, trying single image..."
        if ICT/test_single_image.sh; then
            echo "✅ ICT (Single): SUCCESS"
            ICT_STATUS="SUCCESS (Single)"
        else
            echo "❌ ICT: FAILED"
            ICT_STATUS="FAILED"
        fi
    fi
else
    # Use single image test
    if ICT/test_single_image.sh; then
        echo "✅ ICT (Single): SUCCESS"
        ICT_STATUS="SUCCESS (Single)"
    else
        echo "❌ ICT: FAILED"
        ICT_STATUS="FAILED"
    fi
fi
echo ""

# Generate test report
echo "📊 TEST RESULTS SUMMARY"
echo "======================="
echo "Hardware Level: $HARDWARE_LEVEL"
echo "AOT-GAN:       $AOT_STATUS"
echo "RePaint:       $REPAINT_STATUS"
echo "ICT:           $ICT_STATUS"
echo ""

# Count successful models
SUCCESS_COUNT=0
if [[ "$AOT_STATUS" == *"SUCCESS"* ]]; then ((SUCCESS_COUNT++)); fi
if [[ "$REPAINT_STATUS" == *"SUCCESS"* ]]; then ((SUCCESS_COUNT++)); fi
if [[ "$ICT_STATUS" == *"SUCCESS"* ]]; then ((SUCCESS_COUNT++)); fi

echo "✅ Successfully tested $SUCCESS_COUNT out of 3 models"
echo ""

if [ $SUCCESS_COUNT -eq 3 ]; then
    echo "🎉 All models working! You can now run full tests."
elif [ $SUCCESS_COUNT -eq 2 ]; then
    echo "👍 Most models working! Check failed model for issues."
elif [ $SUCCESS_COUNT -eq 1 ]; then
    echo "⚠️  Only one model working. Check hardware requirements."
else
    echo "❌ No models working. Check installation and hardware."
fi

echo ""
echo "📁 Results saved in: OAI_dataset/output/"
echo "   - AOT_single_test/"
echo "   - RePaint_single_test/"
echo "   - ICT_single_test/"

#!/bin/bash

# Master Comprehensive Test Script (BALANCED 4-SAMPLE COMPREHENSIVE FINAL VERSION)
# Tests ALL available pre-trained models across ALL architectures with balanced 4-sample subset
# Focuses on working models (AOT-GAN and ICT) with clear reporting

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE (BALANCED 4-SAMPLE FINAL)"
echo "=============================================================="
echo "Testing ALL available pre-trained models on balanced OAI dataset subset"
echo "Optimized for 4GB GPU memory with single-image processing approach"
echo ""

# Create balanced subset directories
BALANCED_DIR="/tmp/oai_balanced_4"
mkdir -p "$BALANCED_DIR/img" "$BALANCED_DIR/mask" "$BALANCED_DIR/mask_inv"

echo "📁 Creating balanced 4-sample subset..."
echo "   - 2 Non-Osteoporotic samples (6.C.*)"
echo "   - 2 Osteoporotic samples (6.E.*)"

# Select 2 Non-Osteoporotic samples (6.C.*)
echo "Selecting Non-Osteoporotic samples..."
ls /home/john/Documents/git/OAI/OAI_dataset/test/img/6.C.*.png | head -2 | while read img; do
    img_name=$(basename "$img")
    mask_name="${img_name%.png}.png"
    mask_inv_name="${img_name%.png}.png"
    
    cp "$img" "$BALANCED_DIR/img/"
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask/$mask_name" "$BALANCED_DIR/mask/" 2>/dev/null || true
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask_inv/$mask_inv_name" "$BALANCED_DIR/mask_inv/" 2>/dev/null || true
    
    echo "   ✓ $img_name (Non-Osteoporotic)"
done

# Select 2 Osteoporotic samples (6.E.*)
echo "Selecting Osteoporotic samples..."
ls /home/john/Documents/git/OAI/OAI_dataset/test/img/6.E.*.png | head -2 | while read img; do
    img_name=$(basename "$img")
    mask_name="${img_name%.png}.png"
    mask_inv_name="${img_name%.png}.png"
    
    cp "$img" "$BALANCED_DIR/img/"
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask/$mask_name" "$BALANCED_DIR/mask/" 2>/dev/null || true
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask_inv/$mask_inv_name" "$BALANCED_DIR/mask_inv/" 2>/dev/null || true
    
    echo "   ✓ $img_name (Osteoporotic)"
done

echo ""
echo "📊 Balanced subset created:"
echo "   Total images: $(ls $BALANCED_DIR/img/ | wc -l)"
echo "   Non-Osteoporotic: $(ls $BALANCED_DIR/img/6.C.* 2>/dev/null | wc -l)"
echo "   Osteoporotic: $(ls $BALANCED_DIR/img/6.E.* 2>/dev/null | wc -l)"
echo ""

# Create all output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/{AOT_celebahq_4,AOT_places2_4,ICT_FFHQ_4,ICT_ImageNet_4,ICT_Places2_4}

# Set GPU memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory at start
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test AOT-GAN models (these work well already)
echo ""
echo "🎯 Testing AOT-GAN Models (4-sample balanced)"
echo "============================================="

# AOT-GAN CelebA-HQ
echo "Testing AOT-GAN CelebA-HQ..."
cd "$SCRIPT_DIR/AOT-GAN-for-Inpainting/src" && python test.py \
    --dir_image="$BALANCED_DIR/img" \
    --dir_mask="$BALANCED_DIR/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_celebahq_4" \
    --pre_train="../experiments/celebahq/G0000000.pt" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ AOT-GAN CelebA-HQ: SUCCESS"
    AOT_CELEBA_STATUS="SUCCESS"
else
    echo "❌ AOT-GAN CelebA-HQ: FAILED"
    AOT_CELEBA_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# AOT-GAN Places2
echo "Testing AOT-GAN Places2..."
cd "$SCRIPT_DIR/AOT-GAN-for-Inpainting/src" && python test.py \
    --dir_image="$BALANCED_DIR/img" \
    --dir_mask="$BALANCED_DIR/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_places2_4" \
    --pre_train="../experiments/places2/G0000000.pt" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ AOT-GAN Places2: SUCCESS"
    AOT_PLACES_STATUS="SUCCESS"
else
    echo "❌ AOT-GAN Places2: FAILED"
    AOT_PLACES_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test ICT models using single-image approach
echo ""
echo "🎯 Testing ICT Models (4-sample balanced, single-image approach)"
echo "==============================================================="

# Create a simple ICT test script for the balanced subset
cat > "$SCRIPT_DIR/test_ict_balanced_4.py" << 'EOF'
import os
import sys
import subprocess
import tempfile
import shutil

def test_ict_model(model_type, input_dir, output_dir):
    """Test ICT model with single-image approach"""
    print(f"Testing ICT {model_type}...")
    
    # Create temporary single-image directories
    temp_dir = tempfile.mkdtemp()
    temp_img_dir = os.path.join(temp_dir, "img")
    temp_mask_dir = os.path.join(temp_dir, "mask")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_mask_dir, exist_ok=True)
    
    try:
        # Get list of images
        images = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        
        success_count = 0
        for img_name in images:
            # Copy single image and mask
            shutil.copy2(os.path.join(input_dir, img_name), temp_img_dir)
            mask_name = img_name  # Assuming same name for mask
            if os.path.exists(os.path.join(input_dir.replace('img', 'mask'), mask_name)):
                shutil.copy2(os.path.join(input_dir.replace('img', 'mask'), mask_name), temp_mask_dir)
            
            # Run ICT on single image
            cmd = [
                "python", "run.py",
                "--input_image", temp_img_dir,
                "--input_mask", temp_mask_dir,
                "--sample_num", "1",
                "--save_place", output_dir,
                "--visualize_all"
            ]
            
            if model_type == "FFHQ":
                cmd.append("--FFHQ")
            elif model_type == "ImageNet":
                cmd.append("--ImageNet")
            elif model_type == "Places2":
                cmd.append("--Places2_Nature")
            
            # Change to ICT directory
            original_dir = os.getcwd()
            os.chdir("/home/john/Documents/git/OAI/ICT")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    success_count += 1
                    print(f"  ✓ {img_name}")
                else:
                    print(f"  ✗ {img_name} (failed)")
            except subprocess.TimeoutExpired:
                print(f"  ✗ {img_name} (timeout)")
            finally:
                os.chdir(original_dir)
            
            # Clear GPU memory between images
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            # Clean up temp files
            for f in os.listdir(temp_img_dir):
                os.remove(os.path.join(temp_img_dir, f))
            for f in os.listdir(temp_mask_dir):
                os.remove(os.path.join(temp_mask_dir, f))
        
        return success_count == len(images)
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_type = sys.argv[3]
    
    success = test_ict_model(model_type, input_dir, output_dir)
    sys.exit(0 if success else 1)
EOF

# ICT FFHQ
echo "Testing ICT FFHQ (single-image approach)..."
cd "$SCRIPT_DIR" && python test_ict_balanced_4.py "$BALANCED_DIR/img" "../../OAI_dataset/output/ICT_FFHQ_4" "FFHQ" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ICT FFHQ: SUCCESS"
    ICT_FFHQ_STATUS="SUCCESS"
else
    echo "❌ ICT FFHQ: FAILED"
    ICT_FFHQ_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ICT ImageNet
echo "Testing ICT ImageNet (single-image approach)..."
cd "$SCRIPT_DIR" && python test_ict_balanced_4.py "$BALANCED_DIR/img" "../../OAI_dataset/output/ICT_ImageNet_4" "ImageNet" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ICT ImageNet: SUCCESS"
    ICT_IMAGENET_STATUS="SUCCESS"
else
    echo "❌ ICT ImageNet: FAILED"
    ICT_IMAGENET_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ICT Places2
echo "Testing ICT Places2 (single-image approach)..."
cd "$SCRIPT_DIR" && python test_ict_balanced_4.py "$BALANCED_DIR/img" "../../OAI_dataset/output/ICT_Places2_4" "Places2" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ICT Places2: SUCCESS"
    ICT_PLACES_STATUS="SUCCESS"
else
    echo "❌ ICT Places2: FAILED"
    ICT_PLACES_STATUS="FAILED"
fi

# Clean up temporary files
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$BALANCED_DIR"
rm -f "$SCRIPT_DIR/test_ict_balanced_4.py"

# Generate comprehensive report
echo ""
echo "📊 BALANCED 4-SAMPLE COMPREHENSIVE FINAL TEST RESULTS"
echo "===================================================="
echo ""
echo "🏗️  ARCHITECTURE BREAKDOWN:"
echo "---------------------------"
echo "AOT-GAN:"
echo "  ├── CelebA-HQ: $AOT_CELEBA_STATUS - OAI_dataset/output/AOT_celebahq_4/"
echo "  └── Places2:   $AOT_PLACES_STATUS - OAI_dataset/output/AOT_places2_4/"
echo ""
echo "ICT:"
echo "  ├── FFHQ:      $ICT_FFHQ_STATUS - OAI_dataset/output/ICT_FFHQ_4/"
echo "  ├── ImageNet:  $ICT_IMAGENET_STATUS - OAI_dataset/output/ICT_ImageNet_4/"
echo "  └── Places2:   $ICT_PLACES_STATUS - OAI_dataset/output/ICT_Places2_4/"
echo ""
echo "RePaint:"
echo "  ├── Places2:   SKIPPED (GPU memory constraints - 4GB limit)"
echo "  └── CelebA:    SKIPPED (GPU memory constraints - 4GB limit)"
echo ""

# Count successful models
SUCCESS_COUNT=0
if [ "$AOT_CELEBA_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$AOT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_FFHQ_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_IMAGENET_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi

echo "📈 SUMMARY:"
echo "----------"
echo "Total Models Tested: 5 (AOT-GAN: 2, ICT: 3)"
echo "Successful Models: $SUCCESS_COUNT"
echo "Architectures: 2 (AOT-GAN, ICT) - RePaint skipped due to GPU memory constraints"
echo "Sample Size: 4 (2 Non-Osteoporotic + 2 Osteoporotic)"
echo "Approach: Single-image processing for memory efficiency"
echo ""

if [ $SUCCESS_COUNT -eq 5 ]; then
    echo "🎉 ALL TESTED MODELS SUCCESS ON BALANCED 4-SAMPLE SUBSET!"
    echo "✅ AOT-GAN: Both models working perfectly"
    echo "✅ ICT: All three models working perfectly"
    echo "⚠️  RePaint: Skipped due to GPU memory constraints (4GB limit)"
    echo ""
    echo "📁 Results saved in:"
    echo "   - OAI_dataset/output/AOT_celebahq_4/"
    echo "   - OAI_dataset/output/AOT_places2_4/"
    echo "   - OAI_dataset/output/ICT_FFHQ_4/"
    echo "   - OAI_dataset/output/ICT_ImageNet_4/"
    echo "   - OAI_dataset/output/ICT_Places2_4/"
    echo ""
    echo "🔬 BALANCED SAMPLE ANALYSIS:"
    echo "   - Non-Osteoporotic samples: 2 (6.C.*)"
    echo "   - Osteoporotic samples: 2 (6.E.*)"
    echo "   - Perfect balance for comparative analysis"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  $SUCCESS_COUNT out of 5 models succeeded on balanced 4-sample subset"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO MODELS SUCCEEDED on balanced 4-sample subset"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

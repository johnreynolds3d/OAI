#!/bin/bash

# Master Comprehensive Test Script (BALANCED 4-SAMPLE SIMPLE VERSION)
# Tests ALL available pre-trained models across ALL architectures with balanced 4-sample subset
# Uses existing single-image test approach for better compatibility

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE (BALANCED 4-SAMPLE SIMPLE)"
echo "==============================================================="
echo "Testing ALL available pre-trained models on balanced OAI dataset subset"
echo "Using single-image approach for better GPU memory management"
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
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/{AOT_celebahq_4,AOT_places2_4,ICT_FFHQ_4,ICT_ImageNet_4,ICT_Places2_4,RePaint_Places_4,RePaint_CelebA_4}

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

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test RePaint models using single-image approach
echo ""
echo "🎯 Testing RePaint Models (4-sample balanced, single-image approach)"
echo "==================================================================="

# Create a simple RePaint test script for the balanced subset
cat > "$SCRIPT_DIR/test_repaint_balanced_4.py" << 'EOF'
import os
import sys
import subprocess
import tempfile
import shutil
import yaml

def test_repaint_model(model_type, input_dir, output_dir):
    """Test RePaint model with single-image approach"""
    print(f"Testing RePaint {model_type}...")
    
    # Create temporary single-image directories
    temp_dir = tempfile.mkdtemp()
    temp_img_dir = os.path.join(temp_dir, "img")
    temp_mask_dir = os.path.join(temp_dir, "mask_inv")
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
            mask_path = os.path.join(input_dir.replace('img', 'mask_inv'), mask_name)
            if os.path.exists(mask_path):
                shutil.copy2(mask_path, temp_mask_dir)
            
            # Create config for single image
            config = {
                'attention_resolutions': [32, 16, 8],
                'class_cond': False,
                'diffusion_steps': 1000,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,  # Reduced for memory
                'num_head_channels': 32,
                'num_heads': 2,
                'num_res_blocks': 1,
                'resblock_updown': True,
                'use_fp16': True,
                'use_scale_shift_norm': True,
                'classifier_scale': 2.0,
                'lr_kernel_n_std': 2,
                'num_samples': 1,
                'show_progress': True,
                'timestep_respacing': '10',  # Reduced for speed
                'use_kl': False,
                'predict_xstart': False,
                'rescale_timesteps': False,
                'rescale_learned_sigmas': False,
                'classifier_use_fp16': True,
                'classifier_width': 64,
                'classifier_depth': 1,
                'classifier_attention_resolutions': [32, 16, 8],
                'classifier_use_scale_shift_norm': True,
                'classifier_resblock_updown': True,
                'classifier_pool': 'attention',
                'num_heads_upsample': -1,
                'channel_mult': '',
                'dropout': 0.0,
                'use_checkpoint': True,
                'use_new_attention_order': False,
                'clip_denoised': True,
                'use_ddim': False,
                'latex_name': 'RePaint',
                'method_name': 'Repaint',
                'image_size': 128,  # Reduced for memory
                'model_path': f'./data/pretrained/{model_type.lower()}256_300000.pt' if model_type == 'Places' else './data/pretrained/celeba256_250000.pt',
                'name': f'oai_test_{model_type.lower()}_single',
                'inpa_inj_sched_prev': True,
                'n_jobs': 1,
                'print_estimated_vars': True,
                'inpa_inj_sched_prev_cumnoise': False,
                'schedule_jump_params': {
                    't_T': 10,
                    'n_sample': 1,
                    'jump_length': 2,
                    'jump_n_sample': 2
                },
                'data': {
                    'eval': {
                        f'oai_test_{model_type.lower()}_single': {
                            'mask_loader': True,
                            'gt_path': temp_img_dir,
                            'mask_path': temp_mask_dir,
                            'image_size': 128,
                            'class_cond': False,
                            'deterministic': True,
                            'random_crop': False,
                            'random_flip': False,
                            'return_dict': True,
                            'drop_last': False,
                            'batch_size': 1,
                            'return_dataloader': True,
                            'offset': 0,
                            'max_len': 1,
                            'paths': {
                                'srs': os.path.join(output_dir, 'inpainted'),
                                'lrs': os.path.join(output_dir, 'gt_masked'),
                                'gts': os.path.join(output_dir, 'gt'),
                                'gt_keep_masks': os.path.join(output_dir, 'gt_keep_mask')
                            }
                        }
                    }
                }
            }
            
            # Write config file
            config_path = os.path.join(temp_dir, f'config_{model_type.lower()}.yml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create output directories
            os.makedirs(os.path.join(output_dir, 'inpainted'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'gt_masked'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'gt_keep_mask'), exist_ok=True)
            
            # Run RePaint on single image
            cmd = ["python", "test.py", "--conf_path", config_path]
            
            # Change to RePaint directory
            original_dir = os.getcwd()
            os.chdir("/home/john/Documents/git/OAI/RePaint")
            
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
    
    success = test_repaint_model(model_type, input_dir, output_dir)
    sys.exit(0 if success else 1)
EOF

# RePaint Places2
echo "Testing RePaint Places2 (single-image approach)..."
cd "$SCRIPT_DIR" && python test_repaint_balanced_4.py "$BALANCED_DIR/img" "../../OAI_dataset/output/RePaint_Places_4" "Places" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ RePaint Places2: SUCCESS"
    REPAINT_PLACES_STATUS="SUCCESS"
else
    echo "❌ RePaint Places2: FAILED"
    REPAINT_PLACES_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# RePaint CelebA
echo "Testing RePaint CelebA (single-image approach)..."
cd "$SCRIPT_DIR" && python test_repaint_balanced_4.py "$BALANCED_DIR/img" "../../OAI_dataset/output/RePaint_CelebA_4" "CelebA" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ RePaint CelebA: SUCCESS"
    REPAINT_CELEBA_STATUS="SUCCESS"
else
    echo "❌ RePaint CelebA: FAILED"
    REPAINT_CELEBA_STATUS="FAILED"
fi

# Clean up temporary files
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$BALANCED_DIR"
rm -f "$SCRIPT_DIR/test_ict_balanced_4.py"
rm -f "$SCRIPT_DIR/test_repaint_balanced_4.py"

# Generate comprehensive report
echo ""
echo "📊 BALANCED 4-SAMPLE SIMPLE TEST RESULTS"
echo "========================================"
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
echo "  ├── Places2:   $REPAINT_PLACES_STATUS - OAI_dataset/output/RePaint_Places_4/"
echo "  └── CelebA:    $REPAINT_CELEBA_STATUS - OAI_dataset/output/RePaint_CelebA_4/"
echo ""

# Count successful models
SUCCESS_COUNT=0
if [ "$AOT_CELEBA_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$AOT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_FFHQ_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_IMAGENET_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$ICT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$REPAINT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$REPAINT_CELEBA_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi

echo "📈 SUMMARY:"
echo "----------"
echo "Total Models Tested: 7"
echo "Successful Models: $SUCCESS_COUNT"
echo "Architectures: 3 (AOT-GAN, ICT, RePaint)"
echo "Sample Size: 4 (2 Non-Osteoporotic + 2 Osteoporotic)"
echo "Approach: Single-image processing for memory efficiency"
echo ""

if [ $SUCCESS_COUNT -eq 7 ]; then
    echo "🎉 ALL MODELS SUCCESS ON BALANCED 4-SAMPLE SIMPLE SUBSET!"
    echo "Check the output directories above for results."
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  $SUCCESS_COUNT out of 7 models succeeded on balanced 4-sample simple subset"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO MODELS SUCCEEDED on balanced 4-sample simple subset"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

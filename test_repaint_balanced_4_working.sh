#!/bin/bash

# RePaint Test Script (BALANCED 4-SAMPLE WORKING VERSION)
# Based on backup configuration but optimized for 4GB GPU and 4-sample balanced subset

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 RePaint Testing (BALANCED 4-SAMPLE WORKING VERSION)"
echo "====================================================="
echo "Testing RePaint on balanced OAI dataset subset using backup configuration"
echo ""

# Create balanced subset directories
BALANCED_DIR="/tmp/oai_balanced_4"
mkdir -p "$BALANCED_DIR/img" "$BALANCED_DIR/mask_inv"

echo "📁 Creating balanced 4-sample subset..."
echo "   - 2 Non-Osteoporotic samples (6.C.*)"
echo "   - 2 Osteoporotic samples (6.E.*)"

# Select 2 Non-Osteoporotic samples (6.C.*)
echo "Selecting Non-Osteoporotic samples..."
ls /home/john/Documents/git/OAI/OAI_dataset/test/img/6.C.*.png | head -2 | while read img; do
    img_name=$(basename "$img")
    mask_inv_name="${img_name%.png}.png"
    
    cp "$img" "$BALANCED_DIR/img/"
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask_inv/$mask_inv_name" "$BALANCED_DIR/mask_inv/" 2>/dev/null || true
    
    echo "   ✓ $img_name (Non-Osteoporotic)"
done

# Select 2 Osteoporotic samples (6.E.*)
echo "Selecting Osteoporotic samples..."
ls /home/john/Documents/git/OAI/OAI_dataset/test/img/6.E.*.png | head -2 | while read img; do
    img_name=$(basename "$img")
    mask_inv_name="${img_name%.png}.png"
    
    cp "$img" "$BALANCED_DIR/img/"
    cp "/home/john/Documents/git/OAI/OAI_dataset/test/mask_inv/$mask_inv_name" "$BALANCED_DIR/mask_inv/" 2>/dev/null || true
    
    echo "   ✓ $img_name (Osteoporotic)"
done

echo ""
echo "📊 Balanced subset created:"
echo "   Total images: $(ls $BALANCED_DIR/img/ | wc -l)"
echo "   Non-Osteoporotic: $(ls $BALANCED_DIR/img/6.C.* 2>/dev/null | wc -l)"
echo "   Osteoporotic: $(ls $BALANCED_DIR/img/6.E.* 2>/dev/null | wc -l)"
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places_4/{inpainted,gt_masked,gt,gt_keep_mask}

# Set GPU memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test RePaint Places2
echo ""
echo "🎯 Testing RePaint Places2 (4-sample balanced, working version)"
echo "=============================================================="

cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_balanced_4_working.yml

if [ $? -eq 0 ]; then
    echo "✅ RePaint Places2: SUCCESS"
    REPAINT_PLACES_STATUS="SUCCESS"
else
    echo "❌ RePaint Places2: FAILED"
    REPAINT_PLACES_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test RePaint CelebA
echo ""
echo "🎯 Testing RePaint CelebA (4-sample balanced, working version)"
echo "============================================================="

# Create CelebA configuration for balanced subset
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_celeba_balanced_4_working.yml" << 'EOF'
# RePaint CelebA configuration for OAI dataset testing (BALANCED 4-SAMPLE WORKING VERSION)

attention_resolutions: 32,16,8
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 256
num_head_channels: 64
num_heads: 4
num_res_blocks: 2
resblock_updown: true
use_fp16: false
use_scale_shift_norm: true
classifier_scale: 4.0
lr_kernel_n_std: 2
num_samples: 4  # Number of test images (balanced subset)
show_progress: true
timestep_respacing: '25'  # Fast: 25 steps instead of 250
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: false
classifier_width: 128
classifier_depth: 2
classifier_attention_resolutions: 32,16,8
classifier_use_scale_shift_norm: true
classifier_resblock_updown: true
classifier_pool: attention
num_heads_upsample: -1
channel_mult: ''
dropout: 0.0
use_checkpoint: false
use_new_attention_order: false
clip_denoised: true
use_ddim: false  # DDIM not available, use regular sampling with fewer steps
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: ./data/pretrained/celeba256_250000.pt  # Using CelebA model
name: oai_test_celeba_balanced_4_working
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 25  # Match timestep_respacing
  n_sample: 1
  jump_length: 3  # Reduced for faster processing
  jump_n_sample: 3  # Reduced for faster processing
data:
  eval:
    oai_test_celeba_balanced_4_working:
      mask_loader: true
      gt_path: /tmp/oai_balanced_4/img
      mask_path: /tmp/oai_balanced_4/mask_inv
      image_size: 256
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 4  # Number of test images (balanced subset)
      paths:
        srs: ../OAI_dataset/output/RePaint_CelebA_4/inpainted
        lrs: ../OAI_dataset/output/RePaint_CelebA_4/gt_masked
        gts: ../OAI_dataset/output/RePaint_CelebA_4/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_CelebA_4/gt_keep_mask
EOF

# Create output directories for CelebA
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_CelebA_4/{inpainted,gt_masked,gt,gt_keep_mask}

cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_celeba_balanced_4_working.yml

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

# Generate report
echo ""
echo "📊 RePaint BALANCED 4-SAMPLE WORKING TEST RESULTS"
echo "================================================="
echo ""
echo "🏗️  RePaint Results:"
echo "-------------------"
echo "Places2: $REPAINT_PLACES_STATUS - OAI_dataset/output/RePaint_Places_4/"
echo "CelebA:  $REPAINT_CELEBA_STATUS - OAI_dataset/output/RePaint_CelebA_4/"
echo ""

# Count successful models
SUCCESS_COUNT=0
if [ "$REPAINT_PLACES_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi
if [ "$REPAINT_CELEBA_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi

echo "📈 SUMMARY:"
echo "----------"
echo "Total RePaint Models Tested: 2"
echo "Successful Models: $SUCCESS_COUNT"
echo "Sample Size: 4 (2 Non-Osteoporotic + 2 Osteoporotic)"
echo "Configuration: Based on working backup version"
echo ""

if [ $SUCCESS_COUNT -eq 2 ]; then
    echo "🎉 ALL RePaint MODELS SUCCESS ON BALANCED 4-SAMPLE SUBSET!"
    echo "Check the output directories above for results."
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  $SUCCESS_COUNT out of 2 RePaint models succeeded"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO RePaint MODELS SUCCEEDED"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

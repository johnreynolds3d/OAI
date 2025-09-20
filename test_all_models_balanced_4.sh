#!/bin/bash

# Master Comprehensive Test Script (BALANCED 4-SAMPLE VERSION)
# Tests ALL available pre-trained models across ALL architectures with balanced 4-sample subset
# 2 Non-Osteoporotic (6.C.*) + 2 Osteoporotic (6.E.*) samples

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE (BALANCED 4-SAMPLE)"
echo "========================================================"
echo "Testing ALL available pre-trained models on balanced OAI dataset subset"
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

# Clear GPU memory at start
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test AOT-GAN models
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

# Test ICT models
echo ""
echo "🎯 Testing ICT Models (4-sample balanced)"
echo "========================================="

# ICT FFHQ
echo "Testing ICT FFHQ..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image="$BALANCED_DIR/img" \
    --input_mask="$BALANCED_DIR/mask" \
    --sample_num=1 \
    --save_place="../../OAI_dataset/output/ICT_FFHQ_4" \
    --FFHQ \
    --visualize_all 2>/dev/null

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
echo "Testing ICT ImageNet..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image="$BALANCED_DIR/img" \
    --input_mask="$BALANCED_DIR/mask" \
    --sample_num=1 \
    --save_place="../../OAI_dataset/output/ICT_ImageNet_4" \
    --ImageNet \
    --visualize_all 2>/dev/null

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
echo "Testing ICT Places2..."
cd "$SCRIPT_DIR" && python run.py \
    --input_image="$BALANCED_DIR/img" \
    --input_mask="$BALANCED_DIR/mask" \
    --sample_num=1 \
    --save_place="../../OAI_dataset/output/ICT_Places2_4" \
    --Places2_Nature \
    --visualize_all 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ICT Places2: SUCCESS"
    ICT_PLACES_STATUS="SUCCESS"
else
    echo "❌ ICT Places2: FAILED"
    ICT_PLACES_STATUS="FAILED"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test RePaint models
echo ""
echo "🎯 Testing RePaint Models (4-sample balanced)"
echo "============================================="

# Create RePaint configuration for balanced subset
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_balanced_4.yml" << EOF
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
num_samples: 1
show_progress: true
timestep_respacing: '25'
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
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: ./data/pretrained/places256_300000.pt
name: oai_test_balanced_4
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 25
  n_sample: 1
  jump_length: 3
  jump_n_sample: 3
data:
  eval:
    oai_test_balanced_4:
      mask_loader: true
      gt_path: $BALANCED_DIR/img
      mask_path: $BALANCED_DIR/mask_inv
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
      max_len: 4
      paths:
        srs: ../OAI_dataset/output/RePaint_Places_4/inpainted
        lrs: ../OAI_dataset/output/RePaint_Places_4/gt_masked
        gts: ../OAI_dataset/output/RePaint_Places_4/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_Places_4/gt_keep_mask
EOF

# Create output directories for RePaint
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places_4/{inpainted,gt_masked,gt,gt_keep_mask}

# RePaint Places2
echo "Testing RePaint Places2..."
cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_balanced_4.yml 2>/dev/null

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
echo "Testing RePaint CelebA..."
# Create CelebA configuration for balanced subset
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_celeba_balanced_4.yml" << EOF
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
num_samples: 1
show_progress: true
timestep_respacing: '25'
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
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: ./data/pretrained/celeba256_250000.pt
name: oai_test_celeba_balanced_4
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 25
  n_sample: 1
  jump_length: 3
  jump_n_sample: 3
data:
  eval:
    oai_test_celeba_balanced_4:
      mask_loader: true
      gt_path: $BALANCED_DIR/img
      mask_path: $BALANCED_DIR/mask_inv
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
      max_len: 4
      paths:
        srs: ../OAI_dataset/output/RePaint_CelebA_4/inpainted
        lrs: ../OAI_dataset/output/RePaint_CelebA_4/gt_masked
        gts: ../OAI_dataset/output/RePaint_CelebA_4/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_CelebA_4/gt_keep_mask
EOF

# Create output directories for RePaint CelebA
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_CelebA_4/{inpainted,gt_masked,gt,gt_keep_mask}

cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_celeba_balanced_4.yml 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ RePaint CelebA: SUCCESS"
    REPAINT_CELEBA_STATUS="SUCCESS"
else
    echo "❌ RePaint CelebA: FAILED"
    REPAINT_CELEBA_STATUS="FAILED"
fi

# Clean up temporary balanced subset
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$BALANCED_DIR"

# Generate comprehensive report
echo ""
echo "📊 BALANCED 4-SAMPLE TEST RESULTS"
echo "=================================="
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
echo ""

if [ $SUCCESS_COUNT -eq 7 ]; then
    echo "🎉 ALL MODELS SUCCESS ON BALANCED 4-SAMPLE SUBSET!"
    echo "Check the output directories above for results."
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  $SUCCESS_COUNT out of 7 models succeeded on balanced 4-sample subset"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO MODELS SUCCEEDED on balanced 4-sample subset"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

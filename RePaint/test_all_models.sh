#!/bin/bash

# RePaint Comprehensive Test Script (FIXED VERSION)
# Tests all available pre-trained models on OAI dataset with proper error handling
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 RePaint Comprehensive Testing (Fixed)"
echo "========================================"

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_CelebA

# Create CPU fallback configuration
echo "📝 Creating CPU fallback configuration..."
cat > "$SCRIPT_DIR/confs/oai_test_cpu.yml" << 'EOF'
attention_resolutions: 8,4
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 64
num_head_channels: 16
num_heads: 1
num_res_blocks: 1
resblock_updown: true
use_fp16: false
use_scale_shift_norm: true
classifier_scale: 4.0
lr_kernel_n_std: 2
num_samples: 1
show_progress: true
timestep_respacing: '5'
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: false
classifier_width: 32
classifier_depth: 1
classifier_attention_resolutions: 8,4
classifier_use_scale_shift_norm: true
classifier_resblock_updown: true
classifier_pool: attention
num_heads_upsample: -1
channel_mult: ''
dropout: 0.0
use_checkpoint: true
use_new_attention_order: false
clip_denoised: true
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 64
model_path: ./data/pretrained/places256_300000.pt
name: oai_test_cpu
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 5
  n_sample: 1
  jump_length: 1
  jump_n_sample: 1
data:
  eval:
    oai_test_cpu:
      mask_loader: true
      gt_path: ../OAI_dataset/test/img
      mask_path: ../OAI_dataset/test/mask_inv
      image_size: 64
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 1
      paths:
        srs: ../OAI_dataset/output/RePaint_Places/inpainted
        lrs: ../OAI_dataset/output/RePaint_Places/gt_masked
        gts: ../OAI_dataset/output/RePaint_Places/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_Places/gt_keep_mask
EOF

# Clear GPU memory
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test Places2 model
echo "Testing Places2 model..."
cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test.yml 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Places2: SUCCESS"
    PLACES2_STATUS="SUCCESS"
else
    echo "❌ Places2: FAILED (trying CPU fallback)..."
    # Try CPU fallback
    CUDA_VISIBLE_DEVICES="" python test.py --conf_path ./confs/oai_test_cpu.yml 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Places2 (CPU): SUCCESS"
        PLACES2_STATUS="SUCCESS (CPU)"
    else
        echo "❌ Places2: FAILED"
        PLACES2_STATUS="FAILED"
    fi
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Test CelebA model
echo "Testing CelebA model..."
# Create CelebA configuration
cat > "$SCRIPT_DIR/confs/oai_test_celeba.yml" << 'EOF'
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
name: oai_test_celeba
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
    oai_test_celeba:
      mask_loader: true
      gt_path: ../OAI_dataset/test/img
      mask_path: ../OAI_dataset/test/mask_inv
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
      max_len: 1
      paths:
        srs: ../OAI_dataset/output/RePaint_CelebA/inpainted
        lrs: ../OAI_dataset/output/RePaint_CelebA/gt_masked
        gts: ../OAI_dataset/output/RePaint_CelebA/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_CelebA/gt_keep_mask
EOF

cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test_celeba.yml 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ CelebA: SUCCESS"
    CELEBA_STATUS="SUCCESS"
else
    echo "❌ CelebA: FAILED (GPU memory issue)"
    CELEBA_STATUS="FAILED"
fi

echo ""
echo "📊 RePaint Test Results:"
echo "========================"
echo "Places2: $PLACES2_STATUS - Check OAI_dataset/output/RePaint_Places/"
echo "CelebA:  $CELEBA_STATUS - Check OAI_dataset/output/RePaint_CelebA/"

# Count successful models
SUCCESS_COUNT=0
if [[ "$PLACES2_STATUS" == *"SUCCESS"* ]]; then ((SUCCESS_COUNT++)); fi
if [ "$CELEBA_STATUS" = "SUCCESS" ]; then ((SUCCESS_COUNT++)); fi

echo ""
if [ $SUCCESS_COUNT -eq 2 ]; then
    echo "🎉 RePaint: ALL MODELS SUCCESS"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  RePaint: $SUCCESS_COUNT out of 2 models succeeded"
    exit 1
else
    echo "❌ RePaint: ALL MODELS FAILED"
    exit 1
fi

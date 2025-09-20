#!/bin/bash

# RePaint Comprehensive Test Script
# Tests all available pre-trained models on OAI dataset
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 RePaint Comprehensive Testing"
echo "================================"

# Create output directories
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_CelebA

# Test Places2 model
echo "Testing Places2 model..."
cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test.yml

if [ $? -eq 0 ]; then
    echo "✅ Places2: SUCCESS"
else
    echo "❌ Places2: FAILED (trying CPU fallback)..."
    # Try CPU fallback
    CUDA_VISIBLE_DEVICES="" python test.py --conf_path ./confs/oai_test_cpu.yml
    if [ $? -eq 0 ]; then
        echo "✅ Places2 (CPU): SUCCESS"
    else
        echo "❌ Places2: FAILED"
    fi
fi

# Test CelebA model (if available)
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
num_samples: 22
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
      max_len: 22
      paths:
        srs: ../OAI_dataset/output/RePaint_CelebA/inpainted
        lrs: ../OAI_dataset/output/RePaint_CelebA/gt_masked
        gts: ../OAI_dataset/output/RePaint_CelebA/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_CelebA/gt_keep_mask
EOF

cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test_celeba.yml

if [ $? -eq 0 ]; then
    echo "✅ CelebA: SUCCESS"
else
    echo "❌ CelebA: FAILED"
fi

echo ""
echo "📊 RePaint Test Results:"
echo "========================"
echo "Places2: Check OAI_dataset/output/RePaint_Places/"
echo "CelebA:  Check OAI_dataset/output/RePaint_CelebA/"
echo ""
echo "🎉 RePaint testing completed!"

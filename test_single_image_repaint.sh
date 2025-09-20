#!/bin/bash

# RePaint single image test script (CPU version)
# Tests one image using CPU to avoid GPU memory issues
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create temporary directory with single image
TEMP_DIR="/tmp/repaint_single_test"
mkdir -p "$TEMP_DIR/img" "$TEMP_DIR/mask_inv"

# Copy only the first image and mask
FIRST_IMG=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/img/*.png | head -1)
FIRST_MASK=$(ls /home/john/Documents/git/OAI/OAI_dataset/test/mask_inv/*.png | head -1)

if [ -n "$FIRST_IMG" ] && [ -n "$FIRST_MASK" ]; then
    cp "$FIRST_IMG" "$TEMP_DIR/img/"
    cp "$FIRST_MASK" "$TEMP_DIR/mask_inv/"
    
    echo "Testing RePaint with single image: $(basename "$FIRST_IMG")"
    
    # Create single-image configuration
    cat > "$TEMP_DIR/single_test.yml" << EOF
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
name: single_test
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
    single_test:
      mask_loader: true
      gt_path: $TEMP_DIR/img
      mask_path: $TEMP_DIR/mask_inv
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
        srs: $TEMP_DIR/output/inpainted
        lrs: $TEMP_DIR/output/gt_masked
        gts: $TEMP_DIR/output/gt
        gt_keep_masks: $TEMP_DIR/output/gt_keep_mask
EOF
    
    # Run RePaint on CPU
    cd "$SCRIPT_DIR/RePaint" && CUDA_VISIBLE_DEVICES="" python test.py --conf_path "$TEMP_DIR/single_test.yml"
    
    # Copy result to main output directory
    mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_single_test
    cp -r "$TEMP_DIR/output"/* /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_single_test/ 2>/dev/null || true
    
    echo "✅ RePaint single image test completed"
    echo "   Result saved to: OAI_dataset/output/RePaint_single_test/"
    
    # Clean up
    rm -rf "$TEMP_DIR"
else
    echo "❌ No test images found!"
    exit 1
fi

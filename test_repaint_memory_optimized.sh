#!/bin/bash

# RePaint Test Script (MEMORY OPTIMIZED VERSION)
# Optimized for 4GB GPU memory while maintaining the working configuration

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 RePaint Testing (MEMORY OPTIMIZED VERSION)"
echo "============================================="
echo "Testing RePaint with memory optimizations for 4GB GPU"
echo "Based on backup evidence but adapted for current hardware"
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places_optimized/{inpainted,gt_masked,gt,gt_keep_mask}

# Set environment variables for aggressive memory management
echo "🔧 Setting up memory-optimized environment..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Clear GPU memory before starting
echo "🧹 Clearing GPU memory..."
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
    print(f'GPU Available: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated')
    print(f'GPU Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB reserved')
else:
    print('No CUDA GPU available')
"

# Create memory-optimized configuration
echo "📝 Creating memory-optimized configuration..."
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_memory_optimized.yml" << 'EOF'
# RePaint configuration for OAI dataset testing (MEMORY OPTIMIZED VERSION)
# Based on working backup but optimized for 4GB GPU memory

attention_resolutions: 16,8
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 128
num_head_channels: 32
num_heads: 2
num_res_blocks: 1
resblock_updown: true
use_fp16: true
use_scale_shift_norm: true
classifier_scale: 2.0
lr_kernel_n_std: 2
num_samples: 4
show_progress: true
timestep_respacing: '10'
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: true
classifier_width: 64
classifier_depth: 1
classifier_attention_resolutions: 16,8
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
image_size: 128
model_path: ./data/pretrained/places256_300000.pt
name: oai_test_memory_optimized
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: true
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 10
  n_sample: 1
  jump_length: 2
  jump_n_sample: 2
data:
  eval:
    oai_test_memory_optimized:
      mask_loader: true
      gt_path: ../OAI_dataset/test/img
      mask_path: ../OAI_dataset/test/mask_inv
      image_size: 128
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
        srs: ../OAI_dataset/output/RePaint_Places_optimized/inpainted
        lrs: ../OAI_dataset/output/RePaint_Places_optimized/gt_masked
        gts: ../OAI_dataset/output/RePaint_Places_optimized/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_Places_optimized/gt_keep_mask
EOF

# Test RePaint with memory-optimized configuration
echo ""
echo "🎯 Testing RePaint Places2 (Memory Optimized)"
echo "============================================="

# Run RePaint with memory-optimized configuration
echo "Starting RePaint with memory-optimized configuration..."
echo "Configuration: confs/oai_test_memory_optimized.yml"
echo "Model: places256_300000.pt"
echo "Image size: 128x128 (reduced for memory)"
echo "Timesteps: 10 (ultra-fast mode)"
echo "Channels: 128 (reduced from 256)"
echo ""

# Run the test with proper error handling
cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_memory_optimized.yml

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ RePaint Places2: SUCCESS"
    echo "📊 Checking output..."
    
    # Count generated images
    INPAINTED_COUNT=$(ls ../OAI_dataset/output/RePaint_Places_optimized/inpainted/*.png 2>/dev/null | wc -l)
    echo "   Generated images: $INPAINTED_COUNT"
    
    if [ $INPAINTED_COUNT -gt 0 ]; then
        echo "   ✅ Images successfully generated!"
        echo "   📁 Output directory: ../OAI_dataset/output/RePaint_Places_optimized/inpainted/"
        
        # Show file sizes
        echo "   📏 File sizes:"
        ls -lah ../OAI_dataset/output/RePaint_Places_optimized/inpainted/*.png | head -5 | while read line; do
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
    echo "   Even with memory optimizations, GPU memory insufficient"
    echo "   💡 Try Google Colab with higher GPU memory"
    REPAINT_STATUS="FAILED"
fi

# Generate report
echo ""
echo "📊 RePaint MEMORY OPTIMIZED TEST RESULTS"
echo "========================================"
echo ""
echo "🏗️  RePaint Results:"
echo "-------------------"
echo "Places2: $REPAINT_STATUS - OAI_dataset/output/RePaint_Places_optimized/"
echo ""

if [ "$REPAINT_STATUS" = "SUCCESS" ]; then
    echo "🎉 RePaint SUCCESS with memory optimizations!"
    echo "✅ Generated images with reduced memory footprint"
    echo "📁 Check the output directory for results"
    exit 0
else
    echo "❌ RePaint FAILED even with memory optimizations"
    echo "⚠️  Your 4GB GPU memory is insufficient for RePaint"
    echo "💡 Strongly recommend using Google Colab (12-16GB GPU memory)"
    echo "   The backup evidence shows RePaint worked with similar hardware"
    echo "   but may have had different memory management or PyTorch version"
    exit 1
fi

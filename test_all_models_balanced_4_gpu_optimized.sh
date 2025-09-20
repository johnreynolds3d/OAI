#!/bin/bash

# Master Comprehensive Test Script (BALANCED 4-SAMPLE GPU-OPTIMIZED VERSION)
# Tests ALL available pre-trained models across ALL architectures with balanced 4-sample subset
# Optimized for 4GB GPU memory with reduced batch sizes, iterations, and model parameters

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 COMPREHENSIVE MODEL TESTING SUITE (BALANCED 4-SAMPLE GPU-OPTIMIZED)"
echo "======================================================================"
echo "Testing ALL available pre-trained models on balanced OAI dataset subset"
echo "Optimized for 4GB GPU memory"
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

# Test ICT models with GPU memory optimizations
echo ""
echo "🎯 Testing ICT Models (4-sample balanced, GPU-optimized)"
echo "========================================================"

# Create optimized ICT run script for GPU memory constraints
cat > "$SCRIPT_DIR/run_gpu_optimized.py" << 'EOF'
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--input_mask", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--save_place", type=str, required=True)
    parser.add_argument("--FFHQ", action="store_true")
    parser.add_argument("--ImageNet", action="store_true")
    parser.add_argument("--Places2_Nature", action="store_true")
    parser.add_argument("--visualize_all", action="store_true")
    
    opts = parser.parse_args()
    
    # Create output directory
    os.makedirs(opts.save_place, exist_ok=True)
    
    # Set GPU memory optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Determine model type and set optimized parameters
    if opts.FFHQ:
        # FFHQ with reduced parameters for 4GB GPU
        stage_1_command = (
            "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/FFHQ.pth "
            "--BERT --image_url " + opts.input_image + " "
            "--mask_url " + opts.input_mask + " "
            "--n_layer 30 --n_embd 512 --n_head 8 --top_k 40 --GELU_2 "
            "--save_url " + opts.save_place + "/AP "
            "--image_size 32 --n_samples " + str(opts.sample_num) + " "
            "--batch_size 1"
        )
        stage_2_command = (
            "CUDA_VISIBLE_DEVICES=0 python test.py --mode 2 "
            "--config ./configs/test_ffhq_4gpu.yml "
            "--input_image " + opts.input_image + " "
            "--input_mask " + opts.input_mask + " "
            "--output_dir " + opts.save_place + " "
            "--prior_dir " + opts.save_place + "/AP "
            "--upsample_ckpt ./experiments "
            "--batch_size 1"
        )
    elif opts.ImageNet:
        # ImageNet with reduced parameters
        stage_1_command = (
            "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/ImageNet.pth "
            "--BERT --image_url " + opts.input_image + " "
            "--mask_url " + opts.input_mask + " "
            "--n_layer 12 --n_embd 256 --n_head 4 --top_k 20 --GELU_2 "
            "--save_url " + opts.save_place + "/AP "
            "--image_size 24 --n_samples " + str(opts.sample_num) + " "
            "--batch_size 1"
        )
        stage_2_command = (
            "CUDA_VISIBLE_DEVICES=0 python test.py --mode 2 "
            "--config ./configs/test_imagenet_4gpu.yml "
            "--input_image " + opts.input_image + " "
            "--input_mask " + opts.input_mask + " "
            "--output_dir " + opts.save_place + " "
            "--prior_dir " + opts.save_place + "/AP "
            "--upsample_ckpt ./experiments "
            "--batch_size 1"
        )
    elif opts.Places2_Nature:
        # Places2 with reduced parameters
        stage_1_command = (
            "CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path ../ckpts_ICT/Transformer/Places2_Nature.pth "
            "--BERT --image_url " + opts.input_image + " "
            "--mask_url " + opts.input_mask + " "
            "--n_layer 12 --n_embd 256 --n_head 4 --top_k 20 --GELU_2 "
            "--save_url " + opts.save_place + "/AP "
            "--image_size 24 --n_samples " + str(opts.sample_num) + " "
            "--batch_size 1"
        )
        stage_2_command = (
            "CUDA_VISIBLE_DEVICES=0 python test.py --mode 2 "
            "--config ./configs/test_places2_4gpu.yml "
            "--input_image " + opts.input_image + " "
            "--input_mask " + opts.input_mask + " "
            "--output_dir " + opts.save_place + " "
            "--prior_dir " + opts.save_place + "/AP "
            "--upsample_ckpt ./experiments "
            "--batch_size 1"
        )
    else:
        print("Error: Must specify one of --FFHQ, --ImageNet, or --Places2_Nature")
        return 1
    
    # Run Stage 1 (Transformer)
    print("Running Stage 1 - Appearance Priors Reconstruction...")
    os.chdir("Transformer")
    result1 = subprocess.run(stage_1_command, shell=True, capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"Stage 1 failed: {result1.stderr}")
        return 1
    print("Finish the Stage 1 - Appearance Priors Reconstruction using Transformer")
    
    # Run Stage 2 (Guided Upsample)
    print("Running Stage 2 - Guided Upsampling...")
    os.chdir("../Guided_Upsample")
    result2 = subprocess.run(stage_2_command, shell=True, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"Stage 2 failed: {result2.stderr}")
        return 1
    print("Finish the Stage 2 - Guided Upsampling")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Create optimized config files for ICT
mkdir -p "$SCRIPT_DIR/ICT/Guided_Upsample/configs"

# FFHQ optimized config
cat > "$SCRIPT_DIR/ICT/Guided_Upsample/configs/test_ffhq_4gpu.yml" << 'EOF'
# FFHQ config optimized for 4GB GPU
model:
  name: "InpaintingModel"
  generator:
    name: "InpaintingGenerator"
    in_channels: 4
    out_channels: 3
    base_channels: 32  # Reduced from 64
    num_blocks: 6      # Reduced from 9
    use_attention: true
    attention_layers: [2, 4]  # Reduced attention layers
  discriminator:
    name: "PatchDiscriminator"
    in_channels: 3
    base_channels: 32  # Reduced from 64
    num_layers: 3      # Reduced from 4

data:
  batch_size: 1
  num_workers: 1
  image_size: 256
  crop_size: 256

training:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
  num_epochs: 200
  save_interval: 10

device: "cuda:0"
EOF

# ImageNet optimized config
cat > "$SCRIPT_DIR/ICT/Guided_Upsample/configs/test_imagenet_4gpu.yml" << 'EOF'
# ImageNet config optimized for 4GB GPU
model:
  name: "InpaintingModel"
  generator:
    name: "InpaintingGenerator"
    in_channels: 4
    out_channels: 3
    base_channels: 24  # Further reduced
    num_blocks: 4      # Further reduced
    use_attention: true
    attention_layers: [2]  # Minimal attention
  discriminator:
    name: "PatchDiscriminator"
    in_channels: 3
    base_channels: 24  # Further reduced
    num_layers: 2      # Further reduced

data:
  batch_size: 1
  num_workers: 1
  image_size: 128      # Reduced image size
  crop_size: 128

training:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
  num_epochs: 200
  save_interval: 10

device: "cuda:0"
EOF

# Places2 optimized config
cat > "$SCRIPT_DIR/ICT/Guided_Upsample/configs/test_places2_4gpu.yml" << 'EOF'
# Places2 config optimized for 4GB GPU
model:
  name: "InpaintingModel"
  generator:
    name: "InpaintingGenerator"
    in_channels: 4
    out_channels: 3
    base_channels: 24  # Further reduced
    num_blocks: 4      # Further reduced
    use_attention: true
    attention_layers: [2]  # Minimal attention
  discriminator:
    name: "PatchDiscriminator"
    in_channels: 3
    base_channels: 24  # Further reduced
    num_layers: 2      # Further reduced

data:
  batch_size: 1
  num_workers: 1
  image_size: 128      # Reduced image size
  crop_size: 128

training:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
  num_epochs: 200
  save_interval: 10

device: "cuda:0"
EOF

# ICT FFHQ
echo "Testing ICT FFHQ (GPU-optimized)..."
cd "$SCRIPT_DIR" && python run_gpu_optimized.py \
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
echo "Testing ICT ImageNet (GPU-optimized)..."
cd "$SCRIPT_DIR" && python run_gpu_optimized.py \
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
echo "Testing ICT Places2 (GPU-optimized)..."
cd "$SCRIPT_DIR" && python run_gpu_optimized.py \
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

# Test RePaint models with GPU memory optimizations
echo ""
echo "🎯 Testing RePaint Models (4-sample balanced, GPU-optimized)"
echo "============================================================"

# Create RePaint configuration optimized for 4GB GPU
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_balanced_4_gpu_optimized.yml" << EOF
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
num_samples: 1
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
name: oai_test_balanced_4_gpu_optimized
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
    oai_test_balanced_4_gpu_optimized:
      mask_loader: true
      gt_path: $BALANCED_DIR/img
      mask_path: $BALANCED_DIR/mask_inv
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
        srs: ../OAI_dataset/output/RePaint_Places_4/inpainted
        lrs: ../OAI_dataset/output/RePaint_Places_4/gt_masked
        gts: ../OAI_dataset/output/RePaint_Places_4/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_Places_4/gt_keep_mask
EOF

# Create output directories for RePaint
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_Places_4/{inpainted,gt_masked,gt,gt_keep_mask}

# RePaint Places2
echo "Testing RePaint Places2 (GPU-optimized)..."
cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_balanced_4_gpu_optimized.yml 2>/dev/null

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
echo "Testing RePaint CelebA (GPU-optimized)..."
# Create CelebA configuration optimized for 4GB GPU
cat > "$SCRIPT_DIR/RePaint/confs/oai_test_celeba_balanced_4_gpu_optimized.yml" << EOF
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
num_samples: 1
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
model_path: ./data/pretrained/celeba256_250000.pt
name: oai_test_celeba_balanced_4_gpu_optimized
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
    oai_test_celeba_balanced_4_gpu_optimized:
      mask_loader: true
      gt_path: $BALANCED_DIR/img
      mask_path: $BALANCED_DIR/mask_inv
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
        srs: ../OAI_dataset/output/RePaint_CelebA_4/inpainted
        lrs: ../OAI_dataset/output/RePaint_CelebA_4/gt_masked
        gts: ../OAI_dataset/output/RePaint_CelebA_4/gt
        gt_keep_masks: ../OAI_dataset/output/RePaint_CelebA_4/gt_keep_mask
EOF

# Create output directories for RePaint CelebA
mkdir -p /home/john/Documents/git/OAI/OAI_dataset/output/RePaint_CelebA_4/{inpainted,gt_masked,gt,gt_keep_mask}

cd "$SCRIPT_DIR/RePaint" && python test.py --conf_path ./confs/oai_test_celeba_balanced_4_gpu_optimized.yml 2>/dev/null

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
rm -f "$SCRIPT_DIR/run_gpu_optimized.py"

# Generate comprehensive report
echo ""
echo "📊 BALANCED 4-SAMPLE GPU-OPTIMIZED TEST RESULTS"
echo "==============================================="
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
echo "GPU Memory: Optimized for 4GB"
echo ""

if [ $SUCCESS_COUNT -eq 7 ]; then
    echo "🎉 ALL MODELS SUCCESS ON BALANCED 4-SAMPLE GPU-OPTIMIZED SUBSET!"
    echo "Check the output directories above for results."
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  $SUCCESS_COUNT out of 7 models succeeded on balanced 4-sample GPU-optimized subset"
    echo "Check the output directories above for results."
    exit 1
else
    echo "❌ NO MODELS SUCCEEDED on balanced 4-sample GPU-optimized subset"
    echo "Check GPU memory and model compatibility."
    exit 1
fi

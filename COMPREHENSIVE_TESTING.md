# 🚀 Comprehensive Model Testing Suite

This repository contains a complete testing framework for three state-of-the-art image inpainting models on the OAI (Osteoarthritis Initiative) dataset.

## 📋 Available Models

### AOT-GAN (Attention-aware Outpainting Generative Adversarial Network)
- **CelebA-HQ**: `AOT-GAN-for-Inpainting/test_all_models.sh`
- **Places2**: Included in the same script

### ICT (Image Completion Transformer)
- **FFHQ**: `ICT/test_all_models.sh`
- **ImageNet**: Included in the same script
- **Places2**: Included in the same script

### RePaint (Denoising Diffusion Probabilistic Models)
- **Places2**: `RePaint/test_all_models.sh`
- **CelebA**: Included in the same script

## 🎯 Quick Start

### Test All Models (Recommended)
```bash
cd /home/john/Documents/git/OAI
./test_all_models_comprehensive.sh
```

### Test Individual Architectures
```bash
# Test AOT-GAN only
AOT-GAN-for-Inpainting/test_all_models.sh

# Test ICT only
ICT/test_all_models.sh

# Test RePaint only
RePaint/test_all_models.sh
```

## 📁 Output Structure

All results are saved in `OAI_dataset/output/` with the following structure:

```
OAI_dataset/output/
├── AOT_celebahq/          # AOT-GAN CelebA-HQ results
├── AOT_places2/           # AOT-GAN Places2 results
├── ICT_FFHQ/              # ICT FFHQ results
├── ICT_ImageNet/          # ICT ImageNet results
├── ICT_Places2/           # ICT Places2 results
├── RePaint_Places/        # RePaint Places2 results
└── RePaint_CelebA/        # RePaint CelebA results
```

## 🔧 Hardware Requirements

The testing framework automatically adapts to your hardware:

- **High-end GPU (8GB+)**: Full testing with all models
- **Medium GPU (4-8GB)**: Memory-optimized testing
- **Low GPU (2-4GB)**: Single-image validation
- **CPU only**: CPU-optimized testing

## 📊 Test Results

Each test provides:
- ✅ **SUCCESS**: Model completed successfully
- ❌ **FAILED**: Model encountered errors
- 📁 **Output**: Generated inpainted images saved to respective directories

## 🎉 Features

- **Comprehensive Coverage**: Tests 7 different pre-trained model variants
- **Hardware Adaptive**: Automatically adjusts to available resources
- **Dedicated Outputs**: Each model variant has its own output directory
- **Error Handling**: Graceful fallback for failed models
- **Progress Tracking**: Real-time status updates during testing

## 🚀 Getting Started

1. Ensure you have the required dependencies installed
2. Run the comprehensive test: `./test_all_models_comprehensive.sh`
3. Check the output directories for results
4. Compare different model architectures and pre-trained variants

## 📈 Model Comparison

| Architecture | Pre-trained Models | Best For |
|-------------|-------------------|----------|
| AOT-GAN | CelebA-HQ, Places2 | Fast, reliable inpainting |
| ICT | FFHQ, ImageNet, Places2 | High-quality, complex scenes |
| RePaint | Places2, CelebA | Diffusion-based, diverse outputs |

---

**Happy Testing! 🎨✨**

# Test Scripts for OAI Dataset

This document lists all available test scripts for running different inpainting models on the OAI dataset.

## AOT-GAN Models

### CelebA-HQ Pre-trained Model
```bash
cd AOT-GAN-for-Inpainting
./test_celebahq.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask`
- **Output**: `OAI_dataset/output/AOT_celebahq/`
- **Pre-trained model**: `experiments/celebahq/G0000000.pt`

### Places2 Pre-trained Model
```bash
cd AOT-GAN-for-Inpainting
./test_places2.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask`
- **Output**: `OAI_dataset/output/AOT_places2/`
- **Pre-trained model**: `experiments/places2/G0000000.pt`

## ICT Models

### FFHQ Pre-trained Model
```bash
cd ICT
./test_ffhq.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask`
- **Output**: `OAI_dataset/output/ICT_FFHQ/`
- **Pre-trained models**: 
  - Transformer: `ckpts_ICT/Transformer/FFHQ.pth`
  - Upsample: `ckpts_ICT/Upsample/FFHQ/`

### ImageNet Pre-trained Model
```bash
cd ICT
./test_imagenet.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask`
- **Output**: `OAI_dataset/output/ICT_ImageNet/`
- **Pre-trained models**: 
  - Transformer: `ckpts_ICT/Transformer/ImageNet.pth`
  - Upsample: `ckpts_ICT/Upsample/ImageNet/`

### Places2 Pre-trained Model
```bash
cd ICT
./test_places2.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask`
- **Output**: `OAI_dataset/output/ICT_Places2/`
- **Pre-trained models**: 
  - Transformer: `ckpts_ICT/Transformer/Places2_Nature.pth`
  - Upsample: `ckpts_ICT/Upsample/Places2_Nature/`

## RePaint Model

### Places2 Pre-trained Model
```bash
cd RePaint
./test_oai.sh
```
- **Input**: `OAI_dataset/test/img` and `OAI_dataset/test/mask_inv`
- **Output**: `OAI_dataset/output/RePaint_Places/`
- **Configuration**: `confs/oai_test.yml`
- **Pre-trained model**: `data/pretrained/places256_300000.pt`

## Output Files

Each model generates different types of output files:

### AOT-GAN Output
- `{image_name}_masked.png` - Input image with mask applied
- `{image_name}_pred.png` - Predicted inpainted region
- `{image_name}_comp.png` - Final composite result

### ICT Output
- Various intermediate and final results in the specified output directory

### RePaint Output
- Inpainted results in the specified output directory

## Notes

- All scripts are executable and follow the `{model_name}/test_{pretrained_model}.sh` format
- Output directories are automatically created if they don't exist
- All scripts use the OAI test dataset with 22 images
- Make sure you have sufficient GPU memory and CUDA available for running the models

## ⚠️ ICT Model Status

**Current Status**: ICT models have been partially fixed but still have issues:

### ✅ **Fixed Issues**:
- Architecture mismatches resolved (using correct model parameters)
- Missing kmeans_centers.npy files restored from backup
- Path issues corrected
- Fast test scripts created for quicker testing

### ⚠️ **Remaining Issues**:
- **GPU Memory**: Requires significant GPU memory (4GB+ recommended)
- **Complex Dependencies**: Needs edge detection files and multiple processing stages
- **Processing Time**: Even with optimizations, still takes considerable time

### 🚀 **Recommended Models for Testing**:
- ✅ **AOT-GAN**: Fully working, fast, reliable (tested with CelebA-HQ and Places2)
- ✅ **RePaint**: Fully working, moderate speed (tested with Places2)
- ⚠️ **ICT**: Complex setup, requires significant resources

### 📝 **ICT Testing Options**:
- `test_ffhq_fast.sh` - FFHQ model with smaller image size (24x24)
- `test_places2_fast.sh` - Places2 model with smaller image size (16x16)  
- `test_imagenet_fast.sh` - ImageNet model with smaller image size (16x16)
- `test_single_image.sh` - Test with just one image for quick validation

**For reliable testing, use AOT-GAN or RePaint. ICT is available but requires more setup and resources.**

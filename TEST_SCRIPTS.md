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

## ⚠️ ICT Model Compatibility Issues

**Current Status**: ICT models may have compatibility issues due to architecture mismatches between pre-trained models and the inference code.

**Recommended Models for Testing**:
- ✅ **AOT-GAN**: Fully working (tested with CelebA-HQ and Places2)
- ✅ **RePaint**: Fully working (tested with Places2)
- ⚠️ **ICT**: May require model retraining or architecture adjustments

If you encounter errors with ICT models, consider using AOT-GAN or RePaint instead, or training ICT models specifically for your OAI dataset.

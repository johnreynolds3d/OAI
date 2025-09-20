# Google Colab Setup Guide for OAI Dataset Testing

This guide will help you run the comprehensive balanced 4-sample testing framework in Google Colab with full GPU memory access.

## 🚀 Quick Start

### 1. Open Google Colab
- Go to [Google Colab](https://colab.research.google.com/)
- Create a new notebook
- **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better)

### 2. Clone the Repository
```python
!git clone https://github.com/johnreynolds3d/OAI.git
%cd OAI
```

### 3. Install Dependencies
```python
# Install PyTorch (if not already installed)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install opencv-python pillow matplotlib numpy scipy
!pip install pyyaml
!pip install tqdm
```

### 4. Upload Your OAI Dataset
```python
# Upload your OAI_dataset folder to Colab
# You can use the file upload widget or drag and drop
from google.colab import files
uploaded = files.upload()

# Or use this to upload a zip file
!unzip -q OAI_dataset.zip  # if you uploaded as zip
```

### 5. Run the Comprehensive Test
```python
# Make the script executable and run
!chmod +x test_all_models_balanced_4_comprehensive_final.sh
!./test_all_models_balanced_4_comprehensive_final.sh
```

## 🔧 Colab-Specific Optimizations

### Memory Management
```python
# Clear GPU memory before each test
import torch
torch.cuda.empty_cache()

# Check GPU memory
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

### Environment Variables
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

## 📊 Expected Results in Colab

With Colab's higher GPU memory (12-16GB), you should get:

### ✅ **All Models Working (7/7):**
- **AOT-GAN**: Both CelebA-HQ and Places2
- **ICT**: All three models (FFHQ, ImageNet, Places2)  
- **RePaint**: Both Places2 and CelebA (now working with more memory!)

### 📁 **Output Structure:**
```
OAI_dataset/output/
├── AOT_celebahq_4/          # AOT-GAN CelebA-HQ results
├── AOT_places2_4/           # AOT-GAN Places2 results
├── ICT_FFHQ_4/              # ICT FFHQ results
├── ICT_ImageNet_4/          # ICT ImageNet results
├── ICT_Places2_4/           # ICT Places2 results
├── RePaint_Places_4/        # RePaint Places2 results
└── RePaint_CelebA_4/        # RePaint CelebA results
```

## 🎯 Balanced Sample Analysis

The framework tests on a perfectly balanced subset:
- **2 Non-Osteoporotic samples** (6.C.*)
- **2 Osteoporotic samples** (6.E.*)

This provides ideal conditions for comparative analysis between the two groups.

## 🔍 Troubleshooting

### If RePaint Still Fails:
```python
# Try reducing image size in RePaint configs
# Edit RePaint/confs/oai_test_balanced_4_working.yml
# Change image_size: 256 to image_size: 128
```

### If ICT Fails:
```python
# Ensure all dependencies are installed
!pip install transformers
!pip install einops
```

### Memory Issues:
```python
# Restart runtime and try again
# Runtime → Restart runtime
```

## 📈 Performance Expectations

- **AOT-GAN**: ~2-3 minutes total
- **ICT**: ~5-10 minutes total (single-image approach)
- **RePaint**: ~10-15 minutes total (with full GPU memory)
- **Total Time**: ~20-30 minutes for all models

## 🎉 Success Indicators

Look for these messages:
- `✅ AOT-GAN CelebA-HQ: SUCCESS`
- `✅ AOT-GAN Places2: SUCCESS`
- `✅ ICT FFHQ: SUCCESS`
- `✅ ICT ImageNet: SUCCESS`
- `✅ ICT Places2: SUCCESS`
- `✅ RePaint Places2: SUCCESS`
- `✅ RePaint CelebA: SUCCESS`

## 📝 Notes

1. **Colab Session Time**: Free Colab sessions have time limits. Consider upgrading to Colab Pro for longer sessions.

2. **File Persistence**: Colab files are temporary. Download results or save to Google Drive.

3. **GPU Availability**: Sometimes Colab doesn't provide GPU. Check with `torch.cuda.is_available()`.

4. **Repository Updates**: The latest changes are already pushed to the main branch.

## 🚀 Ready to Test!

You now have everything needed to run the comprehensive testing framework in Google Colab with full GPU memory access. The balanced 4-sample approach will give you perfect comparative analysis between Osteoporotic and non-Osteoporotic samples across all three model architectures.

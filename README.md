# OAI: X-ray Inpainting for Osteoporosis Prediction

This repository contains three state-of-the-art image inpainting models (AOT-GAN, RePaint, and ICT) applied to osteoporosis analysis using X-ray images. The project includes pretrained models, data processing utilities, and comprehensive evaluation tools.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Model Weights](#model-weights)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🔧 Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 2.8.0+
- 8GB+ GPU memory recommended

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/johnreynolds3d/OAI.git
cd OAI
```

### 2. Install Dependencies

#### For ICT (Image Completion Transformer)
```bash
cd ICT
pip install -r requirements.txt
cd ..
```

#### For AOT-GAN and RePaint
```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install opencv-python scikit-image matplotlib pillow tqdm
pip install --upgrade gdown  # For downloading pretrained models
```

### 3. Download Pretrained Models

#### AOT-GAN Models
```bash
# Download AOT-GAN pretrained models
# CELEBA-HQ: https://drive.google.com/drive/folders/1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ?usp=sharing
# Places2: https://drive.google.com/drive/folders/1bSOH-2nBfeFRyDEmiX81CEiWkghss3i?usp=sharing

# Place downloaded models in:
# AOT-GAN-for-Inpainting/experiments/celebahq/
# AOT-GAN-for-Inpainting/experiments/places2/
```

#### RePaint Models
```bash
cd RePaint
bash download.sh
cd ..
```

#### ICT Models
```bash
# ICT models are already included in the repository:
# ICT/ckpts_ICT/Transformer/ (FFHQ, ImageNet, Places2 models)
# ICT/ckpts_ICT/Upsample/ (Upsampling models)
```

## 📊 Dataset Setup

### 1. Prepare the OAI Dataset
```bash
cd OAI_dataset
python split.py  # Creates train/valid/test splits and generates masks
```

### 2. Dataset Structure
```
OAI_dataset/
├── img/                    # Original X-ray images (539 files)
├── data.csv               # Bone mineral density labels
├── split.py               # Dataset splitting script
├── resnet50.py            # BMD analysis model
├── train/                 # Training split (172 images)
│   ├── img/              # Training images
│   ├── mask/             # Generated masks
│   └── mask_inv/         # Inverted masks (for RePaint)
├── valid/                 # Validation split (22 images)
└── test/                  # Test split (22 images)
```

## 🚀 Usage

### AOT-GAN Inpainting
```bash
# Test with CelebA-HQ pretrained model
cd AOT-GAN-for-Inpainting
bash test.sh

# Test with Places2 pretrained model
cd AOT-GAN-for-Inpainting/src && python test.py \
    --dir_image="../../OAI_dataset/test/img" \
    --dir_mask="../../OAI_dataset/test/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_places2" \
    --pre_train="../experiments/places2/G0000000.pt"
```

### RePaint Inpainting
```bash
# Test with OAI dataset
cd RePaint
bash test_oai.sh
```

### ICT (Image Completion Transformer)
```bash
# Test with OAI-trained model
cd ICT
bash test_oai_trained.sh

# Test with pretrained models
bash test_ffhq.sh      # FFHQ pretrained
bash test_imagenet.sh  # ImageNet pretrained
bash test_places2.sh   # Places2 pretrained
```

### Bone Mineral Density Analysis
```bash
cd OAI_dataset
python resnet50.py  # Analyze BMD using ResNet50 model
```

## 📁 Project Structure

```
OAI/
├── AOT-GAN-for-Inpainting/     # AOT-GAN implementation
│   ├── src/                    # Source code
│   ├── experiments/            # Pretrained models
│   └── test.sh                 # Test script
├── ICT/                        # Image Completion Transformer
│   ├── Transformer/            # Main ICT implementation
│   ├── Guided_Upsample/        # Guided upsampling
│   ├── ckpts_ICT/             # Pretrained checkpoints
│   └── run_oai.py             # OAI-specific runner
├── RePaint/                    # RePaint implementation
│   ├── guided_diffusion/       # Diffusion model code
│   ├── data/pretrained/        # Pretrained models
│   └── test_oai.sh            # OAI test script
├── OAI_dataset/               # Osteoporosis dataset
│   ├── img/                   # X-ray images
│   ├── data.csv              # BMD labels
│   ├── split.py              # Dataset splitting
│   └── resnet50.py           # BMD analysis
├── utils/                     # Data processing utilities
│   ├── create_mask.py        # Mask generation
│   ├── prepare_repaint_data.py # RePaint data prep
│   └── analyze_osteoporosis.py # Analysis tools
└── README.md                  # This file
```

## 🔬 Research Applications

This repository enables:
- **Osteoporosis Detection**: Bone mineral density analysis using ResNet50
- **Image Inpainting**: Three different inpainting approaches for medical images
- **Comparative Analysis**: Direct comparison of AOT-GAN, RePaint, and ICT
- **Medical Image Processing**: Specialized tools for X-ray image analysis

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{oai2024,
  title={OAI: Osteoporosis Analysis and Inpainting},
  author={John Reynolds},
  year={2025},
  url={https://github.com/johnreynolds3d/OAI}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues, please open an issue on GitHub or contact [john@johnreynolds3d.com].
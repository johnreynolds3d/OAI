#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR" && python run_oai.py --input_image ../OAI_dataset/test/img \
    --input_mask ../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../OAI_dataset/output/ICT_Places2 \
    --transformer_ckpt ./ckpts_ICT/Transformer/Places2_Nature.pth \
    --upsample_ckpt ./ckpts_ICT/Upsample/Places2_Nature

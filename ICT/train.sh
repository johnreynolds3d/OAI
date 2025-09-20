#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR/Transformer" && python main.py \
    --name ICT_OAI_6_epochs \
    --ckpt_path experiments \
    --data_path ../../OAI_dataset/img \
    --validation_path ../../OAI_dataset/validation \
    --mask_path ../../OAI_dataset/pconv \
    --BERT \
    --batch_size 4 \
    --train_epoch 6 \
    --nodes 1 \
    --gpus 1 \
    --node_rank 0 \
    --n_layer 12 \
    --n_embd 256 \
    --n_head 8 \
    --GELU_2 \
    --image_size 32 \
    --lr 3e-4 \
    --print_freq 25 \
    --dynamic_weight \
    --AMP

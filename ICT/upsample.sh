#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR/Guided_Upsample" && python train.py \
    --model 2 \
    --checkpoints ./experiments \
    --config_file ./config_list/config_template.yml \
    --Generator 4 \
    --use_degradation_2

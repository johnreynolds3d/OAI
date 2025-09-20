#!/bin/bash

# RePaint test script for OAI dataset (FAST VERSION)
# Optimized for limited GPU memory with smaller image size and FP16
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test_fast.yml

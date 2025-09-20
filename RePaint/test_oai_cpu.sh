#!/bin/bash

# RePaint test script for OAI dataset (CPU VERSION)
# Runs on CPU with very small image size for testing
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR" && CUDA_VISIBLE_DEVICES="" python test.py --conf_path ./confs/oai_test_cpu.yml

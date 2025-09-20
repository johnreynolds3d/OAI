#!/bin/bash

# RePaint test script for OAI dataset (MEMORY OPTIMIZED)
# Uses original architecture with memory optimizations (FP16, checkpointing, single image)
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory and run from there
cd "$SCRIPT_DIR" && python test.py --conf_path ./confs/oai_test_memory_optimized.yml

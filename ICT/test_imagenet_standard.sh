#!/bin/bash
cd ICT && python run.py --input_image /home/john/Documents/OAI/OAI_dataset/test/img \
    --input_mask /home/john/Documents/OAI/OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place /home/john/Documents/OAI/OAI_dataset/output/ICT_ImageNet \
    --ImageNet \
    --visualize_all

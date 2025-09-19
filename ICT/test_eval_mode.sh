#!/bin/bash
cd ICT && python run_oai.py --input_image /home/john/Documents/OAI/OAI_dataset/test/img \
    --input_mask /home/john/Documents/OAI/OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place /home/john/Documents/OAI/OAI_dataset/output/ICT_Eval \
    --transformer_ckpt /home/john/Documents/OAI/ICT/Transformer/experiments/ICT_OAI_5_epochs/best.pth \
    --upsample_ckpt /home/john/Documents/OAI/ICT/Guided_Upsample/experiments \
    --visualize_all

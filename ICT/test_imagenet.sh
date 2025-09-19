#!/bin/bash
cd ICT && python run_oai.py --input_image ../OAI_dataset/test/img \
    --input_mask ../OAI_dataset/test/mask \
    --sample_num 1 \
    --save_place ../OAI_dataset/output/ICT_ImageNet \
    --transformer_ckpt ../ckpts_ICT/Transformer/ImageNet.pth \
    --upsample_ckpt ../ckpts_ICT/Upsample/ImageNet

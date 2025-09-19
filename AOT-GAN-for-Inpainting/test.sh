cd AOT-GAN-for-Inpainting/src && python test.py \
    --dir_image="../../OAI_dataset/test/img" \
    --dir_mask="../../OAI_dataset/test/mask" \
    --image_size=256 \
    --outputs="../../OAI_dataset/output/AOT_celebahq" \
    --pre_train="../experiments/celebahq/G0000000.pt"
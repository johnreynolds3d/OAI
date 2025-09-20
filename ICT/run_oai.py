import os
import argparse
import shutil
import sys
from subprocess import call


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image", type=str, default="./", help="The test input image path"
    )
    parser.add_argument(
        "--input_mask", type=str, default="./", help="The test input mask path"
    )
    parser.add_argument(
        "--sample_num", type=int, default=10, help="# of completion results"
    )
    parser.add_argument(
        "--save_place", type=str, default="./save", help="Please use the absolute path"
    )
    parser.add_argument(
        "--transformer_ckpt",
        type=str,
        default="./Transformer/experiments/ICT_OAI_5_epochs/best.pth",
        help="Path to your trained Transformer checkpoint",
    )
    parser.add_argument(
        "--upsample_ckpt",
        type=str,
        default="./Guided_Upsample/experiments",
        help="Path to your trained Guided_Upsample checkpoint",
    )
    parser.add_argument(
        "--visualize_all",
        action="store_true",
        help="show the diverse results in one row",
    )

    opts = parser.parse_args()

    ### Stage1: Reconstruction of Appearance Priors using Transformer

    prior_url = os.path.join(opts.save_place, "AP")
    if os.path.exists(prior_url):
        print("Please change the save path")
        sys.exit(1)
    os.chdir("./Transformer")

    if opts.visualize_all:
        suffix_opt = " --same_face"
        test_batch_size = str(opts.sample_num)
    else:
        suffix_opt = ""
        test_batch_size = str(1)

    # Use your trained OAI Transformer checkpoint
    stage_1_command = f"CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt_path {opts.transformer_ckpt} \
                            --BERT --image_url {opts.input_image} \
                            --mask_url {opts.input_mask} \
                            --n_layer 12 --n_embd 256 --n_head 8 --top_k 40 --GELU_2 \
                            --save_url {prior_url} \
                            --image_size 32 --n_samples {opts.sample_num}"

    run_cmd(stage_1_command)

    print("Finish the Stage 1 - Appearance Priors Reconstruction using Transformer")

    os.chdir("../Guided_Upsample")

    # Use your trained OAI Guided_Upsample checkpoint
    # Use edge detection files from the same directory as input images
    edge_dir = os.path.join(os.path.dirname(opts.input_image), "edge")
    stage_2_command = f"CUDA_VISIBLE_DEVICES=0 python test.py --input {opts.input_image} \
                                    --mask {opts.input_mask} \
                                    --prior {edge_dir} \
                                    --output {opts.save_place} \
                                    --checkpoints {opts.upsample_ckpt} \
                                    --test_batch_size {test_batch_size} --model 2 --Generator 4 --condition_num {opts.sample_num}{suffix_opt}"

    run_cmd(stage_2_command)

    print("Finish the Stage 2 - Guided Upsampling")
    print("Please check the results ...")

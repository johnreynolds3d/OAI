#!/usr/bin/env python3
"""
Balanced Dataset Splitter for OAI Dataset with Integrated Mask Generation
Creates train/validation/test splits with equal representation of osteoporotic and non-osteoporotic samples.
Also generates corresponding masks for each split.

Split: 80% train, 10% validation, 10% test
Each subset maintains equal balance of osteoporotic vs non-osteoporotic samples.
Masks are generated with 37x37 squares (1/6 of image size) avoiding outer 20% boundaries.
"""

import os
import shutil
import pandas as pd
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from collections import Counter


def generate_mask_for_image(image_path, output_path, image_size=224):
    """
    Generate a mask with one small square for a given image.

    Args:
        image_path: Path to the original image
        output_path: Path where the mask should be saved
        image_size: Size of the image (assumed square)
    """

    # Create a white background image
    img = np.zeros([image_size, image_size, 3], np.uint8)
    img.fill(255)

    # Create a black mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Calculate the safe zone (avoiding outer 20% of boundaries)
    margin = int(image_size * 0.2)  # 20% margin
    safe_zone_min = margin
    safe_zone_max = image_size - margin

    # Fixed square size: 1/6 of image dimensions
    square_size = int(image_size / 6)  # 37x37 for 224x224 images

    # Random position within safe zone
    x1 = random.randint(safe_zone_min, safe_zone_max - square_size)
    y1 = random.randint(safe_zone_min, safe_zone_max - square_size)
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw white rectangle on the mask (white = masked area)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Save the mask
    cv2.imwrite(output_path, mask)

    return mask


def create_inverted_mask(mask_path, inverted_mask_path):
    """
    Create an inverted mask for RePaint.
    RePaint expects inverted masks where 1 = keep, 0 = inpaint.

    Args:
        mask_path: Path to the original mask
        inverted_mask_path: Path to save the inverted mask
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Invert mask: 0 becomes 255, 255 becomes 0
    inverted_mask = 255 - mask

    # Save inverted mask
    cv2.imwrite(inverted_mask_path, inverted_mask)

    return inverted_mask


def generate_masks_for_split(split_df, split_name, img_dir, mask_dir, mask_inv_dir):
    """
    Generate masks and inverted masks for all images in a split.

    Args:
        split_df: DataFrame containing the split data
        split_name: Name of the split (train/valid/test)
        img_dir: Directory containing the images for this split
        mask_dir: Directory where masks should be saved
        mask_inv_dir: Directory where inverted masks should be saved
    """

    print(f"\n🎭 Generating masks for {split_name} set...")

    # Ensure mask directories exist
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_inv_dir, exist_ok=True)

    successful = 0
    failed = 0

    for _, row in split_df.iterrows():
        try:
            # Create mask filename (same as image but ensure .png extension)
            mask_filename = os.path.splitext(row["filename"])[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)
            inverted_mask_path = os.path.join(mask_inv_dir, mask_filename)

            # Generate mask
            image_path = os.path.join(img_dir, row["filename"])
            if os.path.exists(image_path):
                mask = generate_mask_for_image(image_path, mask_path)
                # Create inverted mask for RePaint
                create_inverted_mask(mask_path, inverted_mask_path)
                successful += 1
            else:
                print(f"Warning: {image_path} not found!")
                failed += 1

        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
            failed += 1

    print(f"   {split_name} masks: {successful} successful, {failed} failed")
    return successful, failed


def create_balanced_splits():
    """
    Create balanced train/validation/test splits ensuring equal representation
    of osteoporotic and non-osteoporotic samples in each subset.
    Also generate corresponding masks for each split.
    """

    # Load the CSV data
    print("Loading BMD data...")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data.csv")
    df = pd.read_csv(csv_path, header=None, names=["BMD", "filename"])
    print(f"Total samples: {len(df)}")

    # Define osteoporosis threshold (using 25th percentile as we determined earlier)
    osteo_threshold = df["BMD"].quantile(0.25)
    print(f"Osteoporosis threshold (25th percentile): {osteo_threshold:.4f}")

    # Classify samples
    df["is_osteo"] = df["BMD"] <= osteo_threshold
    df["class"] = df["is_osteo"].map({True: "osteoporotic", False: "normal"})

    # Check class distribution
    class_counts = df["class"].value_counts()
    print(f"\nClass distribution:")
    print(
        f"Osteoporotic: {class_counts['osteoporotic']} ({class_counts['osteoporotic']/len(df)*100:.1f}%)"
    )
    print(
        f"Normal: {class_counts['normal']} ({class_counts['normal']/len(df)*100:.1f}%)"
    )

    # Separate osteoporotic and normal samples
    osteo_samples = df[df["is_osteo"] == True].reset_index(drop=True)
    normal_samples = df[df["is_osteo"] == False].reset_index(drop=True)

    print(f"\nOsteoporotic samples: {len(osteo_samples)}")
    print(f"Normal samples: {len(normal_samples)}")

    # Calculate split sizes for each class
    # We'll use the smaller class as the limiting factor to ensure balance
    min_class_size = min(len(osteo_samples), len(normal_samples))

    # Calculate how many samples we can use from each class
    # We'll use 80% of the smaller class to ensure we can balance all splits
    max_samples_per_class = int(min_class_size * 0.8)

    print(
        f"\nUsing {max_samples_per_class} samples from each class for balanced splits"
    )

    # Randomly sample equal numbers from each class
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For mask generation reproducibility
    osteo_subset = osteo_samples.sample(
        n=max_samples_per_class, random_state=42
    ).reset_index(drop=True)
    normal_subset = normal_samples.sample(
        n=max_samples_per_class, random_state=42
    ).reset_index(drop=True)

    # Combine the balanced dataset
    balanced_df = pd.concat([osteo_subset, normal_subset], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # Shuffle

    print(f"Balanced dataset size: {len(balanced_df)}")
    print(
        f"Osteoporotic in balanced set: {len(balanced_df[balanced_df['is_osteo'] == True])}"
    )
    print(
        f"Normal in balanced set: {len(balanced_df[balanced_df['is_osteo'] == False])}"
    )

    # Create splits: 80% train, 10% validation, 10% test
    # First split: 80% train, 20% temp (for val+test)
    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.2, random_state=42, stratify=balanced_df["is_osteo"]
    )

    # Second split: Split the 20% temp into 10% validation and 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["is_osteo"]
    )

    # Verify splits
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(balanced_df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(balanced_df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(balanced_df)*100:.1f}%)")

    # Verify class balance in each split
    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        osteo_count = len(split_df[split_df["is_osteo"] == True])
        normal_count = len(split_df[split_df["is_osteo"] == False])
        total = len(split_df)
        print(f"\n{split_name} set class balance:")
        print(f"  Osteoporotic: {osteo_count} ({osteo_count/total*100:.1f}%)")
        print(f"  Normal: {normal_count} ({normal_count/total*100:.1f}%)")

    # Create output directories inside OAI_dataset
    output_dirs = [
        os.path.join(script_dir, "train/img"),
        os.path.join(script_dir, "train/mask"),
        os.path.join(script_dir, "train/mask_inv"),
        os.path.join(script_dir, "valid/img"),
        os.path.join(script_dir, "valid/mask"),
        os.path.join(script_dir, "valid/mask_inv"),
        os.path.join(script_dir, "test/img"),
        os.path.join(script_dir, "test/mask"),
        os.path.join(script_dir, "test/mask_inv"),
    ]
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {os.path.relpath(dir_name, script_dir)}/")

    # Copy images to respective directories
    print(f"\n📁 Copying images...")

    def copy_images(df, target_dir):
        """Copy images from the dataframe to the target directory"""
        copied_count = 0
        missing_count = 0

        for _, row in df.iterrows():
            src_path = os.path.join(script_dir, "img", row["filename"])
            dst_path = os.path.join(target_dir, row["filename"])

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: {src_path} not found!")
                missing_count += 1

        return copied_count, missing_count

    # Copy images for each split
    train_copied, train_missing = copy_images(
        train_df, os.path.join(script_dir, "train/img")
    )
    val_copied, val_missing = copy_images(val_df, os.path.join(script_dir, "valid/img"))
    test_copied, test_missing = copy_images(
        test_df, os.path.join(script_dir, "test/img")
    )

    print(f"\nCopy results:")
    print(f"Train: {train_copied} copied, {train_missing} missing")
    print(f"Validation: {val_copied} copied, {val_missing} missing")
    print(f"Test: {test_copied} copied, {test_missing} missing")

    # Generate masks for each split
    print(f"\n🎭 Generating masks for all splits...")
    print(f"Mask specifications:")
    print(f"  Square size: 37x37 pixels (1/6 of 224x224 image)")
    print(f"  Safe zone: avoiding outer 20% of boundaries")
    print(f"  Background: Black (0), Square: White (255)")
    print(f"  Creating both regular masks and inverted masks for RePaint")

    # Generate masks for each split
    train_mask_success, train_mask_failed = generate_masks_for_split(
        train_df,
        "train",
        os.path.join(script_dir, "train/img"),
        os.path.join(script_dir, "train/mask"),
        os.path.join(script_dir, "train/mask_inv"),
    )
    val_mask_success, val_mask_failed = generate_masks_for_split(
        val_df,
        "validation",
        os.path.join(script_dir, "valid/img"),
        os.path.join(script_dir, "valid/mask"),
        os.path.join(script_dir, "valid/mask_inv"),
    )
    test_mask_success, test_mask_failed = generate_masks_for_split(
        test_df,
        "test",
        os.path.join(script_dir, "test/img"),
        os.path.join(script_dir, "test/mask"),
        os.path.join(script_dir, "test/mask_inv"),
    )

    # Save split information to CSV files
    print(f"\n💾 Saving split information...")
    train_df.to_csv(os.path.join(script_dir, "train_split_info.csv"), index=False)
    val_df.to_csv(os.path.join(script_dir, "valid_split_info.csv"), index=False)
    test_df.to_csv(os.path.join(script_dir, "test_split_info.csv"), index=False)

    print(f"\nSplit information saved to:")
    print(f"- train_split_info.csv")
    print(f"- valid_split_info.csv")
    print(f"- test_split_info.csv")

    # Create a summary report
    summary = {
        "total_samples": len(df),
        "balanced_samples": len(balanced_df),
        "osteo_threshold": osteo_threshold,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_osteo": len(train_df[train_df["is_osteo"] == True]),
        "train_normal": len(train_df[train_df["is_osteo"] == False]),
        "val_osteo": len(val_df[val_df["is_osteo"] == True]),
        "val_normal": len(val_df[val_df["is_osteo"] == False]),
        "test_osteo": len(test_df[test_df["is_osteo"] == True]),
        "test_normal": len(test_df[test_df["is_osteo"] == False]),
        "train_masks": train_mask_success,
        "val_masks": val_mask_success,
        "test_masks": test_mask_success,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(script_dir, "split_summary.csv"), index=False)
    print(f"- split_summary.csv")

    print(f"\n✅ Dataset split and mask generation completed successfully!")
    print(f"📁 Directory structure:")
    print(f"   train/img/   - {len(train_df)} images (80%)")
    print(f"   train/mask/  - {train_mask_success} masks")
    print(f"   valid/img/   - {len(val_df)} images (10%)")
    print(f"   valid/mask/  - {val_mask_success} masks")
    print(f"   test/img/    - {len(test_df)} images (10%)")
    print(f"   test/mask/   - {test_mask_success} masks")
    print(f"\n🎯 Each split maintains balanced osteoporotic/normal representation!")
    print(f"🎭 All masks generated with 37x37 squares avoiding outer 20% boundaries!")


if __name__ == "__main__":
    create_balanced_splits()

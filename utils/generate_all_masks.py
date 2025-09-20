#!/usr/bin/env python3
"""
DEPRECATED: This script is no longer needed as standalone functionality.
Mask generation has been integrated into split.py.

This script is kept for reference but should not be used directly.
Use split.py instead, which creates masks for train/valid/test splits.
"""

import os
import numpy as np
import cv2
import random
from pathlib import Path


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


def generate_all_masks():
    """
    Generate masks for all images in the OAI dataset.
    """

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define paths
    img_dir = "img"
    mask_dir = "mask"

    # Ensure mask directory exists
    os.makedirs(mask_dir, exist_ok=True)

    # Get all image files
    image_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()  # Sort for consistent ordering

    print(f"Found {len(image_files)} images to process")
    print(f"Generating masks in: {os.path.abspath(mask_dir)}")
    print(f"Safe zone: avoiding outer 20% of {224}x{224} image boundaries")
    print(f"Square size: {int(224/6)}x{int(224/6)} pixels (1/6 of image dimensions)")
    print(f"Number of squares per mask: 1 (fixed)")
    print()

    # Process each image
    successful = 0
    failed = 0

    for i, image_file in enumerate(image_files, 1):
        try:
            # Create output filename (same as input but ensure .png extension)
            mask_filename = os.path.splitext(image_file)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)

            # Generate mask
            image_path = os.path.join(img_dir, image_file)
            mask = generate_mask_for_image(image_path, mask_path)

            successful += 1

            if i % 50 == 0 or i == len(image_files):
                print(
                    f"Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%) - {successful} successful, {failed} failed"
                )

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            failed += 1

    print(f"\n✅ Mask generation completed!")
    print(f"📊 Results:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/len(image_files)*100:.1f}%")

    # Verify some masks were created
    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png") and f != "create_mask.py"
    ]
    print(f"   Masks created: {len(mask_files)}")

    if len(mask_files) > 0:
        print(f"\n📁 Sample mask files:")
        for mask_file in sorted(mask_files)[:5]:
            print(f"   - {mask_file}")
        if len(mask_files) > 5:
            print(f"   ... and {len(mask_files) - 5} more")


def create_sample_visualization():
    """
    Create a sample visualization showing the mask generation process.
    """
    print(f"\n🎨 Creating sample visualization...")

    # Create a sample image and mask
    sample_img = np.zeros([224, 224, 3], np.uint8)
    sample_img.fill(255)

    # Generate a sample mask
    mask = np.zeros(sample_img.shape[:2], np.uint8)
    margin = int(224 * 0.2)  # 20% margin

    # Add one sample square (1/6 of image size)
    square_size = int(224 / 6)  # 37x37
    cv2.rectangle(
        mask,
        (margin + 20, margin + 20),
        (margin + 20 + square_size, margin + 20 + square_size),
        255,
        -1,
    )

    # Create visualization
    # Show the safe zone boundary
    cv2.rectangle(
        sample_img, (margin, margin), (224 - margin, 224 - margin), (0, 255, 0), 2
    )

    # Apply mask to image
    mask_inv = cv2.bitwise_not(mask)
    masked_img = cv2.bitwise_and(sample_img, sample_img, mask=mask_inv)

    # Save sample visualization
    cv2.imwrite("sample_visualization.png", masked_img)
    cv2.imwrite("sample_mask.png", mask)

    print(f"   Sample visualization saved as 'sample_visualization.png'")
    print(f"   Sample mask saved as 'sample_mask.png'")


if __name__ == "__main__":
    print("⚠️  DEPRECATION WARNING ⚠️")
    print(
        "This script is deprecated. Mask generation has been integrated into split.py"
    )
    print("Please use: python split.py")
    print("=" * 60)
    print()

    print("🎭 OAI Dataset Mask Generator (DEPRECATED)")
    print("=" * 50)

    # Generate all masks
    generate_all_masks()

    # Create sample visualization
    create_sample_visualization()

    print(f"\n🎯 All masks generated successfully!")
    print(f"📁 Masks are saved in: {os.path.abspath('.')}")
    print(f"\n⚠️  Remember: Use 'python split.py' for future mask generation!")

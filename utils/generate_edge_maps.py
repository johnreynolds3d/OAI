#!/usr/bin/env python3
"""
Generate edge detection maps for ICT model preprocessing.
This script creates Canny edge detection files that the ICT model requires.
"""

import os
import cv2
import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from PIL import Image
import argparse

def generate_edge_map(image_path, output_path, sigma=2, threshold=0.5):
    """
    Generate Canny edge detection map for an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save edge map
        sigma: Standard deviation of Gaussian filter for Canny
        threshold: Edge detection threshold
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Convert to grayscale
    gray = rgb2gray(img_array)
    
    # Apply Canny edge detection
    edges = canny(gray, sigma=sigma, low_threshold=threshold*0.5, high_threshold=threshold)
    
    # Convert back to RGB (3 channels)
    edges_rgb = gray2rgb(edges.astype(np.uint8) * 255)
    
    # Save edge map
    edge_img = Image.fromarray(edges_rgb)
    edge_img.save(output_path)
    
    return edges_rgb

def main():
    parser = argparse.ArgumentParser(description='Generate edge detection maps for ICT model')
    parser.add_argument('--input_dir', type=str, required=True, help='Input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output edge maps directory')
    parser.add_argument('--sigma', type=float, default=2, help='Canny sigma parameter')
    parser.add_argument('--threshold', type=float, default=0.5, help='Canny threshold parameter')
    
    args = parser.parse_args()
    
    # Create output directory structure
    condition_dir = os.path.join(args.output_dir, 'condition_1')
    os.makedirs(condition_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Generating edge maps for {len(image_files)} images...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sigma: {args.sigma}, Threshold: {args.threshold}")
    
    successful = 0
    failed = 0
    
    for i, filename in enumerate(image_files):
        try:
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(condition_dir, filename)
            
            # Generate edge map
            generate_edge_map(input_path, output_path, args.sigma, args.threshold)
            
            successful += 1
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed += 1
    
    print(f"\n✅ Edge map generation complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Edge maps saved to: {condition_dir}")

if __name__ == '__main__':
    main()

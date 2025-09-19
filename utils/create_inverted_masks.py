#!/usr/bin/env python3
"""
Create inverted masks for RePaint model.
RePaint expects inverted masks where 1 = keep, 0 = inpaint.
"""

import os
import cv2
import numpy as np
from PIL import Image

def create_inverted_masks(input_dir, output_dir):
    """
    Create inverted masks for RePaint.
    
    Args:
        input_dir: Directory containing original masks
        output_dir: Directory to save inverted masks
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of mask files
    mask_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Creating inverted masks for {len(mask_files)} files...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for filename in mask_files:
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Load mask
            mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # Invert mask: 0 becomes 1, 255 becomes 0
            inverted_mask = 255 - mask
            
            # Save inverted mask
            cv2.imwrite(output_path, inverted_mask)
            
            successful += 1
            if (successful + failed) % 10 == 0:
                print(f"Processed {successful + failed}/{len(mask_files)} masks...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed += 1
    
    print(f"\n✅ Inverted mask creation complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Inverted masks saved to: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create inverted masks for RePaint')
    parser.add_argument('--input_dir', type=str, required=True, help='Input masks directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output inverted masks directory')
    
    args = parser.parse_args()
    
    create_inverted_masks(args.input_dir, args.output_dir)

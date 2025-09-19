#!/usr/bin/env python3
"""
Prepare OAI data for RePaint by renaming files to the expected format.
RePaint expects filenames with underscores instead of dots and _mirror suffix.
"""

import os
import shutil
import argparse

def prepare_repaint_data(input_dir, output_dir, mask_dir, mask_inv_dir):
    """
    Prepare data for RePaint by renaming files to expected format.
    
    Args:
        input_dir: Input images directory
        output_dir: Output images directory  
        mask_dir: Input masks directory
        mask_inv_dir: Input inverted masks directory
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask_inv'), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Preparing {len(image_files)} files for RePaint...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for filename in image_files:
        try:
            # Convert filename format: 6.C.1_9103449_20090217_001.png -> 6-C-1_9103449_20090217_001_mirror.png
            base_name = os.path.splitext(filename)[0]
            new_name = base_name.replace('.', '-') + '_mirror.png'
            
            # Copy and rename image
            src_img = os.path.join(input_dir, filename)
            dst_img = os.path.join(output_dir, new_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy and rename mask
            mask_filename = os.path.splitext(filename)[0] + '.png'
            src_mask = os.path.join(mask_dir, mask_filename)
            dst_mask = os.path.join(output_dir, 'mask', new_name)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            
            # Copy and rename inverted mask
            src_mask_inv = os.path.join(mask_inv_dir, mask_filename)
            dst_mask_inv = os.path.join(output_dir, 'mask_inv', new_name)
            if os.path.exists(src_mask_inv):
                shutil.copy2(src_mask_inv, dst_mask_inv)
            
            successful += 1
            if (successful + failed) % 10 == 0:
                print(f"Processed {successful + failed}/{len(image_files)} files...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed += 1
    
    print(f"\n✅ RePaint data preparation complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Files prepared in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare OAI data for RePaint')
    parser.add_argument('--input_dir', type=str, required=True, help='Input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Input masks directory')
    parser.add_argument('--mask_inv_dir', type=str, required=True, help='Input inverted masks directory')
    
    args = parser.parse_args()
    
    prepare_repaint_data(args.input_dir, args.output_dir, args.mask_dir, args.mask_inv_dir)

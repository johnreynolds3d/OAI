#!/usr/bin/env python3
"""
Generate a comprehensive list of all images corresponding to positive osteoporosis classification.
"""

import pandas as pd
import os

def generate_osteoporosis_image_lists():
    # Load the CSV data
    df = pd.read_csv("data.csv", header=None, names=['BMD', 'filename'])
    
    # Define different thresholds for osteoporosis classification
    thresholds = {
        'Q1 (25th percentile)': df['BMD'].quantile(0.25),
        '20th percentile': df['BMD'].quantile(0.20),
        '15th percentile': df['BMD'].quantile(0.15),
        'Statistical outlier': df['BMD'].mean() - 1.5 * df['BMD'].std()
    }
    
    print("=== Complete List of Images with Positive Osteoporosis Classification ===\n")
    
    for threshold_name, threshold_value in thresholds.items():
        osteo_images = df[df['BMD'] <= threshold_value].sort_values('BMD')
        
        print(f"=== {threshold_name} (BMD <= {threshold_value:.4f}) ===")
        print(f"Total images: {len(osteo_images)}")
        print(f"Percentage of dataset: {len(osteo_images)/len(df)*100:.1f}%")
        print("\nAll images with positive osteoporosis classification:")
        print("-" * 60)
        
        for i, (_, row) in enumerate(osteo_images.iterrows(), 1):
            print(f"{i:3d}. BMD: {row['BMD']:.4f} - {row['filename']}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    generate_osteoporosis_image_lists()

#!/usr/bin/env python3
"""
Analyze BMD values to identify images corresponding to positive osteoporosis classification.
Osteoporosis is typically defined as BMD T-score <= -2.5, but since we have raw BMD values,
we'll use statistical thresholds based on the data distribution.
"""

import pandas as pd
import numpy as np
import os

def analyze_osteoporosis_classification():
    # Load the CSV data
    df = pd.read_csv("data.csv", header=None, names=['BMD', 'filename'])
    
    print("=== BMD Data Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"BMD range: {df['BMD'].min():.4f} - {df['BMD'].max():.4f}")
    print(f"BMD mean: {df['BMD'].mean():.4f}")
    print(f"BMD std: {df['BMD'].std():.4f}")
    print(f"BMD median: {df['BMD'].median():.4f}")
    
    # Define osteoporosis thresholds
    # Method 1: Bottom quartile (25th percentile)
    q25 = df['BMD'].quantile(0.25)
    
    # Method 2: Bottom 20th percentile
    q20 = df['BMD'].quantile(0.20)
    
    # Method 3: Bottom 15th percentile
    q15 = df['BMD'].quantile(0.15)
    
    # Method 4: Statistical outlier (mean - 1.5*std)
    outlier_threshold = df['BMD'].mean() - 1.5 * df['BMD'].std()
    
    print(f"\n=== Osteoporosis Classification Thresholds ===")
    print(f"25th percentile (Q1): {q25:.4f}")
    print(f"20th percentile: {q20:.4f}")
    print(f"15th percentile: {q15:.4f}")
    print(f"Statistical outlier (mean - 1.5*std): {outlier_threshold:.4f}")
    
    # Find images with positive osteoporosis classification using different thresholds
    thresholds = {
        'Q1 (25th percentile)': q25,
        '20th percentile': q20,
        '15th percentile': q15,
        'Statistical outlier': outlier_threshold
    }
    
    for threshold_name, threshold_value in thresholds.items():
        osteo_images = df[df['BMD'] <= threshold_value].sort_values('BMD')
        print(f"\n=== {threshold_name} (BMD <= {threshold_value:.4f}) ===")
        print(f"Number of osteoporotic images: {len(osteo_images)}")
        print(f"Percentage: {len(osteo_images)/len(df)*100:.1f}%")
        
        if len(osteo_images) > 0:
            print(f"Lowest BMD values:")
            print(osteo_images[['BMD', 'filename']].head(10).to_string(index=False))
            
            # Check if these images exist in the img folder
            img_folder = "img"
            existing_images = []
            missing_images = []
            
            for _, row in osteo_images.iterrows():
                img_path = os.path.join(img_folder, row['filename'])
                if os.path.exists(img_path):
                    existing_images.append((row['BMD'], row['filename']))
                else:
                    missing_images.append(row['filename'])
            
            print(f"\nImages found in {img_folder}/: {len(existing_images)}")
            print(f"Images missing: {len(missing_images)}")
            
            if existing_images:
                print(f"\nTop 10 lowest BMD images (existing):")
                for i, (bmd, filename) in enumerate(existing_images[:10]):
                    print(f"{i+1:2d}. BMD: {bmd:.4f} - {filename}")
    
    # Additional analysis: BMD distribution by quartiles
    print(f"\n=== BMD Distribution by Quartiles ===")
    df['quartile'] = pd.qcut(df['BMD'], q=4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
    quartile_counts = df['quartile'].value_counts().sort_index()
    print(quartile_counts)
    
    # Show some examples from each quartile
    print(f"\n=== Examples from each quartile ===")
    for quartile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
        quartile_data = df[df['quartile'] == quartile].sort_values('BMD')
        print(f"\n{quartile}:")
        print(quartile_data[['BMD', 'filename']].head(3).to_string(index=False))

if __name__ == "__main__":
    analyze_osteoporosis_classification()

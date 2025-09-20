#!/usr/bin/env python3
"""
Hardware Capabilities Detection Script
Detects GPU memory, CPU cores, and recommends optimal testing configurations
"""

import torch
import psutil
import subprocess
import sys
import os

def detect_hardware():
    """Detect hardware capabilities and return recommendations"""
    
    print("🔍 Detecting Hardware Capabilities...")
    print("=" * 50)
    
    # GPU Detection
    gpu_available = torch.cuda.is_available()
    gpu_memory = 0
    gpu_name = "None"
    
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_name = torch.cuda.get_device_properties(0).name
        print(f"✅ GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
    else:
        print("❌ GPU: Not available")
    
    # CPU Detection
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"✅ CPU: {cpu_cores} cores, {cpu_threads} threads")
    print(f"   RAM: {ram_gb:.1f} GB")
    
    # Recommendations
    print("\n🎯 Testing Recommendations:")
    print("=" * 30)
    
    if gpu_memory >= 8:
        print("🟢 HIGH-END GPU: All models can run with full settings")
        return "high_end"
    elif gpu_memory >= 4:
        print("🟡 MEDIUM GPU: Use memory-optimized settings")
        return "medium"
    elif gpu_memory >= 2:
        print("🟠 LOW GPU: Use single-image tests with small models")
        return "low"
    else:
        print("🔴 CPU ONLY: Use CPU-optimized single-image tests")
        return "cpu_only"

if __name__ == "__main__":
    capability = detect_hardware()
    print(f"\n📊 Detected capability level: {capability}")
    sys.exit(0)

#!/usr/bin/env python
"""
Compute mean and std for multiple datasets combined.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.dataset_process import mean_variance_multi

if __name__ == "__main__":
    # Data directories
    data_dirs = [
        "raw_data/babel_272/all_stream_stand_30fps",
        "raw_data/aistpp/motions_272_30fps",
        "raw_data/finedance/motion_272_30fps"
    ]
    
    # Save directory
    save_dir = "raw_data/multi_272_stand"
    
    # Compute mean and std
    print("=" * 60)
    print("Computing Mean and Std for Multi-Dataset (272-dim)")
    print("=" * 60)
    print(f"\nData directories:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    print(f"\nSave directory: {save_dir}")
    print("=" * 60)
    
    mean, std = mean_variance_multi(data_dirs, save_dir, rep='272')
    
    print("\n" + "=" * 60)
    print("✓ Computation completed successfully!")
    print("=" * 60)


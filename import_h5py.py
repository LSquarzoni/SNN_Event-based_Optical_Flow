import h5py
import os

# List all h5 files
h5_files = [f for f in os.listdir('/usr/scratch/badile13/amarchei/MVSEC_evflow/h5/') if f.endswith('.h5')]
print(f"Found {len(h5_files)} .h5 files")

# Inspect the first file
if h5_files:
    file_path = f'/usr/scratch/badile13/amarchei/MVSEC_evflow/h5/{h5_files[0]}'
    with h5py.File(file_path, 'r') as f:
        print(f"File: {h5_files[0]}")
        print(f"Keys: {list(f.keys())}")
        for key in f.keys():
            print(f"  {key}: {f[key].shape if hasattr(f[key], 'shape') else 'Group'}")
"""
Script to convert raw event camera dataset (UZH FPV format) to H5 format.

Input format:
- events.txt: timestamp x y polarity (space-separated)
- images.txt: id timestamp image_path
- img/ folder: contains PNG images

Output: H5 files with events (ps, ts, xs, ys) and images (image000000000, ...)

Supports center cropping to target resolution.
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import time


def parse_events_txt(events_file, target_width, target_height, orig_width, orig_height, crop_method='center'):
    """
    Parse events.txt file and apply coordinate transformation.
    
    Format: timestamp x y polarity
    Returns: ps, ts, xs, ys arrays
    """
    print(f"Reading events from {events_file}...")
    
    # Calculate crop offsets
    if crop_method == 'center':
        offset_x = (orig_width - target_width) // 2
        offset_y = (orig_height - target_height) // 2
    elif crop_method == 'top_left':
        offset_x = 0
        offset_y = 0
    else:
        raise ValueError(f"Unknown crop method: {crop_method}")
    
    print(f"Crop offsets: x={offset_x}, y={offset_y}")
    
    # Read events (skip header if present)
    events_data = []
    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                ts, x, y, p = parts
                events_data.append([float(ts), int(x), int(y), int(p)])
    
    events_data = np.array(events_data)
    print(f"Read {len(events_data)} events")
    
    # Extract and transform
    ts = events_data[:, 0]
    xs = events_data[:, 1].astype(np.int16)
    ys = events_data[:, 2].astype(np.int16)
    ps = events_data[:, 3].astype(np.int16)
    
    # Convert polarity to boolean: True for positive, False for negative
    ps = (ps == 1)
    
    # Apply crop offset
    xs = xs - offset_x
    ys = ys - offset_y
    
    # Filter events within bounds
    valid_mask = (xs >= 0) & (xs < target_width) & (ys >= 0) & (ys < target_height)
    
    xs = xs[valid_mask]
    ys = ys[valid_mask]
    ts = ts[valid_mask]
    ps = ps[valid_mask]
    
    # Safety clip to ensure no out-of-bounds coordinates
    xs = np.clip(xs, 0, target_width - 1)
    ys = np.clip(ys, 0, target_height - 1)
    
    print(f"After cropping: {len(xs)} events ({100*len(xs)/len(events_data):.1f}% retained)")
    
    # Normalize timestamps to start at 0
    t0 = ts[0] if len(ts) > 0 else 0
    ts = ts - t0
    
    return ps, ts, xs, ys, t0


def parse_images_txt(images_file):
    """
    Parse images.txt file.
    
    Format: id timestamp image_path
    Returns: list of (id, timestamp, image_path) tuples
    """
    images_data = []
    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                img_id = int(parts[0])
                timestamp = float(parts[1])
                img_path = parts[2]
                images_data.append((img_id, timestamp, img_path))
    
    return sorted(images_data, key=lambda x: x[0])


def crop_image(image, target_width, target_height, crop_method='center'):
    """
    Crop image to target size.
    """
    height, width = image.shape[:2]
    
    if crop_method == 'center':
        offset_x = (width - target_width) // 2
        offset_y = (height - target_height) // 2
    elif crop_method == 'top_left':
        offset_x = 0
        offset_y = 0
    else:
        raise ValueError(f"Unknown crop method: {crop_method}")
    
    cropped = image[offset_y:offset_y+target_height, offset_x:offset_x+target_width]
    
    return cropped


def convert_sequence_to_h5(sequence_dir, output_h5, target_width=256, target_height=256, 
                          crop_method='center', compression='gzip'):
    """
    Convert a single sequence to H5 format.
    
    Args:
        sequence_dir: Path to sequence directory
        output_h5: Output H5 file path
        target_width, target_height: Target resolution after cropping
        crop_method: 'center' or 'top_left'
        compression: H5 compression method
    """
    sequence_dir = Path(sequence_dir)
    
    # Check required files
    events_file = sequence_dir / 'events.txt'
    images_file = sequence_dir / 'images.txt'
    img_dir = sequence_dir / 'img'
    
    if not events_file.exists():
        raise FileNotFoundError(f"events.txt not found in {sequence_dir}")
    if not images_file.exists():
        raise FileNotFoundError(f"images.txt not found in {sequence_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"img/ directory not found in {sequence_dir}")
    
    # Get original resolution from first image
    images_data = parse_images_txt(images_file)
    if len(images_data) == 0:
        raise ValueError("No images found in images.txt")
    
    first_img_path = sequence_dir / images_data[0][2]
    first_img = np.array(Image.open(first_img_path))
    orig_height, orig_width = first_img.shape[:2]
    
    print(f"\n{'='*60}")
    print(f"Converting: {sequence_dir.name}")
    print(f"{'='*60}")
    print(f"Original resolution: {orig_width}x{orig_height}")
    print(f"Target resolution: {target_width}x{target_height}")
    print(f"Crop method: {crop_method}")
    
    # Parse events
    ps, ts, xs, ys, t0 = parse_events_txt(
        events_file, target_width, target_height, orig_width, orig_height, crop_method
    )
    
    # Calculate duration
    duration = ts[-1] if len(ts) > 0 else 0
    tk = t0 + duration
    
    # Create H5 file
    print(f"\nCreating H5 file: {output_h5}")
    with h5py.File(output_h5, 'w') as f:
        # Root attributes
        f.attrs['t0'] = t0
        f.attrs['tk'] = tk
        f.attrs['duration'] = duration
        f.attrs['num_events'] = len(xs)
        f.attrs['num_pos'] = np.sum(ps)
        f.attrs['num_neg'] = np.sum(~ps)
        f.attrs['num_imgs'] = len(images_data)
        f.attrs['sensor_resolution'] = np.array([target_height, target_width])
        f.attrs['original_resolution'] = np.array([orig_height, orig_width])
        f.attrs['crop_method'] = crop_method
        
        # Events group
        events_group = f.create_group('events')
        print(f"Writing {len(xs)} events...")
        events_group.create_dataset('xs', data=xs, compression=compression)
        events_group.create_dataset('ys', data=ys, compression=compression)
        events_group.create_dataset('ts', data=ts, compression=compression)
        events_group.create_dataset('ps', data=ps.astype(np.bool_), compression=compression)
        
        # Images group
        images_group = f.create_group('images')
        print(f"Writing {len(images_data)} images...")
        
        for idx, (img_id, img_ts, img_path) in enumerate(tqdm(images_data, desc="Processing images")):
            # Read and crop image
            img_full_path = sequence_dir / img_path
            img = np.array(Image.open(img_full_path))
            
            # Crop to target resolution
            img_cropped = crop_image(img, target_width, target_height, crop_method)
            
            # Create dataset with formatted name
            img_name = f"image{idx:09d}"
            img_dataset = images_group.create_dataset(
                img_name, data=img_cropped, compression=compression
            )
            
            # Add timestamp attribute (normalized)
            img_dataset.attrs['timestamp'] = img_ts - t0
    
    print(f"✓ Successfully created: {output_h5}")
    print(f"  Events: {len(xs)}")
    print(f"  Images: {len(images_data)}")
    print(f"  Duration: {duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw UZH FPV dataset to H5 format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing sequence folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for H5 files'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=256,
        help='Target width after cropping (default: 256)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=256,
        help='Target height after cropping (default: 256)'
    )
    parser.add_argument(
        '--crop_method',
        type=str,
        default='center',
        choices=['center', 'top_left'],
        help='Cropping method (default: center)'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='gzip',
        choices=['gzip', 'lzf', None],
        help='H5 compression method (default: gzip)'
    )
    parser.add_argument(
        '--sequences',
        type=str,
        nargs='*',
        help='Specific sequences to convert (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find sequence directories
    input_dir = Path(args.input_dir)
    if args.sequences:
        sequence_dirs = [input_dir / seq for seq in args.sequences]
    else:
        sequence_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    sequence_dirs = sorted(sequence_dirs)
    
    print(f"Found {len(sequence_dirs)} sequences to convert")
    print(f"Target resolution: {args.width}x{args.height}")
    print(f"Crop method: {args.crop_method}")
    
    # Convert each sequence
    success_count = 0
    for seq_dir in sequence_dirs:
        output_h5 = output_dir / f"{seq_dir.name}_data.h5"
        
        try:
            convert_sequence_to_h5(
                seq_dir,
                output_h5,
                args.width,
                args.height,
                args.crop_method,
                args.compression
            )
            success_count += 1
        except Exception as e:
            print(f"\n✗ ERROR processing {seq_dir.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {success_count}/{len(sequence_dirs)} sequences")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

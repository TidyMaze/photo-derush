#!/usr/bin/env python3
"""Check metadata (timestamp, camera) for specific groups."""

import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos, extract_timestamp, extract_camera_id
from PIL import Image


def load_last_dir():
    """Load last directory from config."""
    config_file = Path.home() / ".photo-derush-cache" / "last_dir.txt"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                directory = f.read().strip()
                if os.path.isdir(directory):
                    return directory
        except Exception:
            pass
    return os.path.expanduser('~/Pictures/photo-dataset')


def main():
    image_dir = load_last_dir()
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    # Parse group IDs from command line or use defaults
    if len(sys.argv) > 2:
        group_ids = [int(gid.strip('#')) for gid in sys.argv[2:]]
    else:
        group_ids = [508, 509, 510, 511]
    
    print(f"Checking groups: {group_ids}")
    print(f"Loading images from: {image_dir}")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    
    if not filenames:
        print("No images found!")
        sys.exit(1)
    
    print(f"Found {len(filenames)} images")
    
    # Load EXIF data
    exif_data = {}
    for fname in filenames:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
            exif = img.getexif()
            exif_data[fname] = {}
            if exif:
                for tag_id, value in exif.items():
                    tag = img.getexif().get(tag_id)
                    if tag:
                        exif_data[fname][str(tag)] = str(value)
        except Exception:
            exif_data[fname] = {}
    
    print("Computing grouping...")
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=None,
        quality_metrics=None,
        session_gap_min=30,
        burst_gap_sec=1.0,
        phash_threshold=5,
        progress_reporter=None,
    )
    
    # Organize by group
    groups_data: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id in group_ids:
            groups_data[group_id].append((filename, info))
    
    print(f"\n{'='*80}")
    print(f"Group Metadata Analysis")
    print(f"{'='*80}\n")
    
    all_images = []
    for group_id in group_ids:
        images = groups_data.get(group_id, [])
        if not images:
            print(f"Group #{group_id}: NOT FOUND")
            continue
        
        print(f"Group #{group_id}: {len(images)} images")
        for filename, info in images:
            path = os.path.join(image_dir, filename)
            timestamp = extract_timestamp(exif_data.get(filename, {}), path)
            camera_id = extract_camera_id(exif_data.get(filename, {}))
            session_id = info.get("session_id")
            burst_id = info.get("burst_id")
            
            print(f"  - {os.path.basename(filename)}")
            print(f"    Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Camera: {camera_id}")
            print(f"    Session: {session_id}, Burst: {burst_id}")
            
            all_images.append((filename, timestamp, camera_id, session_id, burst_id))
    
    # Check if they're from same session/burst
    print(f"\n{'='*80}")
    print(f"Time-Based Grouping Analysis")
    print(f"{'='*80}\n")
    
    if all_images:
        timestamps = [ts for _, ts, _, _, _ in all_images]
        cameras = [cam for _, _, cam, _, _ in all_images]
        sessions = [sess for _, _, _, sess, _ in all_images]
        bursts = [burst for _, _, _, _, burst in all_images]
        
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        time_diff = (max_ts - min_ts).total_seconds()
        
        print(f"Time span: {time_diff:.1f} seconds ({time_diff/60:.1f} minutes)")
        print(f"Cameras: {set(cameras)}")
        print(f"Sessions: {set(sessions)}")
        print(f"Bursts: {set(bursts)}")
        
        if len(set(sessions)) == 1:
            print(f"\n✅ All images are in the same SESSION (#{sessions[0]})")
        if len(set(bursts)) == 1:
            print(f"✅ All images are in the same BURST (#{bursts[0]})")
        if time_diff <= 1.0:
            print(f"✅ All images taken within 1 second (burst)")
        elif time_diff <= 1800:
            print(f"✅ All images taken within 30 minutes (same session)")


if __name__ == "__main__":
    main()


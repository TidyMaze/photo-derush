#!/usr/bin/env python3
"""Debug why specific groups aren't merging."""

import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos
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
    keep_probabilities = {}
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
            keep_probabilities[fname] = 0.5
        except Exception:
            exif_data[fname] = {}
            keep_probabilities[fname] = 0.5
    
    print("Computing grouping...")
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=keep_probabilities,
        quality_metrics=None,
        session_gap_min=10,
        burst_gap_sec=15.0,
        phash_threshold=8,
        progress_reporter=None,
    )
    
    # Get all unique group IDs
    all_group_ids = sorted(set(g.get('group_id') for g in group_info.values() if g.get('group_id') is not None))
    print(f"\nTotal groups found: {len(all_group_ids)}")
    print(f"Group ID range: {min(all_group_ids)} - {max(all_group_ids)}")
    
    # Check specific groups
    groups_to_check = [0, 1, 5, 6, 10, 11, 14, 15, 18, 19, 20, 21]
    
    print(f"\n{'='*80}")
    print(f"Checking groups: {groups_to_check}")
    print(f"{'='*80}\n")
    
    # Organize by group
    groups_data: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None:
            groups_data[group_id].append((filename, info))
    
    # Check each group
    for gid in groups_to_check:
        images = groups_data.get(gid, [])
        if images:
            bursts = set(info.get('burst_id') for _, info in images)
            sessions = set(info.get('session_id') for _, info in images)
            print(f"Group #{gid}: {len(images)} images")
            print(f"  Bursts: {sorted(bursts)}")
            print(f"  Sessions: {sorted(sessions)}")
            print(f"  Sample images:")
            for fname, info in images[:3]:
                print(f"    - {os.path.basename(fname)} (burst={info.get('burst_id')}, session={info.get('session_id')})")
            if len(images) > 3:
                print(f"    ... and {len(images) - 3} more")
        else:
            print(f"Group #{gid}: NOT FOUND")
    
    # Check which bursts have multiple groups
    print(f"\n{'='*80}")
    print(f"Bursts with multiple groups (should be merged):")
    print(f"{'='*80}\n")
    
    burst_to_groups = defaultdict(set)
    for filename, info in group_info.items():
        gid = info.get('group_id')
        burst_id = info.get('burst_id')
        if gid is not None and burst_id is not None:
            burst_to_groups[burst_id].add(gid)
    
    for burst_id in sorted(burst_to_groups.keys())[:30]:
        groups = burst_to_groups[burst_id]
        if len(groups) > 1:
            print(f"Burst {burst_id}: groups {sorted(groups)}")
            # Check if any of these are in our target list
            target_groups = [g for g in groups if g in groups_to_check]
            if target_groups:
                print(f"  ⚠️  Contains target groups: {target_groups}")
    
    # Check consecutive groups
    print(f"\n{'='*80}")
    print(f"Consecutive groups analysis:")
    print(f"{'='*80}\n")
    
    for i in range(len(groups_to_check) - 1):
        gid1 = groups_to_check[i]
        gid2 = groups_to_check[i + 1]
        
        images1 = groups_data.get(gid1, [])
        images2 = groups_data.get(gid2, [])
        
        if images1 and images2:
            bursts1 = set(info.get('burst_id') for _, info in images1)
            bursts2 = set(info.get('burst_id') for _, info in images2)
            common_bursts = bursts1 & bursts2
            
            if common_bursts:
                print(f"Groups #{gid1} and #{gid2}: ✅ Share bursts {sorted(common_bursts)}")
            else:
                print(f"Groups #{gid1} and #{gid2}: ❌ No common bursts")
                print(f"  Group #{gid1} bursts: {sorted(bursts1)}")
                print(f"  Group #{gid2} bursts: {sorted(bursts2)}")
                # Check if they're consecutive bursts
                all_bursts = sorted(bursts1 | bursts2)
                consecutive = False
                for j in range(len(all_bursts) - 1):
                    if all_bursts[j+1] == all_bursts[j] + 1:
                        consecutive = True
                        print(f"  ⚠️  Consecutive bursts found: {all_bursts[j]} and {all_bursts[j+1]}")
                if not consecutive:
                    print(f"  ℹ️  Not consecutive bursts")


if __name__ == "__main__":
    main()


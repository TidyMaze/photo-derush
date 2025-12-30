#!/usr/bin/env python3
"""Check hash distances for specific groups."""

import sys
import os
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos


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


def check_group_distances(group_info: dict, image_dir: str, group_ids: list[int]):
    """Check hash distances between specific groups."""
    # Get all images for each group
    groups_images: dict[int, list[str]] = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None and group_id in group_ids:
            groups_images[group_id].append(filename)
    
    # Compute hashes for all images
    image_hashes: dict[str, imagehash.ImageHash] = {}
    for group_id, filenames in groups_images.items():
        print(f"\nGroup #{group_id}: {len(filenames)} images")
        for filename in filenames:
            path = os.path.join(image_dir, filename)
            try:
                with Image.open(path) as img:
                    image_hashes[filename] = imagehash.phash(img)
                    print(f"  - {os.path.basename(filename)}")
            except Exception as e:
                print(f"  - {os.path.basename(filename)} (error: {e})")
    
    # Compute distances between groups
    print(f"\n{'='*80}")
    print(f"Hash Distances Between Groups")
    print(f"{'='*80}\n")
    
    distances: dict[tuple[int, int], list[float]] = {}
    
    for i, gid1 in enumerate(group_ids):
        for gid2 in group_ids[i+1:]:
            images1 = groups_images.get(gid1, [])
            images2 = groups_images.get(gid2, [])
            
            if not images1 or not images2:
                continue
            
            # Compute all pairwise distances
            all_distances = []
            for f1 in images1:
                h1 = image_hashes.get(f1)
                if not h1:
                    continue
                for f2 in images2:
                    h2 = image_hashes.get(f2)
                    if not h2:
                        continue
                    dist = h1 - h2
                    all_distances.append(dist)
            
            if all_distances:
                min_dist = min(all_distances)
                avg_dist = sum(all_distances) / len(all_distances)
                max_dist = max(all_distances)
                distances[(gid1, gid2)] = all_distances
                print(f"Group #{gid1} ↔ Group #{gid2}:")
                print(f"  Min: {min_dist:.1f}, Avg: {avg_dist:.1f}, Max: {max_dist:.1f} ({len(all_distances)} pairs)")
                if min_dist <= 8:
                    print(f"  ✅ Should merge (min distance {min_dist:.1f} <= 8)")
                elif min_dist <= 12:
                    print(f"  ⚠️  Consider merging (min distance {min_dist:.1f} <= 12)")
                else:
                    print(f"  ❌ Too different (min distance {min_dist:.1f} > 12)")
    
    return distances


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
    
    # Dummy data
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
        session_gap_min=30,
        burst_gap_sec=1.0,
        phash_threshold=5,  # Current threshold
        progress_reporter=None,
    )
    
    # Check if groups exist
    found_groups = set()
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id in group_ids:
            found_groups.add(group_id)
    
    missing = set(group_ids) - found_groups
    if missing:
        print(f"\n⚠️  Groups not found: {missing}")
        print(f"Found groups: {found_groups}")
        print(f"\nAvailable group IDs in dataset:")
        all_group_ids = sorted(set(g.get("group_id") for g in group_info.values() if g.get("group_id") is not None))
        print(f"  Range: {min(all_group_ids)} - {max(all_group_ids)}")
        print(f"  Total: {len(all_group_ids)} groups")
    
    if found_groups:
        check_group_distances(group_info, image_dir, list(found_groups))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Compare groups visually to identify which should be merged."""

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


def compute_hash_distances_between_groups(group_info: dict, image_dir: str, group_ids: list[int]):
    """Compute average hash distance between groups."""
    # Get all images for each group
    groups_images: dict[int, list[str]] = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None and group_id in group_ids:
            groups_images[group_id].append(filename)
    
    # Compute hashes for all images
    image_hashes: dict[str, imagehash.ImageHash] = {}
    for group_id, filenames in groups_images.items():
        for filename in filenames:
            path = os.path.join(image_dir, filename)
            try:
                with Image.open(path) as img:
                    image_hashes[filename] = imagehash.phash(img)
            except Exception:
                pass
    
    # Compute average hash distance between groups
    distances: dict[tuple[int, int], float] = {}
    
    for gid1 in group_ids:
        for gid2 in group_ids:
            if gid1 >= gid2:
                continue  # Only compute once
            
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
                avg_dist = sum(all_distances) / len(all_distances)
                min_dist = min(all_distances)
                distances[(gid1, gid2)] = (avg_dist, min_dist, len(all_distances))
    
    return distances


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
        phash_threshold=8,
        progress_reporter=None,
    )
    
    # Organize by group
    groups_by_id = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None:
            groups_by_id[group_id].append((filename, info))
    
    # Sort by size
    sorted_groups = sorted(groups_by_id.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Top 20 Groups by Size")
    print(f"{'='*80}\n")
    
    # Show top 20 groups
    top_20_ids = [gid for gid, _ in sorted_groups[:20]]
    
    for i, (group_id, images) in enumerate(sorted_groups[:20], 1):
        best_image = next((f for f, info in images if info.get("is_group_best")), "N/A")
        print(f"{i:2d}. Group #{group_id}: {len(images)} images, best: {os.path.basename(best_image)}")
        for filename, info in images[:3]:
            print(f"      - {os.path.basename(filename)}")
        if len(images) > 3:
            print(f"      ... and {len(images) - 3} more")
    
    # Compute distances between top groups
    print(f"\n{'='*80}")
    print(f"Hash Distances Between Top Groups (threshold=8)")
    print(f"{'='*80}\n")
    
    distances = compute_hash_distances_between_groups(group_info, image_dir, top_20_ids)
    
    # Sort by minimum distance (most similar first)
    sorted_distances = sorted(distances.items(), key=lambda x: x[1][1])  # Sort by min_dist
    
    print("Groups that might be merged (sorted by minimum distance):")
    print(f"{'Group 1':<10} {'Group 2':<10} {'Min Dist':<10} {'Avg Dist':<10} {'Pairs':<10} {'Should Merge?'}")
    print("-" * 80)
    
    merge_candidates = []
    for (gid1, gid2), (avg_dist, min_dist, pair_count) in sorted_distances[:30]:
        should_merge = min_dist <= 12  # Suggest merge if min distance <= 12
        merge_marker = "✅ YES" if should_merge else "❌ NO"
        print(f"#{gid1:<9} #{gid2:<9} {min_dist:<10.1f} {avg_dist:<10.1f} {pair_count:<10} {merge_marker}")
        
        if should_merge:
            merge_candidates.append((gid1, gid2, min_dist, avg_dist))
    
    if merge_candidates:
        print(f"\n{'='*80}")
        print(f"Suggested Merges (min distance <= 12):")
        print(f"{'='*80}\n")
        for gid1, gid2, min_dist, avg_dist in merge_candidates:
            size1 = len(groups_by_id[gid1])
            size2 = len(groups_by_id[gid2])
            print(f"  Merge Group #{gid1} ({size1} images) + Group #{gid2} ({size2} images)")
            print(f"    Min distance: {min_dist:.1f}, Avg distance: {avg_dist:.1f}")
    else:
        print(f"\n{'='*80}")
        print(f"No obvious merge candidates found (all min distances > 12)")
        print(f"Consider reducing threshold from 8 to 6 or 5 for stricter grouping")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()


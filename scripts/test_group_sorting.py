#!/usr/bin/env python3
"""Test script to verify group-based sorting in the grid view."""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repository import RatingsTagsRepository
from src.model import ImageModel
from src.viewmodel import PhotoViewModel


def load_last_dir():
    """Load last directory from config (same as app.py)."""
    config_file = Path.home() / ".photo-derush-cache" / "last_dir.txt"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                directory = f.read().strip()
                if os.path.isdir(directory):
                    return directory
        except Exception:
            pass
    return os.path.expanduser('~')


def main():
    # Get the directory from config (same as app.py)
    directory = load_last_dir()
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return 1
    
    print(f"Loading images from: {directory}")
    
    # Create model and viewmodel
    repo = RatingsTagsRepository()
    model = ImageModel(directory, max_images=10000, repo=repo)
    viewmodel = PhotoViewModel(directory, max_images=10000)
    
    # Load images
    print("Loading images...")
    viewmodel.load_images()
    
    # Wait for images to be added
    time.sleep(2)
    
    print("Waiting 20 seconds for predictions and grouping to complete...")
    time.sleep(20)
    
    # Force trigger grouping if not done
    if not getattr(viewmodel, "_grouping_computed", False):
        print("Triggering grouping computation...")
        viewmodel._compute_grouping_async()
        time.sleep(10)
    
    # Get filtered images (these should be sorted by group)
    filtered_images = viewmodel.current_filtered_images()
    
    if not filtered_images:
        print("No images found!")
        return 1
    
    print(f"\nFound {len(filtered_images)} images in grid")
    
    # Get group info
    group_info = getattr(viewmodel, "_group_info", {})
    
    if not group_info:
        print("\n⚠️  No group info available yet. Grouping may still be computing...")
        print("   Images are sorted by uncertainty (fallback behavior)")
        return 0
    
    print(f"\nGroup info available for {len(group_info)} images")
    
    # Check sorting
    print("\n" + "="*80)
    print("Checking Group-Based Sorting")
    print("="*80)
    
    # Group images by group_id
    groups: dict[int | None, list[tuple[str, float, datetime]]] = {}
    group_earliest_date: dict[int | None, datetime] = {}
    
    # Helper to get timestamp
    def get_timestamp(fname: str) -> datetime:
        path = model.get_image_path(fname)
        if path:
            try:
                exif = model.load_exif(path)
                dt_original = exif.get("DateTimeOriginal")
                if dt_original and isinstance(dt_original, str):
                    try:
                        return datetime.strptime(dt_original, "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        pass
                import os
                return datetime.fromtimestamp(os.path.getmtime(path))
            except Exception:
                pass
        return datetime.now()
    
    # Collect group data
    for fname in filtered_images:
        ginfo = group_info.get(fname, {})
        group_id = ginfo.get("group_id")
        pick_score = ginfo.get("pick_score", 0.0)
        ts = get_timestamp(fname)
        
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append((fname, pick_score, ts))
        
        # Track earliest date per group
        if group_id not in group_earliest_date or ts < group_earliest_date[group_id]:
            group_earliest_date[group_id] = ts
    
    # Check if groups are sorted by earliest date (ASC)
    group_ids_in_order = []
    for fname in filtered_images:
        ginfo = group_info.get(fname, {})
        group_id = ginfo.get("group_id")
        if group_id not in group_ids_in_order:
            group_ids_in_order.append(group_id)
    
    # Verify groups are sorted by earliest date
    is_groups_sorted = True
    violations = []
    
    for i in range(len(group_ids_in_order) - 1):
        gid1 = group_ids_in_order[i]
        gid2 = group_ids_in_order[i + 1]
        date1 = group_earliest_date.get(gid1, datetime.now())
        date2 = group_earliest_date.get(gid2, datetime.now())
        if date1 > date2:
            is_groups_sorted = False
            violations.append((i, gid1, date1, gid2, date2))
    
    # Verify within-group sorting (pick_score DESC)
    is_within_group_sorted = True
    within_group_violations = []
    
    current_group = None
    current_group_items = []
    
    for idx, fname in enumerate(filtered_images):
        ginfo = group_info.get(fname, {})
        group_id = ginfo.get("group_id")
        pick_score = ginfo.get("pick_score", 0.0)
        
        if group_id != current_group:
            # Check previous group
            if current_group is not None and len(current_group_items) > 1:
                for j in range(len(current_group_items) - 1):
                    score1 = current_group_items[j][1]
                    score2 = current_group_items[j + 1][1]
                    if score1 < score2:
                        is_within_group_sorted = False
                        within_group_violations.append((
                            current_group,
                            current_group_items[j][0],
                            score1,
                            current_group_items[j + 1][0],
                            score2
                        ))
            current_group = group_id
            current_group_items = [(fname, pick_score)]
        else:
            current_group_items.append((fname, pick_score))
    
    # Check last group
    if current_group is not None and len(current_group_items) > 1:
        for j in range(len(current_group_items) - 1):
            score1 = current_group_items[j][1]
            score2 = current_group_items[j + 1][1]
            if score1 < score2:
                is_within_group_sorted = False
                within_group_violations.append((
                    current_group,
                    current_group_items[j][0],
                    score1,
                    current_group_items[j + 1][0],
                    score2
                ))
    
    # Print results
    print(f"\nTotal groups: {len(groups)}")
    print(f"Groups sorted by date (ASC): {'✅' if is_groups_sorted else '❌'}")
    print(f"Within-group sorted by pick_score (DESC): {'✅' if is_within_group_sorted else '❌'}")
    
    if not is_groups_sorted:
        print(f"\n❌ GROUP DATE SORTING VIOLATIONS: {len(violations)}")
        print("\nFirst 5 violations:")
        for idx, (pos, gid1, date1, gid2, date2) in enumerate(violations[:5], 1):
            print(f"  {idx}. Position {pos}: Group {gid1} ({date1}) > Group {gid2} ({date2})")
    
    if not is_within_group_sorted:
        print(f"\n❌ WITHIN-GROUP SORTING VIOLATIONS: {len(within_group_violations)}")
        print("\nFirst 5 violations:")
        for idx, (gid, fname1, score1, fname2, score2) in enumerate(within_group_violations[:5], 1):
            print(f"  {idx}. Group {gid}: {fname1[:40]:40s} (score={score1:.4f}) < {fname2[:40]:40s} (score={score2:.4f})")
    
    # Print first 30 images with their group info
    print(f"\n{'='*80}")
    print("First 30 images (should be grouped together, sorted by date ASC, pick_score DESC within group):")
    print(f"{'='*80}")
    print(f"{'Pos':<5} {'Group':<8} {'Pick Score':<12} {'Date':<20} {'Filename'}")
    print("-" * 80)
    
    for i, fname in enumerate(filtered_images[:30], 1):
        ginfo = group_info.get(fname, {})
        group_id = ginfo.get("group_id", "None")
        pick_score = ginfo.get("pick_score", 0.0)
        ts = get_timestamp(fname)
        date_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:<5} {str(group_id):<8} {pick_score:<12.4f} {date_str:<20} {fname[:35]}")
    
    # Show group statistics
    print(f"\n{'='*80}")
    print("Group Statistics:")
    print(f"{'='*80}")
    group_sizes = sorted([(gid, len(items)) for gid, items in groups.items()], key=lambda x: x[1], reverse=True)
    print(f"{'Group ID':<12} {'Size':<8} {'Earliest Date':<20}")
    print("-" * 50)
    for gid, size in group_sizes[:20]:
        date_str = group_earliest_date.get(gid, datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{str(gid):<12} {size:<8} {date_str:<20}")
    
    return 0 if (is_groups_sorted and is_within_group_sorted) else 1


if __name__ == "__main__":
    sys.exit(main())


"""TDD Test for Burst Grouping and 1-Best-per-Burst Selection.

Tests GroupingService and CLI Bridge predicting on real collection folder.
"""

import os
import pytest
import json
from datetime import datetime
from src.grouping_service import compute_grouping_for_photos, extract_timestamp
from src.photo_grouping import PhotoMetadata, detect_bursts

def find_test_dir():
    for root, dirs, files in os.walk(r'E:\Google Drive'):
        if root.endswith(r'2021\08') or root.endswith(r'2026\soiree max'):
            return root
    return None

TEST_DIR = find_test_dir()

@pytest.mark.skipif(not TEST_DIR, reason="Working folder 2021/08 or 2026/soiree max not found")
def test_burst_detection_real_folder():
    """Test burst detection on real photos in working folder."""
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.arw'))]
    assert len(files) > 0, f"No photos found in {TEST_DIR}"

    # Compute grouping for all photos in folder
    exif_data = {}  # Fallback to mtime/EXIF
    grouping = compute_grouping_for_photos(
        filenames=files[:100],  # Test first 100 photos
        image_dir=TEST_DIR,
        exif_data=exif_data,
        burst_gap_sec=2.0,  # 2 second threshold for bursts
    )

    assert len(grouping) == len(files[:100])

    # Inspect produced burst groups
    burst_groups = {}
    for fn, info in grouping.items():
        bid = info["burst_id"]
        if bid not in burst_groups:
            burst_groups[bid] = []
        burst_groups[bid].append((fn, info))

    # Verify that every multi-photo burst group has EXACTLY ONE is_burst_best == True
    multi_photo_bursts = {bid: items for bid, items in burst_groups.items() if len(items) > 1}
    print(f"\n[TDD TEST] Analyzed folder: {TEST_DIR}")
    print(f"[TDD TEST] Found {len(burst_groups)} total burst groups, {len(multi_photo_bursts)} multi-photo bursts.")

    # Display sample produced groups
    sample_count = 0
    for bid, items in multi_photo_bursts.items():
        best_count = sum(1 for fn, info in items if info.get("is_burst_best"))
        assert best_count == 1, f"Burst {bid} with {len(items)} photos must have exactly 1 best pick, found {best_count}"
        if sample_count < 5:
            print(f"\n  Burst Group #{bid} ({len(items)} photos):")
            for fn, info in items:
                best_str = "[BEST IN BURST]" if info.get("is_burst_best") else "[BURST DUPLICATE]"
                print(f"     - {fn} {best_str} (score={info.get('pick_score', 0):.2f})")
            sample_count += 1

@pytest.mark.skipif(not TEST_DIR, reason="Working folder not found")
def test_cli_bridge_burst_deduplication():
    """Test cli_bridge predict command with burst deduplication enabled."""
    from src.cli_bridge import cmd_predict

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.arw'))][:50]
    files_json = json.dumps(files)

    result_json_str = cmd_predict(TEST_DIR, labels_file=None, files_json=files_json, burst_limit=True)
    result = json.loads(result_json_str)

    assert result["status"] == "success"
    assert "predictions" in result or "threshold" in result
    assert "groups" in result, "cli_bridge output must contain burst 'groups' metadata!"

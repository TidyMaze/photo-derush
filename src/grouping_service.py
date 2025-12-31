"""Service for computing photo grouping (sessions, bursts, near-duplicates, best-picks).

This service orchestrates the grouping pipeline:
1. Session detection
2. Burst detection
3. Near-duplicate grouping
4. Best-pick recommendation
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime

try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

from src.photo_grouping import (
    BURST_GAP_SEC,
    PHASH_HAMMING_THRESHOLD,
    SESSION_GAP_MIN,
    PhotoMetadata,
    compute_pick_score,
    detect_sessions,
    group_near_duplicates,
    recommend_best_pick,
)


def extract_camera_id(exif: dict) -> str:
    """Extract camera identifier from EXIF data."""
    make = exif.get("Make", "").strip()
    model = exif.get("Model", "").strip()
    serial = exif.get("BodySerialNumber", "").strip()

    if make and model:
        camera_id = f"{make} {model}"
        if serial:
            camera_id += f" {serial}"
        return camera_id
    return "unknown"


def extract_timestamp(exif: dict, path: str) -> datetime:
    """Extract timestamp from EXIF, filename, or fallback to file mtime."""
    # Try to read EXIF directly from file if exif dict doesn't have DateTimeOriginal
    # DateTimeOriginal (tag 36867) is in EXIF sub-IFD, not main IFD
    dt_original = None

    # First, try to get from provided exif dict (might have string keys)
    if isinstance(exif, dict):
        dt_original = exif.get("DateTimeOriginal") or exif.get(36867) or exif.get("36867")
        # Fallback to DateTime (tag 306 in main IFD)
        if not dt_original:
            dt_original = exif.get("DateTime") or exif.get(306) or exif.get("306")

    # If not found, try reading EXIF directly from file to access EXIF sub-IFD
    if not dt_original:
        try:
            from PIL import Image
            # OPTIMIZATION: Use shared image cache to avoid repeated file opens
            from .image_cache import get_cached_image_for_exif
            img = get_cached_image_for_exif(path)
            if img is None:
                return None
            exif_obj = img.getexif()
            if exif_obj:
                # Try to get EXIF sub-IFD (0x8769 is EXIFIFD constant)
                try:
                    exif_ifd = exif_obj.get_ifd(0x8769)
                    dt_original = exif_ifd.get(36867)  # DateTimeOriginal
                except Exception:
                    pass
                # Fallback to DateTime in main IFD
                if not dt_original:
                    dt_original = exif_obj.get(306)  # DateTime
        except Exception:
            pass

    if dt_original:
        try:
            # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
            if isinstance(dt_original, str):
                return datetime.strptime(dt_original, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    # Fallback to file modification time (do NOT parse from filename)
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime)
    except Exception:
        return datetime.now()


def compute_grouping_for_photos(
    filenames: list[str],
    image_dir: str,
    exif_data: dict[str, dict],  # filename -> exif dict
    keep_probabilities: dict[str, float] | None = None,  # filename -> keep prob (0-1)
    quality_metrics: dict[str, dict] | None = None,  # filename -> {sharpness, exposure, noise, ...}
    session_gap_min: int = SESSION_GAP_MIN,
    burst_gap_sec: float = BURST_GAP_SEC,
    phash_threshold: int = PHASH_HAMMING_THRESHOLD,
    progress_reporter=None,  # Optional ProgressReporter for UI progress
) -> dict[str, dict]:
    """Compute grouping for all photos.

    Args:
        filenames: List of image filenames
        image_dir: Directory containing images
        exif_data: Dict mapping filename to EXIF data
        keep_probabilities: Optional dict mapping filename to keep probability
        quality_metrics: Optional dict mapping filename to quality metrics
        session_gap_min: Session gap threshold in minutes
        burst_gap_sec: Burst gap threshold in seconds
        phash_threshold: Perceptual hash Hamming distance threshold

    Returns:
        Dict mapping filename to grouping info:
        {
            "session_id": int,
            "burst_id": int,
            "group_id": int,
            "group_size": int,
            "is_group_best": bool,
            "pick_score": float,
        }
    """
    if not filenames:
        return {}

    import time
    grouping_start_time = time.perf_counter()
    logging.info(f"[grouping_service] Computing grouping for {len(filenames)} photos")

    # Step 1: Build PhotoMetadata list
    photos: list[PhotoMetadata] = []
    filename_to_idx: dict[str, int] = {}

    if progress_reporter:
        progress_reporter.update(2, 8)
        progress_reporter.detail("Building metadata...")
    logging.info(f"[grouping_service] Step 1/8: Building metadata for {len(filenames)} photos...")
    for idx, filename in enumerate(filenames):
        if (idx + 1) % 100 == 0:
            logging.info(f"[grouping_service] Step 1: Processed {idx + 1}/{len(filenames)} photos")
            if progress_reporter:
                progress_reporter.detail("Building metadata...")
        path = os.path.join(image_dir, filename)
        exif = exif_data.get(filename, {})
        timestamp = extract_timestamp(exif, path)
        camera_id = extract_camera_id(exif)

        photos.append(PhotoMetadata(
            filename=filename,
            timestamp=timestamp,
            camera_id=camera_id,
            path=path,
        ))
        filename_to_idx[filename] = idx

    # Sort by (camera_id, timestamp) for session detection
    if progress_reporter:
        progress_reporter.update(3, 8)
        progress_reporter.detail("Detecting sessions...")
    logging.info("[grouping_service] Step 2/8: Sorting photos by camera and timestamp...")
    photos.sort(key=lambda p: (p.camera_id, p.timestamp))

    # Step 2: Detect sessions
    logging.info("[grouping_service] Step 3/8: Detecting sessions...")
    sessions = detect_sessions(photos, session_gap_min=session_gap_min)
    session_count = len(set(sessions))
    logging.info(f"[grouping_service] Step 3: Found {session_count} sessions")
    if progress_reporter:
        progress_reporter.detail(f"Found {session_count} sessions")

    # Step 3: Detect bursts (within each session)
    if progress_reporter:
        progress_reporter.update(4, 8)
        progress_reporter.detail("Detecting bursts...")
    logging.info("[grouping_service] Step 4/8: Detecting bursts...")
    bursts: list[int] = []
    burst_id = 0
    last_timestamp: datetime | None = None

    for photo, session_id in zip(photos, sessions):
        # Start new burst if session changed or time gap > threshold
        if last_timestamp is None:
            bursts.append(burst_id)
        elif (photo.timestamp - last_timestamp).total_seconds() > burst_gap_sec:
            burst_id += 1
            bursts.append(burst_id)
        else:
            bursts.append(burst_id)
        last_timestamp = photo.timestamp
    burst_count = len(set(bursts))
    logging.info(f"[grouping_service] Step 4: Found {burst_count} bursts")
    if progress_reporter:
        progress_reporter.detail(f"Found {burst_count} bursts")

    # Step 4: Near-duplicate grouping
    if progress_reporter:
        progress_reporter.update(5, 8)
        progress_reporter.detail("Computing perceptual hashes...")
    logging.info("[grouping_service] Step 5/8: Computing perceptual hashes...")

    # Initialize hash cache
    from .phash_cache import PerceptualHashCache
    hash_cache = PerceptualHashCache()
    cache_hits = 0
    cache_misses = 0

    def hash_fn(filename: str) -> str:
        nonlocal cache_hits, cache_misses
        path = os.path.join(image_dir, filename)
        # Check cache first
        cached_hash = hash_cache.get_hash(path)
        if cached_hash:
            cache_hits += 1
            return cached_hash

        # Compute hash
        cache_misses += 1
        try:
            # OPTIMIZATION: Use shared image cache to avoid repeated file opens
            from .image_cache import get_cached_image
            img = get_cached_image(path)
            if img is None:
                return None
            phash = imagehash.phash(img)
            hash_string = str(phash)
            # Cache it
            hash_cache.set_hash(path, hash_string)
            return hash_string
        except Exception as e:
            logging.debug(f"[grouping_service] Failed to hash {filename}: {e}")
            error_hash = f"error_{filename}"
            # Don't cache errors
            return error_hash

    # OPTIMIZATION: Pre-compute all hashes in parallel for better performance
    # This helps when cache is cold or partially warm
    sorted_filenames_for_grouping = [p.filename for p in photos]
    if len(sorted_filenames_for_grouping) > 50:  # Only parallelize for larger datasets
        from concurrent.futures import ThreadPoolExecutor
        import os as os_module
        
        def compute_hash_parallel(filename: str) -> tuple[str, str]:
            """Compute hash for a single file, returns (filename, hash_string)."""
            return (filename, hash_fn(filename))
        
        logging.info(f"[grouping_service] Computing {len(sorted_filenames_for_grouping)} hashes in parallel...")
        with ThreadPoolExecutor(max_workers=min(4, os_module.cpu_count() or 2)) as executor:
            # Pre-compute all hashes in parallel
            list(executor.map(compute_hash_parallel, sorted_filenames_for_grouping))
        logging.info(f"[grouping_service] Hash computation complete (cache hits: {cache_hits}, misses: {cache_misses})")

    # Save cache after all hashes are computed
    hash_cache.save()

    if IMAGEHASH_AVAILABLE:
        if progress_reporter:
            progress_reporter.update(6, 8)
            progress_reporter.detail("Grouping near-duplicates...")
        logging.info(f"[grouping_service] Step 6/8: Grouping near-duplicates (hamming_threshold={phash_threshold})...")
        # Use sorted filenames to match the order of photos, sessions, bursts
        sorted_filenames_for_grouping = [p.filename for p in photos]
        # Pass image_dir and hash_cache to group_near_duplicates for cache access
        groups = group_near_duplicates(
            sorted_filenames_for_grouping,  # Use sorted order to match photos/sessions/bursts
            hash_fn,
            hamming_threshold=phash_threshold,
            progress_reporter=progress_reporter,
            image_dir=image_dir,  # For cache key generation
        )
        group_count = len(set(groups))
        logging.info(f"[grouping_service] Step 6: Created {group_count} groups from near-duplicate detection")

        # Post-process: Merge groups from same burst if visually somewhat similar
        # Use a more lenient threshold (30) for burst merging vs strict threshold (8) for initial grouping
        # This allows images from same burst to merge if they're reasonably similar, but prevents
        # merging completely different images (distance > 30) even if in same burst
        BURST_MERGE_THRESHOLD = 30  # More lenient threshold for burst merging

        burst_to_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)  # burst_id -> [(group_id, filename), ...]
        for idx, (burst_id, group_id) in enumerate(zip(bursts, groups)):
            filename = sorted_filenames_for_grouping[idx]
            burst_to_groups[burst_id].append((group_id, filename))

        # For each burst, check if groups should merge based on visual similarity
        group_parent: dict[int, int] = {}

        def find(gid: int) -> int:
            """Find root of group (with path compression)."""
            if gid not in group_parent:
                group_parent[gid] = gid
            if group_parent[gid] != gid:
                group_parent[gid] = find(group_parent[gid])
            return group_parent[gid]

        def union(gid1: int, gid2: int):
            """Merge two groups (always keep smaller ID as root)."""
            root1 = find(gid1)
            root2 = find(gid2)
            if root1 != root2:
                if root1 < root2:
                    group_parent[root2] = root1
                else:
                    group_parent[root1] = root2

        # Compute hashes for all images (needed for burst merging)
        all_hashes: dict[str, str] = {}
        for filename in sorted_filenames_for_grouping:
            path = os.path.join(image_dir, filename)
            h = hash_fn(filename)
            if h:
                all_hashes[filename] = h

        # For each burst, check pairwise hash distances between groups
        merged_bursts = 0
        for burst_id, group_items in burst_to_groups.items():
            if len(group_items) <= 1:
                continue

            # Group items by group_id
            groups_in_burst: dict[int, list[str]] = defaultdict(list)
            for group_id, filename in group_items:
                groups_in_burst[group_id].append(filename)

            if len(groups_in_burst) <= 1:
                continue  # Only one group in this burst, nothing to merge

            # Check pairwise distances between groups in this burst
            group_list = list(groups_in_burst.keys())
            should_merge = False

            for i, gid1 in enumerate(group_list):
                for gid2 in group_list[i+1:]:
                    # Find minimum distance between any images in these two groups
                    min_dist = float('inf')
                    for f1 in groups_in_burst[gid1]:
                        for f2 in groups_in_burst[gid2]:
                            h1 = all_hashes.get(f1)
                            h2 = all_hashes.get(f2)
                            if h1 and h2:
                                try:
                                    hash1 = imagehash.hex_to_hash(h1)
                                    hash2 = imagehash.hex_to_hash(h2)
                                    dist = hash1 - hash2
                                    if dist < min_dist:
                                        min_dist = dist
                                except Exception:
                                    pass

                    # If groups are visually similar enough, merge them
                    if min_dist <= BURST_MERGE_THRESHOLD:
                        union(gid1, gid2)
                        should_merge = True
                        logging.debug(f"[grouping_service] Burst {burst_id}: merging groups {gid1} and {gid2} (distance {min_dist} <= {BURST_MERGE_THRESHOLD})")

            if should_merge:
                merged_bursts += 1

        # Build final mapping: each group -> its root
        merged_final: dict[int, int] = {}
        for gid in set(groups):
            merged_final[gid] = find(gid)

        merged_count = 0
        for idx, group_id in enumerate(groups):
            if group_id in merged_final and merged_final[group_id] != group_id:
                groups[idx] = merged_final[group_id]
                merged_count += 1

        if merged_bursts > 0:
            final_group_count = len(set(groups))
            logging.info(f"[grouping_service] Merged {merged_bursts} bursts with visually similar groups: {group_count} → {final_group_count} groups")

        if progress_reporter:
            progress_reporter.detail(f"Created {len(set(groups))} groups")
        # Save cache after grouping completes
        hash_cache.save()
        if cache_hits > 0 or cache_misses > 0:
            total = cache_hits + cache_misses
            hit_rate = (cache_hits / total * 100) if total > 0 else 0
            logging.info(f"[grouping_service] Hash cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")
    else:
        # Fallback: use burst_id as group_id
        if progress_reporter:
            progress_reporter.update(6, 8)
            progress_reporter.detail("Using bursts as groups...")
        logging.info("[grouping_service] Step 6/8: Using bursts as groups (imagehash not available)")
        groups = bursts

    # Step 5: Compute pick_scores
    if progress_reporter:
        progress_reporter.update(7, 8)
        progress_reporter.detail("Computing pick scores...")
    logging.info("[grouping_service] Step 7/8: Computing pick scores...")
    pick_scores: list[float] = []
    timestamps: list[datetime] = []

    for idx, (photo, group_id) in enumerate(zip(photos, groups)):
        if (idx + 1) % 100 == 0:
            logging.info(f"[grouping_service] Step 7: Computed scores for {idx + 1}/{len(photos)} photos")
            if progress_reporter:
                progress_reporter.detail("Computing scores...")
        keep_prob = keep_probabilities.get(photo.filename, 0.5) if keep_probabilities else 0.5
        metrics = quality_metrics.get(photo.filename, {}) if quality_metrics else {}

        pick_score = compute_pick_score(
            global_keep_score=keep_prob,
            sharpness=metrics.get("sharpness"),
            exposure_quality=metrics.get("exposure_quality"),
            noise_level=metrics.get("noise_level"),
            face_quality=metrics.get("face_quality"),
            motion_blur=metrics.get("motion_blur"),
        )

        pick_scores.append(pick_score)
        timestamps.append(photo.timestamp)

    # Step 6: Recommend best picks
    if progress_reporter:
        progress_reporter.update(8, 8)
        progress_reporter.detail("Recommending best picks...")
    logging.info("[grouping_service] Step 8/8: Recommending best picks...")
    best_flags = recommend_best_pick(groups, pick_scores, timestamps=timestamps)
    best_count = sum(best_flags)
    logging.info(f"[grouping_service] Step 8: Selected {best_count} best picks")
    if progress_reporter:
        progress_reporter.detail(f"Selected {best_count} best picks")

    # Step 7: Compute group sizes
    group_sizes: dict[int, int] = {}
    for group_id in groups:
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1

    # Step 8: Build result dict (mapped back to original filename order)
    logging.info("[grouping_service] Building final result dictionary...")
    result: dict[str, dict] = {}

    # Create mapping from sorted order back to original filenames
    sorted_filenames = [p.filename for p in photos]

    for idx, filename in enumerate(sorted_filenames):
        result[filename] = {
            "session_id": sessions[idx],
            "burst_id": bursts[idx],
            "group_id": groups[idx],
            "group_size": group_sizes[groups[idx]],
            "is_group_best": best_flags[idx],
            "pick_score": pick_scores[idx],
        }

    final_group_count = len(set(g.get('group_id', 0) for g in result.values() if g.get('group_id') is not None))
    final_best_count = sum(1 for g in result.values() if g.get('is_group_best', False))
    grouping_elapsed = time.perf_counter() - grouping_start_time
    logging.info(f"[grouping_service] ✅ Completed: {final_group_count} groups, {final_best_count} best picks, {len(result)} total photos in {grouping_elapsed:.2f}s")
    if grouping_elapsed > 60:
        logging.warning(f"[grouping_service] ⚠️  Grouping took {grouping_elapsed:.2f}s (exceeds 60s limit)")
    return result


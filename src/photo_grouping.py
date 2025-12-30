"""Photo grouping: session detection, burst detection, near-duplicate clustering, best-pick recommendation.

Groups photos by:
1. Session (time gaps > 30 min or camera changes)
2. Burst (within session, time gaps > 1 sec)
3. Near-duplicate (perceptual hash similarity)
4. Best-pick recommendation (heuristic or ranker)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("[photo_grouping] networkx not available, falling back to simple grouping")


@dataclass
class PhotoMetadata:
    """Metadata for a photo used in grouping."""
    filename: str
    timestamp: datetime
    camera_id: str = "default"
    path: str = ""


# Default thresholds
SESSION_GAP_MIN = 10  # minutes
BURST_GAP_SEC = 10.0  # seconds
PHASH_HAMMING_THRESHOLD = 4  # bits (for near-duplicate detection) - stricter to avoid over-grouping


def detect_sessions(
    photos: list[PhotoMetadata], session_gap_min: int = SESSION_GAP_MIN
) -> list[int]:
    """Detect photo sessions based on time gaps and camera changes.

    Args:
        photos: List of PhotoMetadata, must be sorted by (camera_id, timestamp)
        session_gap_min: Time gap in minutes that starts a new session

    Returns:
        List of session IDs (one per photo)
    """
    if not photos:
        return []

    sessions: list[int] = []
    session_id = 0
    last_timestamp: datetime | None = None
    last_camera: str | None = None

    for photo in photos:
        # New session if:
        # 1. First photo
        # 2. Camera changed
        # 3. Time gap > threshold
        if last_timestamp is None or last_camera is None:
            # First photo: start session 0
            sessions.append(session_id)
        elif photo.camera_id != last_camera:
            # Camera changed: new session
            session_id += 1
            sessions.append(session_id)
        elif (photo.timestamp - last_timestamp).total_seconds() / 60 > session_gap_min:
            # Time gap > threshold: new session
            session_id += 1
            sessions.append(session_id)
        else:
            # Same session: use current session_id
            sessions.append(session_id)

        last_timestamp = photo.timestamp
        last_camera = photo.camera_id

    return sessions


def detect_bursts(
    photos: list[PhotoMetadata], burst_gap_sec: float = BURST_GAP_SEC
) -> list[int]:
    """Detect photo bursts within a session.

    Args:
        photos: List of PhotoMetadata, must be sorted by timestamp within same session
        burst_gap_sec: Time gap in seconds that starts a new burst

    Returns:
        List of burst IDs (one per photo)
    """
    if not photos:
        return []

    bursts: list[int] = []
    burst_id = 0
    last_timestamp: datetime | None = None

    for photo in photos:
        if last_timestamp is None:
            bursts.append(burst_id)
        elif (photo.timestamp - last_timestamp).total_seconds() > burst_gap_sec:
            burst_id += 1
            bursts.append(burst_id)
        else:
            bursts.append(burst_id)

        last_timestamp = photo.timestamp

    return bursts


def group_near_duplicates(
    filenames: list[str],
    hash_fn: Callable[[str], str],
    hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
    progress_reporter=None,  # Optional ProgressReporter for UI progress
    image_dir: str | None = None,  # Optional image directory for cache key generation
) -> list[int]:
    """Group photos by perceptual hash similarity using connected components.

    Args:
        filenames: List of image filenames
        hash_fn: Function that returns hash string for a filename
        hamming_threshold: Maximum Hamming distance for grouping (default: 10)

    Returns:
        List of group IDs (one per filename)
    """
    if not filenames:
        return []

    if not NETWORKX_AVAILABLE:
        # Fallback: simple grouping by exact hash match
        logging.warning("[photo_grouping] NetworkX not available, using simple hash grouping")
        hash_to_group: dict[str, int] = {}
        group_id = 0
        result = []
        for fname in filenames:
            h = hash_fn(fname)
            if h not in hash_to_group:
                hash_to_group[h] = group_id
                group_id += 1
            result.append(hash_to_group[h])
        return result

    # Build graph: edge between photos if Hamming distance <= threshold
    G = nx.Graph()
    filename_to_hash: dict[str, str] = {}

    # Compute all hashes
    logging.info(f"[photo_grouping] Computing hashes for {len(filenames)} photos...")
    for idx, fname in enumerate(filenames):
        if (idx + 1) % 100 == 0:
            logging.info(f"[photo_grouping] Computed hashes for {idx + 1}/{len(filenames)} photos")
            if progress_reporter:
                progress_reporter.detail(f"Computing hashes: {idx + 1}/{len(filenames)}...")
        try:
            filename_to_hash[fname] = hash_fn(fname)
            G.add_node(fname)
        except Exception as e:
            logging.debug(f"[photo_grouping] Failed to hash {fname}: {e}")
            # Assign unique group for failed hashes
            filename_to_hash[fname] = f"error_{fname}"
            G.add_node(fname)

    # Add edges for similar hashes
    # Convert hash strings to integers for Hamming distance calculation
    try:
        import imagehash

        for i, fname1 in enumerate(filenames):
            h1_str = filename_to_hash[fname1]
            if h1_str.startswith("error_"):
                continue

            try:
                h1 = imagehash.hex_to_hash(h1_str)
            except Exception:
                continue

            for fname2 in filenames[i + 1 :]:
                h2_str = filename_to_hash[fname2]
                if h2_str.startswith("error_"):
                    continue

                try:
                    h2 = imagehash.hex_to_hash(h2_str)
                    if h1 - h2 <= hamming_threshold:
                        G.add_edge(fname1, fname2)
                except Exception:
                    continue
    except ImportError:
        # imagehash not available, use string comparison as fallback
        logging.warning("[photo_grouping] imagehash not available, using exact hash matching")
        for i, fname1 in enumerate(filenames):
            h1_str = filename_to_hash[fname1]
            if isinstance(h1_str, str) and h1_str.startswith("error_"):
                continue
            for fname2 in filenames[i + 1 :]:
                h2_str = filename_to_hash[fname2]
                if isinstance(h2_str, str) and h2_str.startswith("error_"):
                    continue
                if h1_str == h2_str:  # Exact match only
                    G.add_edge(fname1, fname2)

    # Find connected components (each component is a group)
    components = list(nx.connected_components(G))

    # Map filenames to group IDs
    filename_to_group: dict[str, int] = {}
    for group_id, component in enumerate(components):
        for fname in component:
            filename_to_group[fname] = group_id

    # Assign unique group IDs to unconnected nodes (singletons)
    next_group_id = len(components)
    result = []
    for fname in filenames:
        if fname in filename_to_group:
            result.append(filename_to_group[fname])
        else:
            result.append(next_group_id)
            next_group_id += 1

    logging.info(f"[photo_grouping] Created {len(set(result))} groups from {len(filenames)} photos")
    return result


def compute_pick_score(
    global_keep_score: float,
    sharpness: float | None = None,
    exposure_quality: float | None = None,
    noise_level: float | None = None,
    face_quality: float | None = None,
    motion_blur: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute heuristic pick_score for best-pick recommendation.

    Args:
        global_keep_score: Model's keep probability (0-1)
        sharpness: Sharpness metric (higher = better, normalized 0-1)
        exposure_quality: Exposure quality (0-1, higher = better)
        noise_level: Noise level (0-1, lower = better, will be inverted)
        face_quality: Face detection quality (0-1, higher = better, optional)
        motion_blur: Motion blur estimate (0-1, lower = better, will be inverted)
        weights: Optional custom weights dict

    Returns:
        Pick score (higher = better candidate for best pick)
    """
    default_weights = {
        "global_keep": 0.4,
        "sharpness": 0.25,
        "exposure": 0.15,
        "noise": 0.1,
        "face": 0.05,
        "motion_blur": 0.05,
    }
    w = weights or default_weights

    score = w["global_keep"] * global_keep_score

    if sharpness is not None:
        score += w["sharpness"] * sharpness

    if exposure_quality is not None:
        score += w["exposure"] * exposure_quality

    if noise_level is not None:
        # Invert noise (lower noise = higher score)
        score += w["noise"] * (1.0 - min(1.0, noise_level))

    if face_quality is not None:
        score += w["face"] * face_quality

    if motion_blur is not None:
        # Invert motion blur (lower blur = higher score)
        score += w["motion_blur"] * (1.0 - min(1.0, motion_blur))

    return float(score)


def recommend_best_pick(
    groups: list[int],
    pick_scores: list[float],
    timestamps: list[datetime] | None = None,
) -> list[bool]:
    """Recommend best photo per group based on pick_score.

    Args:
        groups: List of group IDs (one per photo)
        pick_scores: List of pick scores (higher = better)
        timestamps: Optional list of timestamps for tie-breaking (earlier = better)

    Returns:
        List of bool flags (True = best pick in group)
    """
    if len(groups) != len(pick_scores):
        raise ValueError(f"groups and pick_scores must have same length: {len(groups)} != {len(pick_scores)}")

    if timestamps and len(timestamps) != len(groups):
        raise ValueError(f"timestamps must have same length as groups: {len(timestamps)} != {len(groups)}")

    best_flags = [False] * len(groups)

    # Group photos by group_id
    group_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(groups):
        group_to_indices[group_id].append(idx)

    # For each group, pick the best photo
    for group_id, indices in group_to_indices.items():
        if len(indices) == 1:
            # Singleton: always best
            best_flags[indices[0]] = True
            continue

        # Find index with highest pick_score
        best_idx = indices[0]
        best_score = pick_scores[best_idx]

        for idx in indices[1:]:
            score = pick_scores[idx]
            if score > best_score:
                best_idx = idx
                best_score = score
            elif score == best_score and timestamps:
                # Tie-breaker: earlier timestamp wins
                if timestamps[idx] < timestamps[best_idx]:
                    best_idx = idx

        best_flags[best_idx] = True

    return best_flags


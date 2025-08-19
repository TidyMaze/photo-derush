import os
import logging
from .utils import compute_perceptual_hash

def _hamming(a, b):
    try:
        return a - b  # imagehash objects support subtraction -> Hamming distance
    except Exception:
        return 9999

def cluster_duplicates(image_paths, directory='.', distance_threshold=4):
    """Group near-duplicate images using perceptual hashes.
    Args:
        image_paths: iterable of image file names (not full paths)
        directory: base dir
        distance_threshold: max Hamming distance to consider duplicate
    Returns:
        clusters: list[list[str]] groups of image names
        image_hashes: dict image_name -> hex hash string
    """
    hashes = {}
    for name in image_paths:
        path = os.path.join(directory, name)
        h = compute_perceptual_hash(path)
        if h is None:
            continue
        hashes[name] = h
    # Simple greedy clustering
    unassigned = set(hashes.keys())
    clusters = []
    while unassigned:
        seed = unassigned.pop()
        seed_hash = hashes[seed]
        group = [seed]
        to_remove = []
        for other in unassigned:
            if _hamming(seed_hash, hashes[other]) <= distance_threshold:
                group.append(other)
                to_remove.append(other)
        for o in to_remove:
            unassigned.discard(o)
        clusters.append(sorted(group))
    # Add singles for images without hash
    nohash = [n for n in image_paths if n not in hashes]
    for n in nohash:
        clusters.append([n])
    clusters.sort(key=lambda g: (len(g)==1, g[0]))
    image_hashes = {k: str(v) for k,v in hashes.items()}
    logging.info('[Duplicate] Formed %d clusters (threshold=%d)', len(clusters), distance_threshold)
    return clusters, image_hashes

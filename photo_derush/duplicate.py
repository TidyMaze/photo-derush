import os
import logging
import numpy as np
import imagehash
from .utils import image_manager
try:
    import faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False

def compute_hash(img_path, hash_type='dhash'):
    img = image_manager.get_image(img_path)
    if img is None:
        logging.warning(f"[compute_hash] Could not load image: {img_path}")
        return None
    if hash_type == 'dhash':
        return imagehash.dhash(img)
    elif hash_type == 'phash':
        return imagehash.phash(img)
    else:
        raise ValueError(f"Unknown hash_type: {hash_type}")

def _hamming(a, b):
    try:
        return a - b  # imagehash objects support subtraction -> Hamming distance
    except Exception:
        return 9999

def cluster_duplicates(image_paths, directory='.', hash_type='dhash', distance_threshold=5, use_faiss=True, hamming_thresh=None):
    """Group near-duplicate images using hashes and optional FAISS for speed."""
    if hamming_thresh is not None:
        distance_threshold = hamming_thresh
    hashes = {}
    hash_arrs = []
    valid_names = []
    for name in image_paths:
        path = os.path.join(directory, name)
        h = compute_hash(path, hash_type=hash_type)
        if h is None:
            continue
        hashes[name] = h
        h_bytes = int(str(h), 16).to_bytes(8, 'big')
        hash_arrs.append(np.frombuffer(h_bytes, dtype='uint8'))
        valid_names.append(name)
    if use_faiss and _faiss_available and hash_arrs:
        hashes_np = np.stack(hash_arrs).astype('uint8')
        index = faiss.IndexBinaryFlat(64)
        index.add(hashes_np)
        clusters = []
        assigned = set()
        for i in range(len(valid_names)):
            if i in assigned:
                continue
            lims, D, I = index.range_search(hashes_np[i][np.newaxis, :], distance_threshold)
            cluster = [valid_names[j] for j in I[lims[0]:lims[1]] if j not in assigned]
            for j in I[lims[0]:lims[1]]:
                assigned.add(j)
            if len(cluster) > 1:
                clusters.append(sorted(cluster))
        # Add singles
        singles = [valid_names[i] for i in range(len(valid_names)) if i not in assigned]
        for s in singles:
            clusters.append([s])
    else:
        # Fallback: simple greedy clustering
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
    logging.info('[Duplicate] Formed %d clusters (threshold=%d, hash_type=%s)', len(clusters), distance_threshold, hash_type)
    return clusters, image_hashes

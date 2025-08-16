import os
import time
import logging
import numpy as np
import imagehash
from PIL import Image
from photo_derush.image_manager import image_manager
import faiss

MAX_IMAGES = 200  # reused if needed

def compute_dhash(image_path):
    img = image_manager.get_image(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image for dhash: {image_path}")
    return imagehash.dhash(img)

def compute_duplicate_groups(hashes):
    if not hashes or not all(h is not None for h in hashes):
        return {}, {}, {}
    hashes_np = np.stack(hashes).astype('uint8')
    index = faiss.IndexBinaryFlat(64)
    index.add(hashes_np)
    group_ids = {}
    group_cardinality = {}
    hash_map = {}
    current_group = 1
    assigned = set()
    total = len(hashes_np)
    log_interval = max(1, total // 20)
    start_group_time = time.perf_counter()
    logging.info("[Grouping] (async) Starting duplicate grouping over %d hashes", total)
    for i in range(total):
        if i in assigned:
            if (i + 1) % log_interval == 0 or (i + 1) == total:
                elapsed = time.perf_counter() - start_group_time
                logging.info("[Grouping] (async) %d/%d (%.1f%%) processed (elapsed %.1fs)",
                             i + 1, total, (i + 1)/total*100.0, elapsed)
            continue
        lims, D, I = index.range_search(hashes_np[i][np.newaxis, :], 5)
        cluster = set(j for j in I[lims[0]:lims[1]] if j != i)
        cluster.add(i)
        if len(cluster) > 1:
            for j in cluster:
                group_ids[j] = current_group
                assigned.add(j)
            group_cardinality[current_group] = len(cluster)
            current_group += 1
        if (i + 1) % log_interval == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - start_group_time
            logging.info("[Grouping] (async) %d/%d (%.1f%%) processed, groups=%d (elapsed %.1fs)",
                         i + 1, total, (i + 1)/total*100.0, current_group - 1, elapsed)
    for idx in range(len(hashes)):
        if idx not in group_ids:
            group_ids[idx] = None
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_map[idx] = ''.join(f'{b:02x}' for b in h)
        else:
            hash_map[idx] = None
    logging.info("[Grouping] (async) Completed in %.2fs. Groups=%d", time.perf_counter() - start_group_time, current_group - 1)
    return group_ids, group_cardinality, hash_map


def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]


def prepare_images_and_groups(directory: str, max_images: int = MAX_IMAGES):
    images = list_images(directory)
    logging.info("[Prep] (async) Found %d images in %s", len(images), directory)
    if not images:
        return [], {}, {"total_images": 0, "duplicate_group_count": 0, "duration_seconds": 0.0}
    logging.info("[Hashing] (async) Starting hash + group computation for %d images", len(images))
    start_hash_time = time.perf_counter()
    image_hashes = {}
    hashes = []
    total = len(images)
    log_interval = max(1, total // 20)
    for idx, img in enumerate(images):
        img_path = os.path.join(directory, img)
        try:
            dh = compute_dhash(img_path)
            h_bytes = int(str(dh), 16).to_bytes(8, 'big')
            hash_arr = np.frombuffer(h_bytes, dtype='uint8')
            image_hashes[img] = hash_arr
            hashes.append(hash_arr)
        except Exception as ex:
            logging.warning("[Hashing] (async) Failed for %s: %s", img, ex)
            image_hashes[img] = None
            hashes.append(None)
        if (idx + 1) % log_interval == 0 or (idx + 1) == total:
            elapsed = time.perf_counter() - start_hash_time
            logging.info("[Hashing] (async) %d/%d (%.1f%%) hashed (elapsed %.1fs)",
                         idx + 1, total, (idx + 1)/total*100.0, elapsed)
    group_ids, group_cardinality, hash_map = compute_duplicate_groups(hashes)
    duration = time.perf_counter() - start_hash_time
    duplicate_group_count = len([g for g in set(group_ids.values()) if g is not None])
    logging.info("[Hashing] (async) Finished in %.2fs. Duplicate groups: %d", duration, duplicate_group_count)
    image_info = {img: {"hash": hash_map.get(idx), "group": group_ids.get(idx)} for idx, img in enumerate(images)}
    stats = {"total_images": len(images), "duplicate_group_count": duplicate_group_count, "duration_seconds": duration}
    return images, image_info, stats

import os
import time
import logging
from photo_derush.duplicate import cluster_duplicates

MAX_IMAGES = 200  # reused if needed

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
    clusters, image_hashes = cluster_duplicates(images, directory, hash_type='dhash', distance_threshold=5, use_faiss=True)
    duration = time.perf_counter() - start_hash_time
    duplicate_group_count = len([c for c in clusters if len(c) > 1])
    image_info = {img: {"hash": image_hashes.get(img), "group": None} for img in images}
    # Assign group ids for each image
    group_id_map = {}
    for group_idx, group in enumerate(clusters, 1):
        if len(group) > 1:
            for img in group:
                image_info[img]["group"] = group_idx
            group_id_map[group_idx] = len(group)
        else:
            img = group[0]
            image_info[img]["group"] = None
    stats = {"total_images": len(images), "duplicate_group_count": duplicate_group_count, "duration_seconds": duration}
    logging.info("[Hashing] (async) Finished in %.2fs. Duplicate groups: %d", duration, duplicate_group_count)
    return images, image_info, stats

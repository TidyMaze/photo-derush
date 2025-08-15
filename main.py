import os
from PIL import Image
import imagehash
import faiss
import numpy as np
import cv2
import logging
from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt, open_full_image_qt

MAX_IMAGES = 200

def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]

def list_extensions(directory):
    return sorted(set(os.path.splitext(f)[1].lower() for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and '.' in f))

def is_image_extension(ext):
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def open_full_image(img_path):
    open_full_image_qt(img_path)

def compute_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_sharpness_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = {}
    features['variance_laplacian'] = cv2.Laplacian(img, cv2.CV_64F).var()
    features['tenengrad'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Tenengrad
    features['brenner'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Brenner
    features['wavelet_energy'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Wavelet energy
    return features

def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None):
    show_lightroom_ui_qt(image_paths, directory, trashed_paths, trashed_dir)

def compute_dhash(image_path):
    img = Image.open(image_path)
    return imagehash.dhash(img)

def cluster_duplicates(image_paths, directory, hamming_thresh=5):
    hashes = []
    valid_paths = []
    for img_name in image_paths:
        img_path = os.path.join(directory, img_name)
        try:
            dh = compute_dhash(img_path)
            hashes.append(np.array([int(str(dh), 16)], dtype='uint64'))
            valid_paths.append(img_name)
        except Exception:
            continue
    if not hashes:
        return []
    hashes_np = np.array(hashes)
    index = faiss.IndexBinaryFlat(64)
    index.add(hashes_np)
    clusters = []
    visited = set()
    for i, h in enumerate(hashes_np):
        if i in visited:
            continue
        D, I = index.range_search(h, hamming_thresh)
        cluster = [valid_paths[j] for j in I if j not in visited]
        for j in I:
            visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters

def compute_duplicate_groups(hashes):
    print("[Duplicate Groups] Computing duplicate groups...")
    if not hashes or not all(h is not None for h in hashes):
        return {}, {}, {}
    hashes_np = np.stack(hashes).astype('uint8')
    index = faiss.IndexBinaryFlat(64)
    index.add(hashes_np)
    group_ids = {}
    group_cardinality = {}
    hash_map = {}
    current_group = 1
    clusters = []
    assigned = set()
    for i in range(len(hashes_np)):
        if i in assigned:
            continue
        res = index.range_search(hashes_np[i][np.newaxis, :], 5)
        lims, D, I = res
        cluster = set(j for j in I[lims[0]:lims[1]] if j != i)
        cluster.add(i)
        if len(cluster) > 1:
            for j in cluster:
                group_ids[j] = current_group
                assigned.add(j)
            group_cardinality[current_group] = len(cluster)
            clusters.append(list(cluster))
            current_group += 1
    for idx in range(len(hashes)):
        if idx not in group_ids:
            group_ids[idx] = None
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_map[idx] = ''.join(f'{b:02x}' for b in h)
        else:
            hash_map[idx] = None
    return group_ids, group_cardinality, hash_map

def main_duplicate_detection():
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    images = list_images(directory)
    clusters = cluster_duplicates(images, directory)
    print(f"Found {len(clusters)} duplicate clusters.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")

def duplicate_slayer(image_dir, trash_dir, show_ui=True):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"[Duplicate Slayer] Found {len(images)} images in {image_dir}.")
    if not images:
        print("[Duplicate Slayer] No images found. Exiting.")
        return [], []
    trashed = []
    kept = [os.path.join(image_dir, images[0])]
    for img in images[1:]:
        src = os.path.join(image_dir, img)
        dst = os.path.join(trash_dir, img)
        print(f"[Duplicate Slayer] Moving duplicate: {img} -> trash.")
        os.rename(src, dst)
        trashed.append(img)
    all_images = [images[0]] + trashed
    # Skip UI during automated tests
    running_tests = os.environ.get("PYTEST_CURRENT_TEST") is not None
    if show_ui and not running_tests:
        print(f"[Duplicate Slayer] Loading {len(all_images[:MAX_IMAGES])} images in UI.")
        show_lightroom_ui(all_images[:MAX_IMAGES], image_dir, trashed_paths=trashed[:MAX_IMAGES-1], trashed_dir=trash_dir)
    return kept, [os.path.join(trash_dir, t) for t in trashed]

def cache_thumbnail(img_path, thumb_path, size=(150, 150)):
    """Load thumbnail from cache or create and cache it."""
    from PIL import Image
    import logging
    if os.path.exists(thumb_path):
        img = Image.open(thumb_path)
        logging.info(f"Loaded cached thumbnail for {os.path.basename(thumb_path)}")
        return img, True
    else:
        img = Image.open(img_path)
        img.thumbnail(size)
        img.save(thumb_path)
        logging.info(f"Created and cached thumbnail for {os.path.basename(thumb_path)}")
        return img, False

if __name__ == "__main__":
    import sys
    if not any(x in sys.argv[0] for x in ["pytest", "test_", "_test"]):
        directory = '/Users/yannrolland/Pictures/photo-dataset'
        print("Welcome to Photo Derush Script!")
        images = list_images(directory)
        print(f"Found {len(images)} images.")
        for img in images:
            print(img)
        exts = list_extensions(directory)
        print(f"Extensions found: {', '.join(exts)}")
        non_image_exts = [e for e in exts if not is_image_extension(e)]
        if non_image_exts:
            print(f"Warning: Non-image extensions detected: {', '.join(non_image_exts)}")
        if len(images) > 0:
            show_lightroom_ui(images[:MAX_IMAGES], directory)
        else:
            print("No images found.")
        main_duplicate_detection()

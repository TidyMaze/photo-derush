"""Dataset building module.
Single responsibility: construct labeled feature matrices from repository.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .features import FEATURE_COUNT, batch_extract_features
from .model import RatingsTagsRepository


def build_dataset(
    image_dir: str, repo: RatingsTagsRepository, progress_callback=None, displayed_filenames: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import time
    dataset_start = time.perf_counter()
    logging.info(f"[dataset] ===== BUILD DATASET START =====")
    
    from .features import load_feature_cache
    cache_load_start = time.perf_counter()
    feature_cache = load_feature_cache()
    cache_load_time = time.perf_counter() - cache_load_start
    logging.info(f"[dataset] Feature cache loaded in {cache_load_time*1000:.1f}ms: {len(feature_cache)} entries")
    
    X: list[list[float]] = []
    y: list[int] = []
    filenames: list[str] = []
    if not os.path.isdir(image_dir):
        logging.error("Image directory does not exist: %s", image_dir)
        return np.zeros((0, FEATURE_COUNT)), np.zeros((0,), dtype=int), []
    
    file_list_start = time.perf_counter()
    if displayed_filenames is not None:
        all_files = displayed_filenames
        logging.info(f"[dataset] Using displayed filenames: {len(all_files)}")
    else:
        all_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        logging.info(f"[dataset] Using all files: {len(all_files)}")
    file_list_time = time.perf_counter() - file_list_start
    logging.info(f"[dataset] File listing completed in {file_list_time*1000:.1f}ms")
    
    label_check_start = time.perf_counter()
    manual_count = 0
    auto_skipped = 0
    labeled_files = []
    labels = []
    for fname in all_files:
        state = repo.get_state(fname)
        if state not in ("keep", "trash"):
            continue
        source = repo.get_label_source(fname)
        if source != "manual":
            auto_skipped += 1
            logging.debug(f"[dataset] Skip auto-labeled {fname} state={state} source={source}")
            continue
        manual_count += 1
        labeled_files.append(fname)
        labels.append(1 if state == "keep" else 0)
    label_check_time = time.perf_counter() - label_check_start
    logging.info(f"[dataset] Label checking completed in {label_check_time*1000:.1f}ms: {manual_count} manual, {auto_skipped} auto skipped")
    
    if auto_skipped:
        logging.info(f"[dataset] Dataset: {manual_count} manual, {auto_skipped} auto skipped")
    if not labeled_files:
        logging.info(f"[dataset] No labeled files found")
        return np.zeros((0, FEATURE_COUNT)), np.zeros((0,), dtype=int), []
    
    image_paths = [os.path.join(image_dir, f) for f in labeled_files]
    logging.info(f"[dataset] Starting feature extraction for {len(image_paths)} images...")
    feats_all = batch_extract_features(image_paths, progress_callback=progress_callback)
    
    result_build_start = time.perf_counter()
    for fname, feats, label in zip(labeled_files, feats_all, labels):
        if feats is not None:
            X.append(feats)
            y.append(label)
            filenames.append(fname)
    result_build_time = time.perf_counter() - result_build_start
    logging.info(f"[dataset] Result building completed in {result_build_time*1000:.1f}ms: {len(X)} valid samples")
    
    if not X:
        logging.warning(f"[dataset] No valid features extracted")
        return np.zeros((0, FEATURE_COUNT)), np.zeros((0,), dtype=int), []
    
    dataset_time = time.perf_counter() - dataset_start
    logging.info(f"[dataset] ===== BUILD DATASET COMPLETE: {dataset_time:.2f}s total =====")
    return np.array(X, dtype=float), np.array(y, dtype=int), filenames


__all__ = ["build_dataset"]

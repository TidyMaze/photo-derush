#!/usr/bin/env python3
"""Extract features for AVA dataset images.

This script processes AVA images and extracts features compatible with the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import batch_extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("extract_ava")


def find_ava_images(ava_images_dir: str, image_ids: list[str]) -> dict[str, str]:
    """Find AVA image files by ID.
    
    AVA images are typically named: <image_id>.jpg
    They may be in subdirectories.
    """
    image_map = {}
    
    # Search for images
    for root, dirs, files in os.walk(ava_images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Extract image ID from filename (remove extension)
                file_id = os.path.splitext(file)[0]
                if file_id in image_ids:
                    full_path = os.path.join(root, file)
                    image_map[file_id] = full_path
    
    return image_map


def main():
    parser = argparse.ArgumentParser(description="Extract features for AVA images")
    parser.add_argument("--ava-images-dir", required=True, help="Directory containing AVA images")
    parser.add_argument("--ava-labels", required=True, help="AVA labels JSON file")
    parser.add_argument("--output", default=".cache/ava_features.joblib", help="Output features file")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("AVA FEATURE EXTRACTION")
    log.info("="*80)
    
    # Load AVA labels
    log.info(f"Loading AVA labels from {args.ava_labels}...")
    with open(args.ava_labels) as f:
        ava_data = json.load(f)
    
    if args.max_images:
        ava_data = ava_data[:args.max_images]
    
    log.info(f"Processing {len(ava_data)} AVA images")
    
    # Extract image IDs
    image_ids = [img['image_id'] for img in ava_data]
    
    # Find image files
    log.info(f"Searching for images in {args.ava_images_dir}...")
    image_map = find_ava_images(args.ava_images_dir, image_ids)
    
    log.info(f"Found {len(image_map)}/{len(image_ids)} images")
    
    if len(image_map) == 0:
        log.error("No images found! Check --ava-images-dir path")
        return 1
    
    # Get image paths in order
    image_paths = []
    labels = []
    found_ids = []
    
    for img_data in ava_data:
        img_id = img_data['image_id']
        if img_id in image_map:
            image_paths.append(image_map[img_id])
            labels.append(1 if img_data['label'] == 'keep' else 0)
            found_ids.append(img_id)
    
    log.info(f"Extracting features for {len(image_paths)} images...")
    
    # Extract features
    features = batch_extract_features(image_paths)
    
    # Filter out None results
    valid_features = []
    valid_labels = []
    valid_ids = []
    
    for i, feat in enumerate(features):
        if feat is not None:
            valid_features.append(feat)
            valid_labels.append(labels[i])
            valid_ids.append(found_ids[i])
    
    log.info(f"Successfully extracted features for {len(valid_features)} images")
    
    # Save features
    features_data = {
        'features': np.array(valid_features),
        'labels': np.array(valid_labels),
        'image_ids': valid_ids,
        'image_paths': [image_map[img_id] for img_id in valid_ids],
    }
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    joblib.dump(features_data, args.output)
    
    log.info(f"\nFeatures saved to: {args.output}")
    log.info(f"  Images: {len(valid_features)}")
    log.info(f"  Keep: {np.sum(valid_labels == 1)} ({np.sum(valid_labels == 1)/len(valid_labels)*100:.1f}%)")
    log.info(f"  Trash: {np.sum(valid_labels == 0)} ({np.sum(valid_labels == 0)/len(valid_labels)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



#!/usr/bin/env python3
"""Download and prepare AVA dataset for keep/trash training.

AVA dataset: ~250,000 images with aesthetic quality scores (1-10).
We convert quality scores to keep/trash labels.

Usage:
    poetry run python scripts/download_ava_dataset.py [--output-dir OUTPUT] [--max-images N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("ava_downloader")


def download_ava_metadata(output_dir: str):
    """Download AVA dataset metadata (image IDs and scores).
    
    Note: AVA dataset requires manual download from:
    https://github.com/mtobeiyf/ava_downloader
    Or use the official AVA dataset from:
    https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a
    """
    metadata_path = os.path.join(output_dir, "AVA_metadata.txt")
    
    if os.path.exists(metadata_path):
        log.info(f"Metadata already exists: {metadata_path}")
        return metadata_path
    
    log.warning("AVA metadata file not found.")
    log.warning("AVA dataset requires manual setup:")
    log.warning("1. Clone: git clone https://github.com/mtobeiyf/ava_downloader.git")
    log.warning("2. Or download from: https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a")
    log.warning(f"3. Place AVA.txt in: {metadata_path}")
    return None


def parse_ava_metadata(metadata_path: str):
    """Parse AVA metadata file.
    
    Format: image_id score1 score2 ... score10 challenge_id
    """
    images = []
    with open(metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            # Format: index image_id score1_count ... score10_count challenge_id
            # Skip index (parts[0]), image_id is parts[1]
            image_id = parts[1]
            # Score counts are parts[2:12] (10 values for ratings 1-10)
            score_counts = [int(s) for s in parts[2:12]]
            total_votes = sum(score_counts)
            if total_votes == 0:
                continue
            # Weighted mean: (1*count1 + 2*count2 + ... + 10*count10) / total_votes
            weighted_sum = sum((i+1) * count for i, count in enumerate(score_counts))
            mean_score = weighted_sum / total_votes
            images.append({
                'image_id': image_id,
                'scores': score_counts,
                'mean_score': mean_score,
            })
    return images


def convert_to_keep_trash(images: list[dict], threshold: float = 6.0):
    """Convert quality scores to keep/trash labels.
    
    Args:
        images: List of image dicts with 'mean_score'
        threshold: Score threshold (≥ threshold → keep, < threshold → trash)
    
    Returns:
        List of dicts with 'image_id', 'label', 'score'
    """
    labeled = []
    for img in images:
        label = "keep" if img['mean_score'] >= threshold else "trash"
        labeled.append({
            'image_id': img['image_id'],
            'label': label,
            'score': img['mean_score'],
            'scores': img['scores'],
        })
    return labeled


def download_ava_images(labeled_images: list[dict], output_dir: str, max_images: int = None):
    """Download AVA images (requires ava_downloader script).
    
    Note: This is a placeholder. Actual download requires the ava_downloader tool.
    """
    log.warning("Image download requires the ava_downloader tool.")
    log.warning("Install with: git clone https://github.com/mtobeiyf/ava_downloader.git")
    log.warning("Then use: python ava_downloader/download_ava.py")
    
    # Save labeled metadata for use with downloader
    labels_path = os.path.join(output_dir, "ava_keep_trash_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labeled_images[:max_images] if max_images else labeled_images, f, indent=2)
    
    log.info(f"Saved {len(labeled_images)} labels to {labels_path}")
    log.info("Use this file with ava_downloader to download only labeled images")
    
    return labels_path


def main():
    parser = argparse.ArgumentParser(description="Download and prepare AVA dataset")
    parser.add_argument("--output-dir", default=".cache/ava_dataset", help="Output directory")
    parser.add_argument("--threshold", type=float, default=6.0, help="Quality score threshold for keep/trash")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--download-images", action="store_true", help="Attempt to download images (requires ava_downloader)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info("="*80)
    log.info("AVA DATASET PREPARATION")
    log.info("="*80)
    log.info("AVA dataset: ~250,000 images with aesthetic quality scores")
    log.info(f"Converting scores to keep/trash (threshold: {args.threshold})")
    
    # Download metadata
    metadata_path = download_ava_metadata(args.output_dir)
    if not metadata_path:
        return 1
    
    # Parse metadata
    log.info("Parsing AVA metadata...")
    images = parse_ava_metadata(metadata_path)
    log.info(f"Found {len(images)} images in metadata")
    
    if args.max_images:
        images = images[:args.max_images]
        log.info(f"Limited to {len(images)} images")
    
    # Convert to keep/trash
    log.info(f"Converting quality scores to keep/trash (threshold: {args.threshold})...")
    labeled = convert_to_keep_trash(images, args.threshold)
    
    keep_count = sum(1 for img in labeled if img['label'] == 'keep')
    trash_count = sum(1 for img in labeled if img['label'] == 'trash')
    
    log.info(f"Labels: {keep_count} keep, {trash_count} trash")
    log.info(f"Keep rate: {keep_count/len(labeled)*100:.1f}%")
    
    # Save labels
    labels_path = os.path.join(args.output_dir, "ava_keep_trash_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labeled, f, indent=2)
    
    log.info(f"\nSaved labels to: {labels_path}")
    
    # Statistics
    keep_scores = [img['score'] for img in labeled if img['label'] == 'keep']
    trash_scores = [img['score'] for img in labeled if img['label'] == 'trash']
    
    log.info(f"\nScore statistics:")
    log.info(f"  Keep: mean={np.mean(keep_scores):.2f}, std={np.std(keep_scores):.2f}")
    log.info(f"  Trash: mean={np.mean(trash_scores):.2f}, std={np.std(trash_scores):.2f}")
    
    # Download instructions
    log.info(f"\n{'='*80}")
    log.info("NEXT STEPS")
    log.info(f"{'='*80}")
    log.info("1. Download AVA images using ava_downloader:")
    log.info("   git clone https://github.com/mtobeiyf/ava_downloader.git")
    log.info("   cd ava_downloader")
    log.info("   python download_ava.py")
    log.info("")
    log.info("2. Or use the labels file to download only needed images")
    log.info(f"   Labels file: {labels_path}")
    log.info("")
    log.info("3. After downloading, integrate with your training pipeline")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


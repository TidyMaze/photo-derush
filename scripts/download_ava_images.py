#!/usr/bin/env python3
"""Download AVA dataset images.

Supports multiple download methods:
1. Kaggle (requires kaggle API)
2. HuggingFace (requires datasets library)
3. Direct download from URLs (subset)
4. Manual download instructions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("download_ava")


def download_via_kaggle(output_dir: str, max_images: int = None):
    """Download AVA images via Kaggle API."""
    try:
        import kaggle
    except ImportError:
        log.error("Kaggle API not installed. Install with: pip install kaggle")
        log.info("Then set up credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False
    
    log.info("Downloading AVA dataset from Kaggle...")
    try:
        dataset = "nicolacarrassi/ava-aesthetic-visual-assessment"
        kaggle.api.dataset_download_files(
            dataset,
            path=output_dir,
            unzip=True,
        )
        log.info(f"Downloaded to {output_dir}")
        return True
    except Exception as e:
        log.error(f"Kaggle download failed: {e}")
        return False


def download_via_huggingface(output_dir: str, max_images: int = None, batch_size: int = 100):
    """Download AVA images via HuggingFace datasets with batch processing."""
    try:
        from datasets import load_dataset
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError:
        log.error("HuggingFace datasets not installed. Install with: pip install datasets")
        return False
    
    log.info("Downloading AVA dataset from HuggingFace...")
    try:
        # Use streaming for large datasets
        dataset = load_dataset("Iceclear/AVA", split="train", streaming=True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Batch processing for faster downloads
        batch = []
        count = 0
        skipped = 0
        
        def save_image(item_data):
            """Save a single image."""
            image_id, image = item_data
            if image:
                try:
                    image_path = os.path.join(output_dir, f"{image_id}.jpg")
                    if not os.path.exists(image_path):  # Skip if already exists
                        image.save(image_path)
                        return True
                    return False  # Already exists
                except Exception as e:
                    log.debug(f"Failed to save {image_id}: {e}")
                    return False
            return False
        
        log.info(f"Processing images in batches of {batch_size} (10 parallel workers)...")
        
        for item in dataset:
            if max_images and count >= max_images:
                break
            
            image_id = item.get('image_id')
            if not image_id:
                image_id = str(count)
            
            image = item.get('image')
            
            batch.append((image_id, image))
            
            # Process batch when full
            if len(batch) >= batch_size:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(save_image, item_data) for item_data in batch]
                    for future in as_completed(futures):
                        if future.result():
                            count += 1
                        else:
                            skipped += 1
                
                if count % 500 == 0:
                    log.info(f"Downloaded {count} images (skipped {skipped} existing/failed)...")
                
                batch = []
        
        # Process remaining batch
        if batch:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(save_image, item_data) for item_data in batch]
                for future in as_completed(futures):
                    if future.result():
                        count += 1
                    else:
                        skipped += 1
        
        log.info(f"Downloaded {count} images to {output_dir} (skipped {skipped})")
        return count > 0
    except Exception as e:
        log.error(f"HuggingFace download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_subset_from_urls(output_dir: str, ava_labels: str, max_images: int = 100):
    """Download a small subset of AVA images directly from URLs.
    
    Note: This is a limited approach. For full dataset, use Kaggle/HuggingFace/Torrent.
    """
    log.warning("Direct URL download is limited. For full dataset, use:")
    log.warning("  - Kaggle: pip install kaggle && kaggle datasets download nicolacarrassi/ava-aesthetic-visual-assessment")
    log.warning("  - HuggingFace: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('Iceclear/AVA')\"")
    log.warning("  - Torrent: http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460")
    
    # AVA images are hosted at: http://www.dpchallenge.com/image.php?IMAGE_ID=<id>
    # But direct download may not work. Better to use official sources.
    return False


def download_via_torrent(output_dir: str):
    """Provide instructions for torrent download."""
    log.info("="*80)
    log.info("TORRENT DOWNLOAD INSTRUCTIONS")
    log.info("="*80)
    log.info("")
    log.info("1. Download torrent file:")
    log.info("   http://academictorrents.com/download/71631f83b11d3d79d8f84efe0a7e12f0ac001460.torrent")
    log.info("")
    log.info("2. Open with torrent client (uTorrent, Transmission, etc.)")
    log.info("")
    log.info("3. Set download directory to:")
    log.info(f"   {output_dir}")
    log.info("")
    log.info("4. Wait for download to complete (~32GB, 255k images)")
    log.info("")
    log.info("Alternative: Use magnet link:")
    log.info("   magnet:?xt=urn:btih:71631f83b11d3d79d8f84efe0a7e12f0ac001460")
    log.info("")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download AVA dataset images")
    parser.add_argument("--output-dir", default=".cache/ava_dataset/images", help="Output directory for images")
    parser.add_argument("--method", choices=["kaggle", "huggingface", "torrent", "auto"], default="auto", help="Download method")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to download (for testing)")
    parser.add_argument("--ava-labels", default=".cache/ava_dataset/ava_keep_trash_labels.json", help="AVA labels file")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info("="*80)
    log.info("AVA IMAGE DOWNLOAD")
    log.info("="*80)
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Method: {args.method}")
    log.info("")
    
    success = False
    
    if args.method == "auto":
        # Try methods in order
        log.info("Trying download methods in order...")
        
        # Try HuggingFace first (easiest)
        log.info("\n1. Trying HuggingFace...")
        if download_via_huggingface(args.output_dir, args.max_images):
            success = True
        else:
            # Try Kaggle
            log.info("\n2. Trying Kaggle...")
            if download_via_kaggle(args.output_dir, args.max_images):
                success = True
            else:
                # Provide torrent instructions
                log.info("\n3. Torrent download required...")
                download_via_torrent(args.output_dir)
    
    elif args.method == "huggingface":
        success = download_via_huggingface(args.output_dir, args.max_images)
    elif args.method == "kaggle":
        success = download_via_kaggle(args.output_dir, args.max_images)
    elif args.method == "torrent":
        download_via_torrent(args.output_dir)
        return 0
    
    if success:
        log.info("")
        log.info("="*80)
        log.info("DOWNLOAD COMPLETE")
        log.info("="*80)
        log.info(f"Images saved to: {args.output_dir}")
        log.info("")
        log.info("Next steps:")
        log.info("1. Extract features:")
        log.info(f"   poetry run python scripts/extract_ava_features.py --ava-images-dir {args.output_dir} --ava-labels {args.ava_labels}")
        log.info("")
        log.info("2. Train on combined dataset:")
        log.info("   poetry run python scripts/train_with_ava.py --max-ava 10000")
        return 0
    else:
        log.info("")
        log.info("="*80)
        log.info("MANUAL DOWNLOAD REQUIRED")
        log.info("="*80)
        log.info("")
        log.info("Recommended: Download via torrent or MEGA")
        log.info("")
        log.info("Torrent:")
        log.info("  http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460")
        log.info("")
        log.info("MEGA:")
        log.info("  https://mega.nz/folder/9b520Lzb#2gIa1fgAzr677dcHKxjmtQ")
        log.info("  (64 7z files, ~32GB total)")
        log.info("")
        log.info("After downloading, extract to:")
        log.info(f"  {args.output_dir}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


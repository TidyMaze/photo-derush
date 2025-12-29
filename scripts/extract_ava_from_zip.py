#!/usr/bin/env python3
"""Extract AVA images from zip file and match with labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("extract_ava")


def extract_images_from_zip(zip_path: str, output_dir: str, ava_labels: str, max_images: int = None):
    """Extract images from AVA zip file matching the labels."""
    
    # Load labels
    log.info(f"Loading labels from {ava_labels}...")
    with open(ava_labels) as f:
        label_data = json.load(f)
    
    if max_images:
        label_data = label_data[:max_images]
    
    # Create image ID set for fast lookup
    image_ids = {img['image_id'] for img in label_data}
    log.info(f"Looking for {len(image_ids)} image IDs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    log.info(f"Opening zip file: {zip_path}...")
    extracted = 0
    skipped = 0
    
    def extract_image(member_info):
        """Extract a single image if it matches our labels."""
        member, zip_file = member_info
        # AVA images are typically named: <image_id>.jpg
        if member.filename.endswith(('.jpg', '.jpeg')):
            # Extract image ID from filename
            image_id = os.path.splitext(os.path.basename(member.filename))[0]
            
            if image_id in image_ids:
                output_path = os.path.join(output_dir, f"{image_id}.jpg")
                if not os.path.exists(output_path):
                    try:
                        with zip_file.open(member) as source:
                            with open(output_path, 'wb') as target:
                                target.write(source.read())
                        return True
                    except Exception as e:
                        log.debug(f"Failed to extract {image_id}: {e}")
                        return False
                return False  # Already exists
        return False
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        members = zip_file.namelist()
        log.info(f"Zip file contains {len(members)} files")
        
        # Filter image members
        image_members = [
            (zip_file.getinfo(m), zip_file) 
            for m in members 
            if m.endswith(('.jpg', '.jpeg')) and os.path.splitext(os.path.basename(m))[0] in image_ids
        ]
        
        log.info(f"Found {len(image_members)} matching images in zip")
        
        # Extract in parallel
        log.info("Extracting images (10 parallel workers)...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(extract_image, image_members))
            extracted = sum(results)
            skipped = len(results) - extracted
        
        log.info(f"Extracted {extracted} images (skipped {skipped} existing/failed)")
    
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract AVA images from zip file")
    parser.add_argument("zip_path", help="Path to AVA_dataset.zip file")
    parser.add_argument("--output-dir", default=".cache/ava_dataset/images", help="Output directory")
    parser.add_argument("--ava-labels", default=".cache/ava_dataset/ava_keep_trash_labels.json", help="AVA labels file")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to extract")
    args = parser.parse_args()
    
    if not os.path.exists(args.zip_path):
        log.error(f"Zip file not found: {args.zip_path}")
        log.info("Searching for zip file...")
        # Try common locations
        possible = [
            os.path.expanduser("~/Downloads/AVA_dataset.zip"),
            os.path.expanduser("~/Downloads/ava_dataset.zip"),
            ".cache/ava_dataset/AVA_dataset.zip",
        ]
        for p in possible:
            if os.path.exists(p):
                log.info(f"Found: {p}")
                args.zip_path = p
                break
        else:
            log.error("Zip file not found. Please provide path to AVA_dataset.zip")
            return 1
    
    log.info("="*80)
    log.info("EXTRACTING AVA IMAGES FROM ZIP")
    log.info("="*80)
    log.info(f"Zip file: {args.zip_path}")
    log.info(f"Output: {args.output_dir}")
    log.info("")
    
    extracted = extract_images_from_zip(args.zip_path, args.output_dir, args.ava_labels, args.max_images)
    
    if extracted > 0:
        log.info("")
        log.info("="*80)
        log.info("EXTRACTION COMPLETE")
        log.info("="*80)
        log.info(f"Extracted {extracted} images to {args.output_dir}")
        log.info("")
        log.info("Next steps:")
        log.info("1. Extract features:")
        log.info(f"   poetry run python scripts/extract_ava_features.py --ava-images-dir {args.output_dir} --ava-labels {args.ava_labels}")
        log.info("")
        log.info("2. Train on combined dataset:")
        log.info("   poetry run python scripts/train_with_ava.py --max-ava 10000")
        return 0
    else:
        log.warning("No images extracted. Check zip file and labels.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



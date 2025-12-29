#!/usr/bin/env python3
"""Setup AVA dataset for training.

This script helps download and prepare AVA dataset metadata.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("setup_ava")


def clone_ava_downloader(output_dir: str):
    """Clone ava_downloader repository."""
    ava_dir = os.path.join(output_dir, "ava_downloader")
    
    # Check in AVA_dataset subdirectory
    ava_txt_path = os.path.join(ava_dir, "AVA_dataset", "AVA.txt")
    if os.path.exists(ava_txt_path):
        log.info(f"AVA.txt already exists at {ava_txt_path}")
        return ava_dir
    
    if os.path.exists(ava_dir):
        log.info(f"ava_downloader directory exists: {ava_dir}")
        return ava_dir
    
    log.info("Cloning ava_downloader repository...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/mtobeiyf/ava_downloader.git", ava_dir],
            check=True,
            capture_output=True,
        )
        log.info(f"Cloned to {ava_dir}")
        return ava_dir
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to clone: {e}")
        return None
    except FileNotFoundError:
        log.error("git not found. Please install git or manually clone the repository")
        return None


def copy_ava_metadata(ava_dir: str, target_dir: str):
    """Copy AVA.txt to target directory."""
    # Check in AVA_dataset subdirectory
    source = os.path.join(ava_dir, "AVA_dataset", "AVA.txt")
    if not os.path.exists(source):
        # Fallback to root
        source = os.path.join(ava_dir, "AVA.txt")
    target = os.path.join(target_dir, "AVA_metadata.txt")
    
    if not os.path.exists(source):
        log.error(f"AVA.txt not found at {source}")
        return False
    
    import shutil
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, target)
    log.info(f"Copied AVA.txt to {target}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup AVA dataset")
    parser.add_argument("--output-dir", default=".cache/ava_dataset", help="Output directory")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning if directory exists")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("AVA DATASET SETUP")
    log.info("="*80)
    
    # Clone repository
    if not args.skip_clone:
        ava_dir = clone_ava_downloader(args.output_dir)
        if not ava_dir:
            log.error("Failed to clone ava_downloader")
            log.info("")
            log.info("Manual setup:")
            log.info("1. git clone https://github.com/mtobeiyf/ava_downloader.git")
            log.info(f"2. cp ava_downloader/AVA.txt {args.output_dir}/AVA_metadata.txt")
            return 1
    else:
        ava_dir = os.path.join(args.output_dir, "ava_downloader")
        if not os.path.exists(ava_dir):
            log.error(f"ava_downloader directory not found: {ava_dir}")
            return 1
    
    # Copy metadata
    if copy_ava_metadata(ava_dir, args.output_dir):
        log.info("")
        log.info("✓ AVA metadata ready!")
        log.info("")
        log.info("Next steps:")
        log.info("1. Prepare labels:")
        log.info("   poetry run python scripts/download_ava_dataset.py --max-images 10000")
        log.info("")
        log.info("2. Download images (requires ava_downloader tool)")
        log.info("")
        log.info("3. Train on combined dataset")
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


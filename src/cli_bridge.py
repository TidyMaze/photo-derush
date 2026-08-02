#!/usr/bin/env python3
"""
CLI Bridge for Darktable Lua Plugin.
Provides JSON-RPC style interface to expose photo-derush ML, grouping, and feature extraction.
"""

import argparse
import json
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import extract_features
from src.grouping_service import compute_grouping_for_photos
from src.model import ImageModel as PhotoModel


def cmd_scan(args):
    """Scan folder and extract basic photo stats."""
    directory = args.directory
    model = PhotoModel(directory=directory, max_images=None)
    image_names = model.get_image_files()
    output = {
        "status": "success",
        "directory": directory,
        "count": len(image_names),
        "images": image_names[:100]
    }
    print(json.dumps(output))


def cmd_group(args):
    """Run pHash burst grouping on directory."""
    directory = args.directory
    model = PhotoModel(directory=directory, max_images=None)
    image_names = model.get_image_files()

    # Preload EXIF for grouping calculation
    exif_data = {fn: model.get_exif(model.get_image_path(fn)) for fn in image_names if model.get_image_path(fn)}
    group_info = compute_grouping_for_photos(
        filenames=image_names,
        image_dir=directory,
        exif_data=exif_data
    )

    output = {
        "status": "success",
        "total_images": len(image_names),
        "groups": group_info
    }
    print(json.dumps(output))


def main():
    parser = argparse.ArgumentParser(description="Derush Darktable CLI Bridge")
    subparsers = parser.add_subparsers(dest="command")

    # Scan command
    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--directory", required=True, help="Path to photo directory")

    # Group command
    group_parser = subparsers.add_parser("group")
    group_parser.add_argument("--directory", required=True, help="Path to photo directory")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "group":
        cmd_group(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

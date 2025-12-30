#!/usr/bin/env python3
"""Check which files have valid EXIF DateTimeOriginal."""

import sys
import os
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_last_dir():
    """Load last directory from config."""
    config_file = Path.home() / ".photo-derush-cache" / "last_dir.txt"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                directory = f.read().strip()
                if os.path.isdir(directory):
                    return directory
        except Exception:
            pass
    return os.path.expanduser('~/Pictures/photo-dataset')


def main():
    image_dir = load_last_dir()
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    print(f"Checking EXIF data in: {image_dir}\n")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    
    if not filenames:
        print("No images found!")
        sys.exit(1)
    
    print(f"Found {len(filenames)} images\n")
    
    # Check EXIF data
    has_exif = []
    no_exif = []
    invalid_exif = []
    
    for fname in filenames:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
            exif = img.getexif()
            
            if not exif:
                no_exif.append(fname)
                continue
            
            # Look for DateTimeOriginal (tag 306) or DateTime (tag 306)
            dt_original = None
            dt_original_tag = None
            
            # Try to find DateTimeOriginal tag
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    dt_original = value
                    dt_original_tag = tag_id
                    break
            
            # If not found, try DateTime
            if not dt_original:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name == "DateTime":
                        dt_original = value
                        dt_original_tag = tag_id
                        break
            
            if dt_original:
                try:
                    # Try to parse it
                    if isinstance(dt_original, str):
                        from datetime import datetime
                        datetime.strptime(dt_original, "%Y:%m:%d %H:%M:%S")
                        has_exif.append((fname, dt_original))
                    else:
                        invalid_exif.append((fname, f"Not a string: {type(dt_original)}"))
                except ValueError as e:
                    invalid_exif.append((fname, f"Invalid format: {dt_original}"))
            else:
                no_exif.append(fname)
                
        except Exception as e:
            invalid_exif.append((fname, f"Error: {e}"))
    
    print(f"{'='*80}")
    print(f"EXIF DateTimeOriginal Summary")
    print(f"{'='*80}\n")
    
    print(f"✅ Files with valid EXIF DateTimeOriginal: {len(has_exif)}")
    print(f"❌ Files without EXIF DateTimeOriginal: {len(no_exif)}")
    print(f"⚠️  Files with invalid EXIF DateTimeOriginal: {len(invalid_exif)}\n")
    
    if has_exif:
        print(f"{'='*80}")
        print(f"Files WITH valid EXIF DateTimeOriginal (first 20):")
        print(f"{'='*80}\n")
        for fname, dt in has_exif[:20]:
            print(f"  {os.path.basename(fname):50s} {dt}")
        if len(has_exif) > 20:
            print(f"  ... and {len(has_exif) - 20} more")
    
    if no_exif:
        print(f"\n{'='*80}")
        print(f"Files WITHOUT EXIF DateTimeOriginal (first 20):")
        print(f"{'='*80}\n")
        for fname in no_exif[:20]:
            print(f"  {os.path.basename(fname)}")
        if len(no_exif) > 20:
            print(f"  ... and {len(no_exif) - 20} more")
    
    if invalid_exif:
        print(f"\n{'='*80}")
        print(f"Files with INVALID EXIF DateTimeOriginal (first 10):")
        print(f"{'='*80}\n")
        for fname, reason in invalid_exif[:10]:
            print(f"  {os.path.basename(fname):50s} {reason}")
        if len(invalid_exif) > 10:
            print(f"  ... and {len(invalid_exif) - 10} more")
    
    # Show percentage
    total = len(filenames)
    exif_percent = (len(has_exif) / total * 100) if total > 0 else 0
    print(f"\n{'='*80}")
    print(f"Summary: {exif_percent:.1f}% of images have valid EXIF DateTimeOriginal")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


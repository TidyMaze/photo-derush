#!/usr/bin/env python3
"""Visualize images grouped by burst, session, and date."""

import sys
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos, extract_timestamp
from src.photo_grouping import PhotoMetadata, detect_sessions, detect_bursts


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


def visualize_by_burst(group_info: dict, image_dir: str, output_path: str = "studies/outputs/bursts.png"):
    """Visualize images grouped by burst."""
    # Group images by burst_id
    bursts: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        burst_id = info.get('burst_id')
        if burst_id is not None:
            bursts[burst_id].append((filename, info))
    
    # Sort bursts by ID
    sorted_bursts = sorted(bursts.items(), key=lambda x: x[0])
    
    # Take top 10 bursts by size
    sorted_bursts = sorted(bursts.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    n_bursts = len(sorted_bursts)
    if n_bursts == 0:
        print("No bursts found")
        return
    
    fig = plt.figure(figsize=(20, 4 * n_bursts))
    gs = GridSpec(n_bursts, 1, figure=fig, hspace=0.3)
    
    for burst_idx, (burst_id, images) in enumerate(sorted_bursts):
        ax = fig.add_subplot(gs[burst_idx, 0])
        ax.set_title(f"Burst {burst_id} ({len(images)} images)", fontsize=14, fontweight="bold")
        ax.axis("off")
        
        # Display thumbnails
        cols = min(10, len(images))
        rows = (len(images) + cols - 1) // cols
        
        max_thumb_w = 120
        max_thumb_h = 120
        spacing = 5
        
        loaded_images = []
        thumbnail_sizes = []
        
        for filename, info in images:
            image_path = os.path.join(image_dir, filename)
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    orig_w, orig_h = img.size
                    aspect = orig_w / orig_h
                    
                    if aspect > 1:
                        thumb_w = max_thumb_w
                        thumb_h = max_thumb_w / aspect
                    else:
                        thumb_h = max_thumb_h
                        thumb_w = max_thumb_h * aspect
                    
                    img_resized = img.resize((int(thumb_w), int(thumb_h)), Image.Resampling.LANCZOS)
                    loaded_images.append((img_resized, int(thumb_w), int(thumb_h)))
                    thumbnail_sizes.append((int(thumb_w), int(thumb_h)))
                else:
                    loaded_images.append((None, max_thumb_w, max_thumb_h))
                    thumbnail_sizes.append((max_thumb_w, max_thumb_h))
            except Exception:
                loaded_images.append((None, max_thumb_w, max_thumb_h))
                thumbnail_sizes.append((max_thumb_w, max_thumb_h))
        
        col_widths = [0] * cols
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            col = idx % cols
            col_widths[col] = max(col_widths[col], thumb_w)
        
        row_heights = [0] * rows
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            row = idx // cols
            row_heights[row] = max(row_heights[row], thumb_h)
        
        for img_idx, ((filename, info), (img, thumb_w, thumb_h)) in enumerate(zip(images, loaded_images)):
            row = img_idx // cols
            col = img_idx % cols
            
            x = sum(col_widths[:col]) + col * spacing
            y = -(sum(row_heights[:row]) + row * spacing)
            
            try:
                if img is not None:
                    ax.imshow(img, extent=(float(x), float(x + thumb_w), float(y - thumb_h), float(y)), aspect="equal")
                    ax.text(x + thumb_w / 2, y - thumb_h - 15, os.path.basename(filename)[:20], ha="center", va="top", fontsize=6)
            except Exception:
                pass
        
        total_width = sum(col_widths) + (cols - 1) * spacing if cols > 0 else 0
        total_height = sum(row_heights) + (rows - 1) * spacing if rows > 0 else 0
        ax.set_xlim(-spacing, total_width + spacing)
        ax.set_ylim(-total_height - 20, spacing)
    
    plt.suptitle("Top 10 Bursts by Image Count", fontsize=16, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved burst visualization to {output_path}")
    plt.close()


def visualize_by_session(group_info: dict, image_dir: str, output_path: str = "studies/outputs/sessions.png"):
    """Visualize images grouped by session."""
    sessions: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        session_id = info.get('session_id')
        if session_id is not None:
            sessions[session_id].append((filename, info))
    
    sorted_sessions = sorted(sessions.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    n_sessions = len(sorted_sessions)
    if n_sessions == 0:
        print("No sessions found")
        return
    
    fig = plt.figure(figsize=(20, 4 * n_sessions))
    gs = GridSpec(n_sessions, 1, figure=fig, hspace=0.3)
    
    for session_idx, (session_id, images) in enumerate(sorted_sessions):
        ax = fig.add_subplot(gs[session_idx, 0])
        ax.set_title(f"Session {session_id} ({len(images)} images)", fontsize=14, fontweight="bold")
        ax.axis("off")
        
        cols = min(10, len(images))
        rows = (len(images) + cols - 1) // cols
        
        max_thumb_w = 120
        max_thumb_h = 120
        spacing = 5
        
        loaded_images = []
        thumbnail_sizes = []
        
        for filename, info in images:
            image_path = os.path.join(image_dir, filename)
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    orig_w, orig_h = img.size
                    aspect = orig_w / orig_h
                    
                    if aspect > 1:
                        thumb_w = max_thumb_w
                        thumb_h = max_thumb_w / aspect
                    else:
                        thumb_h = max_thumb_h
                        thumb_w = max_thumb_h * aspect
                    
                    img_resized = img.resize((int(thumb_w), int(thumb_h)), Image.Resampling.LANCZOS)
                    loaded_images.append((img_resized, int(thumb_w), int(thumb_h)))
                    thumbnail_sizes.append((int(thumb_w), int(thumb_h)))
                else:
                    loaded_images.append((None, max_thumb_w, max_thumb_h))
                    thumbnail_sizes.append((max_thumb_w, max_thumb_h))
            except Exception:
                loaded_images.append((None, max_thumb_w, max_thumb_h))
                thumbnail_sizes.append((max_thumb_w, max_thumb_h))
        
        col_widths = [0] * cols
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            col = idx % cols
            col_widths[col] = max(col_widths[col], thumb_w)
        
        row_heights = [0] * rows
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            row = idx // cols
            row_heights[row] = max(row_heights[row], thumb_h)
        
        for img_idx, ((filename, info), (img, thumb_w, thumb_h)) in enumerate(zip(images, loaded_images)):
            row = img_idx // cols
            col = img_idx % cols
            
            x = sum(col_widths[:col]) + col * spacing
            y = -(sum(row_heights[:row]) + row * spacing)
            
            try:
                if img is not None:
                    ax.imshow(img, extent=(float(x), float(x + thumb_w), float(y - thumb_h), float(y)), aspect="equal")
                    ax.text(x + thumb_w / 2, y - thumb_h - 15, os.path.basename(filename)[:20], ha="center", va="top", fontsize=6)
            except Exception:
                pass
        
        total_width = sum(col_widths) + (cols - 1) * spacing if cols > 0 else 0
        total_height = sum(row_heights) + (rows - 1) * spacing if rows > 0 else 0
        ax.set_xlim(-spacing, total_width + spacing)
        ax.set_ylim(-total_height - 20, spacing)
    
    plt.suptitle("Top 10 Sessions by Image Count", fontsize=16, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved session visualization to {output_path}")
    plt.close()


def visualize_by_date(group_info: dict, image_dir: str, exif_data: dict, output_path: str = "studies/outputs/dates.png"):
    """Visualize images grouped by date (YYYY-MM-DD)."""
    dates: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    
    for filename, info in group_info.items():
        path = os.path.join(image_dir, filename)
        timestamp = extract_timestamp(exif_data.get(filename, {}), path)
        date_str = timestamp.strftime('%Y-%m-%d')
        dates[date_str].append((filename, info))
    
    sorted_dates = sorted(dates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    n_dates = len(sorted_dates)
    if n_dates == 0:
        print("No dates found")
        return
    
    fig = plt.figure(figsize=(20, 4 * n_dates))
    gs = GridSpec(n_dates, 1, figure=fig, hspace=0.3)
    
    for date_idx, (date_str, images) in enumerate(sorted_dates):
        ax = fig.add_subplot(gs[date_idx, 0])
        ax.set_title(f"Date {date_str} ({len(images)} images)", fontsize=14, fontweight="bold")
        ax.axis("off")
        
        cols = min(10, len(images))
        rows = (len(images) + cols - 1) // cols
        
        max_thumb_w = 120
        max_thumb_h = 120
        spacing = 5
        
        loaded_images = []
        thumbnail_sizes = []
        
        for filename, info in images:
            image_path = os.path.join(image_dir, filename)
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    orig_w, orig_h = img.size
                    aspect = orig_w / orig_h
                    
                    if aspect > 1:
                        thumb_w = max_thumb_w
                        thumb_h = max_thumb_w / aspect
                    else:
                        thumb_h = max_thumb_h
                        thumb_w = max_thumb_h * aspect
                    
                    img_resized = img.resize((int(thumb_w), int(thumb_h)), Image.Resampling.LANCZOS)
                    loaded_images.append((img_resized, int(thumb_w), int(thumb_h)))
                    thumbnail_sizes.append((int(thumb_w), int(thumb_h)))
                else:
                    loaded_images.append((None, max_thumb_w, max_thumb_h))
                    thumbnail_sizes.append((max_thumb_w, max_thumb_h))
            except Exception:
                loaded_images.append((None, max_thumb_w, max_thumb_h))
                thumbnail_sizes.append((max_thumb_w, max_thumb_h))
        
        col_widths = [0] * cols
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            col = idx % cols
            col_widths[col] = max(col_widths[col], thumb_w)
        
        row_heights = [0] * rows
        for idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            row = idx // cols
            row_heights[row] = max(row_heights[row], thumb_h)
        
        for img_idx, ((filename, info), (img, thumb_w, thumb_h)) in enumerate(zip(images, loaded_images)):
            row = img_idx // cols
            col = img_idx % cols
            
            x = sum(col_widths[:col]) + col * spacing
            y = -(sum(row_heights[:row]) + row * spacing)
            
            try:
                if img is not None:
                    ax.imshow(img, extent=(float(x), float(x + thumb_w), float(y - thumb_h), float(y)), aspect="equal")
                    ax.text(x + thumb_w / 2, y - thumb_h - 15, os.path.basename(filename)[:20], ha="center", va="top", fontsize=6)
            except Exception:
                pass
        
        total_width = sum(col_widths) + (cols - 1) * spacing if cols > 0 else 0
        total_height = sum(row_heights) + (rows - 1) * spacing if rows > 0 else 0
        ax.set_xlim(-spacing, total_width + spacing)
        ax.set_ylim(-total_height - 20, spacing)
    
    plt.suptitle("Top 10 Dates by Image Count", fontsize=16, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved date visualization to {output_path}")
    plt.close()


def main():
    image_dir = load_last_dir()
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    print(f"Loading images from: {image_dir}")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    
    if not filenames:
        print("No images found!")
        sys.exit(1)
    
    print(f"Found {len(filenames)} images")
    
    # Load EXIF data
    exif_data = {}
    keep_probabilities = {}
    for fname in filenames:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
            exif = img.getexif()
            exif_data[fname] = {}
            if exif:
                for tag_id, value in exif.items():
                    tag = img.getexif().get(tag_id)
                    if tag:
                        exif_data[fname][str(tag)] = str(value)
            keep_probabilities[fname] = 0.5
        except Exception:
            exif_data[fname] = {}
            keep_probabilities[fname] = 0.5
    
    print("Computing grouping...")
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=keep_probabilities,
        quality_metrics=None,
        session_gap_min=10,
        burst_gap_sec=15.0,
        phash_threshold=8,
        progress_reporter=None,
    )
    
    print("\nVisualizing by burst...")
    visualize_by_burst(group_info, image_dir, "bursts.png")
    
    print("\nVisualizing by session...")
    visualize_by_session(group_info, image_dir, "sessions.png")
    
    print("\nVisualizing by date...")
    visualize_by_date(group_info, image_dir, exif_data, "dates.png")
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()


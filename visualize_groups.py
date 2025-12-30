#!/usr/bin/env python3
"""Visualize top 10 groups with all images linked to each group."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.grouping_service import compute_grouping_for_photos


def load_group_info_from_app_state():
    """Try to load group_info from app state if available."""
    # Check if there's a saved state file
    state_file = Path.home() / ".photo-derush-cache" / "group_info.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def compute_group_info(image_dir: str):
    """Compute group info for images in directory."""
    print(f"Computing grouping for images in {image_dir}...")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    
    if not filenames:
        print(f"No images found in {image_dir}")
        return None
    
    print(f"Found {len(filenames)} images")
    
    # Load EXIF data (simplified - just use filenames for now)
    exif_data = {}
    for filename in filenames:
        exif_data[filename] = {"DateTimeOriginal": "2025:01:01 12:00:00"}  # Placeholder
    
    # Compute grouping
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=None,
        quality_metrics=None,
        progress_reporter=None,
    )
    
    return group_info, image_dir


def get_top_groups(group_info: dict, top_n: int = 10):
    """Get top N groups by size."""
    # Group images by group_id
    groups: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None:
            groups[group_id].append((filename, info))
    
    # Sort by group size (descending)
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    return sorted_groups[:top_n]


def visualize_groups(top_groups: list, image_dir: str, output_path: str = "group_visualization.png"):
    """Create visualization of top groups with their images."""
    n_groups = len(top_groups)
    
    # Create figure with subplots for each group
    fig = plt.figure(figsize=(20, 4 * n_groups))
    gs = GridSpec(n_groups, 1, figure=fig, hspace=0.3)
    
    for group_idx, (group_id, images) in enumerate(top_groups):
        group_size = len(images)
        best_pick = next((f for f, info in images if info.get("is_group_best", False)), None)
        
        # Create subplot for this group
        ax = fig.add_subplot(gs[group_idx, 0])
        ax.set_title(
            f"Group {group_id} ({group_size} images)" + (f" - Best: {os.path.basename(best_pick) if best_pick else 'N/A'}" if best_pick else ""),
            fontsize=14,
            fontweight="bold"
        )
        ax.axis("off")
        
        # Display thumbnails in a grid
        cols = min(10, group_size)  # Max 10 columns
        rows = (group_size + cols - 1) // cols
        
        max_thumbnail_width = 120
        max_thumbnail_height = 120
        spacing = 5
        
        # First pass: load all images and calculate their thumbnail sizes preserving aspect ratio
        loaded_images = []
        thumbnail_sizes = []
        
        for filename, info in images:
            image_path = os.path.join(image_dir, filename)
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    orig_width, orig_height = img.size
                    aspect_ratio = orig_width / orig_height
                    
                    # Calculate thumbnail size preserving aspect ratio
                    if aspect_ratio > 1:  # Landscape
                        thumb_width = max_thumbnail_width
                        thumb_height = max_thumbnail_width / aspect_ratio
                    else:  # Portrait or square
                        thumb_height = max_thumbnail_height
                        thumb_width = max_thumbnail_height * aspect_ratio
                    
                    # Resize image to exact thumbnail size
                    img_resized = img.resize((int(thumb_width), int(thumb_height)), Image.Resampling.LANCZOS)
                    loaded_images.append((img_resized, int(thumb_width), int(thumb_height)))
                    thumbnail_sizes.append((int(thumb_width), int(thumb_height)))
                else:
                    loaded_images.append((None, max_thumbnail_width, max_thumbnail_height))
                    thumbnail_sizes.append((max_thumbnail_width, max_thumbnail_height))
            except Exception as e:
                loaded_images.append((None, max_thumbnail_width, max_thumbnail_height))
                thumbnail_sizes.append((max_thumbnail_width, max_thumbnail_height))
        
        # Calculate column widths (max width in each column)
        col_widths = [0] * cols
        for img_idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            col = img_idx % cols
            col_widths[col] = max(col_widths[col], thumb_w)
        
        # Calculate row heights (max height in each row)
        row_heights = [0] * rows
        for img_idx, (thumb_w, thumb_h) in enumerate(thumbnail_sizes):
            row = img_idx // cols
            row_heights[row] = max(row_heights[row], thumb_h)
        
        # Second pass: display images with calculated positions
        for img_idx, ((filename, info), (img, thumb_w, thumb_h)) in enumerate(zip(images, loaded_images)):
            row = img_idx // cols
            col = img_idx % cols
            
            # Calculate x position (sum of previous column widths + spacing)
            x = sum(col_widths[:col]) + col * spacing
            # Calculate y position (negative for top-to-bottom, sum of previous row heights)
            y = -(sum(row_heights[:row]) + row * spacing)
            
            try:
                if img is not None:
                    # Display image with exact dimensions (preserves aspect ratio)
                    ax.imshow(img, extent=(float(x), float(x + thumb_w), float(y - thumb_h), float(y)), aspect="equal", interpolation="lanczos")
                    
                    # Highlight best pick
                    if info.get("is_group_best", False):
                        rect = mpatches.Rectangle(
                            (x - 2, y - thumb_h - 2),
                            thumb_w + 4,
                            thumb_h + 4,
                            linewidth=3,
                            edgecolor="gold",
                            facecolor="none"
                        )
                        ax.add_patch(rect)
                        # Add "BEST" label
                        ax.text(x + thumb_w - 5, y - 5, "BEST", fontsize=8, color="gold", weight="bold",
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
                    
                    # Add filename label
                    ax.text(
                        x + thumb_w / 2,
                        y - thumb_h - 15,
                        os.path.basename(filename)[:20] + ("..." if len(os.path.basename(filename)) > 20 else ""),
                        ha="center",
                        va="top",
                        fontsize=8,
                        rotation=0
                    )
                else:
                    # Placeholder for missing image
                    ax.text(x + thumb_w / 2, y - thumb_h / 2, "?", ha="center", va="center", fontsize=20)
            except Exception as e:
                # Error loading image
                ax.text(x + thumb_w / 2, y - thumb_h / 2, "X", ha="center", va="center", fontsize=20, color="red")
        
        # Set limits based on calculated dimensions
        total_width = sum(col_widths) + (cols - 1) * spacing if cols > 0 else 0
        total_height = sum(row_heights) + (rows - 1) * spacing if rows > 0 else 0
        ax.set_xlim(-spacing, total_width + spacing)
        ax.set_ylim(-total_height - 20, spacing)
    
    plt.suptitle("Top 10 Groups by Image Count", fontsize=16, fontweight="bold", y=0.998)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main function."""
    # Try to get image directory from command line or use default
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} is not a directory")
        sys.exit(1)
    
    # Try to load from app state first
    group_info = load_group_info_from_app_state()
    
    if group_info is None:
        # Compute grouping
        result = compute_group_info(image_dir)
        if result is None:
            sys.exit(1)
        group_info, image_dir = result
    
    # Get top 10 groups
    top_groups = get_top_groups(group_info, top_n=10)
    
    if not top_groups:
        print("No groups found")
        sys.exit(1)
    
    print(f"\nTop {len(top_groups)} groups:")
    for group_id, images in top_groups:
        best_pick = next((f for f, info in images if info.get("is_group_best", False)), None)
        print(f"  Group {group_id}: {len(images)} images, best: {os.path.basename(best_pick) if best_pick else 'N/A'}")
    
    # Create visualization
    visualize_groups(top_groups, image_dir, "top_10_groups.png")
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()


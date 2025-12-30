#!/usr/bin/env python3
"""Visualize all images grouped by burst and session with colored rectangles."""

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos


def load_last_dir():
    """Load last directory from config."""
    config_file = Path.home() / ".photo-derush-cache" / "last_dir.txt"
    if config_file.exists():
        try:
            with open(config_file) as f:
                directory = f.read().strip()
                if os.path.isdir(directory):
                    return directory
        except Exception:
            pass
    return os.path.expanduser('~/Pictures/photo-dataset')


def visualize_hierarchical(group_info: dict, image_dir: str, exif_data: dict, output_path: str = "bursts_sessions_hierarchical.png"):
    """Visualize images grouped by session -> burst with colored rectangles."""

    # Organize by session -> burst -> images
    sessions: dict[int, dict[int, list[tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))

    for filename, info in group_info.items():
        session_id = info.get('session_id')
        burst_id = info.get('burst_id')
        if session_id is not None and burst_id is not None:
            sessions[session_id][burst_id].append((filename, info))

    # Sort sessions by ID
    sorted_sessions = sorted(sessions.items(), key=lambda x: x[0])

    # Calculate layout: arrange sessions horizontally, bursts vertically within sessions
    # Each burst is a row, sessions are columns

    # First, find max bursts per session and max images per burst
    max_bursts_per_session = max(len(bursts) for _, bursts in sorted_sessions) if sorted_sessions else 0
    max_images_per_burst = 0
    for _, bursts in sorted_sessions:
        for _, images in bursts.items():
            max_images_per_burst = max(max_images_per_burst, len(images))

    # Limit for visualization
    max_sessions_to_show = 10
    max_bursts_per_session = min(max_bursts_per_session, 20)
    max_images_per_burst = min(max_images_per_burst, 15)

    sorted_sessions = sorted_sessions[:max_sessions_to_show]

    # Create figure
    plt.figure(figsize=(max_sessions_to_show * 3, max_bursts_per_session * 2))

    # Color maps
    session_colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_sessions)))
    burst_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Track positions
    x_offset = 0
    session_widths = []

    for session_idx, (session_id, bursts) in enumerate(sorted_sessions):
        session_color = session_colors[session_idx]

        # Sort bursts by ID
        sorted_bursts = sorted(bursts.items(), key=lambda x: x[0])

        # Calculate width needed for this session (max images in any burst)
        session_max_images = max(len(images) for _, images in sorted_bursts) if sorted_bursts else 0
        session_width = min(session_max_images, max_images_per_burst)
        session_widths.append(session_width)

        # Draw session rectangle
        session_rect = mpatches.Rectangle(
            (x_offset - 0.1, -0.1),
            session_width + 0.2,
            len(sorted_bursts) + 0.2,
            linewidth=3,
            edgecolor=session_color,
            facecolor='none',
            linestyle='--',
            alpha=0.7
        )
        plt.gca().add_patch(session_rect)

        # Add session label
        plt.text(x_offset + session_width / 2, len(sorted_bursts) + 0.3,
                f'Session {session_id}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=session_color)

        # Draw bursts
        for burst_idx, (burst_id, images) in enumerate(sorted_bursts):
            if burst_idx >= max_bursts_per_session:
                break

            burst_color = burst_colors[burst_id % len(burst_colors)]
            y_pos = len(sorted_bursts) - burst_idx - 1

            # Draw burst rectangle
            burst_rect = mpatches.Rectangle(
                (x_offset - 0.05, y_pos - 0.05),
                session_width + 0.1,
                0.9,
                linewidth=2,
                edgecolor=burst_color,
                facecolor='none',
                alpha=0.8
            )
            plt.gca().add_patch(burst_rect)

            # Add burst label
            plt.text(x_offset - 0.15, y_pos + 0.4, f'B{burst_id}',
                    ha='right', va='center', fontsize=8,
                    color=burst_color, fontweight='bold')

            # Draw images in burst
            for img_idx, (filename, info) in enumerate(images[:max_images_per_burst]):
                if img_idx >= session_width:
                    break

                image_path = os.path.join(image_dir, filename)
                try:
                    img = Image.open(image_path)
                    orig_w, orig_h = img.size
                    aspect = orig_w / orig_h

                    # Thumbnail size
                    thumb_h = 0.8
                    thumb_w = thumb_h * aspect

                    # Position
                    x_pos = x_offset + img_idx + (1 - thumb_w) / 2
                    y_pos_img = y_pos + (1 - thumb_h) / 2

                    # Resize image
                    thumb_pixels_h = int(thumb_h * 100)  # Scale for display
                    thumb_pixels_w = int(thumb_w * 100)
                    img_resized = img.resize((thumb_pixels_w, thumb_pixels_h), Image.Resampling.LANCZOS)

                    # Display image
                    plt.imshow(img_resized, extent=(x_pos, x_pos + thumb_w, y_pos_img, y_pos_img + thumb_h),
                             aspect='auto', interpolation='lanczos')

                except Exception:
                    # Placeholder for failed images
                    x_pos = x_offset + img_idx + 0.1
                    y_pos_img = y_pos + 0.1
                    plt.text(x_pos + 0.2, y_pos_img + 0.3, '?', ha='center', va='center',
                            fontsize=20, color='red')

        x_offset += session_width + 0.5  # Add spacing between sessions

    # Set axis limits
    plt.xlim(-0.5, x_offset - 0.3)
    plt.ylim(-0.5, max_bursts_per_session + 0.5)
    plt.axis('off')
    plt.title('Images Grouped by Session → Burst', fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='gray', linestyle='--', linewidth=2, label='Session'),
        mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=1, label='Burst'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved hierarchical visualization to {output_path}")
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

    print("\nVisualizing hierarchical grouping...")
    visualize_hierarchical(group_info, image_dir, exif_data, "bursts_sessions_hierarchical.png")

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()


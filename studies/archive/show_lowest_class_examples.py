#!/usr/bin/env python3
"""Show multiple examples from the lowest AVA class."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_ava_multiclass_labels(cache_dir: Path):
    """Load AVA labels with full score distribution (1-10)."""
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
        print(f"AVA metadata not found at {ava_metadata_path}")
        return None
    
    image_scores = {}
    
    with open(ava_metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            image_id = parts[1]
            score_counts = [int(s) for s in parts[2:12]]
            total_votes = sum(score_counts)
            if total_votes == 0:
                continue
            
            weighted_sum = sum((i+1) * count for i, count in enumerate(score_counts))
            mean_score = weighted_sum / total_votes
            rounded_score = int(round(mean_score))
            rounded_score = max(1, min(10, rounded_score))
            
            image_scores[image_id] = {
                'class': rounded_score - 1,  # 0-9 for sklearn
                'mean_score': mean_score,
            }
    
    return image_scores


def find_image_path(image_id: str, images_dir: Path) -> Path | None:
    """Find image file by ID."""
    possible_names = [
        f"{image_id}.jpg",
        f"{image_id}.JPG",
        f"{image_id}.jpeg",
        f"{image_id}.JPEG",
    ]
    
    for name in possible_names:
        path = images_dir / name
        if path.exists():
            return path
        
        for subdir in images_dir.iterdir():
            if subdir.is_dir():
                path = subdir / name
                if path.exists():
                    return path
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir():
                        path = subsubdir / name
                        if path.exists():
                            return path
    
    return None


def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Load AVA features
    ava_features_path = cache_dir / "ava_features.joblib"
    if not ava_features_path.exists():
        print(f"AVA features not found at {ava_features_path}")
        return 1
    
    print("Loading AVA features...")
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    ava_paths = ava_data.get('image_paths', [])
    
    print(f"AVA features: {len(X_ava)} samples")
    
    # Load multi-class labels
    image_scores = load_ava_multiclass_labels(cache_dir)
    if image_scores is None:
        return 1
    
    # Match features to labels
    y_ava = []
    valid_indices = []
    image_id_map = {}
    
    for i, img_id in enumerate(ava_ids):
        img_id_str = str(img_id)
        if img_id_str in image_scores:
            y_ava.append(image_scores[img_id_str]['class'])
            valid_indices.append(i)
            image_id_map[i] = img_id_str
        elif f"{img_id}.jpg" in image_scores:
            y_ava.append(image_scores[f"{img_id}.jpg"]['class'])
            valid_indices.append(i)
            image_id_map[i] = f"{img_id}.jpg"
    
    if len(y_ava) == 0:
        ava_labeled_path = cache_dir / "ava_dataset" / "ava_keep_trash_labels.json"
        if ava_labeled_path.exists():
            import json
            with open(ava_labeled_path) as f:
                labeled = json.load(f)
            y_ava = []
            for i in range(len(X_ava)):
                if i < len(labeled):
                    mean_score = labeled[i].get('score', 5.0)
                    rounded = int(round(mean_score))
                    rounded = max(1, min(10, rounded))
                    y_ava.append(rounded - 1)
                    valid_indices.append(i)
                    image_id_map[i] = labeled[i].get('image_id', str(i))
            print("Using fallback labels from keep/trash file")
    
    y_ava = np.array(y_ava)
    
    # Find lowest class
    present_classes = sorted([cls for cls in range(10) if np.sum(y_ava == cls) > 0])
    lowest_class = present_classes[0]
    lowest_class_indices = np.where(y_ava == lowest_class)[0]
    
    print(f"\nLowest class: {lowest_class} (Score {lowest_class+1})")
    print(f"Number of examples: {len(lowest_class_indices)}")
    
    # Find images directory
    images_dir = cache_dir / "ava_dataset" / "images"
    if not images_dir.exists():
        print(f"Images directory not found at {images_dir}")
        return 1
    
    # Show all examples (or up to 12)
    n_examples = min(len(lowest_class_indices), 12)
    n_cols = 4
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx in range(n_examples):
        class_idx = lowest_class_indices[idx]
        example_idx = valid_indices[class_idx]
        image_id = image_id_map.get(example_idx, str(ava_ids[example_idx]))
        
        # Try to find image
        image_path = None
        if example_idx < len(ava_paths) and ava_paths[example_idx]:
            image_path = Path(ava_paths[example_idx])
            if not image_path.exists():
                image_path = None
        
        if image_path is None:
            image_path = find_image_path(image_id, images_dir)
        
        ax = axes[idx]
        
        if image_path and image_path.exists():
            try:
                img = Image.open(image_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'ID: {image_id}', fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error\n{image_id}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')
                ax.set_title(f'ID: {image_id}', fontsize=9)
        else:
            ax.text(0.5, 0.5, f'Not found\n{image_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')
            ax.set_title(f'ID: {image_id}', fontsize=9)
    
    # Hide unused axes
    for idx in range(n_examples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Lowest Class Examples: Class {lowest_class} (Score {lowest_class+1}) - {len(lowest_class_indices)} total examples', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = cache_dir / "ava_lowest_class_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


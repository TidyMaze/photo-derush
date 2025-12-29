#!/usr/bin/env python3
"""Plot class distribution for AVA dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_ava_multiclass_labels(cache_dir: Path):
    """Load AVA labels with full score distribution (1-10)."""
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
        print(f"AVA metadata not found at {ava_metadata_path}")
        return None, None
    
    print(f"Loading AVA metadata from {ava_metadata_path}...")
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
                'score_counts': score_counts,
            }
    
    print(f"Loaded scores for {len(image_scores)} images")
    return image_scores, ava_metadata_path


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
    
    print(f"AVA features: {len(X_ava)} samples")
    
    # Load multi-class labels
    image_scores, metadata_path = load_ava_multiclass_labels(cache_dir)
    if image_scores is None:
        return 1
    
    # Match features to labels
    y_ava = []
    for i, img_id in enumerate(ava_ids):
        img_id_str = str(img_id)
        if img_id_str in image_scores:
            y_ava.append(image_scores[img_id_str]['class'])
        elif f"{img_id}.jpg" in image_scores:
            y_ava.append(image_scores[f"{img_id}.jpg"]['class'])
        else:
            # Try fallback
            continue
    
    if len(y_ava) == 0:
        # Fallback: use keep/trash labels
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
            print("Using fallback labels from keep/trash file")
    
    y_ava = np.array(y_ava)
    
    print(f"\nTotal samples: {len(y_ava)}")
    print(f"\nClass distribution (0-9, representing scores 1-10):")
    
    class_counts = {}
    for cls in range(10):
        count = int(np.sum(y_ava == cls))
        class_counts[cls] = count
        if count > 0:
            print(f"  Class {cls} (Score {cls+1}): {count:5d} ({count/len(y_ava)*100:5.1f}%)")
        else:
            print(f"  Class {cls} (Score {cls+1}): {count:5d} (MISSING)")
    
    # Plot
    classes = list(range(10))
    counts = [class_counts[cls] for cls in classes]
    colors = ['red' if c == 0 else 'blue' for c in counts]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Class (0-9, representing scores 1-10)', fontsize=12)
    plt.ylabel('Number of Examples', fontsize=12)
    plt.title('AVA Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(classes, [f'{c}\n(Score {c+1})' for c in classes])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (cls, count) in enumerate(zip(classes, counts)):
        if count > 0:
            plt.text(cls, count + max(counts)*0.01, str(count), 
                    ha='center', va='bottom', fontsize=9)
    
    # Highlight missing classes
    missing_classes = [cls for cls in classes if class_counts[cls] == 0]
    if missing_classes:
        plt.text(0.02, 0.98, f'Missing classes: {[c+1 for c in missing_classes]} (scores)', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    output_path = cache_dir / "ava_class_distribution.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


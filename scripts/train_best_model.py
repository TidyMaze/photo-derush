#!/usr/bin/env python3
"""Train model with all features (highest score configuration)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training_core import train_keep_trash_model, DEFAULT_MODEL_PATH
from src.model import RatingsTagsRepository

def main():
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        print("No repository found")
        return 1
    
    print("="*80)
    print("TRAINING MODEL WITH ALL FEATURES (BEST CONFIGURATION)")
    print("="*80)
    print(f"\nTraining with all 78 features (highest score: 79.01%)")
    print(f"Model will be saved to: {DEFAULT_MODEL_PATH}\n")
    
    result = train_keep_trash_model(
        image_dir=image_dir,
        model_path=DEFAULT_MODEL_PATH,
        repo=repo,
        progress_callback=lambda current, total, detail: print(f"  {current}/{total} {detail}") if current % max(1, total // 10) == 0 or current == total else None
    )
    
    if result:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Model saved to: {DEFAULT_MODEL_PATH}")
        print(f"✓ Model uses all 78 features (no feature_indices filtering)")
        return 0
    else:
        print("Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())


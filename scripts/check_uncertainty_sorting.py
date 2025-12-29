#!/usr/bin/env python3
"""Check that images in grid are sorted by uncertainty in strictly descending order."""
import sys
import time
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viewmodel import PhotoViewModel
from src.model import ImageModel
from src.repository import RatingsTagsRepository

def calculate_uncertainty(prob):
    """Calculate uncertainty score from probability."""
    if prob is None:
        return 1.0  # No prediction = highest uncertainty
    return 0.5 - abs(prob - 0.5)

def load_last_dir():
    """Load last directory from config (same as app.py)."""
    CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
            last_dir = data.get('last_dir')
            if last_dir and os.path.isdir(last_dir):
                return last_dir
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return os.path.expanduser('~')

def main():
    # Get the directory from config (same as app.py)
    directory = load_last_dir()
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return 1
    
    print(f"Loading images from: {directory}")
    
    # Create model and viewmodel
    repo = RatingsTagsRepository()
    model = ImageModel(directory, max_images=10000, repo=repo)
    viewmodel = PhotoViewModel(directory, max_images=10000)
    
    # Load images
    print("Loading images...")
    viewmodel.load_images()
    
    # Wait a bit for images to be added
    time.sleep(2)
    
    print("Waiting 10 seconds for predictions to load...")
    time.sleep(10)
    
    # Get filtered images (these should be sorted by uncertainty)
    filtered_images = viewmodel.current_filtered_images()
    
    if not filtered_images:
        print("No images found!")
        return 1
    
    print(f"\nFound {len(filtered_images)} images in grid")
    print("\nChecking uncertainty sorting...")
    
    # Get probabilities
    probs = getattr(viewmodel._auto, "predicted_probabilities", {})
    
    # Calculate uncertainty for each image
    uncertainties = []
    for fname in filtered_images:
        prob = probs.get(fname)
        uncertainty = calculate_uncertainty(prob)
        uncertainties.append((fname, uncertainty, prob))
    
    # Check if sorted in descending order
    is_sorted = True
    violations = []
    
    for i in range(len(uncertainties) - 1):
        current_unc = uncertainties[i][1]
        next_unc = uncertainties[i + 1][1]
        if current_unc < next_unc:
            is_sorted = False
            violations.append((i, uncertainties[i][0], current_unc, uncertainties[i + 1][0], next_unc))
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Uncertainty Sorting Check Results")
    print(f"{'='*80}")
    print(f"Total images: {len(uncertainties)}")
    print(f"Sorted correctly: {is_sorted}")
    
    if not is_sorted:
        print(f"\n❌ VIOLATIONS FOUND: {len(violations)}")
        print("\nFirst 10 violations:")
        for idx, (pos, fname1, unc1, fname2, unc2) in enumerate(violations[:10], 1):
            print(f"  {idx}. Position {pos}: {fname1[:40]:40s} (unc={unc1:.4f}) > {fname2[:40]:40s} (unc={unc2:.4f})")
    else:
        print("\n✅ All images are sorted by uncertainty in descending order!")
    
    # Print first 20 images with their uncertainties
    print(f"\n{'='*80}")
    print("First 20 images (should be highest uncertainty first):")
    print(f"{'='*80}")
    print(f"{'Pos':<5} {'Uncertainty':<12} {'Probability':<12} {'Filename'}")
    print("-" * 80)
    for i, (fname, uncertainty, prob) in enumerate(uncertainties[:20], 1):
        prob_str = f"{prob:.4f}" if prob is not None else "None"
        print(f"{i:<5} {uncertainty:<12.4f} {prob_str:<12} {fname[:50]}")
    
    return 0 if is_sorted else 1

if __name__ == "__main__":
    sys.exit(main())


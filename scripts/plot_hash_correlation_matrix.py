#!/usr/bin/env python3
"""Plot correlation matrix of perceptual hash distances between all images."""

import os
import sys
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phash_cache import PerceptualHashCache


def compute_hash_matrix(image_dir: str, output_path: str = "hash_correlation_matrix.png"):
    """Compute and plot hash distance matrix for all images."""
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = sorted([
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ])
    
    if not filenames:
        print("No images found!")
        return
    
    print(f"Found {len(filenames)} images")
    print("Computing hashes...")
    
    # Load hash cache
    hash_cache = PerceptualHashCache()
    
    # Compute hashes for all images
    hashes = {}
    hash_strings = []
    
    for idx, filename in enumerate(filenames):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(filenames)} images...")
        
        path = os.path.join(image_dir, filename)
        
        # Try cache first
        cached_hash = hash_cache.get_hash(path)
        if cached_hash:
            hash_str = cached_hash
        else:
            # Compute hash
            try:
                with Image.open(path) as img:
                    phash = imagehash.phash(img)
                    hash_str = str(phash)
                    hash_cache.set_hash(path, hash_str)
            except Exception as e:
                print(f"  Warning: Failed to hash {filename}: {e}")
                hash_str = None
        
        if hash_str:
            hashes[filename] = hash_str
            hash_strings.append(hash_str)
    
    print(f"Computed {len(hashes)} hashes")
    print("Computing distance matrix...")
    
    # Compute distance matrix
    n = len(hashes)
    distance_matrix = np.zeros((n, n), dtype=np.int32)
    
    filenames_list = list(hashes.keys())
    
    for i in range(n):
        if (i + 1) % 50 == 0:
            print(f"  Computed distances for {i + 1}/{n} images...")
        
        hash1_str = hash_strings[i]
        hash1 = imagehash.hex_to_hash(hash1_str)
        
        for j in range(i, n):
            hash2_str = hash_strings[j]
            hash2 = imagehash.hex_to_hash(hash2_str)
            dist = hash1 - hash2
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    
    print("Plotting correlation matrix...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Plot heatmap
    im = ax.imshow(distance_matrix, cmap='viridis_r', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hash Distance (Hamming)', rotation=270, labelpad=20)
    
    # Set labels (sample every Nth filename to avoid clutter)
    step = max(1, len(filenames_list) // 50)  # Show max 50 labels
    tick_positions = list(range(0, len(filenames_list), step))
    tick_labels = [os.path.basename(filenames_list[i])[:20] + '...' if len(os.path.basename(filenames_list[i])) > 20 else os.path.basename(filenames_list[i]) for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)
    
    ax.set_xlabel('Image Index', fontsize=10)
    ax.set_ylabel('Image Index', fontsize=10)
    ax.set_title(f'Perceptual Hash Distance Matrix ({len(filenames_list)} images)\n'
                 f'Distance range: {distance_matrix.min()} - {distance_matrix.max()}', 
                 fontsize=12)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation matrix to {output_path}")
    
    # Print statistics
    print("\nDistance Statistics:")
    print(f"  Min distance: {distance_matrix.min()}")
    print(f"  Max distance: {distance_matrix.max()}")
    print(f"  Mean distance: {distance_matrix.mean():.2f}")
    print(f"  Median distance: {np.median(distance_matrix):.2f}")
    print(f"  Std deviation: {distance_matrix.std():.2f}")
    
    # Count pairs within different thresholds
    print("\nPairs within thresholds:")
    for threshold in [5, 8, 10, 15, 20, 30]:
        count = np.sum(distance_matrix <= threshold) - n  # Subtract diagonal (self-distances)
        count = count // 2  # Divide by 2 since matrix is symmetric
        print(f"  Distance <= {threshold}: {count} pairs ({count / (n * (n-1) / 2) * 100:.2f}%)")
    
    # Plot histogram of pair distances
    print("\nPlotting distance histogram...")
    hist_output = output_path.replace('.png', '_histogram.png')
    
    # Extract upper triangle (excluding diagonal) to avoid double counting
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    bins = np.arange(0, distance_matrix.max() + 2) - 0.5
    counts, bins, patches = ax.hist(upper_triangle, bins=bins, edgecolor='black', alpha=0.7)
    
    # Color bars based on thresholds
    for i, (count, patch) in enumerate(zip(counts, patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center <= 8:
            patch.set_facecolor('green')
        elif bin_center <= 20:
            patch.set_facecolor('yellow')
        elif bin_center <= 30:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    ax.set_xlabel('Hash Distance (Hamming)', fontsize=12)
    ax.set_ylabel('Number of Pairs', fontsize=12)
    ax.set_title(f'Distribution of Hash Distances Between Image Pairs\n'
                 f'Total pairs: {len(upper_triangle):,}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines for thresholds
    ax.axvline(x=8, color='blue', linestyle='--', linewidth=2, label='Main threshold (8)')
    ax.axvline(x=30, color='purple', linestyle='--', linewidth=2, label='Burst merge threshold (30)')
    ax.legend(fontsize=10)
    
    # Add text annotations for key thresholds
    for threshold in [8, 30]:
        count = np.sum(upper_triangle <= threshold)
        percentage = count / len(upper_triangle) * 100
        ax.text(threshold, counts[int(threshold)] * 1.1, 
                f'{count:,} pairs\n({percentage:.1f}%)',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(hist_output, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to {hist_output}")
    
    return distance_matrix, filenames_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_hash_correlation_matrix.py <image_dir> [output_path]")
        print("Example: python plot_hash_correlation_matrix.py ~/Pictures/photo-dataset")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hash_correlation_matrix.png"
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    compute_hash_matrix(image_dir, output_path)


if __name__ == "__main__":
    main()


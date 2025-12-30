#!/usr/bin/env python3
"""Visualize what PCA components represent by finding images that activate them.

Usage:
    poetry run python scripts/visualize_pca_components.py [--embeddings PATH] [--pca-dim 128] [--n-components 10]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def find_extreme_images(embeddings_pca: np.ndarray, filenames: list[str], component_idx: int, n_images: int = 5):
    """Find images with highest and lowest values for a PCA component."""
    component_values = embeddings_pca[:, component_idx]
    
    # Highest (positive) values
    top_indices = np.argsort(component_values)[::-1][:n_images]
    top_images = [(filenames[i], component_values[i]) for i in top_indices]
    
    # Lowest (negative) values
    bottom_indices = np.argsort(component_values)[:n_images]
    bottom_images = [(filenames[i], component_values[i]) for i in bottom_indices]
    
    return top_images, bottom_images


def visualize_component(
    component_idx: int,
    top_images: list[tuple[str, float]],
    bottom_images: list[tuple[str, float]],
    image_dir: str,
    explained_var: float,
    output_path: str = None,
):
    """Visualize a PCA component by showing images that activate it."""
    fig, axes = plt.subplots(2, len(top_images), figsize=(3 * len(top_images), 6))
    if len(top_images) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(
        f'PCA Component {component_idx}\n'
        f'Explains {explained_var*100:.2f}% variance\n'
        f'Top row: High activation | Bottom row: Low activation',
        fontsize=14
    )
    
    # Top images (high activation)
    for i, (fname, value) in enumerate(top_images):
        ax = axes[0, i]
        img_path = os.path.join(image_dir, fname)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.set_title(f'{value:+.2f}', fontsize=10, pad=5)
            except Exception:
                ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
        ax.axis('off')
    
    # Bottom images (low activation)
    for i, (fname, value) in enumerate(bottom_images):
        ax = axes[1, i]
        img_path = os.path.join(image_dir, fname)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.set_title(f'{value:+.2f}', fontsize=10, pad=5)
            except Exception:
                ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def analyze_component_patterns(
    embeddings_pca: np.ndarray,
    filenames: list[str],
    component_idx: int,
    image_dir: str,
):
    """Analyze patterns in images that activate a component."""
    component_values = embeddings_pca[:, component_idx]
    
    # Find extremes
    top_5_indices = np.argsort(component_values)[::-1][:5]
    bottom_5_indices = np.argsort(component_values)[:5]
    
    print(f"\nComponent {component_idx} Analysis:")
    print(f"  Value range: [{component_values.min():.2f}, {component_values.max():.2f}]")
    print(f"  Mean: {component_values.mean():.2f}, Std: {component_values.std():.2f}")
    print()
    
    print("  Top 5 images (high activation):")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"    {i}. {filenames[idx]} (value: {component_values[idx]:+.2f})")
    
    print("\n  Bottom 5 images (low activation):")
    for i, idx in enumerate(bottom_5_indices, 1):
        print(f"    {i}. {filenames[idx]} (value: {component_values[idx]:+.2f})")
    
    # Try to infer what the component captures
    print("\n  Interpretation:")
    print("    Compare the top and bottom images to identify common patterns")
    print("    High activation images share visual characteristics")
    print("    Low activation images share opposite characteristics")


def correlate_with_metadata(
    embeddings_pca: np.ndarray,
    filenames: list[str],
    component_idx: int,
    image_dir: str,
):
    """Try to correlate component values with image properties."""
    component_values = embeddings_pca[:, component_idx]
    
    # Analyze file sizes, names, etc.
    print(f"\nComponent {component_idx} Correlations:")
    
    # Check if filenames contain hints
    high_val_indices = np.where(component_values > component_values.mean() + component_values.std())[0]
    low_val_indices = np.where(component_values < component_values.mean() - component_values.std())[0]
    
    high_filenames = [filenames[i] for i in high_val_indices[:10]]
    low_filenames = [filenames[i] for i in low_val_indices[:10]]
    
    print(f"  High activation sample: {high_filenames[0] if high_filenames else 'N/A'}")
    print(f"  Low activation sample: {low_filenames[0] if low_filenames else 'N/A'}")


def create_component_summary(
    embeddings: np.ndarray,
    embeddings_pca: np.ndarray,
    filenames: list[str],
    pca: PCA,
    explained_var: np.ndarray,
    image_dir: str,
    n_components: int = 10,
    output_dir: str = None,
):
    """Create visual summary of top PCA components."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PCA COMPONENT VISUALIZATION")
    print("=" * 80)
    print()
    print(f"Analyzing top {n_components} components...")
    print()
    
    for comp_idx in range(min(n_components, embeddings_pca.shape[1])):
        var_explained = explained_var[comp_idx]
        
        # Find extreme images
        top_images, bottom_images = find_extreme_images(
            embeddings_pca, filenames, comp_idx, n_images=5
        )
        
        # Analyze patterns
        analyze_component_patterns(embeddings_pca, filenames, comp_idx, image_dir)
        
        # Visualize
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"component_{comp_idx:03d}.png")
        
        visualize_component(
            comp_idx,
            top_images,
            bottom_images,
            image_dir,
            var_explained,
            output_path,
        )
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize PCA components")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA target dimensions")
    parser.add_argument("--n-components", type=int, default=10, help="Number of components to visualize")
    parser.add_argument("--output-dir", default=None, help="Output directory for visualizations")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    # Find embeddings file
    embeddings_path = args.embeddings
    if not embeddings_path:
        cache_dir = Path(__file__).resolve().parent.parent / ".cache"
        possible = [
            cache_dir / "embeddings_resnet18_full.joblib",
            cache_dir / "embeddings_resnet18.joblib",
        ]
        for p in possible:
            if p.exists():
                embeddings_path = str(p)
                break
    
    if not embeddings_path or not os.path.exists(embeddings_path):
        print(f"Error: No embeddings file found. Tried: {possible}")
        return 1
    
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings, filenames = load_embeddings(embeddings_path)
    print(f"Loaded {len(filenames)} embeddings of dimension {embeddings.shape[1]}\n")
    
    # Apply PCA
    print(f"Applying PCA to reduce to {args.pca_dim} dimensions...")
    pca = PCA(n_components=args.pca_dim, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_
    
    print(f"PCA complete. Explained variance: {explained_var.sum()*100:.2f}%\n")
    
    # Create visualizations
    output_dir = args.output_dir or f".cache/pca_components_{args.pca_dim}d"
    create_component_summary(
        embeddings,
        embeddings_pca,
        filenames,
        pca,
        explained_var,
        image_dir,
        n_components=args.n_components,
        output_dir=output_dir,
    )
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nTo interpret components:")
    print("  1. Look at images with high activation (top row)")
    print("  2. Look at images with low activation (bottom row)")
    print("  3. Identify common visual patterns in each group")
    print("  4. The component captures the difference between these patterns")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


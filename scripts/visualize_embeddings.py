#!/usr/bin/env python3
"""Visualize ResNet18 embeddings using t-SNE and UMAP.

Usage:
    poetry run python scripts/visualize_embeddings.py [IMAGE_DIR] [--method tsne|umap|pca] [--output output.png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    import umap
except ImportError:
    umap = None

from sklearn.decomposition import PCA


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def visualize_2d(
    embeddings: np.ndarray,
    filenames: list[str],
    image_dir: str,
    method: str = 'tsne',
    output_path: str = None,
    n_samples: int = None,
    show_images: bool = True,
):
    """Visualize embeddings in 2D using dimensionality reduction."""
    
    if n_samples and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        filenames = [filenames[i] for i in indices]
    else:
        indices = np.arange(len(embeddings))
    
    print(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]}D to 2D using {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.2%}")
    elif method == 'tsne':
        if TSNE is None:
            raise ImportError("scikit-learn required for t-SNE. Install with: poetry add scikit-learn")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords = reducer.fit_transform(embeddings)
        print("t-SNE complete")
    elif method == 'umap':
        if umap is None:
            raise ImportError("umap-learn required for UMAP. Install with: poetry add umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
        coords = reducer.fit_transform(embeddings)
        print("UMAP complete")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure
    if show_images:
        fig, ax = plt.subplots(figsize=(20, 20))
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
    
    # Normalize coordinates for display
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = 0.1
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    
    if show_images:
        # Show thumbnail images at their coordinates
        thumb_size = 0.02 * max(x_range, y_range)
        for i, (x, y) in enumerate(coords):
            img_path = os.path.join(image_dir, filenames[i])
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    ax.imshow(img, extent=[x - thumb_size/2, x + thumb_size/2, 
                                          y - thumb_size/2, y + thumb_size/2], 
                             zorder=10)
                except Exception as e:
                    # Fallback: just plot a point
                    ax.scatter(x, y, s=20, alpha=0.6, c='blue', zorder=5)
            else:
                ax.scatter(x, y, s=20, alpha=0.6, c='red', zorder=5)
    else:
        # Just scatter plot
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.6, c='blue')
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(f'Image Embeddings Visualization ({method.upper()})\n{len(embeddings)} images', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_similarity_matrix(
    embeddings: np.ndarray,
    filenames: list[str],
    output_path: str = None,
    n_samples: int = 50,
):
    """Visualize similarity matrix between embeddings."""
    
    if n_samples and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        filenames = [filenames[i] for i in indices]
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity, cmap='viridis', aspect='auto')
    ax.set_title(f'Embedding Similarity Matrix ({len(embeddings)} images)', fontsize=14)
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('Image Index', fontsize=12)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix to {output_path}")
    else:
        plt.show()


def find_similar_images(
    embeddings: np.ndarray,
    filenames: list[str],
    query_idx: int,
    top_k: int = 5,
):
    """Find most similar images to a query image."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_emb = embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Exclude self
    similarities[query_idx] = -1
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nQuery image: {filenames[query_idx]}")
    print(f"Top {top_k} most similar images:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {filenames[idx]} (similarity: {similarities[idx]:.4f})")
    
    return top_indices


def main():
    parser = argparse.ArgumentParser(description="Visualize image embeddings")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--method", choices=['tsne', 'umap', 'pca'], default='tsne', 
                       help="Dimensionality reduction method")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--n-samples", type=int, default=None, 
                       help="Limit number of images to visualize")
    parser.add_argument("--no-images", action='store_true', 
                       help="Don't show thumbnail images, just scatter plot")
    parser.add_argument("--similarity", action='store_true', 
                       help="Show similarity matrix instead")
    parser.add_argument("--find-similar", type=int, default=None,
                       help="Find similar images to image at this index")
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
    print(f"Loaded {len(filenames)} embeddings of dimension {embeddings.shape[1]}")
    
    if args.find_similar is not None:
        find_similar_images(embeddings, filenames, args.find_similar)
        return 0
    
    if args.similarity:
        output = args.output or f".cache/embeddings_similarity_{args.method}.png"
        visualize_similarity_matrix(embeddings, filenames, output, args.n_samples)
    else:
        output = args.output or f".cache/embeddings_visualization_{args.method}.png"
        visualize_2d(
            embeddings, 
            filenames, 
            image_dir, 
            method=args.method,
            output_path=output,
            n_samples=args.n_samples,
            show_images=not args.no_images,
        )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


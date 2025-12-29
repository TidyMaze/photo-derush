#!/usr/bin/env python3
"""Analyze PCA components of image embeddings to understand what they capture.

Usage:
    poetry run python scripts/analyze_pca_embeddings.py [--embeddings PATH] [--pca-dim 128]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def analyze_pca(embeddings: np.ndarray, n_components: int = 128):
    """Analyze PCA transformation of embeddings."""
    print("=" * 80)
    print("PCA ANALYSIS OF IMAGE EMBEDDINGS")
    print("=" * 80)
    print()
    
    print(f"Original embeddings: {embeddings.shape[0]} images × {embeddings.shape[1]} dimensions")
    print(f"Target dimensions: {n_components}")
    print()
    
    # Fit PCA
    print("Fitting PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    print(f"Transformed embeddings: {embeddings_pca.shape[0]} images × {embeddings_pca.shape[1]} dimensions")
    print()
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("EXPLAINED VARIANCE:")
    print(f"  Total explained variance: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.2f}%)")
    print(f"  First component: {explained_var[0]:.4f} ({explained_var[0]*100:.2f}%)")
    print(f"  First 10 components: {cumulative_var[9]:.4f} ({cumulative_var[9]*100:.2f}%)")
    print(f"  First 50 components: {cumulative_var[49]:.4f} ({cumulative_var[49]*100:.2f}%)")
    print()
    
    # Component importance
    print("TOP 10 MOST IMPORTANT COMPONENTS:")
    top_indices = np.argsort(explained_var)[::-1][:10]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Component {idx}: {explained_var[idx]:.4f} ({explained_var[idx]*100:.2f}% variance)")
    print()
    
    # Variance distribution
    print("VARIANCE DISTRIBUTION:")
    print(f"  Mean variance per component: {explained_var.mean():.6f}")
    print(f"  Std variance per component: {explained_var.std():.6f}")
    print(f"  Min variance: {explained_var.min():.6f}")
    print(f"  Max variance: {explained_var.max():.6f}")
    print()
    
    # Component loadings (what original dimensions contribute most)
    print("UNDERSTANDING PCA COMPONENTS:")
    print("  Each PCA component is a linear combination of original 512 dimensions")
    print("  Components are ordered by variance explained (most important first)")
    print("  Higher variance = component captures more variation in the data")
    print()
    
    # Show what original dimensions contribute to first few components
    print("FIRST 3 COMPONENTS - TOP CONTRIBUTING ORIGINAL DIMENSIONS:")
    for comp_idx in range(min(3, n_components)):
        component = pca.components_[comp_idx]
        top_contrib = np.argsort(np.abs(component))[::-1][:5]
        print(f"\n  Component {comp_idx} (explains {explained_var[comp_idx]*100:.2f}% variance):")
        for orig_idx in top_contrib:
            weight = component[orig_idx]
            print(f"    Original dim {orig_idx}: weight = {weight:+.4f}")
    print()
    
    return pca, embeddings_pca, explained_var, cumulative_var


def visualize_pca(explained_var: np.ndarray, cumulative_var: np.ndarray, output_path: str = None):
    """Visualize PCA explained variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax1.plot(range(1, len(explained_var) + 1), explained_var, 'b-', linewidth=2)
    ax1.set_xlabel('Component Number', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot: Variance Explained by Each Component', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, min(50, len(explained_var)))  # Show first 50 components
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 'r-', linewidth=2)
    ax2.axhline(y=95, color='g', linestyle='--', label='95% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(cumulative_var))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def compare_original_vs_pca(embeddings: np.ndarray, embeddings_pca: np.ndarray):
    """Compare original and PCA-reduced embeddings."""
    print("COMPARISON: ORIGINAL vs PCA-REDUCED")
    print("=" * 80)
    print()
    
    # Statistics
    print("ORIGINAL EMBEDDINGS (512 dims):")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    print()
    
    print("PCA-REDUCED EMBEDDINGS (128 dims):")
    print(f"  Mean: {embeddings_pca.mean():.4f}")
    print(f"  Std: {embeddings_pca.std():.4f}")
    print(f"  Min: {embeddings_pca.min():.4f}")
    print(f"  Max: {embeddings_pca.max():.4f}")
    print()
    
    # Information content
    print("INFORMATION PRESERVATION:")
    print("  PCA preserves the directions of maximum variance")
    print("  Lower dimensions capture the most important patterns")
    print("  Higher dimensions capture fine-grained details")
    print()


def interpret_components(pca: PCA, n_show: int = 5):
    """Interpret what PCA components might represent."""
    print("INTERPRETING PCA COMPONENTS")
    print("=" * 80)
    print()
    print("What each component represents:")
    print("  - Each component is a weighted combination of original 512 ResNet18 features")
    print("  - Components are orthogonal (uncorrelated)")
    print("  - Higher variance = more important for distinguishing images")
    print()
    print("Component interpretation:")
    print("  - Component 0: Captures the most variation (dominant visual pattern)")
    print("  - Component 1: Second most important pattern (orthogonal to Component 0)")
    print("  - Component 2: Third most important pattern, etc.")
    print()
    print("Practical meaning:")
    print("  - Early components: Major visual differences (indoor/outdoor, people/no people)")
    print("  - Middle components: Moderate differences (lighting, composition)")
    print("  - Late components: Fine details (texture, subtle variations)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze PCA components of embeddings")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA target dimensions")
    parser.add_argument("--output", default=None, help="Output path for visualization")
    args = parser.parse_args()
    
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
    
    # Analyze PCA
    pca, embeddings_pca, explained_var, cumulative_var = analyze_pca(embeddings, args.pca_dim)
    
    # Compare
    compare_original_vs_pca(embeddings, embeddings_pca)
    
    # Interpret
    interpret_components(pca)
    
    # Visualize
    output = args.output or f".cache/pca_analysis_{args.pca_dim}d.png"
    visualize_pca(explained_var, cumulative_var, output)
    
    # Save PCA transformer for reference
    pca_path = Path(__file__).resolve().parent.parent / ".cache" / f"pca_transformer_{args.pca_dim}d.joblib"
    joblib.dump(pca, pca_path)
    print(f"\nPCA transformer saved to: {pca_path}")
    print("  (Can be used to transform new embeddings)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


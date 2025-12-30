#!/usr/bin/env python3
"""Plot correlation matrix of perceptual hash distances between all images."""

import os
import sys
from pathlib import Path
from collections import defaultdict

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform

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
    
    # For large datasets, create alternative visualizations
    if n > 200:
        print(f"\nLarge dataset ({n} images), creating alternative visualizations...")
        
        # 1. Sampled correlation matrix (every Nth image)
        sample_step = max(1, n // 200)  # Sample to ~200 images max
        sampled_indices = list(range(0, n, sample_step))
        sampled_matrix = distance_matrix[np.ix_(sampled_indices, sampled_indices)]
        sampled_filenames = [filenames_list[i] for i in sampled_indices]
        
        print(f"Creating sampled matrix ({len(sampled_indices)} images)...")
        _plot_correlation_matrix(sampled_matrix, sampled_filenames, 
                                 output_path.replace('.png', '_sampled.png'),
                                 f'Sampled Correlation Matrix ({len(sampled_indices)}/{n} images)')
        
        # 2. Clustered heatmap
        print("Creating clustered heatmap...")
        _plot_clustered_heatmap(distance_matrix, filenames_list, 
                                output_path.replace('.png', '_clustered.png'))
        
        # 3. Network graph of similar pairs
        print("Creating similarity network graph...")
        _plot_similarity_network(distance_matrix, filenames_list,
                                output_path.replace('.png', '_network.png'))
        
        # 4. Distance distribution histogram
        print("Plotting distance histogram...")
        hist_output = output_path.replace('.png', '_histogram.png')
        _plot_distance_histogram(distance_matrix, hist_output)
    else:
        # For smaller datasets, show full matrix
        print("Plotting full correlation matrix...")
        _plot_correlation_matrix(distance_matrix, filenames_list, output_path,
                                 f'Perceptual Hash Distance Matrix ({n} images)')
        
        # Always create histogram
        hist_output = output_path.replace('.png', '_histogram.png')
        _plot_distance_histogram(distance_matrix, hist_output)
    
    # Print statistics
    print("\nDistance Statistics:")
    print(f"  Min distance: {distance_matrix.min()}")
    print(f"  Max distance: {distance_matrix.max()}")
    print(f"  Mean distance: {distance_matrix.mean():.2f}")
    print(f"  Median distance: {np.median(distance_matrix):.2f}")
    print(f"  Std deviation: {distance_matrix.std():.2f}")
    
    # Count pairs within different thresholds
    print("\nPairs within thresholds:")
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]
    total_pairs = len(upper_triangle)
    for threshold in [5, 8, 10, 15, 20, 30]:
        count = np.sum(upper_triangle <= threshold)
        print(f"  Distance <= {threshold}: {count:,} pairs ({count / total_pairs * 100:.2f}%)")
    
    return distance_matrix, filenames_list


def _plot_correlation_matrix(distance_matrix, filenames_list, output_path, title):
    """Plot correlation matrix heatmap."""
    n = len(filenames_list)
    fig, ax = plt.subplots(figsize=(min(20, n/10), min(20, n/10)))
    
    im = ax.imshow(distance_matrix, cmap='viridis_r', aspect='auto', interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hash Distance (Hamming)', rotation=270, labelpad=20)
    
    # Set labels (sample every Nth filename)
    step = max(1, n // 50)
    tick_positions = list(range(0, n, step))
    tick_labels = [os.path.basename(filenames_list[i])[:15] + '...' 
                   if len(os.path.basename(filenames_list[i])) > 15 
                   else os.path.basename(filenames_list[i]) 
                   for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)
    
    ax.set_xlabel('Image Index', fontsize=10)
    ax.set_ylabel('Image Index', fontsize=10)
    ax.set_title(f'{title}\nDistance range: {distance_matrix.min()} - {distance_matrix.max()}', 
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")


def _plot_clustered_heatmap(distance_matrix, filenames_list, output_path):
    """Plot clustered heatmap using hierarchical clustering."""
    n = len(filenames_list)
    
    # Sample if too large
    if n > 300:
        sample_step = n // 300
        sampled_indices = list(range(0, n, sample_step))
        sampled_matrix = distance_matrix[np.ix_(sampled_indices, sampled_indices)]
        sampled_filenames = [filenames_list[i] for i in sampled_indices]
        n = len(sampled_indices)
    else:
        sampled_matrix = distance_matrix
        sampled_filenames = filenames_list
    
    # Convert to condensed distance matrix for linkage
    condensed_distances = squareform(sampled_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Get dendrogram leaf order
    dendro_order = leaves_list(linkage_matrix)
    
    # Reorder matrix by dendrogram order
    reordered_matrix = sampled_matrix[np.ix_(dendro_order, dendro_order)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 4], hspace=0.1, wspace=0.1)
    
    # Dendrogram
    ax_dendro = fig.add_subplot(gs[0, 1])
    dendrogram(linkage_matrix, ax=ax_dendro, orientation='top', no_labels=True, color_threshold=0.7*max(linkage_matrix[:,2]))
    ax_dendro.set_ylabel('Distance', fontsize=10)
    ax_dendro.set_title('Hierarchical Clustering Dendrogram', fontsize=12)
    
    # Heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    im = ax_heatmap.imshow(reordered_matrix, cmap='viridis_r', aspect='auto', interpolation='nearest')
    ax_heatmap.set_xlabel('Image Index (clustered)', fontsize=10)
    ax_heatmap.set_ylabel('Image Index (clustered)', fontsize=10)
    ax_heatmap.set_title('Clustered Hash Distance Matrix', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Hash Distance', rotation=270, labelpad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def _plot_similarity_network(distance_matrix, filenames_list, output_path, threshold=30):
    """Plot network graph showing connections between similar images."""
    try:
        import networkx as nx
    except ImportError:
        print("  Skipping network graph (networkx not available)")
        return
    
    n = len(filenames_list)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, filename in enumerate(filenames_list):
        G.add_node(i, label=os.path.basename(filename)[:20])
    
    # Add edges for similar pairs
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= threshold:
                G.add_edge(i, j, weight=distance_matrix[i, j])
                edge_count += 1
    
    if edge_count == 0:
        print(f"  No edges found below threshold {threshold}")
        return
    
    print(f"  Found {edge_count} edges (distance <= {threshold})")
    
    # Sample nodes if graph is too large
    if n > 200:
        # Keep nodes with most connections
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:200]
        nodes_to_keep = [node for node, _ in top_nodes]
        G = G.subgraph(nodes_to_keep).copy()
        print(f"  Sampled to {len(G.nodes())} nodes")
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color='gray', ax=ax)
    
    # Draw nodes (size by degree)
    node_sizes = [G.degree(node) * 50 + 100 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                           alpha=0.7, ax=ax)
    
    # Draw labels (only for nodes with high degree)
    high_degree_nodes = [node for node in G.nodes() if G.degree(node) > 2]
    labels = {node: G.nodes[node]['label'] for node in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    ax.set_title(f'Similarity Network (distance <= {threshold})\n'
                 f'{len(G.nodes())} nodes, {len(G.edges())} edges', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def _plot_distance_histogram(distance_matrix, output_path):
    """Plot histogram of distance distribution."""
    n = len(distance_matrix)
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
        if threshold < len(counts):
            ax.text(threshold, counts[int(threshold)] * 1.1, 
                    f'{count:,} pairs\n({percentage:.1f}%)',
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


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


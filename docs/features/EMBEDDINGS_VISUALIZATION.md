# Visualizing Image Embeddings

## Quick Start

```bash
# t-SNE visualization (recommended)
poetry run python scripts/visualize_embeddings.py --method tsne

# With thumbnail images (slower but more informative)
poetry run python scripts/visualize_embeddings.py --method tsne --n-samples 50

# Just scatter plot (faster)
poetry run python scripts/visualize_embeddings.py --method tsne --no-images

# Save to file
poetry run python scripts/visualize_embeddings.py --method tsne --output embeddings_viz.png
```

## Methods

### 1. t-SNE (Recommended)
Best for visualizing clusters and local structure. Non-linear, preserves local neighborhoods.

```bash
poetry run python scripts/visualize_embeddings.py --method tsne
```

**When to use:**
- Exploring image clusters
- Finding groups of similar images
- Understanding local relationships

### 2. UMAP
Similar to t-SNE but faster and preserves global structure better.

```bash
# First install: poetry add umap-learn
poetry run python scripts/visualize_embeddings.py --method umap
```

**When to use:**
- Large datasets (faster than t-SNE)
- Need to preserve global structure
- Want both local and global relationships

### 3. PCA
Linear, fast, preserves global variance. Good for quick overview.

```bash
poetry run python scripts/visualize_embeddings.py --method pca
```

**When to use:**
- Quick overview
- Understanding main variance directions
- Fast visualization

## Visualization Types

### Scatter Plot with Thumbnails
Shows images at their embedding coordinates. Similar images cluster together.

```bash
poetry run python scripts/visualize_embeddings.py --method tsne --n-samples 50
```

### Scatter Plot Only
Faster, shows just points without thumbnails.

```bash
poetry run python scripts/visualize_embeddings.py --method tsne --no-images
```

### Similarity Matrix
Shows pairwise cosine similarity between all images. Bright = similar, dark = different.

```bash
poetry run python scripts/visualize_embeddings.py --similarity --n-samples 50
```

### Find Similar Images
Find the most similar images to a query image.

```bash
# Find 5 most similar images to image at index 0
poetry run python scripts/visualize_embeddings.py --find-similar 0
```

## Examples

### Full Dataset Visualization
```bash
poetry run python scripts/visualize_embeddings.py \
    --method tsne \
    --no-images \
    --output .cache/embeddings_full_tsne.png
```

### Sample with Thumbnails
```bash
poetry run python scripts/visualize_embeddings.py \
    --method tsne \
    --n-samples 30 \
    --output .cache/embeddings_sample_tsne.png
```

### Similarity Analysis
```bash
# Find similar images to first image
poetry run python scripts/visualize_embeddings.py --find-similar 0

# Show similarity matrix
poetry run python scripts/visualize_embeddings.py --similarity --n-samples 50
```

## What to Look For

### Clusters
- **Tight clusters**: Very similar images (same scene, same objects)
- **Loose clusters**: Related images (similar composition, lighting)
- **Isolated points**: Unique images (distinct from others)

### Patterns
- **Spatial organization**: Images with similar visual features group together
- **Gradients**: Smooth transitions indicate related visual properties
- **Outliers**: Images that are visually distinct

### Interpretation
- **Close together** = Visually similar (same objects, scenes, composition)
- **Far apart** = Visually different
- **Clusters** = Groups of related images (e.g., all portraits, all landscapes)

## Performance Tips

1. **Use `--n-samples`** for large datasets (t-SNE is O(n²))
2. **Use `--no-images`** for faster rendering
3. **Use PCA** for quick overviews
4. **Use UMAP** for large datasets (faster than t-SNE)

## Output Files

Visualizations are saved to `.cache/` by default:
- `embeddings_visualization_tsne.png`
- `embeddings_visualization_umap.png`
- `embeddings_visualization_pca.png`
- `embeddings_similarity_tsne.png`

Or specify custom path with `--output`.


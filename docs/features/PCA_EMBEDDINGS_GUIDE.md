# Understanding PCA-Transformed Embeddings

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that:
- Transforms high-dimensional data (512 dims) into lower dimensions (128 dims)
- Preserves the most important information (variance)
- Creates orthogonal (uncorrelated) components

## How PCA Works on Embeddings

### Original Embeddings (512 dimensions)
- Each dimension represents a learned visual feature from ResNet18
- Dimensions are correlated (overlapping information)
- Many dimensions may be redundant

### PCA Transformation (512 → 128 dimensions)
1. **Finds principal directions**: Components that capture maximum variance
2. **Orders by importance**: Component 0 > Component 1 > Component 2 > ...
3. **Creates linear combinations**: Each component = weighted sum of original 512 dims
4. **Preserves information**: 95%+ of variance in first 128 components

## Interpreting PCA Components

### Component Importance

**Component 0** (First Principal Component):
- Captures the **most variation** in the dataset
- Represents the dominant visual pattern that distinguishes images
- Example: Indoor vs outdoor, presence of people, etc.

**Component 1** (Second Principal Component):
- Captures the **second most variation** (orthogonal to Component 0)
- Represents the next most important distinguishing pattern
- Example: Lighting conditions, time of day, etc.

**Component 2-N**:
- Progressively less important patterns
- Fine-grained visual differences
- Texture, subtle composition, etc.

### Variance Explained

- **Explained Variance Ratio**: How much of total variance each component captures
- **Cumulative Variance**: Running total of variance explained
- **Rule of thumb**: 
  - 90%+ cumulative variance = good representation
  - 95%+ cumulative variance = excellent representation

## What Information is Preserved?

### Preserved
- **Major visual patterns**: Objects, scenes, composition
- **Distinguishing features**: What makes images different
- **Global structure**: Overall image characteristics

### Lost/Reduced
- **Fine details**: Subtle texture variations
- **Redundant information**: Correlated features
- **Noise**: Random variations

## Practical Interpretation

### For Model Training
- **Early components (0-20)**: Most important for classification
- **Middle components (20-80)**: Moderate importance
- **Late components (80-128)**: Fine details, may help with edge cases

### For Visualization
- **Component 0 vs Component 1**: 2D plot shows main image clusters
- **Component 0 vs Component 2**: Alternative view of data structure
- **t-SNE/UMAP**: Can be applied to PCA-reduced embeddings for better visualization

### For Understanding
- **High variance component**: Captures major differences between images
- **Low variance component**: Captures subtle variations
- **Component loadings**: Show which original features contribute most

## Example Analysis

```
Component 0: 12.5% variance
  → Dominant pattern: Presence of people
  → Top contributing original dims: 45, 123, 78 (person detection features)

Component 1: 8.3% variance
  → Second pattern: Indoor vs outdoor
  → Top contributing original dims: 234, 156, 89 (scene classification features)

Component 2: 6.1% variance
  → Third pattern: Lighting conditions
  → Top contributing original dims: 12, 345, 201 (brightness/contrast features)
```

## Using PCA Components

### In Model Training
- **Input**: 128 PCA components (instead of 512 original)
- **Benefit**: Faster training, less overfitting, similar accuracy
- **Trade-off**: Slight information loss (usually <5%)

### For Analysis
- **Visualization**: Plot first 2-3 components to see data structure
- **Clustering**: Use PCA components for similarity analysis
- **Feature importance**: Early components are more important

## Key Takeaways

1. **PCA components are ordered by importance** (variance explained)
2. **Early components capture major patterns**, late components capture details
3. **128 components preserve 95%+ of information** from 512 original
4. **Each component is a weighted combination** of original features
5. **Components are orthogonal** (uncorrelated, independent)

## Visualization

Run the analysis script to see:
- **Scree plot**: Variance explained by each component
- **Cumulative variance**: Total variance explained by N components
- **Component loadings**: Which original dimensions contribute most

```bash
poetry run python scripts/analyze_pca_embeddings.py --pca-dim 128
```

This helps understand:
- How many components are needed (elbow in scree plot)
- What percentage of variance is preserved
- Which components are most important


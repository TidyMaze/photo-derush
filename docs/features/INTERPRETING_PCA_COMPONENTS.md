# Interpreting PCA Components: Visual Analysis

## The Challenge

Unlike CNN activations which can be visualized with activation maps, **PCA components are abstract linear combinations** of the original 512 ResNet18 features. They don't directly correspond to image regions.

## Solution: Find Images That Activate Components

Instead of visualizing what activates a component in a single image, we can:

1. **Find images with high component values** (positive activation)
2. **Find images with low component values** (negative activation)
3. **Compare the two groups** to identify what the component captures

## How It Works

### Step 1: Compute Component Values
For each image, compute its value for each PCA component:
```
Component_0_value = PCA_transform(image_embedding)[0]
Component_1_value = PCA_transform(image_embedding)[1]
...
```

### Step 2: Find Extremes
- **High activation**: Images with highest component values
- **Low activation**: Images with lowest component values

### Step 3: Visual Comparison
Compare the two groups to identify patterns:
- What do high-activation images have in common?
- What do low-activation images have in common?
- What's the difference between them?

## Example Interpretation

### Component 0 (12.23% variance)

**High Activation Images:**
- Images with people
- Outdoor scenes
- Bright lighting

**Low Activation Images:**
- Images without people
- Indoor scenes
- Dim lighting

**Interpretation**: Component 0 captures "presence of people in outdoor settings"

### Component 1 (6.03% variance)

**High Activation Images:**
- Close-up portraits
- Single subject
- Shallow depth of field

**Low Activation Images:**
- Wide landscapes
- Multiple subjects
- Deep depth of field

**Interpretation**: Component 1 captures "portrait vs landscape composition"

## Visualization Script

Run the visualization script to see this in action:

```bash
poetry run python scripts/visualize_pca_components.py \
    --n-components 10 \
    --output-dir .cache/pca_components
```

This creates:
- **Component 0 visualization**: Top 5 high-activation images vs bottom 5 low-activation images
- **Component 1 visualization**: Same for component 1
- **...and so on** for top N components

## Reading the Visualizations

Each visualization shows:
- **Top row**: Images with highest component values (what the component "likes")
- **Bottom row**: Images with lowest component values (what the component "dislikes")
- **Title**: Component number and variance explained

**To interpret:**
1. Look at the top row - what do these images have in common?
2. Look at the bottom row - what do these images have in common?
3. What's the difference? That's what the component captures.

## Limitations

### Unlike CNN Activation Maps
- **CNN activations**: Show which image regions activate a neuron (spatial visualization)
- **PCA components**: Show which images activate a component (no spatial info)

### Why This Difference?
- **CNN**: Processes spatial information (pixels → features)
- **PCA**: Operates on already-extracted features (no spatial structure)

### What We Can't Do
- ❌ Show which parts of an image activate a component
- ❌ Generate images that maximize a component (would need gradient access)
- ❌ Create activation heatmaps

### What We Can Do
- ✅ Find images that activate components
- ✅ Identify patterns in high/low activation groups
- ✅ Understand what visual features each component captures
- ✅ Correlate components with image properties

## Advanced Techniques

### 1. Statistical Analysis
Correlate component values with:
- Image metadata (EXIF data)
- Object detection results
- Image properties (brightness, contrast, etc.)

### 2. Clustering
Group images by component values to find:
- Natural clusters in component space
- Images with similar component activations

### 3. Component Combinations
Analyze combinations of components:
- Component 0 + Component 1: 2D visualization
- Multiple components: t-SNE/UMAP visualization

### 4. Gradient-Based Visualization (Future)
If we had access to gradients:
- Generate images that maximize a component
- Create "dream" images for each component
- Visualize what each component "wants to see"

## Practical Workflow

1. **Run visualization script** to see top components
2. **Examine each component** by comparing high/low activation images
3. **Document patterns** you observe
4. **Use insights** to understand what the model learns

## Example Workflow

```bash
# 1. Generate visualizations
poetry run python scripts/visualize_pca_components.py --n-components 10

# 2. Open .cache/pca_components_128d/component_000.png
#    Compare top row (high) vs bottom row (low)

# 3. Document findings:
#    Component 0: People vs no people
#    Component 1: Portrait vs landscape
#    Component 2: Indoor vs outdoor
#    ...

# 4. Use for model understanding:
#    - Which components are most important?
#    - What visual patterns does the model learn?
#    - How do components relate to keep/trash decisions?
```

## Key Insight

**PCA components are interpretable through comparison**, not direct visualization. By finding images that activate components differently, we can understand what each component captures - even if we can't visualize it spatially like CNN activations.


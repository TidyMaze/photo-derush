# ResNet18 Image Embeddings Description

## Overview

The embeddings are **512-dimensional dense vector representations** of images extracted using a pre-trained ResNet18 convolutional neural network. They capture high-level visual features learned from millions of images.

## Technical Details

### Model Architecture
- **Model**: ResNet18 (18-layer deep CNN)
- **Pre-training**: ImageNet dataset (1.2M images, 1000 object classes)
- **Output**: 512-dimensional feature vectors
- **Method**: Removed final classification layer, extracted penultimate layer activations

### Image Preprocessing
Before extracting embeddings, images are:
1. **Resized** to 256×256 pixels
2. **Center-cropped** to 224×224 (ImageNet standard)
3. **Normalized** using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406] (RGB)
   - Std: [0.229, 0.224, 0.225] (RGB)
4. **Converted** to tensor format

### Embedding Extraction Process
```
Image (any size) 
  → Resize(256) 
  → CenterCrop(224) 
  → Normalize 
  → ResNet18 (18 conv layers) 
  → Global Average Pooling 
  → 512-dim vector
```

## What Embeddings Capture

### Visual Features
- **Low-level**: Edges, textures, colors, gradients
- **Mid-level**: Shapes, patterns, object parts
- **High-level**: Objects, scenes, composition, lighting

### Specific Capabilities
- **Object recognition**: Detects presence of objects (people, animals, vehicles, etc.)
- **Scene understanding**: Indoor/outdoor, natural/urban, time of day
- **Composition**: Rule of thirds, symmetry, balance
- **Visual quality**: Blur, noise, exposure issues
- **Aesthetic patterns**: Learned from millions of photos

## Comparison with Handcrafted Features

### Handcrafted Features (78 features)
- **EXIF metadata**: ISO, aperture, shutter speed, focal length
- **Color histograms**: RGB distribution (24 bins)
- **Quality metrics**: Sharpness, saturation, entropy, noise
- **Geometric**: Width, height, aspect ratio
- **Temporal**: Hour, day of week, month

### Embeddings (512 → 128 dims)
- **Learned representations**: Patterns discovered from data
- **Visual semantics**: What the image "looks like" at a high level
- **Complementary**: Captures things handcrafted features miss

## Usage in Model

### Feature Combination
```
78 handcrafted features + 128 embedding features = 206 total features
```

### Dimensionality Reduction
- Original embeddings: 512 dimensions
- PCA reduction: 512 → 128 dimensions (for efficiency)
- Explained variance: ~95%+ (minimal information loss)

### Performance Impact
- **Without embeddings**: 84.00% accuracy (CatBoost, 78 features)
- **With embeddings**: 88.00% accuracy (+4.00%)
- **Final combined**: 86.67% accuracy (with tuned hyperparameters + threshold)

## Why They Work

1. **Transfer Learning**: Pre-trained on diverse ImageNet dataset
2. **Rich Representations**: Capture visual patterns beyond explicit features
3. **Complementarity**: Add information not in EXIF/histogram features
4. **Generalization**: Learned from millions of images, not just your dataset

## Example Embedding Values

```
Sample embedding (first 10 of 512 dimensions):
[2.36, 0.81, 0.36, 0.76, 0.12, 0.87, 0.05, 0.03, 2.59, 0.33, ...]

Statistics:
- Mean: 0.87
- Std: 0.84
- Range: 0.0 to 11.66
```

Each dimension represents activation of a learned feature detector. Higher values indicate stronger presence of that visual pattern.

## Storage

- **File**: `.cache/embeddings_resnet18.joblib`
- **Format**: Joblib (Python pickle format)
- **Contents**: 
  - `embeddings`: numpy array (401 images × 512 dims)
  - `filenames`: list of image filenames
- **Size**: ~814 KB

## Integration

The embeddings are:
1. **Pre-computed** once for all images (cached)
2. **Aligned** with handcrafted features by filename
3. **Concatenated** to create combined feature vector
4. **Used** by CatBoost model for prediction

This allows the model to leverage both explicit metadata/statistics AND learned visual representations.


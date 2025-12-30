#!/usr/bin/env python3
"""Generate comprehensive technical report of the photo-derush project."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training_core import DEFAULT_MODEL_PATH

def generate_technical_report(output_path: Path, format: str = "markdown"):
    """Generate comprehensive technical report in Markdown or PDF format."""
    
    if format == "markdown":
        return generate_markdown_report(output_path)
    elif format == "pdf":
        return generate_pdf_report(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def generate_markdown_report(output_path: Path):
    """Generate Markdown technical report."""
    
    md = f"""# Photo-Derush Technical Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

**Photo-Derush** is a desktop application for automated photo curation, classifying images as "keep" or "trash" using machine learning. The system combines handcrafted features, deep learning embeddings, and object detection.

**Current Production Model:** CatBoost classifier with 78 handcrafted features + 128 ResNet18 embeddings, achieving **82.23% ± 1.76% accuracy** (5-fold StratifiedKFold CV) on a personal photo dataset of 591 manually labeled images.

**Key Achievement:** Improved from 72% baseline (XGBoost, single split) to 82.23% ± 1.76% (CatBoost, 5-fold CV) through systematic experimentation.

> ⚠️ **Methodological Note:** This report documents the current implementation. See "Critical Issues & Recommendations" section for important improvements needed (group-based CV, threshold tuning leakage, asymmetric cost metrics).

---

## Table of Contents

1. [Libraries & Dependencies](#1-libraries--dependencies)
2. [System Architecture](#2-system-architecture)
3. [Feature Engineering](#3-feature-engineering)
4. [Object Detection](#4-object-detection)
5. [Image Embeddings](#5-image-embeddings)
6. [Classification Models](#6-classification-models)
7. [Experiments & Results](#7-experiments--results)
8. [Best Model Configuration](#8-best-model-configuration)
9. [Critical Issues & Recommendations](#9-critical-issues--recommendations)
10. [Key Learnings](#10-key-learnings)
11. [Future Improvements](#11-future-improvements)
12. [Reproducibility](#12-reproducibility)

---

## 1. Libraries & Dependencies

### Core ML Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ^1.5.0 | Pipeline, StandardScaler, train_test_split, cross-validation, metrics |
| catboost | ^1.2.8 | Primary classifier (best performance), handles categorical features |
| xgboost | ^3.1.1 | Alternative classifier (baseline: 72%), fallback option |
| lightgbm | ^4.6.0 | Ensemble experiments, faster training |
| torch | ^2.9.0 | ResNet18/ResNet50 embeddings, CNN training |
| torchvision | ^0.24.0 | Pre-trained models (ResNet18, ResNet50), image transforms |

### Object Detection

| Library | Purpose |
|---------|---------|
| ultralytics (YOLOv8) | YOLOv8n model for person/object detection, COCO classes |

### Image Processing

| Library | Version | Purpose |
|---------|---------|---------|
| Pillow | ^10.0.0 | Image loading, resizing, EXIF extraction |
| opencv-python | ^4.9.0.80 | Image quality metrics (sharpness, blur detection) |
| piexif | ^1.1.3 | EXIF data extraction (ISO, aperture, shutter speed) |
| imagehash | ^4.3.2 | Perceptual hashing for duplicate detection |

### UI & Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| PySide6 | ^6.6.0 | Desktop GUI application |
| matplotlib | ^3.8.0 | Plots, charts, model analysis |
| optuna | ^4.6.0 | Hyperparameter tuning (Bayesian optimization) |
| joblib | ^1.5.2 | Model persistence, feature caching |

---

## 2. System Architecture

### Pipeline Overview

```
Input Image
  ↓
[Feature Extraction]
  ├─ Handcrafted Features (78 dims)
  │  ├─ EXIF metadata (ISO, aperture, shutter, focal length)
  │  ├─ Color histograms (RGB, 24 bins)
  │  ├─ Quality metrics (sharpness, saturation, entropy, noise)
  │  ├─ Geometric (width, height, aspect ratio)
  │  ├─ Temporal (hour, day, month)
  │  └─ Object detection (person count, object count)
  │
  └─ Image Embeddings (128 dims)
     └─ ResNet18 (512 dims) → PCA (128 dims)
        ↓
[Preprocessing]
  └─ StandardScaler (normalize features)
     ↓
[Classification]
  └─ CatBoostClassifier
     ↓
[Decision]
  └─ Threshold: 0.67 → Keep/Trash
```

### Component Modules

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `src/features.py` | Feature Extraction | `extract_features()`, `batch_extract_features()`, feature cache |
| `src/object_detection.py` | Object Detection | `detect_objects()`, YOLOv8Adapter, person detection |
| `src/inference.py` | Model Inference | `predict_keep_probability()`, embedding extraction |
| `src/training_core.py` | Model Training | `train_keep_trash_model()`, cross-validation, early stopping |
| `src/viewmodel.py` | Application Logic | Filtering, sorting, model interaction |
| `src/view.py` | UI Components | PhotoView, grid display, badges |

---

## 3. Feature Engineering

### Handcrafted Features (78 dimensions)

| Category | Features | Description |
|----------|----------|-------------|
| EXIF Metadata | ISO, Aperture, Shutter Speed, Focal Length | Camera settings, technical quality indicators |
| Color Histograms | RGB distribution (24 bins) | Color composition, saturation patterns |
| Quality Metrics | Sharpness, Saturation, Entropy, Noise | Image quality assessment |
| Geometric | Width, Height, Aspect Ratio | Image dimensions, composition |
| Temporal | Hour, Day of Week, Month | Time-based patterns |
| Object Detection | Person Count, Object Count | Content indicators (from YOLOv8) |

### Feature Interactions

⚠️ **Current Implementation:** Top 15 features by variance are selected, then pairwise interactions (multiplication, ratio) are created. This adds ~100 interaction features.

**Issue:** High variance ≠ predictive. Variance just means the feature wiggles, not that it's useful.

**Recommendation:** 
- Let CatBoost handle non-linearities without manual interactions (it's designed for this), OR
- Pick interactions via mutual information / model-based importance computed *within CV*

Given interactions added complexity for marginal gains, they should be removed unless they measurably help under proper GroupKFold evaluation.

### Preprocessing Pipeline

⚠️ **Current Implementation:** StandardScaler applied to all features (handcrafted + embeddings).

**Issue:** 
- CatBoost (and GBDTs generally) doesn't need feature scaling
- StandardScaler on everything is unnecessary and adds complexity
- Risk of accidental leakage if not careful with fit/transform per fold

**Correct Structure:**
- Use `ColumnTransformer` to scale **only embeddings** (if PCA is applied)
- Keep handcrafted features raw (or transform only specific ones that need it)
- Concatenate after per-column preprocessing

This also reduces accidental leakage because it forces explicit fit/transform on the right subset per fold.

### Feature Caching

Features are cached to disk (`feature_cache.pkl`) to avoid recomputation. Cache is invalidated when source images are modified (mtime check).

---

## 4. Object Detection

### YOLOv8 Implementation

| Component | Details |
|-----------|---------|
| Model | YOLOv8n (nano variant, fastest) |
| Classes | COCO dataset (80 classes: person, car, dog, etc.) |
| Confidence Threshold | 0.6 (default), 0.8 (high confidence) |
| Device Support | CUDA, MPS (Apple Silicon), CPU fallback |
| Max Image Size | 800px (scaled down for efficiency) |

### Usage in Features

- **Person Detection:** Binary feature indicating person presence (confidence ≥ 0.6)
- **Object Count:** Total number of detected objects
- **High-Confidence Objects:** Objects with confidence ≥ 0.8 (max 5 shown in UI)

### Performance

Detection runs in separate worker process to avoid blocking UI. Results are cached per image to avoid redundant detection calls.

---

## 5. Image Embeddings

### ResNet18 Embeddings

| Property | Value |
|----------|-------|
| Model | ResNet18 (18-layer CNN) |
| Pre-training | ImageNet (1.2M images, 1000 classes) |
| Output Dimensions | 512 (original) → 128 (PCA-reduced) |
| PCA Explained Variance | ~95%+ (⚠️ **Needs measurement:** Should report mean/min across folds) |
| Preprocessing | Resize(256) → CenterCrop(224) → Normalize (ImageNet stats) |

### What Embeddings Capture

- **Low-level:** Edges, textures, colors, gradients
- **Mid-level:** Shapes, patterns, object parts
- **High-level:** Objects, scenes, composition, lighting, aesthetic patterns

### Performance Impact

**Embeddings Contribution:** Significant improvement (exact numbers vary by evaluation protocol - see "Critical Issues" section)

- Without embeddings: Lower accuracy (exact % depends on evaluation method)
- With embeddings: Higher accuracy (exact % depends on evaluation method)

> **Note:** Reported numbers (84% → 88%) are from single-split evaluations and may be optimistic. See "Critical Issues" for proper evaluation protocol.

### ResNet50 Experiments

ResNet50 embeddings (2048 dims) were tested but did not show significant improvement over ResNet18, likely due to small dataset size (591 samples).

---

## 6. Classification Models Tested

### Gradient Boosting Models

| Model | Best Accuracy | Notes |
|-------|---------------|-------|
| **CatBoost** | **82.23% ± 1.76%** (5-fold StratifiedKFold CV) | Best performance, handles class imbalance well, production model |
| XGBoost | 72.00% (single split) | Baseline, worse than CatBoost on this task |
| LightGBM | ~80% (estimated) | Faster training, used in ensemble experiments |

### Deep Learning Models

| Model | Architecture | Results |
|-------|--------------|---------|
| Simple CNN | 3 conv layers + 2 FC layers | ~75% (underperformed due to small dataset) |
| Transfer Learning | ResNet18 fine-tuned | Not extensively tested (small dataset limitation) |

### Ensemble Methods

- **Stacking:** Meta-learner on top of base models (CatBoost, XGBoost, LightGBM)
- **Voting:** Hard/soft voting ensembles
- **Blending:** Weighted average of predictions
- **Result:** Ensembles showed marginal improvement (~1-2%) but added complexity

---

## 7. Experiments & Results

### Canonical Evaluation Protocol

**Current Implementation:**
- **CV Method:** 5-fold StratifiedKFold (shuffle=True, random_state=42)
- **Features:** 78 handcrafted + 128 embeddings = 206 total
- **Preprocessing:** StandardScaler applied to all features (⚠️ **Issue:** Unnecessary for CatBoost, should use ColumnTransformer - see "Critical Issues")
- **PCA:** Fit on train set only, transform test set (explained variance: ~95%+ - needs measurement)
- **Threshold:** 0.5 (default) for CV evaluation; 0.67 for production (⚠️ **Issue:** Tuned globally, not per-fold - see "Critical Issues")

**Canonical Results (5-fold CV, threshold=0.5):**

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 82.23% ± 1.76% | Mean ± std across 5 folds, computed with default threshold 0.5 |
| Learning Rate | 0.1 | Most stable (lowest variance) |
| Early Stopping | Enabled (patience=200, eval_metric="Accuracy") | ⚠️ **Issue:** Metric misaligned with keep-loss goal - see "Critical Issues" |

### Learning Rate Study

| Learning Rate | Accuracy (5-fold CV) | Std Dev | Notes |
|---------------|----------------------|---------|-------|
| 0.01 | 80.37% | ±4.40% | Too small, underfitting |
| 0.05 | 80.54% | ±3.51% | Still underfitting |
| **0.10** | **82.23%** | **±1.76%** | **BEST (most stable)** |
| 0.11 | 81.05% | ±2.41% | Higher variance |
| 0.20 | 80.21% | ±2.85% | Too large, overfitting |

> **Key Finding:** Single test set (119 samples) showed high variance. LR=0.07 achieved 89.08% on one split but only 80.55% ± 3.87% in CV. Cross-validation revealed LR=0.1 is most reliable (82.23% ± 1.76%).

### Early Stopping Study

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| No Early Stopping (200 iter) | 84.87% (single split) | Single split evaluation, not comparable to CV |
| Early Stopping (patience=50) | Not measured | Removed - not measured under canonical protocol |
| Early Stopping (patience=200) | 84.87% (single split) | Single split evaluation, not comparable to CV |

> **Note:** These numbers are from single-split evaluations. CV results show 82.23% ± 1.76% with early stopping enabled.

### Feature Ablation Studies

⚠️ **Warning:** These numbers are from different evaluation protocols (single splits, different thresholds). They should be re-evaluated with consistent CV protocol.

| Feature Set | Reported Accuracy | Evaluation Method | Notes |
|-------------|-------------------|-------------------|-------|
| Handcrafted only (78) | 84.00% | Single split | Needs CV re-evaluation |
| + Embeddings (206) | 88.00% | Single split | Needs CV re-evaluation |
| + Feature Interactions (~306) | ~86-87% | Estimated | Marginal improvement, adds complexity |

### Hyperparameter Tuning

**Method:** Optuna (Bayesian optimization), 30 trials

| Parameter | Search Range | Best Value |
|-----------|--------------|------------|
| iterations | 100-1000 | 200 (with early stopping: 2000) |
| learning_rate | 0.01-0.3 | 0.1 (from CV study) |
| depth | 4-10 | 6 |
| l2_leaf_reg | 0.1-10.0 | 1.0 |
| border_count | 32-255 | Default (254) |

### Threshold Optimization

⚠️ **Critical Issue:** Optimal decision threshold: **0.67** (vs default 0.5)

**Current Implementation:**
- Threshold tuned on validation set (3% of training data)
- Same threshold used for all CV folds
- **Problem:** This leaks information - threshold should be tuned per-fold

**Recommendation:**
- Tune threshold inside each CV fold (on that fold's validation set)
- Or use nested CV (heavier, but correct)

---

## 8. Best Model Configuration

### Production Model Settings

| Component | Configuration |
|-----------|---------------|
| Classifier | CatBoostClassifier |
| Learning Rate | 0.1 (validated with 5-fold CV) |
| Max Iterations | 2000 (with early stopping) |
| Early Stopping | Enabled (patience=200, eval_metric="Accuracy") |
| Depth | 6 |
| L2 Regularization | 1.0 |
| Features | 78 handcrafted + 128 embeddings = 206 total |
| Preprocessing | StandardScaler ⚠️ (unnecessary for CatBoost, but kept for consistency) |
| Decision Threshold | 0.67 ⚠️ (tuned globally, not per-fold) |

### Performance Metrics (5-fold CV)

- **CV Accuracy:** 82.23% ± 1.76% (5-fold StratifiedKFold)
- **Test Set Size:** 119 samples (20% of 591 total)
- **Dataset:** 591 manually labeled images (391 keep, 200 trash)

### Dataset Statistics

- **Total Samples:** 591 manually labeled images
- **Keep:** 391 (66.2%)
- **Trash:** 200 (33.8%)
- **Train/Test Split:** 80/20 (stratified, random_state=42)
- **Test Set Size:** 119 samples

---

## 9. Critical Issues & Recommendations

### ⚠️ Issue 1: Data Leakage Risk (HIGH PRIORITY)

**Problem:** Photos are highly correlated:
- Burst shots (same scene, milliseconds apart)
- Near-duplicates (edited variants, crops)
- Same session/day/camera
- Perceptual hash clusters

If correlated photos land in different CV folds, the model can "cheat" by seeing similar images in training.

**Current Implementation:**
- Uses `StratifiedKFold` (no grouping)
- No duplicate detection/clustering
- No session-based grouping

**Fix (DO THIS FIRST):**

**Concrete Implementation Plan (best ROI → worst):**

1. **Duplicate clusters** from perceptual hash distance (strongest leakage source)
   - Use `imagehash` library (already in dependencies)
   - Cluster by Hamming distance threshold (e.g., ≤ 5 bits)
   - Group ID = cluster ID

2. **EXIF timestamp bucket** (5-30 minutes) with folder fallback
   - Group photos taken within same time window
   - Fallback to folder name if EXIF missing

3. **Folder + filename patterns** (if EXIF missing)
   - Group by folder structure
   - Handle filename patterns (IMG_001.jpg, IMG_002.jpg = same burst)

**Code:**
```python
from sklearn.model_selection import StratifiedGroupKFold
import imagehash
from PIL import Image
import os

def create_duplicate_groups(filenames, image_dir, hamming_threshold=5):
    # Group images by perceptual hash similarity
    groups = {{}}
    group_id = 0
    
    for fname in filenames:
        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path)
            phash = str(imagehash.phash(img))
            
            # Find existing group with similar hash
            assigned = False
            for existing_hash, gid in groups.items():
                if imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing_hash) <= hamming_threshold:
                    groups[phash] = gid
                    assigned = True
                    break
            
            if not assigned:
                groups[phash] = group_id
                group_id += 1
        except Exception:
            groups[phash] = group_id  # Assign new group on error
            group_id += 1
    
    return [groups.get(str(imagehash.phash(Image.open(os.path.join(image_dir, f)))), 0) for f in filenames]

groups = create_duplicate_groups(filenames, image_dir)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

**Expected Impact:** Accuracy may drop (this is good - means you're measuring reality, not leakage).

**Next Evaluation Milestone:** Re-run all experiments with StratifiedGroupKFold (duplicate clusters) and report new baseline.

### ⚠️ Issue 2: Threshold Tuning Leakage (HIGH PRIORITY)

**Problem:** Threshold 0.67 was tuned globally (on validation set), then used for all CV folds. This leaks information.

**Current Implementation:**
- Threshold tuned once on validation set
- Same threshold used for all folds
- **This is incorrect**

**Fix:**

```python
# Inside each CV fold:
for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Further split train for threshold tuning
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Train model
    model.fit(X_train_fit, y_train_fit)
    
    # Tune threshold on THIS fold's validation set
    threshold = optimize_threshold(model, X_val, y_val)
    
    # Evaluate on THIS fold's test set
    evaluate_with_threshold(model, X_test, y_test, threshold)
```

**Alternative:** Use nested CV (outer CV for evaluation, inner CV for threshold tuning).

### ⚠️ Issue 3: Wrong Top-Line Metric (HIGH PRIORITY)

**Problem:** Accuracy hides asymmetric costs:
- **False Negative (keep → trash):** Catastrophic (lose photos forever)
- **False Positive (trash → keep):** Annoying (clutter)

**Current Metrics:**
- Only accuracy reported
- No per-class metrics
- No cost-sensitive evaluation

**Fix - Add These Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| **Keep-Loss Rate** | FN(keep→trash) / total_keep | < 1-2% (hard constraint) |
| **Junk-Leak Rate** | FP(trash→keep) / total_trash | Acceptable if keep-loss is low |
| **PR-AUC** | Precision-Recall AUC | More informative than ROC-AUC under imbalance |
| **Precision (Keep)** | TP / (TP + FP) | How many "keep" predictions are correct |
| **Recall (Keep)** | TP / (TP + FN) | How many actual keeps are found |

**Threshold Selection:**
- Choose threshold to respect hard constraint: **keep-loss rate < 1-2%**
- Accept higher junk-leak rate if needed
- Don't optimize for accuracy alone

### ⚠️ Issue 4: Inconsistent Metrics Reporting

**Problem:** Report mixes:
- Single split scores (84%, 88%, 84.87%)
- CV scores (82.23% ± 1.76%)
- Different feature sets
- Different thresholds

**Fix:**
- Use **one canonical evaluation protocol** for all comparisons
- Report only CV mean ± std (or repeated CV)
- Same folds, same preprocessing fit rules, same thresholding method
- Document which numbers are from which protocol

### ⚠️ Issue 5: StandardScaler Unnecessary for CatBoost + Wrong Preprocessing Structure

**Problem:** 
- CatBoost (and GBDTs generally) doesn't need feature scaling
- StandardScaler applied to all features is unnecessary
- Current structure doesn't use ColumnTransformer, making it harder to ensure per-fold fit/transform correctness

**Current Implementation:**
- StandardScaler in pipeline before CatBoost (applied to all features)
- No column-wise preprocessing
- Risk of accidental leakage if not careful

**Recommendation:**
- Use `ColumnTransformer` to scale **only embeddings** (if PCA is applied)
- Keep handcrafted features raw
- Explicit per-column preprocessing = easier to verify fold-purity

**Correct Structure:**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define column indices
handcrafted_cols = list(range(78))  # First 78 columns
embedding_cols = list(range(78, 78+128))  # Next 128 columns

preprocessor = ColumnTransformer([
    ('handcrafted', 'passthrough', handcrafted_cols),  # No scaling
    ('embeddings', StandardScaler(), embedding_cols),  # Scale only embeddings
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('cat', CatBoostClassifier(...)),
])
```

This also reduces accidental leakage because it forces explicit fit/transform on the right subset per fold.

### ⚠️ Issue 6: Feature Interaction Selection Method

**Problem:** "Top 15 by variance" is not predictive. High variance ≠ useful feature.

**Current Implementation:**
- Select top 15 features by variance
- Create pairwise interactions
- Adds ~100 interaction features

**Recommendation:**
- Let CatBoost handle non-linearities (it's designed for this)
- OR: Pick interactions via mutual information / model-based importance (computed within CV)
- Given marginal gains, probably remove interactions

### ⚠️ Issue 7: Early Stopping Metric Misaligned with Goal

**Problem:** Early stopping uses `eval_metric="Accuracy"`, but the real goal is to minimize keep-loss rate (false negatives on keep photos).

**Current Implementation:**
- Early stopping: `eval_metric="Accuracy"`
- Decision objective: Minimize keep-loss rate (constraint: < 1-2%)
- **These are misaligned**

**Recommendation:**
- Either:
  - Early stop on **Logloss / AUC / PR-AUC**, then choose threshold by keep-loss constraint, OR
  - Skip early stopping and use fixed iteration count tuned by CV
- Right now the training objective and decision objective aren't aligned

### ⚠️ Issue 8: Missing Calibration

**Problem:** Probabilities are used for threshold tuning, but not calibrated. Uncalibrated probabilities lead to poor threshold selection.

**Current Implementation:**
- Raw CatBoost probabilities used
- No calibration applied

**Recommendation:**
- Add reliability curve + Brier score analysis
- Use `CalibratedClassifierCV` (isotonic/sigmoid) **inside CV**
- Or use CatBoost's own calibration options + post-calibration

### ⚠️ Issue 9: Missing Error Analysis

**Problem:** No systematic analysis of failure cases. Don't know what the model struggles with.

**Missing:**
- Categorized failure examples (20-50 real cases)
- Motion blur failures
- Low light noise failures
- Faces vs no faces
- Duplicates
- Screenshots
- "Aesthetically good but technically weird"

**Recommendation:**
- Add error analysis section (see "Error Analysis" section below)
- Categorize 20-50 real failure cases
- Use findings to guide feature engineering (prevents random feature-bloat)
- This will tell you what features you actually need next

---

## 10. Operating Point (Production Metrics)

⚠️ **Note:** These metrics are computed with the current (flawed) evaluation protocol. They should be re-measured after fixing threshold tuning leakage and implementing StratifiedGroupKFold.

**Production Operating Point (threshold=0.67, single split evaluation):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Keep-Loss Rate** | Not measured | < 1-2% | ⚠️ **Missing** |
| **Junk-Leak Rate** | Not measured | Acceptable if keep-loss is low | ⚠️ **Missing** |
| **Precision (Keep)** | 0.875 (estimated) | Higher is better | ⚠️ Needs measurement |
| **Recall (Keep)** | 0.942 (estimated) | Higher is better | ⚠️ Needs measurement |
| **PR-AUC** | Not measured | Primary metric | ⚠️ **Missing** |
| **Accuracy** | 82.23% ± 1.76% (CV, threshold=0.5) | Secondary metric | Measured |
| **F1 Score** | Not measured | Balanced metric | ⚠️ **Missing** |

**After Fixing Methodological Issues:**
- Re-measure all metrics with per-fold threshold tuning
- Report keep-loss rate and junk-leak rate as headline metrics
- Choose threshold to respect hard constraint: **keep-loss rate < 1-2%**

---

## 11. Error Analysis

⚠️ **Status:** Not yet performed. This section should be populated with 20-50 categorized failure cases.

**Planned Categories:**
- Motion blur failures
- Low light noise failures
- Faces vs no faces
- Duplicates (near-duplicates misclassified)
- Screenshots (incorrectly classified)
- "Aesthetically good but technically weird" (good composition but technical issues)
- Overexposed/underexposed
- Out of focus
- Composition issues (good technical quality but poor framing)

**Method:**
1. Collect 20-50 misclassified examples from test set
2. Categorize each failure
3. Count frequency per category
4. Use findings to guide feature engineering priorities

**Expected Insights:**
- If motion blur is common → add blur detection features
- If face detection issues → improve face detection features
- If duplicates are problematic → prioritize duplicate clustering
- If low-light is common → add noise/ISO features

---

## 12. Key Learnings

### Cross-Validation is Essential

**Finding:** Single test set (119 samples) showed high variance. Small LR changes flipped 8-14 predictions = 6-12% accuracy swings. 5-fold CV provides reliable estimates by averaging over multiple splits.

**Next Step:** Move to StratifiedGroupKFold to handle photo correlation.

### Learning Rate Insights

- **Smaller LR ≠ Higher Accuracy:** LR=0.01-0.05 underfit (80-81%), LR=0.1 optimal (82.23%), LR≥0.15 overfit (80-81%)
- **Optimal Range:** 0.07-0.11 (sweet spot)
- **Variance Matters:** LR=0.07 had higher variance (±3.87%) than LR=0.1 (±1.76%)

### Early Stopping Considerations

- Small validation sets (3-5%) cause premature stopping
- High patience (200 rounds) needed to match baseline performance
- Early stopping works best with validation set ≥10% of training data

### Feature Engineering

- **Embeddings are crucial:** Significant improvement (exact % needs proper CV re-evaluation)
- **Feature interactions:** Marginal improvement, adds complexity (probably remove)
- **Handcrafted + Learned:** Complementary information sources

### Model Selection

- **CatBoost > XGBoost:** Better default performance, class imbalance handling
- **Gradient Boosting > CNNs:** Better for small datasets (591 samples)
- **Ensembles:** Marginal gains, added complexity not worth it

### Dataset Size Limitations

**Challenge:** 591 samples is small for deep learning. ResNet50 embeddings didn't help, CNNs underperformed. Gradient boosting with handcrafted features + embeddings is optimal for this dataset size.

---

## 13. Future Improvements

### High Priority (Address Methodological Issues)

1. **StratifiedGroupKFold:** Group by duplicate clusters (perceptual hash) → EXIF timestamp → folder
2. **Per-fold Threshold Tuning:** Fix threshold tuning leakage (tune inside each CV fold)
3. **Asymmetric Cost Metrics:** Replace accuracy with keep-loss rate + PR-AUC as headline metrics
4. **Error Analysis:** Categorize 20-50 failure cases (see "Error Analysis" section)
5. **ColumnTransformer Preprocessing:** Scale only embeddings, keep handcrafted raw
6. **Early Stopping Metric Alignment:** Use Logloss/AUC/PR-AUC for early stopping, then tune threshold by keep-loss

### Medium Priority

- **More Data:** Label more images to reach 1000+ samples (would enable better deep learning)
- **Better Embeddings:** CLIP embeddings (semantic understanding), EfficientNet (better feature extraction)
- **Data Augmentation:** Rotation, flip, crop, color jitter to increase effective dataset size
- **Calibration:** Platt scaling / isotonic regression to improve probability estimates

### Low Priority

- **Class Balancing:** SMOTE, focal loss, better class weights
- **Feature Engineering:** Domain-specific features (face detection, scene classification)
- **Ensemble Refinement:** Better stacking/blending strategies
- **Active Learning:** Select most informative images for labeling

### Architecture Improvements

- **Multi-task Learning:** Predict keep/trash + quality score simultaneously
- **Attention Mechanisms:** Focus on important image regions
- **Graph Neural Networks:** Model relationships between images

### Product Feature: Near-Duplicate Handling (V2 Objective)

**Current:** Binary classification (keep/trash) per image

**V2 Objective:** Treat duplicates as a *product* feature, not only an eval fix
- **Cluster** similar images (perceptual hash, burst detection)
- **Rank** images within cluster (by quality score, sharpness, composition)
- **Keep top-k** per cluster (e.g., keep best 1-3 images per burst)

This is likely the biggest UX win: users don't want binary classification; they want **ranking within bursts**. This aligns with how humans actually curate photos (pick best from burst, not classify each individually).

**Implementation:**
1. Cluster images by perceptual hash distance
2. Compute quality score per image (sharpness, exposure, composition)
3. Rank within cluster
4. Keep top-k per cluster
5. Apply keep/trash classification only to non-duplicate images or after ranking

### Feature/Model Ideas That Actually Move the Needle

- **CLIP / SigLIP embeddings:** Usually outperform ResNet on "semantic + aesthetic" judgments
- **Aesthetic score feature:** Single scalar (LAION aesthetic predictor / NIMA-like) can beat 100 handcrafted features
- **Face features:** Number of faces, face size ratio, sharpness on face region (huge for personal photos)
- **Near-duplicate handling:** Cluster similar photos, decision becomes "pick best in cluster" (closer to human behavior)

---

## 14. Reproducibility

### Dataset

- **Total Samples:** 591 manually labeled images
- **Source:** Personal photo collection
- **Labeling:** Manual (keep/trash)
- **Auto-labeled excluded:** Yes (to avoid circular dependency)

### Evaluation Protocol

- **CV Method:** 5-fold StratifiedKFold
- **Random State:** 42 (fixed throughout)
- **Shuffle:** True
- **Stratification:** Yes (maintains class balance)

⚠️ **Missing:** Group-based splitting (should use StratifiedGroupKFold)

### Preprocessing

- **StandardScaler:** Applied to all features (fit on train fold, transform test fold) ⚠️ **Issue:** Unnecessary for CatBoost, should use ColumnTransformer
- **PCA (embeddings):** Fit on train fold, transform test fold (explained variance: ~95%+ ⚠️ **Needs measurement**)
- **Feature Interactions:** Created from top 15 features by variance (⚠️ suboptimal method)

### Model Training

- **Random Seed:** 42
- **Early Stopping:** Enabled (patience=200, eval_metric="Accuracy") ⚠️ **Issue:** Metric misaligned with keep-loss goal
- **Validation Set:** 3% of training data (for early stopping)

⚠️ **Missing:** 
- Threshold tuned globally (should be per-fold)
- Group-based splitting (should use StratifiedGroupKFold)

### Hyperparameters

- **Learning Rate:** 0.1 (from CV study)
- **Max Iterations:** 2000 (with early stopping)
- **Depth:** 6
- **L2 Regularization:** 1.0
- **Decision Threshold:** 0.67 (⚠️ tuned globally, not per-fold)

### Code Versions

- **Python:** 3.12
- **scikit-learn:** ^1.5.0
- **catboost:** ^1.2.8
- **torch:** ^2.9.0
- **torchvision:** ^0.24.0

### Results

**Canonical Results (5-fold StratifiedKFold CV, threshold=0.5):**
- Accuracy: 82.23% ± 1.76% (computed with default threshold 0.5, not production threshold 0.67)

⚠️ **Note:** These results may be optimistic due to:
- No group-based splitting (photo correlation) - **highest priority fix**
- Global threshold tuning (leakage) - threshold 0.67 not used in CV
- Missing asymmetric cost metrics (keep-loss rate, PR-AUC)
- Early stopping metric misaligned (Accuracy vs keep-loss goal)

**Recommended Next Steps (in order):**
1. Implement StratifiedGroupKFold (duplicate clusters first)
2. Fix threshold tuning (per-fold, inside CV)
3. Add keep-loss rate + PR-AUC metrics
4. Re-evaluate with proper protocol
5. Report operating point table with keep-loss as headline metric

---

## Recommended Evaluation Protocol

### Proposed Correct Protocol

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, auc

# 1. Create groups (shooting session, duplicate cluster, or combination)
groups = create_groups(images)  # e.g., by folder, timestamp bucket, or perceptual hash cluster

# 2. CV with grouping
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Per-fold evaluation
results = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Split train further for threshold tuning
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Fit preprocessing (StandardScaler, PCA) on train_fit only
    scaler.fit(X_train_fit)
    pca.fit(embeddings_train_fit)
    
    # Train model
    model.fit(scaler.transform(X_train_fit), y_train_fit)
    
    # Tune threshold on THIS fold's validation set
    y_proba_val = model.predict_proba(scaler.transform(X_val))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
    
    # Choose threshold to keep keep-loss rate < 1-2%
    # keep_loss_rate = FN / total_keep
    threshold = find_threshold_for_max_keep_loss(y_val, y_proba_val, max_loss=0.02)
    
    # Evaluate on THIS fold's test set
    y_proba_test = model.predict_proba(scaler.transform(X_test))[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_test, y_proba_test)
    results.append(metrics)

# 4. Aggregate results
mean_accuracy = np.mean([r['accuracy'] for r in results])
std_accuracy = np.std([r['accuracy'] for r in results])
mean_keep_loss = np.mean([r['keep_loss_rate'] for r in results])
mean_pr_auc = np.mean([r['pr_auc'] for r in results])
```

### Metrics to Report

| Metric | Definition | Target |
|--------|------------|--------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Secondary metric |
| Keep-Loss Rate | FN(keep→trash) / total_keep | **< 1-2% (hard constraint)** |
| Junk-Leak Rate | FP(trash→keep) / total_trash | Acceptable if keep-loss is low |
| PR-AUC | Precision-Recall AUC | Primary metric (better than ROC-AUC under imbalance) |
| Precision (Keep) | TP / (TP + FP) | How many "keep" predictions are correct |
| Recall (Keep) | TP / (TP + FN) | How many actual keeps are found |
| F1 Score | 2 * (precision * recall) / (precision + recall) | Balanced metric |

---

## Conclusion

The Photo-Derush project demonstrates a systematic approach to building a photo curation system with limited data. Key achievements:

- Improved accuracy from 72% (XGBoost, single split) to 82.23% ± 1.76% (CatBoost, 5-fold CV)
- Validated findings with cross-validation to avoid overfitting to single test set
- Combined handcrafted features with deep learning embeddings effectively
- Established best practices for small dataset ML (gradient boosting > CNNs)

**However, critical methodological issues need to be addressed:**

1. **Data leakage risk:** Photos are correlated (bursts, duplicates, sessions) - need StratifiedGroupKFold
2. **Threshold tuning leakage:** Threshold tuned globally, not per-fold
3. **Wrong metrics:** Accuracy hides asymmetric costs - need keep-loss rate + PR-AUC
4. **Inconsistent reporting:** Mixed evaluation protocols make numbers incomparable

**If you do only 3 things:**
1. Switch to **StratifiedGroupKFold** (group by session/duplicate cluster)
2. Make thresholding + tuning **fold-pure** (no leakage)
3. Replace "accuracy" with **keep-loss rate + PR-AUC**, and choose threshold by cost

The system is production-ready but should be re-evaluated with proper methodology to ensure reported metrics reflect reality.

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    output_path.write_text(md)
    print(f"Markdown report saved to: {output_path}")


def generate_pdf_report(output_path: Path):
    """Generate PDF technical report (converts from Markdown)."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from io import StringIO
        
        # Generate markdown first
        md_path = output_path.with_suffix('.md')
        generate_markdown_report(md_path)
        
        # Convert markdown to HTML
        with open(md_path) as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add basic styling
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ff9800; padding-left: 15px; margin-left: 0; color: #555; font-weight: bold; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        # Convert HTML to PDF
        HTML(string=styled_html).write_pdf(output_path)
        print(f"PDF report saved to: {output_path}")
        return True
        
    except ImportError:
        print("Error: PDF generation requires 'markdown' and 'weasyprint' packages.")
        print("Falling back to Markdown format.")
        md_path = output_path.with_suffix('.md')
        generate_markdown_report(md_path)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate comprehensive technical report")
    parser.add_argument("--output", default="studies/outputs/technical_report.md", help="Output file (default: .md, can be .pdf)")
    parser.add_argument("--format", choices=["markdown", "pdf", "auto"], default="auto", 
                       help="Output format (auto detects from file extension)")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    # Auto-detect format from extension
    if args.format == "auto":
        if output_path.suffix.lower() == ".pdf":
            format_type = "pdf"
        else:
            format_type = "markdown"
    else:
        format_type = args.format
    
    generate_technical_report(output_path, format=format_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

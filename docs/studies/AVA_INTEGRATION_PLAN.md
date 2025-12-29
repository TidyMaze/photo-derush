# AVA Dataset Integration Plan

## Current Status

### Model Performance
- **Training Accuracy**: 100% (overfitting)
- **Validation Accuracy**: 78.4%
- **Test Accuracy**: 76%
- **Overfitting Gap**: 21.6 percentage points

### Dataset
- **Current Dataset**: 371 samples (100% accuracy - overfitting)
- **AVA Metadata**: 255,530 images ready
- **AVA Labels Prepared**: 1,000 samples (75.2% keep, 24.8% trash)

## Problem: Overfitting

The model achieves 100% accuracy on the training set but only 76% on test set, indicating severe overfitting. This is expected with only 371 training samples.

## Solution: AVA Dataset Integration

### Phase 1: Download AVA Images (Required)

**Option A: Using ava_downloader tool**
```bash
cd .cache/ava_dataset/ava_downloader
python download_ava.py
```

**Option B: Download from Academic Torrents**
- Visit: https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a
- Download torrent file (~32GB)
- Extract images

**Option C: Download from MEGA**
- Link: https://mega.nz/folder/9b520Lzb#2gIa1fgAzr677dcHKxjmtQ
- Download 64 7z files (~32GB total)
- Extract first file (others will auto-extract)

### Phase 2: Prepare AVA Features

Once images are downloaded, extract features:

```bash
# Extract features for AVA images
poetry run python scripts/extract_ava_features.py \
    --ava-images-dir /path/to/ava/images \
    --ava-labels .cache/ava_dataset/ava_keep_trash_labels.json \
    --output .cache/ava_features.joblib
```

### Phase 3: Train on Combined Dataset

```bash
# Train on current + AVA (start with 10k AVA samples)
poetry run python scripts/train_with_ava.py \
    --ava-labels .cache/ava_dataset/ava_keep_trash_labels.json \
    --ava-images-dir /path/to/ava/images \
    --max-ava 10000

# Scale up to more samples
poetry run python scripts/train_with_ava.py \
    --ava-labels .cache/ava_dataset/ava_keep_trash_labels.json \
    --ava-images-dir /path/to/ava/images \
    --max-ava 50000
```

### Phase 4: Evaluate Improvement

```bash
# Benchmark new model
poetry run python scripts/benchmark_model.py

# Check overfitting
# - Training accuracy should be close to validation accuracy
# - Gap should be < 5%
```

## Expected Improvements

### With 10k AVA Samples
- **Expected Test Accuracy**: 80-85%
- **Overfitting Gap**: < 10%
- **Generalization**: Better performance on unseen data

### With 50k+ AVA Samples
- **Expected Test Accuracy**: 85-90%
- **Overfitting Gap**: < 5%
- **Generalization**: Strong performance across diverse images

## Files Created

1. **`scripts/train_with_ava.py`** - Training script with overfitting detection
2. **`scripts/benchmark_model.py`** - Benchmarking tool
3. **`scripts/setup_ava_dataset.py`** - AVA metadata setup
4. **`scripts/download_ava_dataset.py`** - AVA label preparation
5. **`.cache/ava_dataset/AVA_metadata.txt`** - 255,530 image records
6. **`.cache/ava_dataset/ava_keep_trash_labels.json`** - Labeled samples

## Next Steps

1. **Download AVA images** (32GB, ~255k images)
2. **Extract features** for AVA images
3. **Train on combined dataset** (current + AVA)
4. **Evaluate** improvement and overfitting
5. **Iterate** with more AVA samples if needed

## Monitoring Overfitting

The training script automatically detects overfitting:
- **Gap < 5%**: No significant overfitting ✓
- **Gap 5-10%**: Moderate overfitting ⚠️
- **Gap > 10%**: Significant overfitting ❌

If overfitting persists:
- Add more training data (AVA)
- Increase regularization (l2_leaf_reg)
- Reduce model complexity (depth)
- Use early stopping

## Notes

- AVA dataset provides diverse aesthetic quality labels
- Score range: 3.11 - 7.72 (mean: 5.46)
- Threshold: 6.0 → 75% keep, 25% trash (adjustable)
- Images need to be downloaded separately (not included in metadata)



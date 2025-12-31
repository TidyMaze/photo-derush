# COCO Class Impact Analysis

## Optimization Function

### Primary Goal
**Minimize keep-loss rate** (target: < 2%)

```
keep_loss_rate = FN(keep→trash) / total_keep
```

Where:
- `FN` = False Negatives (keep photos incorrectly predicted as trash)
- `total_keep` = Total number of keep photos

### Secondary Goals
- Maximize accuracy
- Maximize PR-AUC (Precision-Recall AUC)

### Model Training
- **Algorithm**: CatBoost classifier
- **Loss function**: `Logloss` (aligned with keep-loss goal)
- **Threshold tuning**: Optimized to minimize keep-loss < 2%
- **Early stopping**: `eval_metric="Logloss"`, `patience=10`

## Current State

**Only 1 COCO class is used for training:**
- `"person"` (class ID 1)
- Binary feature: 1.0 if person detected, 0.0 otherwise
- Feature index: 77 (FAST mode) or 101 (FULL mode)

**All other COCO classes:**
- Detected and cached (80 total COCO classes)
- Displayed in UI (bounding boxes, labels)
- **NOT used as training features**

## Analysis Script

**Script**: `scripts/analyze_coco_class_impact.py`

**Purpose**: For each COCO class, compute its impact on the optimization function.

**Metrics computed:**
1. **Keep-loss improvement** (primary): `Δ keep_loss_rate = baseline - with_class`
2. **Accuracy improvement**: `Δ accuracy = with_class - baseline`
3. **PR-AUC improvement**: `Δ pr_auc = with_class - baseline`
4. **Presence rate**: Percentage of images containing the class

**Methodology:**
1. Load training dataset
2. For each COCO class:
   - Extract binary feature (1.0 if class detected, 0.0 otherwise)
   - Train baseline model (without class)
   - Train model with class feature
   - Compute metrics with optimal thresholds (keep-loss < 2%)
   - Calculate improvements
3. Sort by keep-loss improvement (primary optimization goal)

**Usage:**
```bash
python scripts/analyze_coco_class_impact.py \
  --image-dir <path_to_images> \
  --output coco_class_impact.json
```

**Output:**
- JSON file with impact metrics for each COCO class
- Console summary of top 10 classes by keep-loss improvement
- Sorted by primary optimization goal (keep-loss reduction)

## Expected Results

The analysis will identify:
- **High-impact classes**: Classes that significantly reduce keep-loss rate
- **Low-impact classes**: Classes with minimal or negative impact
- **Insufficient data**: Classes present in < 1% of images (skipped)

**Decision criteria:**
- If a class reduces keep-loss by > 0.5% → **Consider adding**
- If a class improves accuracy by > 1% → **Consider adding**
- If a class has negative impact → **Skip**

## Implementation Notes

To add a COCO class as a feature:
1. Modify `src/features.py`:
   - Add extraction function similar to `_extract_person_detection()`
   - Add feature to `_do_feature_extraction()`
   - Update `FEATURE_COUNT`
2. Update `src/model_stats.py`:
   - Add feature name mapping
3. Retrain model:
   - New features will be automatically included in training

## References

- **Training code**: `src/training_core.py`
- **Feature extraction**: `src/features.py`
- **Object detection**: `src/object_detection.py`
- **COCO classes**: 80 classes defined in `COCO_CLASSES` dict


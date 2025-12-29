# Large Dataset Options for Keep/Trash Training

## Challenge
No publicly available dataset exists with explicit "keep/trash" labels. However, several approaches can provide larger training data.

## Option 1: AVA Dataset (Aesthetic Visual Analysis) ⭐ RECOMMENDED

**Dataset**: AVA (Aesthetic Visual Analysis)
- **Size**: ~250,000 images
- **Labels**: Aesthetic quality scores (1-10 scale)
- **Source**: 
  - GitHub: https://github.com/mtobeiyf/ava_downloader
  - Academic Torrents: https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a
- **Paper**: "AVA: A large-scale database for aesthetic visual analysis" (CVPR 2012)

**How to use for keep/trash**:
- Convert quality scores to binary: scores ≥ 6 → "keep", scores < 6 → "trash"
- Or use percentile: top 50% → "keep", bottom 50% → "trash"
- Provides ~125k keep + 125k trash labels

**Advantages**:
- Large scale (250k images)
- Real aesthetic quality labels
- Diverse photography styles
- Well-established dataset

**Download Steps**:
```bash
# Option 1: Use ava_downloader script
git clone https://github.com/mtobeiyf/ava_downloader.git
cd ava_downloader
python download_ava.py

# Option 2: Download from Academic Torrents
# Visit: https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a
# Download AVA.txt (metadata) and images
```

## Option 2: Create Synthetic Labels from Existing Datasets

### A. ImageNet + Quality Heuristics
- **Source**: ImageNet (14M images)
- **Approach**: Use heuristics to create keep/trash labels
  - High resolution + good EXIF → keep
  - Low resolution + blur → trash
  - Object detection confidence → quality proxy
- **Size**: Millions of images (filtered)

### B. Open Images Dataset + Quality Metrics
- **Source**: Open Images V7 (9M images)
- **Approach**: Compute quality metrics (sharpness, exposure, etc.)
  - High quality metrics → keep
  - Low quality metrics → trash
- **Size**: Millions of images
- **Download**: https://storage.googleapis.com/openimages/web/index.html

### C. COCO + Quality Assessment
- **Source**: COCO (330k images)
- **Approach**: Use image quality metrics
  - Well-composed, sharp images → keep
  - Blurry, poorly exposed → trash
- **Download**: https://cocodataset.org/

## Option 3: Active Learning with Current Model

**Approach**: Use your current model to identify uncertain predictions, then label those
- **Process**:
  1. Run model on large unlabeled image collection
  2. Find predictions with probability near 0.5 (uncertain)
  3. Manually label these edge cases
  4. Retrain with expanded dataset
- **Advantage**: Focuses labeling effort on most informative examples

## Option 4: Transfer Learning from Quality Assessment Models

**Approach**: Use pre-trained aesthetic quality models
- **Models**: 
  - NIMA (Neural Image Assessment)
  - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
  - AADB (Aesthetic Attributes Database)
- **Process**:
  1. Download pre-trained quality assessment model
  2. Run on large image collection
  3. Convert quality scores to keep/trash labels
  4. Use as training data

## Option 5: Crowdsourcing / Manual Labeling

**Approaches**:
- **Amazon Mechanical Turk**: Pay workers to label images
- **Internal team**: Have team members label images
- **Gamification**: Make labeling part of app workflow

## Recommended Strategy

### Phase 1: AVA Dataset (Quick Win)
1. Download AVA dataset (~250k images)
2. Convert quality scores to keep/trash (score ≥ 6 → keep)
3. Train model on AVA + your current data
4. **Expected**: +5-10% accuracy improvement

### Phase 2: Active Learning (Ongoing)
1. Use trained model to find uncertain predictions
2. Label 100-200 uncertain images per week
3. Retrain periodically
4. **Expected**: +2-5% over time

### Phase 3: Synthetic Labels (Scale Up)
1. Download Open Images or ImageNet subset
2. Apply quality heuristics to create labels
3. Filter for high-confidence labels
4. Add to training set
5. **Expected**: +3-7% with good heuristics

## Implementation Scripts

1. **`scripts/download_ava_dataset.py`** - AVA metadata preparation
2. **`scripts/convert_ava_to_keep_trash.py`** - Quality scores → keep/trash (to be created)
3. **`scripts/active_learning_sampler.py`** - Find uncertain predictions (to be created)
4. **`scripts/synthetic_label_generator.py`** - Heuristics-based labeling (to be created)

## Dataset Comparison

| Dataset | Size | Labels | Keep/Trash | Effort | ROI |
|---------|------|--------|------------|--------|-----|
| **AVA** | 250k | Quality scores | Convert | Low | ⭐⭐⭐⭐⭐ |
| **Open Images** | 9M | Object labels | Heuristics | Medium | ⭐⭐⭐ |
| **ImageNet** | 14M | Object classes | Heuristics | Medium | ⭐⭐⭐ |
| **COCO** | 330k | Object labels | Heuristics | Medium | ⭐⭐⭐ |
| **Active Learning** | Unlimited | Manual | Manual | High | ⭐⭐⭐⭐ |

## Next Steps

1. **Download AVA dataset** (highest ROI)
   ```bash
   git clone https://github.com/mtobeiyf/ava_downloader.git
   cd ava_downloader
   python download_ava.py
   ```

2. **Convert to keep/trash labels**
   ```bash
   poetry run python scripts/download_ava_dataset.py
   ```

3. **Train on combined dataset** (AVA + current data)

4. **Evaluate improvement**

## References

- **AVA Dataset**: https://github.com/mtobeiyf/ava_downloader
- **AVA Paper**: "AVA: A large-scale database for aesthetic visual analysis" (CVPR 2012)
- **AVA Academic Torrents**: https://academictorrents.com/details/71631f83b11d3d79ebc7486382bff9f6d30f502a
- **Open Images**: https://storage.googleapis.com/openimages/web/index.html
- **ImageNet**: https://www.image-net.org/
- **COCO**: https://cocodataset.org/

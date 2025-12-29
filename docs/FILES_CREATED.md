# Files Created by Photo Derush

This document lists all files and directories created by the Photo Derush application.

## Summary Table

| # | File/Directory | Type | Location | Description |
|---|----------------|------|----------|-------------|
| 1 | `~/.photo_app_config.json` | Config | Home | App settings (last directory, preferences) |
| 2 | `~/.photo-derush-keep-trash-model.joblib` | Model | Home | Main trained model |
| 3 | `~/.photo-derush-keep-trash-model.joblib.calib.joblib` | Model | Home | Model calibrator (if calibration enabled) |
| 4 | `~/baseline_current.joblib` | Model | Home | Baseline model (inference.py) |
| 5 | `~/combined_current.joblib` | Model | Home | Combined model (inference.py) |
| 6 | `~/.photo-derush-xgb-params.json` | Model | Home | Best hyperparameters (tuning.py) |
| 7 | `~/.photo-derush-xgb-cv-results.json` | Model | Home | Cross-validation results (tuning.py) |
| 8 | `~/.photo-derush-cache/` | Cache | Home | Main cache directory |
| 9 | `~/.photo-derush-cache/*.png` | Cache | Home | Thumbnail cache files (one per image) |
| 10 | `~/.photo-derush-cache/duplicate_groups_<hash>.pkl` | Cache | Home | Duplicate grouping cache |
| 11 | `feature_cache.pkl` | Cache | Project Root | Feature extraction cache |
| 12 | `.cache/object_detections.joblib` | Cache | Project | Object detection cache |
| 13 | `.cache/overlays/` | Cache | Project | Overlay images directory |
| 14 | `.cache/final_thumbs/` | Cache | Project | Final thumbnail dumps (debug only) |
| 15 | `.cache/baseline_metrics.json` | Cache | Project | Baseline metrics (benchmark.py) |
| 16 | `.cache/benchmark_results.json` | Cache | Project | Benchmark results (benchmark.py) |
| 17 | `<image_dir>/.ratings_tags.json` | Data | Image Dir | Labels, ratings, tags for each image |
| 18 | `sweeps/combined_search_results.json` | Output | Project | Combined search results (scripts) |
| 19 | `plots/*.png` | Output | Project | Generated plots from various scripts |

## File Categories

### Configuration Files (1)
- **`~/.photo_app_config.json`**: Stores app preferences, last used directory, and settings

### Model Files (6)
All model files are stored in the home directory (`~`):
- Main trained model
- Model calibrator (optional)
- Baseline and combined models (for inference)
- Hyperparameter tuning results

### Cache Files (11)
Cache files are split between:
- **Home directory** (`~/.photo-derush-cache/`): Thumbnails and duplicate groups
- **Project directory** (`.cache/`): Object detections, overlays, benchmarks

### Data Files (1)
- **`<image_dir>/.ratings_tags.json`**: Created in the selected image directory, stores all labels, ratings, and tags

### Output Files (2)
- Script-generated files (plots, search results)

## Notes

- Most files use `~` (home directory) or project-relative paths
- Only **one file** is created in the image directory: `.ratings_tags.json`
- Cache files can be safely deleted (will be regenerated)
- Model files should be preserved (trained models)
- The `.ratings_tags.json` file contains all your labels and should be backed up



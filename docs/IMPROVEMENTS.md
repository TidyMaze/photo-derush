# Future Improvements

> **Note**: Many UI features from earlier roadmaps have been removed to focus on the core ML classification workflow. This document outlines potential future enhancements if needed.

## 🎯 Core ML Workflow Enhancements

### Model Improvements
- **Better Feature Engineering**: Extract more meaningful features from images
- **Active Learning**: Prioritize labeling of uncertain predictions
- **Model Versioning**: Track model performance over time
- **Confidence Calibration**: Improve probability estimates

### Workflow Efficiency
- **Keyboard Navigation**: Arrow keys to move between images
- **Batch Operations**: Label multiple images at once (✅ implemented)
- **Filter Presets**: Quick filters for unlabeled/keep/trash (✅ implemented)
- **Search**: Find images by filename (✅ implemented)

## 🎨 UI Polish (If Needed)

### Visual Improvements
- **Better Thumbnail Quality**: Higher resolution thumbnails
- **Loading Indicators**: Show progress for async operations
- **Status Bar**: Real-time statistics (✅ implemented)

### Information Display
- **EXIF Panel**: Show camera metadata (✅ implemented)
- **Object Detection**: Visualize detected objects (✅ implemented)
- **Prediction Display**: Show model confidence (✅ implemented)

## 🚀 Performance

### Optimization
- **Thumbnail Caching**: Cache thumbnails to disk (✅ implemented)
- **Lazy Loading**: Load images on demand (✅ implemented)
- **Background Processing**: Non-blocking ML operations (✅ implemented)

## 📝 Developer Experience

### Code Quality
- **Type Hints**: Add more type annotations
- **Tests**: Unit tests for core functionality
- **Documentation**: API documentation

### Tooling
- **Linting**: Consistent code style (✅ using isort)
- **Formatting**: Auto-format on save
- **CI/CD**: Automated testing

## Model-Specific Improvements

### High Priority
1. **Better Embeddings**: ResNet50, CLIP, or SigLIP embeddings (expected +2-5% accuracy)
2. **Ensemble Methods**: Combine multiple models (expected +2-4% accuracy)
3. **More Data**: Label more photos, focus on edge cases (expected +5-10% accuracy)

### Medium Priority
4. **Data Augmentation**: Rotation, flip, crop, color jitter (expected +1-3% accuracy)
5. **Class Balancing**: SMOTE, better class weights (expected +1-2% accuracy)
6. **Hyperparameter Tuning**: More extensive Optuna trials (expected +0.5-1.5% accuracy)

### Low Priority
7. **Advanced Feature Engineering**: Polynomial features, domain-specific metrics
8. **Different Architectures**: LightGBM, XGBoost, neural networks
9. **Transfer Learning**: Fine-tune ResNet on photo dataset

## Expected Combined Improvement

If all improvements are implemented:
- **Conservative estimate**: 75.31% → 85-90%
- **Optimistic estimate**: 75.31% → 90-95%

## Implementation Priority

1. **High**: Better embeddings (ResNet50/CLIP/SigLIP)
2. **High**: More labeled data (focus on edge cases)
3. **Medium**: Ensemble methods
4. **Medium**: Data augmentation
5. **Medium**: Class balancing (SMOTE)
6. **Low**: Advanced feature engineering
7. **Low**: Different architectures


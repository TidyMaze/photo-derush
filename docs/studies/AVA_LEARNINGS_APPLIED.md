# AVA Experiments: Key Learnings Applied to Keep/Trash Model

## What We Learned from AVA Experiments

### 1. **Feature Interactions Are Critical** ⭐
- **Finding**: Top 10 features include 4 interactions and 2 ratios
- **Impact**: Interactions capture non-linear relationships (e.g., width × symmetry)
- **Current keep/trash model**: Uses embeddings + handcrafted, but NO interactions
- **Opportunity**: Add feature interactions to improve accuracy

### 2. **Embeddings Provide Significant Boost**
- **Finding**: +2.13% improvement with embeddings (48.88% vs 46.75%)
- **Current keep/trash model**: Already uses ResNet18 embeddings ✅
- **Opportunity**: Try ResNet50 embeddings (expected +2-5% more)

### 3. **Regularization Prevents Overfitting**
- **Finding**: Best AVA model uses:
  - `rsm=0.88` (feature dropout)
  - `subsample=0.85` (row dropout)
  - `l2_leaf_reg=3.0` (L2 regularization)
  - `early_stopping_rounds=200`
- **Current keep/trash model**: Uses basic CatBoost params, may not have these
- **Opportunity**: Add stronger regularization

### 4. **Hyperparameter Tuning Matters**
- **Finding**: Best AVA params: `iterations=2500`, `lr=0.018`, `depth=7`
- **Current keep/trash model**: Uses Optuna-tuned params (good ✅)
- **Opportunity**: Re-tune with AVA-learned regularization params

### 5. **PCA Dimension Selection**
- **Finding**: 128-192 dims work well for embeddings
- **Current keep/trash model**: Uses 128 dims ✅
- **Opportunity**: Try 192 dims for ResNet50

### 6. **Class Imbalance Handling**
- **Finding**: Class weighting didn't help much in AVA (too extreme)
- **Current keep/trash model**: Uses `scale_pos_weight` ✅
- **Opportunity**: Current approach is fine

## How to Apply to Keep/Trash Model

### Priority 1: Add Feature Interactions ⭐ HIGHEST IMPACT
**Current**: 206 features (78 handcrafted + 128 embeddings)
**Add**: ~100 interactions + ~20 ratios = ~326 features

**Expected improvement**: +2-4% accuracy (86.67% → 88-90%)

### Priority 2: Improve Regularization
**Add to CatBoost params**:
- `rsm=0.88` (feature dropout)
- `subsample=0.85` (row dropout)  
- `bootstrap_type="Bernoulli"` (required for subsample)
- `early_stopping_rounds=200`
- `use_best_model=True`

**Expected improvement**: Better generalization, less overfitting

### Priority 3: Try ResNet50 Embeddings
**Current**: ResNet18 (512 dims → 128 PCA)
**Upgrade**: ResNet50 (2048 dims → 192 PCA)

**Expected improvement**: +2-5% accuracy

### Priority 4: Re-tune Hyperparameters
**Include new params in Optuna search**:
- `rsm`: [0.7, 0.95]
- `subsample`: [0.7, 0.95]
- `iterations`: [1000, 3000]
- `early_stopping_rounds`: [100, 300]

**Expected improvement**: +1-2% accuracy

## Implementation Plan

1. **Add feature interactions to training pipeline**
2. **Update CatBoost params with regularization**
3. **Test on current dataset**
4. **If successful, try ResNet50 embeddings**
5. **Re-tune hyperparameters with new features**

## Expected Combined Improvement

**Conservative**: 86.67% → 90-92% accuracy
**Optimistic**: 86.67% → 92-95% accuracy

**Key insight**: Feature interactions are the biggest missing piece!


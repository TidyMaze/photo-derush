# Learning Rate Impact Study

## Summary

**Best Learning Rate Found: 0.07**
- **Accuracy: 89.08%** (vs 84.87% baseline with LR=0.1)
- **Improvement: +4.21 percentage points**
- **Exceeds goal: 88.24%** ✅

## Key Findings

### Without Early Stopping (200 iterations)

| Learning Rate | Accuracy | F1 Score | ROC-AUC | Notes |
|--------------|----------|---------|---------|-------|
| 0.03 | 88.24% | 0.9103 | 0.9142 | Matches original goal |
| **0.07** | **89.08%** | **0.9172** | **0.9218** | **BEST** |
| 0.11 | 89.08% | 0.9193 | 0.9047 | Tied for best |
| 0.10 (current) | 84.87% | 0.8831 | 0.9203 | Baseline |

### With Early Stopping

Early stopping with current validation set size (3-5%) performs worse:
- LR=0.07: 79.83% (vs 89.08% without)
- LR=0.11: 81.51% (vs 89.08% without)

**Conclusion:** Validation set too small for reliable early stopping. Model needs full 200 iterations to reach potential.

## Recommendations

1. **Update learning rate to 0.07** (or 0.11) - improves accuracy by +4.21pp
2. **Keep early stopping disabled** for best performance (or use larger validation set ~10%)
3. **Alternative:** Use LR=0.07 with early stopping only if validation set is ≥10% of training data

## Implementation

Updated `src/training_core.py`:
- Changed default `learning_rate` from 0.1 to 0.07
- Expected improvement: 84.87% → 89.08% (+4.21pp)


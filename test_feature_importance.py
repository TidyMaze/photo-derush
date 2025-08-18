import numpy as np
from ml.personal_learner import PersonalLearner

def test_explain_feature_importance_basic():
    # Simple dataset: feature 0 is always the label, others are noise
    X = np.zeros((20, 3))
    y = np.zeros(20, dtype=int)
    X[:10, 0] = 1
    y[:10] = 1
    # Add some noise to other features
    rng = np.random.default_rng(42)
    X[:, 1:] = rng.normal(0, 0.1, size=(20, 2))
    learner = PersonalLearner(n_features=3)
    importances = learner.train_and_explain(X, y)
    # Should return 3 features, sorted by importance
    assert len(importances) == 3
    assert all(imp[1] >= 0 for imp in importances)
    # Feature 0 should be most important
    assert importances[0][0] == learner.feature_names[0] if learner.feature_names else 'feature_0'
    # Importances should be sorted descending
    assert all(importances[i][1] >= importances[i+1][1] for i in range(len(importances)-1))

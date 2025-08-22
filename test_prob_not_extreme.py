import numpy as np
import pytest
from ml.personal_learner import PersonalLearner

def test_probabilities_not_extreme_after_many_updates():
    rng = np.random.default_rng(123)
    n_features = 20
    learner = PersonalLearner(n_features=n_features)
    # Simulate 60 incremental labels with random feature vectors
    for i in range(60):
        x = rng.normal(size=(1, n_features)).astype(np.float64)
        y = np.array([i % 2], dtype=np.int64)  # alternate labels
        learner.partial_fit(x, y)
    # Predict on 10 random samples
    X_test = rng.normal(size=(10, n_features))
    probs = learner.predict_keep_prob(X_test)
    # With clipping and regularization, they should avoid hard 0/1
    assert np.all(probs > 0.0) and np.all(probs < 1.0), f"Extreme probs found: {probs}"

def test_predict_keep_prob_single_class_raises():
    learner = PersonalLearner(n_features=5)
    X = np.random.normal(size=(10, 5))
    y = np.zeros(10, dtype=int)  # Only one class
    learner.partial_fit(X, y)
    X_test = np.random.normal(size=(2, 5))
    with pytest.raises(ValueError, match=r"predict_proba returned shape.*expected at least 2 columns"):
        learner.predict_keep_prob(X_test)

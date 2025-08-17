import numpy as np
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

import numpy as np
from sklearn.linear_model import SGDClassifier

class PersonalLearner:
    def __init__(self, n_features, classes=[0, 1]):
        self.n_features = n_features
        self.classes = np.array(classes)
        self.model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
        self._is_initialized = False

    def partial_fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if not self._is_initialized:
            self.model.partial_fit(X, y, classes=self.classes)
            self._is_initialized = True
        else:
            self.model.partial_fit(X, y)

    def predict_proba(self, X):
        X = np.array(X)
        if not self._is_initialized:
            # Return 0.5 for all if not trained
            return np.full((len(X), 2), 0.5)
        return self.model.predict_proba(X)

    def predict_keep_prob(self, X):
        proba = self.predict_proba(X)
        # Probability of class 1 (keep)
        return proba[:, 1]

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


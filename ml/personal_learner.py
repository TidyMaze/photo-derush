import numpy as np
from sklearn.linear_model import SGDClassifier
import logging

logger = logging.getLogger(__name__)

class PersonalLearner:
    def __init__(self, n_features, classes=[0, 1]):
        self.n_features = n_features
        self.classes = np.array(classes)
        # Big banner log for new model creation
        logger.info("=" * 72)
        logger.info("[ModelInit] NEW PersonalLearner created n_features=%d classes=%s", n_features, self.classes.tolist())
        logger.info("=" * 72)
        self.model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
        self._is_initialized = False

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        init_before = self._is_initialized
        if not self._is_initialized:
            logger.info("[Learner] Initial partial_fit: %d samples, classes=%s", len(X), np.unique(y))
            self.model.partial_fit(X, y, classes=self.classes)
            self._is_initialized = True
        else:
            logger.info("[Learner] Incremental partial_fit: %d samples, classes=%s", len(X), np.unique(y))
            self.model.partial_fit(X, y)
        logger.info("[Learner] partial_fit complete (was_initialized=%s, now_initialized=%s)", init_before, self._is_initialized)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self._is_initialized:
            logger.info("[Learner] predict_proba called before initialization (%d samples) -> uniform", len(X))
            return np.full((len(X), 2), 0.5)
        logger.info("[Learner] predict_proba on %d samples", len(X))
        return self.model.predict_proba(X)

    def predict_keep_prob(self, X):
        proba = self.predict_proba(X)
        keep_probs = proba[:, 1]
        logger.info("[Learner] predict_keep_prob -> min=%.4f max=%.4f avg=%.4f", float(np.min(keep_probs)), float(np.max(keep_probs)), float(np.mean(keep_probs)))
        return keep_probs

    def save(self, path):
        import joblib
        logger.info("[Learner] Saving model to %s", path)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        logger.info("[Learner] Loading model from %s", path)
        return joblib.load(path)

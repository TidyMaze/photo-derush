import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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
        self.model = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            verbose=1,
            learning_rate='constant',
            eta0=0.001
        )
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._is_initialized = False

    def partial_fit(self, X, y):
        # Backward compatibility: older persisted models may lack scaler
        if not hasattr(self, 'scaler') or self.scaler is None:
            from sklearn.preprocessing import StandardScaler as _SS
            self.scaler = _SS(with_mean=True, with_std=True)
            logger.info("[ModelUpgrade] Added missing scaler to legacy PersonalLearner instance")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Show feature batch as DataFrame before any scaling/training
        try:
            import pandas as _pd
            cols = {f"f{i}": X[:, i] for i in range(X.shape[1])}
            cols['label'] = y
            _df = _pd.DataFrame(cols)
            if len(_df) > 20:
                _display = _df.head(20)
                tail_note = f"\n[... truncated {len(_df)-20} more rows ...]"
            else:
                _display = _df
                tail_note = ""
            logger.info("[Learner][Preview] Training batch (n=%d, n_features=%d):\n%s%s", len(_df), X.shape[1], _display.to_string(index=False), tail_note)
        except Exception as e:  # noqa: PERF203
            logger.debug("[Learner][Preview] Skipped DataFrame preview: %s", e)
        # Update scaler incrementally before model update
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)
        init_before = self._is_initialized
        if not self._is_initialized:
            logger.info("[Learner] Initial partial_fit: %d samples, classes=%s", len(Xs), np.unique(y))
            self.model.partial_fit(Xs, y, classes=self.classes)
            self._is_initialized = True
        else:
            logger.info("[Learner] Incremental partial_fit: %d samples, classes=%s", len(Xs), np.unique(y))
            self.model.partial_fit(Xs, y)
        logger.info("[Learner] partial_fit complete (was_initialized=%s, now_initialized=%s)", init_before, self._is_initialized)

    def predict_proba(self, X):
        # Backward compatibility: ensure scaler exists
        if not hasattr(self, 'scaler') or self.scaler is None:
            from sklearn.preprocessing import StandardScaler as _SS
            self.scaler = _SS(with_mean=True, with_std=True)
            logger.info("[ModelUpgrade] Added missing scaler to legacy PersonalLearner instance (predict path)")
        X = np.asarray(X, dtype=np.float64)
        if not self._is_initialized:
            logger.info("[Learner] predict_proba called before initialization (%d samples) -> uniform", len(X))
            return np.full((len(X), 2), 0.5)
        try:
            Xs = self.scaler.transform(X)
        except Exception as e:  # noqa: PERF203
            logger.warning("[Learner] Scaling failed (%s); using raw features", e)
            Xs = X
        logger.info("[Learner] predict_proba on %d samples", len(Xs))
        return self.model.predict_proba(Xs)

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

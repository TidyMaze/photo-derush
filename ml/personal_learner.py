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
            alpha=0.01,              # stronger regularization to avoid weight blow-up
            learning_rate='adaptive',
            eta0=0.01,
            power_t=0.5,
            early_stopping=False,
        )
        self.scaler = StandardScaler(
            with_mean=True,
            with_std=True
        )
        self._is_initialized = False
        # Will hold canonical feature names (e.g., FEATURE_NAMES) once inferred
        self.feature_names = None
        self._recent_X = []  # minibatch buffer
        self._recent_y = []
        self._buffer_max = 32
        self.proba_clip = (0.01, 0.99)

    def partial_fit(self, X, y):
        # Backward compatibility: ensure new attributes exist on legacy loaded instances
        if not hasattr(self, 'feature_names'):
            self.feature_names = None
        if not hasattr(self, '_recent_X'):
            self._recent_X = []
            self._recent_y = []
            self._buffer_max = 32
        if not hasattr(self, 'proba_clip'):
            self.proba_clip = (0.01, 0.99)
        # Backward compatibility: older persisted models may lack scaler
        if not hasattr(self, 'scaler') or self.scaler is None:
            from sklearn.preprocessing import StandardScaler as _SS
            self.scaler = _SS(with_mean=True, with_std=True)
            logger.info("[ModelUpgrade] Added missing scaler to legacy PersonalLearner instance")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Pad shorter vectors (legacy schema) instead of triggering constant reinitializations
        if X.shape[1] < self.n_features:
            pad_width = self.n_features - X.shape[1]
            X = np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])
        elif X.shape[1] > self.n_features:
            # New larger vectors -> reinitialize model to new dimension once
            logger.info('[Learner][Migration] Expanding feature space %d -> %d', self.n_features, X.shape[1])
            self.n_features = X.shape[1]
            from sklearn.linear_model import SGDClassifier as _SGD
            from sklearn.preprocessing import StandardScaler as _SS
            self.model = _SGD(loss="log_loss", max_iter=1, warm_start=True, verbose=1)
            self.scaler = _SS(with_mean=True, with_std=True)
            self._is_initialized = False
        # Attempt to capture canonical feature names (only if lengths match)
        try:  # noqa: PERF203
            from .features_cv import FEATURE_NAMES as _FN  # local import to avoid hard dependency if module path changes
            if self.feature_names is None and self.n_features == len(_FN):
                self.feature_names = list(_FN)
        except Exception:  # noqa: PERF203
            pass
        # Buffer update (store original post-padding X,y)
        for row, lbl in zip(X, y):
            if len(self._recent_X) >= self._buffer_max:
                self._recent_X.pop(0); self._recent_y.pop(0)
            self._recent_X.append(row.copy())
            self._recent_y.append(int(lbl))
        buf_X = np.asarray(self._recent_X, dtype=np.float64)
        buf_y = np.asarray(self._recent_y, dtype=np.int64)
        # Show feature batch as DataFrame before any scaling/training
        try:  # noqa: PERF203
            import pandas as _pd
            names = self.feature_names if (self.feature_names and len(self.feature_names) == buf_X.shape[1]) else [f"f{i}" for i in range(buf_X.shape[1])]
            cols = {names[i]: buf_X[:, i] for i in range(buf_X.shape[1])}
            cols['label'] = buf_y
            _df = _pd.DataFrame(cols)
            if len(_df) > 20:
                _display = _df.tail(20)  # show most recent
                tail_note = f"\n[... kept only last 20 of buffer size {len(_df)} ...]"
            else:
                _display = _df
                tail_note = ""
            logger.info("[Learner][Preview] Buffer batch (n=%d, n_features=%d):\n%s%s", len(_df), buf_X.shape[1], _display.to_string(index=False), tail_note)
        except Exception as e:  # noqa: PERF203
            logger.info("[Learner][Preview] Skipped DataFrame preview: %s", e)
        # Update scaler with buffer then fit model on buffer
        self.scaler.partial_fit(buf_X)
        Xs = self.scaler.transform(buf_X)
        init_before = self._is_initialized
        if not self._is_initialized:
            logger.info("[Learner] Initial partial_fit (buffer size %d) classes=%s", len(Xs), np.unique(buf_y))
            self.model.partial_fit(Xs, buf_y, classes=self.classes)
            self._is_initialized = True
        else:
            logger.info("[Learner] Incremental partial_fit (buffer size %d)", len(Xs))
            self.model.partial_fit(Xs, buf_y)
        logger.info("[Learner] partial_fit complete (was_initialized=%s, now_initialized=%s)", init_before, self._is_initialized)
        return

    def predict_proba(self, X):
        if not hasattr(self, 'feature_names'):
            self.feature_names = None
        if not hasattr(self, '_recent_X'):
            # Legacy instance before buffer introduction
            self._recent_X = []
            self._recent_y = []
            self._buffer_max = 32
        if not hasattr(self, 'proba_clip'):
            self.proba_clip = (0.01, 0.99)
        # Backward compatibility: ensure scaler exists
        if not hasattr(self, 'scaler') or self.scaler is None:
            from sklearn.preprocessing import StandardScaler as _SS
            self.scaler = _SS(with_mean=True, with_std=True)
            logger.info("[ModelUpgrade] Added missing scaler to legacy PersonalLearner instance (predict path)")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] < self.n_features:
            pad_width = self.n_features - X.shape[1]
            logger.warning('[Learner][Predict] Received lower-dim features (%d) than model (%d); padding with zeros', X.shape[1], self.n_features)
            X = np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])
        elif X.shape[1] > self.n_features:
            logger.warning('[Learner][Predict] Received higher-dim features (%d) than model (%d); truncating', X.shape[1], self.n_features)
            X = X[:, :self.n_features]
        if not self._is_initialized:
            logger.info('[Learner] predict_proba called before initialization (%d samples) -> uniform', len(X))
            return np.full((len(X), 2), 0.5)
        try:
            Xs = self.scaler.transform(X)
        except Exception as e:  # noqa: PERF203
            logger.warning('[Learner] Scaling failed (%s); using raw features', e)
            Xs = X
        logger.info('[Learner] predict_proba on %d samples', len(Xs))
        probs = self.model.predict_proba(Xs)
        # clip probabilities to avoid extreme 0/1 saturation
        low, high = self.proba_clip
        probs = np.clip(probs, low, high)
        return probs

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

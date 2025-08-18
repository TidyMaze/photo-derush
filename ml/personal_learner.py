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
        # Attempt to infer feature names immediately (covers full_retrain path which bypasses partial_fit)
        try:  # noqa: PERF203
            from ml.features_cv import FEATURE_NAMES as _CV_FN  # light import
            if n_features == len(_CV_FN):
                self.feature_names = list(_CV_FN)
            else:
                try:
                    from ml.features import all_feature_names as _ALL_FN
                    all_full = _ALL_FN(include_strings=False)
                    if n_features == len(all_full):
                        self.feature_names = list(all_full)
                except Exception:  # noqa: PERF203
                    pass
        except Exception:  # noqa: PERF203
            pass
        self._recent_X = []  # minibatch buffer
        self._recent_y = []
        self._buffer_max = 32
        self.proba_clip = (0.01, 0.99)
        self.last_retrain_loss_curve = []  # list of per-epoch losses
        self.last_retrain_early_stopped = False
        self.last_retrain_epochs_run = 0

    def full_retrain(self, X, y):
        """Full retrain (scaler + model) on entire dataset X,y using iterative SGD epochs.
        Early stopping based on log-loss improvement. Stores per-epoch losses in last_retrain_loss_curve.
        Returns self."""
        import numpy as _np
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.linear_model import SGDClassifier as _SGD
        from sklearn.metrics import log_loss as _log_loss
        X_arr = _np.asarray(X, dtype=_np.float64)
        y_arr = _np.asarray(y, dtype=_np.int64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.size == 0:
            logger.info('[Learner][FullRetrain] Empty dataset -> skip')
            return self
        if X_arr.shape[1] != self.n_features:
            logger.info('[Learner][FullRetrain] Feature dim change %d -> %d', self.n_features, X_arr.shape[1])
            self.n_features = X_arr.shape[1]
        # If feature_names not yet assigned, attempt now (important for EXIF features visibility)
        if self.feature_names is None:
            try:  # noqa: PERF203
                from ml.features import all_feature_names as _ALL_FN
                all_full = _ALL_FN(include_strings=False)
                if len(all_full) == self.n_features:
                    self.feature_names = list(all_full)
                else:
                    from ml.features_cv import FEATURE_NAMES as _CV_FN
                    if len(_CV_FN) == self.n_features:
                        self.feature_names = list(_CV_FN)
            except Exception:  # noqa: PERF203
                pass
        # --- New: log dataset snapshot as DataFrame prior to training ---
        try:  # noqa: PERF203
            import pandas as _pd
            if self.feature_names is not None and len(self.feature_names) == X_arr.shape[1]:
                cols = list(self.feature_names)
            else:
                cols = [f'f{i}' for i in range(X_arr.shape[1])]
            _df = _pd.DataFrame(X_arr, columns=cols)
            _df['label'] = y_arr
            # Limit very large outputs to avoid log spam
            MAX_ROWS_LOG = 5
            shown_df = _df if len(_df) <= MAX_ROWS_LOG else _df.head(MAX_ROWS_LOG)
            logger.info('[Learner][Dataset][Preview] rows=%d cols=%d (showing %d)\n%s', len(_df), X_arr.shape[1], len(shown_df), shown_df.to_string(index=False))
            if len(_df) > MAX_ROWS_LOG:
                logger.info('[Learner][Dataset][Preview] (truncated additional %d rows)', len(_df) - MAX_ROWS_LOG)
        except Exception as e:  # noqa: PERF203
            logger.info('[Learner][Dataset] Failed to build DataFrame preview: %s', e)
        # ---------------------------------------------------------------
        # Reset scaler & model
        self.scaler = _SS(with_mean=True, with_std=True)
        self.scaler.fit(X_arr)
        Xs = self.scaler.transform(X_arr)
        # Fresh SGD with single-epoch step; we'll iterate manually
        self.model = _SGD(
            loss='log_loss',
            max_iter=1,
            warm_start=True,
            alpha=0.01,
            learning_rate='adaptive',
            eta0=0.001,
            power_t=0.5,
            early_stopping=False,
            verbose=0,
            shuffle=True,
            random_state=42,
        )
        max_epochs = 1000  # higher cap; early stopping will usually cut earlier
        patience = 10
        min_delta = 1e-4
        self.last_retrain_loss_curve = []
        self.last_retrain_early_stopped = False
        self.last_retrain_epochs_run = 0
        best_loss = float('inf')
        epochs_since_improve = 0
        for epoch in range(max_epochs):
            if epoch == 0:
                self.model.partial_fit(Xs, y_arr, classes=self.classes)
            else:
                self.model.partial_fit(Xs, y_arr)
            try:
                probs = self.model.predict_proba(Xs)
                eps = 1e-9
                probs = _np.clip(probs, eps, 1 - eps)
                # Manual log loss (cross-entropy) to avoid potential segfaults in sklearn.log_loss on tiny datasets
                # Equivalent to sklearn's log_loss for binary labels 0/1 with provided probs.
                if len(y_arr) >= 1:
                    chosen = probs[_np.arange(len(y_arr)), y_arr]
                    loss = float(-_np.mean(_np.log(chosen)))
                else:
                    loss = float('nan')
            except Exception as e:  # noqa: PERF203
                loss = float('nan')
                logger.info('[Learner][FullRetrain] Loss computation failed epoch=%d: %s', epoch+1, e)
            self.last_retrain_loss_curve.append(loss)
            self.last_retrain_epochs_run = epoch + 1
            logger.info('[Learner][FullRetrain][Epoch %d/%d] logloss=%.4f', epoch+1, max_epochs, loss)
            # Early stopping logic (only if loss is finite)
            if loss == loss:  # not NaN
                if loss + min_delta < best_loss:
                    best_loss = loss
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= patience:
                        self.last_retrain_early_stopped = True
                        logger.info('[Learner][FullRetrain] Early stopping at epoch %d (best_loss=%.4f)', epoch+1, best_loss)
                        break
        # Update recent buffer
        self._recent_X = [row.copy() for row in Xs[-self._buffer_max:]]
        self._recent_y = [int(lbl) for lbl in y_arr[-self._buffer_max:]]
        self._is_initialized = True
        final_loss = self.last_retrain_loss_curve[-1] if self.last_retrain_loss_curve else float('nan')
        logger.info('[Learner][FullRetrain] Finished epochs=%d early_stopped=%s samples=%d n_features=%d final_loss=%.4f',
                    self.last_retrain_epochs_run, self.last_retrain_early_stopped, len(Xs), self.n_features, final_loss)
        return self

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
        # Try to set feature_names to the full list (including EXIF) if available
        try:
            from ml.features import all_feature_names as _ALL_FN
            all_names = _ALL_FN(include_strings=False)
            if self.feature_names is None and self.n_features == len(all_names):
                self.feature_names = list(all_names)
        except Exception:
            pass
        # Buffer update (store original post-padding X,y)
        for row, lbl in zip(X, y):
            if len(self._recent_X) >= self._buffer_max:
                self._recent_X.pop(0); self._recent_y.pop(0)
            self._recent_X.append(row.copy())
            self._recent_y.append(int(lbl))
        buf_X = np.asarray(self._recent_X, dtype=np.float64)
        buf_y = np.asarray(self._recent_y, dtype=np.int64)
        # Lightweight preview (avoid heavy pandas import to prevent segfaults in constrained env)
        try:
            sample_preview = ' '.join(f"f{i}={buf_X[-1,i]:.4g}" for i in range(min(6, buf_X.shape[1]))) if buf_X.size else ''
            logger.info("[Learner][PreviewLite] buffer_size=%d n_features=%d last_sample: %s label=%s", len(buf_X), buf_X.shape[1], sample_preview, buf_y[-1] if buf_y.size else 'NA')
        except Exception:
            pass
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

    def explain_feature_importance(self, top_n=None):
        """Return a sorted list of (feature, importance) pairs after training."""
        if not hasattr(self.model, 'coef_'):
            logger.warning("[Learner][Explain] Model is not trained yet; no coefficients available.")
            return []
        # Late binding of feature names if still missing (covers legacy saved models)
        if self.feature_names is None:
            try:  # noqa: PERF203
                from ml.features import all_feature_names as _ALL_FN
                all_full = _ALL_FN(include_strings=False)
                if len(all_full) == getattr(self.model.coef_, 'shape', [None, None])[1]:
                    self.feature_names = list(all_full)
                else:
                    from ml.features_cv import FEATURE_NAMES as _CV_FN
                    if len(_CV_FN) == getattr(self.model.coef_, 'shape', [None, None])[1]:
                        self.feature_names = list(_CV_FN)
            except Exception:  # noqa: PERF203
                pass
        importances = np.abs(self.model.coef_)
        # For binary classification, coef_ shape is (1, n_features)
        if importances.ndim == 2 and importances.shape[0] == 1:
            importances = importances[0]
        if self.feature_names is not None and len(self.feature_names) == len(importances):
            features = self.feature_names
        else:
            features = [f"feature_{i}" for i in range(len(importances))]
        importance_pairs = list(zip(features, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            importance_pairs = importance_pairs[:top_n]
        logger.info("[Learner][Explain] Feature importances: %s", importance_pairs)
        return importance_pairs

    def train_and_explain(self, X, y, top_n=None):
        """Train the model and return sorted feature importances."""
        self.full_retrain(X, y)
        return self.explain_feature_importance(top_n=top_n)

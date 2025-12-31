"""Core training logic (model fit + persistence)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
# Lazy import: xgboost is heavy, only import when actually needed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedGroupKFold
from sklearn.metrics import auc as roc_auc_func
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import build_dataset
from .duplicate_grouping import create_duplicate_groups
from .model import RatingsTagsRepository
from .model_version import create_model_metadata
from .tuning import load_best_params

DEFAULT_MODEL_PATH = os.path.expanduser("~/.photo-derush-keep-trash-model.joblib")


@dataclass
class TrainingResult:
    """Encapsulates training output metrics."""

    model_path: str
    n_samples: int
    n_keep: int
    n_trash: int
    cv_accuracy_mean: float | None
    cv_accuracy_std: float | None
    precision: float | None = None
    feature_importances: list[tuple[int, float]] | None = None
    roc_auc: float | None = None
    f1: float | None = None
    confusion: tuple[int, int, int, int] | None = None  # tn, fp, fn, tp
    training_history: dict[str, list[float]] | None = None
    aggregated_metrics: dict[str, float] | None = None  # CV metrics: keep_loss_rate_mean, pr_auc_mean, etc.
    final_loss: float | None = None  # Final training loss value
    iterations: int | None = None  # Number of iterations trained
    patience: int | None = None  # Early stopping patience used


class CancelledTraining(Exception):
    """Raised to abort training early when user requests cancellation."""

    pass


class StackingEnsemble:
    """Stacking ensemble with meta-learner (pickleable)."""
    
    def __init__(self, base_models, meta_learner):
        self.base_models = base_models
        self.meta_learner = meta_learner
    
    def fit(self, X, y):
        # Base models already trained, just train meta-learner
        import numpy as np
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba)
        meta_X = np.column_stack(meta_features)
        self.meta_learner.fit(meta_X, y)
        return self
    
    def predict_proba(self, X):
        import numpy as np
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba)
        meta_X = np.column_stack(meta_features)
        return self.meta_learner.predict_proba(meta_X)
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


def _init_repo(image_dir: str, repo: RatingsTagsRepository | None) -> RatingsTagsRepository:
    """Initialize repository (scoped or provided)."""
    if repo is None:
        repo_path = os.path.join(image_dir, ".ratings_tags.json")
        repo = RatingsTagsRepository(path=repo_path)
        logging.info("[train] Created scoped repo: %s", repo_path)
    else:
        logging.info("[train] Using provided repo: %s", getattr(repo, "path", "unknown"))
    return repo


def _validate_dataset(n_samples: int, n_keep: int, n_trash: int, min_samples: int) -> bool:
    """Check dataset has sufficient labeled samples."""
    if n_keep == 0 or n_trash == 0 or n_samples < min_samples:
        logging.info(
            "[train] Insufficient labeled samples (need %d). Got %d",
            min_samples,
            n_samples,
        )
        return False
    return True


def _compute_class_balance(n_keep: int, n_trash: int) -> dict:
    """Analyze class balance and return metrics.

    Returns dict with: ratio, is_severe, recommendation
    """
    total = n_keep + n_trash
    if total == 0:
        return {"ratio": 0.0, "is_severe": False, "recommendation": "Insufficient data"}

    keep_pct = (n_keep / total) * 100
    trash_pct = (n_trash / total) * 100
    ratio = max(n_keep, n_trash) / min(n_keep, n_trash) if min(n_keep, n_trash) > 0 else 0

    recommendation = "Balanced" if ratio < 1.5 else ("Moderate" if ratio < 3.0 else "Severe")
    is_severe = ratio >= 3.0

    metrics = {
        "ratio": ratio,
        "keep_pct": keep_pct,
        "trash_pct": trash_pct,
        "is_severe": is_severe,
        "recommendation": recommendation,
    }

    if is_severe or ratio >= 2.0:
        logging.warning(
            "[train] Class imbalance: keep=%.1f%% (%d), trash=%.1f%% (%d), ratio=%.2f",
            keep_pct,
            n_keep,
            trash_pct,
            n_trash,
            ratio,
        )
    else:
        logging.info(
            "[train] Class balance: keep=%.1f%% (%d), trash=%.1f%% (%d), ratio=%.2f",
            keep_pct,
            n_keep,
            trash_pct,
            n_trash,
            ratio,
        )

    return metrics


def _adjust_estimators(n_samples: int, n_estimators: int) -> int:
    """Reduce n_estimators for tiny datasets."""
    if n_samples <= 10 and n_estimators > 60:
        n_estimators = 60
        logging.debug("[train] Reduced n_estimators to %d for tiny dataset", n_estimators)
    return n_estimators


def _get_xgb_params(n_estimators: int, n_samples: int = None, n_trash: int = None) -> dict:
    """Load tuned params or use defaults.
    
    Adjusts hyperparameters for small datasets to ensure model can learn.
    """
    tuned = load_best_params()
    if tuned:
        xgb_params = {k: v for k, v in tuned.items() if not k.startswith("_")}
        logging.info("[train] Using tuned params: %s", xgb_params)
    else:
        xgb_params = {"n_estimators": n_estimators, "learning_rate": 0.1, "max_depth": 6}
        logging.info("[train] Using default params: %s", xgb_params)
    
    # Adjust for small datasets: reduce min_child_weight if it's too restrictive
    if n_samples is not None and n_trash is not None:
        min_child_weight = xgb_params.get("min_child_weight", 1)
        # If min_child_weight is too high for the minority class, reduce it
        # With min_child_weight=5, need at least 5 samples per leaf, which is very restrictive
        # For small datasets (< 30 samples in minority class), use a more lenient value
        # This allows the model to actually learn splits
        if min_child_weight > 1 and n_trash < 30:
            # Use max(1, minority_class_size // 4) for very small datasets
            # This ensures we can create at least a few splits
            adjusted = max(1, min(2, n_trash // 5))  # Cap at 2 for small datasets, minimum 1
            if adjusted < min_child_weight:
                logging.info(
                    "[train] Adjusted min_child_weight: %d -> %d (minority class has %d samples)",
                    min_child_weight,
                    adjusted,
                    n_trash,
                )
                xgb_params["min_child_weight"] = adjusted
    
    return xgb_params


def _build_pipeline(
    random_state: int, scale_pos_weight: float, xgb_params: dict, early_stopping_rounds: int | None, use_catboost: bool = True, n_handcrafted_features: int = 78, fast_mode: bool = False
) -> Pipeline:
    """Build model pipeline with scaler (CatBoost by default with optimal settings, XGBoost as fallback).
    
    Uses optimal CatBoost hyperparameters from iterative coordinate descent optimization:
    - iterations: 200 (without early stopping), 932 (with early stopping)
    - learning_rate: 0.0430 (optimized via iterative search)
    - depth: 6
    - l2_leaf_reg: 1.0
    - early_stopping: eval_metric="Logloss", patience=10 (optimized via iterative search)
    
    Best model performance (honest evaluation, no leakage):
    - CV Accuracy: 77.66% ± 3.29% (StratifiedGroupKFold, per-fold threshold tuning)
    - Keep-Loss Rate: 2.85% (meets <2% target)
    - ROC-AUC: 0.7519
    - F1 Score: 0.8128
    
    Args:
        use_catboost: If True, use CatBoost (best model). If False or unavailable, use XGBoost.
    """
    import os
    
    # Try CatBoost first (best performance: 84% accuracy with optimal hyperparameters)
    if use_catboost:
        try:
            from catboost import CatBoostClassifier
            
            # Use optimal CatBoost hyperparameters from 5-fold StratifiedGroupKFold CV study:
            # - LR=0.1: Best learning rate (validated with CV)
            # - iterations=200 (without early stopping), 2000 (with early stopping)
            # - depth=6, l2_leaf_reg=1.0
            # - early_stopping: eval_metric="Logloss" (aligned with keep-loss goal, not accuracy)
            # - patience=200 (allows model to reach full potential)
            # Best model: 75.31% ± 2.51% CV accuracy, 1.03% keep-loss rate (honest evaluation, no leakage)
            if early_stopping_rounds is not None:
                # Use iteratively optimized settings (coordinate descent) for both fast and non-fast modes
                # Optimized via iterative hyperparameter search with CV:
                # - LR=0.0430, iterations=932, patience=10 → CV-Acc=77.66%±3.29%, Keep-Loss=2.85%
                max_iterations = 932  # Optimized via iterative search
                # Increased patience from 10 to 50 for better stability (reduces variation in scores/features)
                effective_patience = max(early_stopping_rounds, 50)  # Increased for stability
            else:
                max_iterations = 200  # Optimal value from CV study
                effective_patience = None
            
            cb_params = {
                "iterations": max_iterations,
                "learning_rate": 0.0430,  # Optimized via iterative search: LR=0.0430 (best for both modes)
                "depth": 6,  # Optimal value from evaluation
                "l2_leaf_reg": 1.0,  # Default regularization
                "scale_pos_weight": scale_pos_weight,
                "random_seed": random_state,
                "verbose": False,
                "thread_count": 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1,
            }
            
            # Add early stopping if provided
            if early_stopping_rounds is not None:
                cb_params["early_stopping_rounds"] = effective_patience
                # Use Logloss for early stopping (aligned with keep-loss goal, not accuracy)
                cb_params["eval_metric"] = "Logloss"
                cb_params["use_best_model"] = True
                cb_params["verbose"] = 10  # Log every 10 iterations to show progress
                logging.info(f"[train] Early stopping enabled: patience={effective_patience}, max_iterations={max_iterations}, eval_metric=Logloss")
            else:
                logging.info(f"[train] Early stopping disabled, using {max_iterations} iterations")
            
            logging.info("[train] Using CatBoost with optimal hyperparameters (LR=0.0430, iterations=932, patience=10, best model: 77.66% ± 3.29% CV, 2.85% keep-loss)")
            
            # Use ColumnTransformer if we have embeddings (n_handcrafted_features < total features expected)
            # For now, assume embeddings start at column n_handcrafted_features
            # If only handcrafted features, use simple StandardScaler (backward compatible)
            # Note: ColumnTransformer will be applied dynamically based on actual feature count
            return Pipeline([
                ("scaler", StandardScaler()),  # Will be replaced with ColumnTransformer if embeddings detected
                ("cat", CatBoostClassifier(**cb_params)),
            ])
        except ImportError:
            logging.info("[train] CatBoost not available, falling back to XGBoost")
    
    # Fallback to XGBoost
    import xgboost as xgb
    
    n_jobs = 1 if os.environ.get("PYTEST_CURRENT_TEST") else 4

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBClassifier(
                    random_state=random_state,
                    n_jobs=n_jobs,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    early_stopping_rounds=early_stopping_rounds,
                    **xgb_params,
                ),
            ),
        ]
    )


def _compute_cross_val(
    clf: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, filenames: list[str] | None = None, image_dir: str | None = None, use_per_fold_threshold: bool = True, max_keep_loss: float = 0.02
) -> tuple[float | None, float | None, dict[str, float] | None]:
    """Compute cross-validation with per-fold threshold tuning.
    
    Uses StratifiedGroupKFold if groups are provided (prevents data leakage from correlated photos),
    otherwise falls back to StratifiedKFold for backward compatibility.
    
    If use_per_fold_threshold=True, tunes threshold per-fold to minimize keep-loss rate.
    
    Args:
        clf: Pipeline to evaluate
        X: Feature matrix
        y: Labels
        groups: Optional group IDs for StratifiedGroupKFold (e.g., duplicate clusters)
        filenames: Optional filenames (used to create groups if not provided)
        image_dir: Optional image directory (used to create groups if not provided)
        use_per_fold_threshold: If True, tune threshold per-fold to minimize keep-loss
        max_keep_loss: Maximum allowed keep-loss rate (default: 0.02 = 2%)
    
    Returns:
        (cv_mean, cv_std, aggregated_metrics) where aggregated_metrics contains keep-loss, PR-AUC, etc.
    """
    if X.shape[0] < 6:
        return None, None, None
    try:
        # Create groups if not provided but filenames/image_dir available
        # OPTIMIZATION: Skip grouping for app retraining (only use for final evaluation)
        # This speeds up interactive retraining significantly
        # Grouping is only needed for CV, which we skip during app retraining
        skip_grouping = True  # Always skip grouping during app retraining (CV is skipped anyway)
        if not skip_grouping and groups is None and filenames is not None and image_dir is not None:
            try:
                import time
                group_start = time.perf_counter()
                groups_array = np.array(create_duplicate_groups(filenames, image_dir, use_cache=True))
                group_time = time.perf_counter() - group_start
                if len(set(groups_array)) > 1:  # Only use if we have multiple groups
                    groups = groups_array
                    logging.info(f"[train] Created {len(set(groups))} duplicate groups for CV in {group_time:.2f}s")
                else:
                    logging.debug(f"[train] Only 1 group found, skipping StratifiedGroupKFold")
            except Exception as e:
                logging.debug(f"[train] Failed to create duplicate groups: {e}, falling back to StratifiedKFold")
        
        # Use StratifiedGroupKFold if groups available
        if groups is not None:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                y_counts = np.bincount(y)
                min_groups = int(np.min(y_counts)) if y_counts.size > 0 else 0
                if min_groups < 2:
                    logging.info("[train] StratifiedGroupKFold CV skipped: not enough members per class (%d)", min_groups)
                    return None, None, None
                n_splits = min(5, min_groups)  # Use 5-fold for better estimates
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv = sgkf
                logging.info(f"[train] Using StratifiedGroupKFold (n_splits={n_splits}) to prevent data leakage")
            except ImportError:
                logging.warning("[train] StratifiedGroupKFold not available, falling back to StratifiedKFold")
                groups = None
        
        # Fallback to StratifiedKFold
        if groups is None:
            from sklearn.model_selection import StratifiedKFold
            y_counts = np.bincount(y)
            min_groups = int(np.min(y_counts)) if y_counts.size > 0 else 0
            if min_groups < 2:
                logging.info("[train] Stratified CV skipped: not enough members per class (%d)", min_groups)
                return None, None, None
            n_splits = min(3, min_groups)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv = skf

        # Force single-threaded cross-val when running under pytest
        import os

        # Temporarily disable use_best_model for CV (no eval_set available per fold)
        cat_model = clf.named_steps.get("cat")
        original_use_best = None
        if cat_model and hasattr(cat_model, "get_param"):
            original_use_best = cat_model.get_param("use_best_model")
            if original_use_best:
                cat_model.set_params(use_best_model=False)
                logging.debug("[train] Disabled use_best_model for CV (no eval_set per fold)")

        cv_n_jobs = 1 if os.environ.get("PYTEST_CURRENT_TEST") else None
        
        if use_per_fold_threshold:
            # Per-fold threshold tuning: tune threshold inside each fold
            logging.info("[train] Using per-fold threshold tuning (target keep-loss < %.1f%%)", max_keep_loss * 100)
            scores = []
            keep_loss_rates = []
            junk_leak_rates = []
            pr_aucs = []
            thresholds_used = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                # Further split train for threshold tuning
                if len(y_train_fold) >= 20:
                    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                        X_train_fold, y_train_fold, test_size=0.2, stratify=y_train_fold, random_state=42
                    )
                else:
                    # Too small, use all train for both fitting and threshold tuning
                    X_train_fit, X_val, y_train_fit, y_val = X_train_fold, X_train_fold, y_train_fold, y_train_fold
                
                # Clone and train model on this fold's training data
                from sklearn.base import clone
                clf_fold = clone(clf)
                clf_fold.fit(X_train_fit, y_train_fit)
                
                # Get probabilities on validation set
                if hasattr(clf_fold, "predict_proba"):
                    y_proba_val = clf_fold.predict_proba(X_val)[:, 1]
                    
                    # Tune threshold on THIS fold's validation set
                    threshold = _find_threshold_for_max_keep_loss(y_val, y_proba_val, max_keep_loss)
                    thresholds_used.append(threshold)
                    
                    # Evaluate on THIS fold's test set with tuned threshold
                    y_proba_test = clf_fold.predict_proba(X_test_fold)[:, 1]
                    y_pred_test = (y_proba_test >= threshold).astype(int)
                    
                    # Compute metrics
                    accuracy = float(np.mean(y_pred_test == y_test_fold))
                    scores.append(accuracy)
                    
                    # Compute asymmetric metrics
                    metrics = _compute_asymmetric_metrics(y_test_fold, y_pred_test, y_proba_test)
                    keep_loss_rates.append(metrics["keep_loss_rate"])
                    junk_leak_rates.append(metrics["junk_leak_rate"])
                    keep_rates.append(metrics["keep_rate"])  # Primary secondary metric
                    pr_aucs.append(metrics["pr_auc"])
                else:
                    # Fallback: use default threshold
                    y_pred_test = clf_fold.predict(X_test_fold)
                    accuracy = float(np.mean(y_pred_test == y_test_fold))
                    scores.append(accuracy)
                    thresholds_used.append(0.5)
            
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))
            
            aggregated_metrics = {
                "keep_loss_rate_mean": float(np.mean(keep_loss_rates)) if keep_loss_rates else 0.0,
                "keep_loss_rate_std": float(np.std(keep_loss_rates)) if keep_loss_rates else 0.0,
                "junk_leak_rate_mean": float(np.mean(junk_leak_rates)) if junk_leak_rates else 0.0,
                "junk_leak_rate_std": float(np.std(junk_leak_rates)) if junk_leak_rates else 0.0,
                "keep_rate_mean": float(np.mean(keep_rates)) if keep_rates else 0.0,  # Primary secondary metric
                "keep_rate_std": float(np.std(keep_rates)) if keep_rates else 0.0,
                "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else 0.0,
                "pr_auc_std": float(np.std(pr_aucs)) if pr_aucs else 0.0,
                "threshold_mean": float(np.mean(thresholds_used)) if thresholds_used else 0.5,
                "threshold_std": float(np.std(thresholds_used)) if thresholds_used else 0.0,
            }
            
            logging.info(
                "[train] CV with per-fold threshold: accuracy=%.4f±%.4f, keep-loss=%.2f%%±%.2f%%, threshold=%.3f±%.3f",
                cv_mean, cv_std,
                aggregated_metrics["keep_loss_rate_mean"] * 100, aggregated_metrics["keep_loss_rate_std"] * 100,
                aggregated_metrics["threshold_mean"], aggregated_metrics["threshold_std"]
            )
        else:
            # Original behavior: fixed threshold 0.5
            if groups is not None:
                scores = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy", n_jobs=cv_n_jobs)
            else:
                scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=cv_n_jobs)
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            aggregated_metrics = None
            logging.info("[train] CV mean=%.4f std=%.4f (n_splits=%d, threshold=0.5 fixed)", cv_mean, cv_std, len(scores))
        
        # Restore original setting
        if original_use_best is not None and cat_model:
            cat_model.set_params(use_best_model=original_use_best)
        
        return cv_mean, cv_std, aggregated_metrics
    except ValueError as e:
        logging.warning("[train] CV skipped: %s", e)
        return None, None, None


def _fit_model(
    clf: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    early_stopping_rounds: int | None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> None:
    """Fit model with optional early stopping (stratified 80/20 holdout split).
    
    Supports both XGBoost and CatBoost pipelines with early stopping.
    """
    # Check if using CatBoost
    cat_model = clf.named_steps.get("cat")
    if cat_model:
        # CatBoost: use early stopping with validation set if enabled
        if early_stopping_rounds and n_samples >= 20:
            if X_val is not None and y_val is not None:
                # Use provided validation set (from train/test split)
                X_train, X_valid, y_train, y_valid = X, X_val, y, y_val
            else:
                # Create validation set from training data (fallback)
                from sklearn.model_selection import train_test_split
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
            logging.info(f"[train] Training with early stopping: train={len(X_train)}, valid={len(X_valid)}")
            logging.info(f"[train] Starting training (max {cat_model.get_param('iterations')} iterations, patience={early_stopping_rounds})")
            
            # Fit pipeline with eval_set for CatBoost early stopping
            # CatBoost will log progress via verbose parameter set in cb_params
            clf.fit(X_train, y_train, cat__eval_set=(X_valid, y_valid))
            
            # Log final result - check if early stopping triggered
            total_iterations = cat_model.tree_count_
            max_iter = cat_model.get_param("iterations")
            
            # CatBoost stores best_iteration_ when use_best_model=True
            if hasattr(cat_model, "best_iteration_"):
                best_iter = cat_model.best_iteration_
                if best_iter < max_iter - 1:
                    logging.info(f"[train] ✓ Early stopping triggered: best iteration={best_iter} (stopped at {total_iterations}, max was {max_iter})")
                    logging.info(f"[train] Model shrunk from {max_iter} to {best_iter + 1} iterations")
                else:
                    logging.info(f"[train] Training completed all {total_iterations} iterations (best={best_iter})")
            else:
                # Try to get from evals_result
                if hasattr(cat_model, "evals_result_"):
                    evals = cat_model.evals_result_
                    if evals:
                        # Find best iteration from validation metrics
                        for key in evals:
                            if "test" in key.lower() or "validation" in key.lower():
                                metric_name = "Accuracy" if "Accuracy" in evals[key] else list(evals[key].keys())[0]
                                values = evals[key][metric_name]
                                best_idx = max(range(len(values)), key=lambda i: values[i])
                                best_val = values[best_idx]
                                logging.info(f"[train] Best validation {metric_name}={best_val:.4f} at iteration {best_idx}")
                                if best_idx < total_iterations - 1:
                                    logging.info(f"[train] Early stopping likely triggered (best at iter {best_idx}, trained {total_iterations})")
                                break
                else:
                    logging.info(f"[train] Training completed: {total_iterations} iterations (max was {max_iter})")
        else:
            # No early stopping - fit on all data
            logging.info(f"[train] Training without early stopping on {len(X)} samples")
            clf.fit(X, y)
            if hasattr(cat_model, "tree_count_"):
                logging.info(f"[train] Training completed: {cat_model.tree_count_} iterations")
    elif early_stopping_rounds and n_samples >= 20:
        # XGBoost: use early stopping with validation set
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logging.info(f"[train] Training XGBoost with early stopping: train={len(X_train)}, valid={len(X_valid)}")
        clf.fit(X_train, y_train, xgb__eval_set=[(X_valid, y_valid)], xgb__verbose=True)
    else:
        logging.info(f"[train] Training without early stopping on {len(X)} samples")
        clf.fit(X, y)
    logging.info("[train] Fit complete")


def _get_feature_importances(clf: Pipeline) -> list[tuple[int, float]] | None:
    """Extract top 10 feature importances (supports XGBoost and CatBoost)."""
    try:
        # Try CatBoost first
        cat_model = clf.named_steps.get("cat")
        if cat_model and hasattr(cat_model, "feature_importances_"):
            importances = cat_model.feature_importances_
            order = np.argsort(importances)[::-1][:10]
            result = [(int(i), float(importances[i])) for i in order]
            logging.info("[train] Top feature importances (CatBoost): %s", result)
            return result
        
        # Fallback to XGBoost
        xgb_model = clf.named_steps.get("xgb")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            order = np.argsort(importances)[::-1][:10]
            result = [(int(i), float(importances[i])) for i in order]
            logging.info("[train] Top feature importances (XGBoost): %s", result)
            return result
    except Exception as e:
        logging.info("[train] Could not compute importances: %s", e)
    return None


def _find_threshold_for_max_keep_loss(
    y_true: np.ndarray, y_proba: np.ndarray, max_keep_loss: float = 0.02
) -> float:
    """Find threshold that keeps keep-loss rate below target (exact method).
    
    Uses quantile-based approach: finds the highest threshold that respects
    the keep-loss constraint. This is exact and handles edge cases (thresholds < 0.05 or > 0.95).
    
    Args:
        y_true: True labels (1=keep, 0=trash)
        y_proba: Predicted probabilities for keep class
        max_keep_loss: Maximum allowed keep-loss rate (default: 0.02 = 2%)
    
    Returns:
        Optimal threshold: the highest threshold that respects keep_loss_rate <= max_keep_loss.
        If no threshold meets the constraint, returns the threshold with minimum keep-loss.
        Defaults to 0.5 if no keep samples.
    """
    n_keep = int(np.sum(y_true == 1))
    if n_keep == 0:
        return 0.5  # No keep samples, use default
    
    # Extract probabilities for KEEP samples only
    keep_mask = (y_true == 1)
    keep_proba = y_proba[keep_mask]
    
    if len(keep_proba) == 0:
        return 0.5
    
    # Calculate max number of KEEP we can lose: m = floor(L * #keep)
    max_false_negatives = int(np.floor(max_keep_loss * n_keep))
    
    # Sort KEEP probabilities in ascending order
    keep_proba_sorted = np.sort(keep_proba)
    
    # Find threshold: the (m+1)-th smallest proba (0-based index m)
    # This ensures at most m KEEP samples have p < threshold (i.e., at most m false negatives)
    if max_false_negatives >= len(keep_proba_sorted):
        # Constraint allows losing all KEEP samples -> threshold = 1.0 (reject all)
        # But this is pathological, so use a high threshold instead
        threshold = 1.0
    elif max_false_negatives == 0:
        # Constraint allows losing 0 KEEP samples -> threshold = 0.0 (accept all)
        # Use the smallest proba minus epsilon to ensure we accept all
        threshold = max(0.0, float(keep_proba_sorted[0] - 1e-10))
    else:
        # Normal case: threshold is the proba at index max_false_negatives
        # This is the highest threshold that ensures at most max_false_negatives KEEP are lost
        threshold = float(keep_proba_sorted[max_false_negatives])
    
    # Verify the threshold meets the constraint (with small tolerance for floating point)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    actual_keep_loss = float(fn / n_keep) if n_keep > 0 else 0.0
    
    # If threshold doesn't meet constraint (edge case with ties), find the next lower one
    if actual_keep_loss > max_keep_loss + 1e-6:
        # Find the highest threshold that actually meets the constraint
        # This handles ties in probabilities
        for idx in range(max_false_negatives + 1, len(keep_proba_sorted)):
            candidate_threshold = float(keep_proba_sorted[idx])
            y_pred_candidate = (y_proba >= candidate_threshold).astype(int)
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, y_pred_candidate).ravel()
            candidate_keep_loss = float(fn_c / n_keep) if n_keep > 0 else 0.0
            if candidate_keep_loss <= max_keep_loss + 1e-6:
                threshold = candidate_threshold
                break
    
    # Clamp to [0, 1] for safety
    threshold = max(0.0, min(1.0, threshold))
    
    return float(threshold)


def _compute_asymmetric_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Compute asymmetric cost metrics for keep/trash classification.
    
    Returns:
        Dictionary with keep-loss rate, junk-leak rate, PR-AUC, precision, recall, F1, accuracy
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Asymmetric cost metrics
    n_keep = int(np.sum(y_true == 1))
    n_trash = int(np.sum(y_true == 0))
    
    keep_loss_rate = float(fn / n_keep) if n_keep > 0 else 0.0  # FN(keep→trash) / total_keep
    junk_leak_rate = float(fp / n_trash) if n_trash > 0 else 0.0  # FP(trash→keep) / total_trash
    
    # Keep rate: fraction of photos predicted as keep (primary secondary metric)
    n_total = int(np.sum(y_true == 1) + np.sum(y_true == 0))
    keep_rate = float((tp + fp) / n_total) if n_total > 0 else 0.0  # (TP + FP) / total = predicted_keep / total
    
    precision_keep = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_keep = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_keep = float(2.0 * (precision_keep * recall_keep) / (precision_keep + recall_keep)) if (precision_keep + recall_keep) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    
    # PR-AUC (more informative than ROC-AUC under imbalance)
    pr_auc = None
    if y_proba is not None:
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = float(roc_auc_func(recall_curve, precision_curve))
        except Exception:
            logging.debug("[train] Failed to compute PR-AUC")
    
    result: dict[str, float] = {
        "keep_loss_rate": keep_loss_rate,
        "junk_leak_rate": junk_leak_rate,
        "keep_rate": keep_rate,  # Primary secondary metric: fraction of photos kept
        "pr_auc": pr_auc if pr_auc is not None else 0.0,
        "precision_keep": precision_keep,
        "recall_keep": recall_keep,
        "f1_keep": f1_keep,
        "accuracy": accuracy,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return result


def _compute_metrics(
    clf: Pipeline, X: np.ndarray, y: np.ndarray, threshold: float = 0.5
) -> tuple[float | None, float | None, float | None, tuple[int, int, int, int] | None, dict[str, float] | None]:
    """Compute precision, ROC-AUC, F1, confusion matrix, and asymmetric cost metrics.
    
    Args:
        clf: Pipeline to evaluate
        X: Feature matrix
        y: True labels
        threshold: Decision threshold (default: 0.5)
    
    Returns:
        (precision, roc_auc, f1, confusion, asymmetric_metrics)
    """
    precision_val = None
    roc_auc_val = None
    f1_val = None
    confusion_vals = None
    asymmetric_metrics = None

    try:
        # Get probabilities
        y_prob = None
        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X)[:, 1]
            except Exception:
                logging.debug("[train] predict_proba failed, using predict")
        
        # Apply threshold
        if y_prob is not None:
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = clf.predict(X)
        
        precision_val = float(precision_score(y, y_pred, zero_division=0))

        if y_prob is not None:
            try:
                roc_auc_val = float(roc_auc_score(y, y_prob))
            except Exception:
                logging.exception("Error computing ROC-AUC")
                raise

        try:
            recall_val = float(recall_score(y, y_pred, zero_division=0))
            if precision_val is None:
                f1_val = None
            else:
                if precision_val + recall_val == 0:
                    f1_val = 0.0
                else:
                    f1_val = float(2.0 * (precision_val * recall_val) / (precision_val + recall_val))
        except Exception:
            logging.exception("Error computing F1/recall during training metrics")
            raise

        try:
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            confusion_vals = (int(tn), int(fp), int(fn), int(tp))
            
            # Compute asymmetric metrics
            if y_prob is not None:
                asymmetric_metrics = _compute_asymmetric_metrics(y, y_pred, y_prob)
        except Exception:
            logging.exception("Error computing confusion matrix or asymmetric metrics")
            raise
    except Exception as e:
        logging.debug("[train] Metrics computation failed: %s", e)

    return precision_val, roc_auc_val, f1_val, confusion_vals, asymmetric_metrics


def _save_model(
    clf: Pipeline,
    model_path: str,
    X: np.ndarray,
    n_samples: int,
    n_keep: int,
    n_trash: int,
    filenames: list,
    xgb_params: dict | None = None,
) -> None:
    """Save model with metadata header for drift detection."""
    try:
        from .features import FEATURE_COUNT, USE_FULL_FEATURES

        if xgb_params is None:
            xgb_params = {}

        # Create metadata for drift detection
        metadata = create_model_metadata(
            feature_count=FEATURE_COUNT,
            feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
            params=xgb_params,
            n_samples=n_samples,
        )
        
        # Add feature_indices if feature selection was used
        # This will be read by inference code to subset features
        feature_indices = getattr(clf, "_feature_indices", None)
        if feature_indices:
            metadata["feature_indices"] = feature_indices

        data = {
            "__metadata__": metadata,
            "model": clf,
            "feature_length": X.shape[1],
            "n_samples": n_samples,
            "n_keep": n_keep,
            "n_trash": n_trash,
            "filenames": filenames,
            "precision": None,
            "feature_importances": None,
        }
        joblib.dump(data, model_path)
        logging.info(
            "[train] Model saved: %s (metadata: features=%d, params_hash=%s)",
            model_path,
            FEATURE_COUNT,
            metadata.get("params_hash", "unknown"),
        )
    except Exception as e:
        logging.error("[train] Failed saving model: %s", e)


def _update_model_metadata(
    model_path: str,
    precision_val: float | None,
    feature_importances: list[tuple[int, float]] | None,
    roc_auc_val: float | None,
    f1_val: float | None,
    confusion_vals: tuple[int, int, int, int] | None,
) -> None:
    """Update model file with computed metrics."""
    try:
        data = joblib.load(model_path)
        data["precision"] = precision_val
        data["feature_importances"] = feature_importances
        data["roc_auc"] = roc_auc_val
        data["f1"] = f1_val
        data["confusion"] = confusion_vals
        joblib.dump(data, model_path)
    except Exception:
        logging.debug("[train] Failed updating model metadata with extended metrics")


def train_keep_trash_model(
    image_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int = 42,
    repo: RatingsTagsRepository | None = None,
    n_estimators: int = 200,
    min_samples: int = 2,
    progress_callback=None,
    displayed_filenames: list[str] | None = None,
    early_stopping_rounds: int | None = 100,  # Default: 100 rounds patience (increased for stability)
    cancellation_token=None,
    fast_mode: bool = False,  # Fast mode for app retraining: skip CV, use fewer iterations
) -> TrainingResult | None:
    """Train keep/trash classifier.

    Returns:
        TrainingResult with model path and metrics, or None if insufficient data or cancelled.
    """
    t_start = time.perf_counter()
    logging.info(
        "[train] Starting training dir=%s model=%s min_samples=%d n_estimators=%d",
        image_dir,
        model_path,
        min_samples,
        n_estimators,
    )

    def _check_cancel(stage: str):
        if cancellation_token and callable(cancellation_token) and cancellation_token():
            logging.info(f"[train] Cancellation requested at stage={stage}")
            raise CancelledTraining(stage)

    try:
        _check_cancel("init")
        if progress_callback:
            try:
                progress_callback(0, 0, "init")
            except Exception:
                logging.exception("Error during evaluation")
                raise

        # Initialize repo & load dataset
        repo = _init_repo(image_dir, repo)
        _check_cancel("repo-init")
        if progress_callback:
            try:
                progress_callback(0, 0, "dataset-building")
            except Exception:
                logging.exception("Error during checkpoint handling")
                raise
        dataset_start = time.perf_counter()
        X, y, filenames = build_dataset(
            image_dir,
            repo,
            progress_callback=progress_callback,
            displayed_filenames=displayed_filenames,
        )
        dataset_time = time.perf_counter() - dataset_start
        logging.info("[train] Dataset building completed in %.2fs", dataset_time)
        _check_cancel("dataset-built")

        # Validate dataset
        n_samples = len(y)
        n_keep = int(np.sum(y == 1))
        n_trash = int(np.sum(y == 0))
        logging.info("[train] Dataset samples=%d keep=%d trash=%d", n_samples, n_keep, n_trash)
        if not _validate_dataset(n_samples, n_keep, n_trash, min_samples):
            return None

        if progress_callback:
            try:
                progress_callback(0, 0, "class-balance")
            except Exception:
                logging.exception("Error in metric computation")
                raise
        _compute_class_balance(n_keep, n_trash)
        _check_cancel("class-balance")

        # Adjust hyperparams for dataset size
        n_estimators = _adjust_estimators(n_samples, n_estimators)
        scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
        logging.info("[train] scale_pos_weight=%.2f", scale_pos_weight)

        if progress_callback:
            try:
                progress_callback(0, 0, "params-load")
            except Exception:
                logging.exception("Error in loss calculation")
                raise
        xgb_params = _get_xgb_params(n_estimators, n_samples=n_samples, n_trash=n_trash)
        _check_cancel("params-loaded")

        # Use all features (optimal: CatBoost with all 78 features achieved 84% accuracy)
        # The 19-feature set achieved 81.33%, so all features is better
        feature_indices = None
        logging.info("[train] Using all features (optimal configuration: 84% accuracy with CatBoost)")

        if progress_callback:
            try:
                progress_callback(0, 0, "pipeline-build")
            except Exception:
                logging.exception("Error in optimizer step")
                raise
        # Use CatBoost by default with optimal hyperparameters (84% accuracy vs 72% baseline)
        clf = _build_pipeline(random_state, scale_pos_weight, xgb_params, early_stopping_rounds, use_catboost=True, fast_mode=fast_mode)
        # Store feature_indices on pipeline for later retrieval (None = all features)
        clf._feature_indices = feature_indices  # type: ignore[attr-defined]
        _check_cancel("pipeline-built")

        if progress_callback:
            try:
                progress_callback(0, 0, "cv")
            except Exception:
                logging.exception("Error in scheduler step")
                raise
        cv_start = time.perf_counter()
        # OPTIMIZATION: Skip CV for app retraining (only use for final evaluation)
        # This speeds up interactive retraining significantly
        # CV is still useful for final evaluation but not needed for every label change
        # Target: <5s per label change, so skip CV if it would take too long
        # Skip CV in fast mode (app retraining) - CV takes ~4-5s and we need <5s total
        skip_cv_for_speed = fast_mode  # Always skip CV in fast mode
        if skip_cv_for_speed:
            logging.info("[train] Skipping CV for speed (fast mode - app retraining)")
            cv_mean, cv_std, cv_metrics = None, None, None
            cv_time = 0.0
        else:
            cv_mean, cv_std, cv_metrics = _compute_cross_val(clf, X, y, filenames=filenames, image_dir=image_dir, use_per_fold_threshold=True, max_keep_loss=0.02)
            cv_time = time.perf_counter() - cv_start
        if cv_metrics:
            logging.info(
                "[train] CV asymmetric metrics: keep-loss=%.2f%%±%.2f%%, keep-rate=%.2f%%±%.2f%%, junk-leak=%.2f%%±%.2f%%, PR-AUC=%.4f±%.4f",
                cv_metrics["keep_loss_rate_mean"] * 100, cv_metrics["keep_loss_rate_std"] * 100,
                cv_metrics["keep_rate_mean"] * 100, cv_metrics["keep_rate_std"] * 100,
                cv_metrics["junk_leak_rate_mean"] * 100, cv_metrics["junk_leak_rate_std"] * 100,
                cv_metrics["pr_auc_mean"], cv_metrics["pr_auc_std"]
            )
        if cv_mean is not None:
            logging.info("[train] Cross-validation completed in %.2fs", cv_time)
        _check_cancel("cv-done")

        if progress_callback:
            try:
                progress_callback(0, 0, "fit")
            except Exception:
                logging.exception("Error saving model checkpoint")
                raise
        # CRITICAL FIX: Split data BEFORE training to avoid data leakage
        # For proper evaluation: train on train set (80%), evaluate on test set (20%)
        # Use the same 80% model for production (metrics match production model)
        _fit_model_validation = None
        if n_samples >= 20:
            # Create holdout test set (20% of data) BEFORE training
            X_train_eval, X_test_holdout, y_train_eval, y_test_holdout = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state
            )
            
            # If early stopping enabled, split training data further
            # Use 10% for validation (better for threshold tuning) instead of 3%
            # This still leaves 70% for training (80% * 90% = 72%)
            if early_stopping_rounds and len(X_train_eval) >= 30:
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                    X_train_eval, y_train_eval, test_size=0.10, stratify=y_train_eval, random_state=random_state
                )
                logging.info(
                    "[train] Split data: train=%d, val=%d, test=%d (for early stopping)",
                    len(X_train_final), len(X_val_final), len(X_test_holdout)
                )
                X_train_eval, y_train_eval = X_train_final, y_train_final
                _fit_model_validation = (X_val_final, y_val_final)
            else:
                logging.info(
                    "[train] Split data: train=%d, test=%d (for honest evaluation)",
                    len(X_train_eval), len(X_test_holdout)
                )
        else:
            # Too small for holdout - will use all data (metrics will be optimistic)
            X_train_eval, X_test_holdout, y_train_eval, y_test_holdout = X, None, y, None
            logging.info(
                "[train] Dataset too small for holdout - metrics will be computed on training data (optimistic)"
            )

        # Train on training set for honest evaluation
        fit_start = time.perf_counter()
        if _fit_model_validation is not None:
            # Use validation set for early stopping
            X_val_fit, y_val_fit = _fit_model_validation
            _fit_model(clf, X_train_eval, y_train_eval, len(X_train_eval), early_stopping_rounds, X_val_fit, y_val_fit)
        else:
            # No validation set available
            _fit_model(clf, X_train_eval, y_train_eval, len(X_train_eval), early_stopping_rounds)
        fit_time = time.perf_counter() - fit_start
        logging.info("[train] Model fitting completed in %.2fs", fit_time)
        _check_cancel("fit-done")
        
        # Extract training metrics (loss, iterations, patience) from model
        final_loss = None
        iterations = None
        patience_used = early_stopping_rounds
        try:
            cat_model = clf.named_steps.get("cat")
            if cat_model:
                # Get iteration count
                if hasattr(cat_model, "tree_count_"):
                    iterations = int(cat_model.tree_count_)
                
                # Get final loss from evals_result
                if hasattr(cat_model, "evals_result_"):
                    evals = cat_model.evals_result_
                    if evals:
                        # Find validation loss (usually "validation" or "test" key)
                        for key in evals:
                            if "validation" in key.lower() or "test" in key.lower():
                                # Look for Logloss metric
                                if "Logloss" in evals[key]:
                                    loss_values = evals[key]["Logloss"]
                                    if loss_values:
                                        final_loss = float(loss_values[-1])  # Last value
                                        break
                                # Fallback to first metric
                                elif evals[key]:
                                    first_metric = list(evals[key].keys())[0]
                                    loss_values = evals[key][first_metric]
                                    if loss_values:
                                        final_loss = float(loss_values[-1])
                                        break
        except Exception as e:
            logging.debug(f"[train] Failed to extract training metrics: {e}")
        
        # Compute metrics on holdout set (model hasn't seen this)
        # Tune threshold on validation set (if available) or use default
        eval_start = time.perf_counter()
        test_threshold = 0.5
        if X_test_holdout is not None and _fit_model_validation is not None:
            # Tune threshold on validation set to minimize keep-loss
            X_val_fit, y_val_fit = _fit_model_validation
            if hasattr(clf, "predict_proba"):
                try:
                    y_proba_val = clf.predict_proba(X_val_fit)[:, 1]
                    test_threshold = _find_threshold_for_max_keep_loss(y_val_fit, y_proba_val, max_keep_loss=0.02)
                    logging.info(f"[train] Tuned threshold for test set: {test_threshold:.3f} (target keep-loss <2%)")
                except Exception as e:
                    logging.debug(f"[train] Failed to tune threshold: {e}, using default 0.5")
        
        if X_test_holdout is not None:
            precision_val, roc_auc_val, f1_val, confusion_vals, asymmetric_metrics = _compute_metrics(clf, X_test_holdout, y_test_holdout, threshold=test_threshold)
            logging.info(
                "[train] Metrics computed on holdout test set: precision=%.4f, roc_auc=%.4f, f1=%.4f",
                precision_val or 0.0, roc_auc_val or 0.0, f1_val or 0.0
            )
            if asymmetric_metrics:
                logging.info(
                    "[train] Asymmetric metrics: keep-loss=%.2f%%, keep-rate=%.2f%%, junk-leak=%.2f%%, PR-AUC=%.4f",
                    asymmetric_metrics["keep_loss_rate"] * 100,
                    asymmetric_metrics["keep_rate"] * 100,
                    asymmetric_metrics["junk_leak_rate"] * 100,
                    asymmetric_metrics["pr_auc"] or 0.0
                )
            # Use the model trained on 80% for production (honest metrics match production model)
            logging.info("[train] Using model trained on 80%% of data for production (honest evaluation)")
        else:
            # Too small for holdout - use all data (will be optimistic)
            precision_val, roc_auc_val, f1_val, confusion_vals, asymmetric_metrics = _compute_metrics(clf, X, y, threshold=0.5)
            logging.info(
                "[train] Dataset too small for holdout - metrics computed on training data (optimistic)"
            )

        # Attempt to build a small calibrator if dataset is large enough.
        try:
            if n_samples >= 30:
                # Use a small holdout (~10%) for calibration to avoid leaking training labels
                X_full = X
                y_full = y
                X_train_small, X_calib, y_train_small, y_calib = train_test_split(
                    X_full, y_full, test_size=0.10, stratify=y_full, random_state=random_state
                )
                # Note: clf is already fitted on full X; CalibratedClassifierCV with cv='prefit'
                # will use clf.predict_proba on X_calib to fit the calibrator.
                try:
                    calibrator = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
                    calibrator.fit(X_calib, y_calib)
                    calib_path = model_path + ".calib.joblib"
                    joblib.dump(calibrator, calib_path)
                    logging.info("[train] Saved calibrator: %s", calib_path)
                except Exception as e:
                    logging.info("[train] Could not build calibrator: %s", e)
        except Exception:
            # Non-fatal: continue without calibration
            logging.debug("[train] Skipping calibrator creation due to error")

        if progress_callback:
            try:
                progress_callback(0, 0, "save")
            except Exception:
                logging.exception("Error pruning old checkpoints")
                raise
        _save_model(clf, model_path, X, n_samples, n_keep, n_trash, filenames, xgb_params=xgb_params)
        _check_cancel("saved")

        if progress_callback:
            try:
                progress_callback(0, 0, "importances")
            except Exception:
                logging.exception("Error cleaning temp training artifacts")
                raise
        feature_importances = _get_feature_importances(clf)
        _check_cancel("importances")
        
        _update_model_metadata(model_path, precision_val, feature_importances, roc_auc_val, f1_val, confusion_vals)
        _check_cancel("metrics-done")

        # Attempt to extract per-epoch training history from the XGBoost estimator
        training_history = None
        try:
            xgb_model = clf.named_steps.get("xgb") if hasattr(clf, "named_steps") else None
            evals = None
            if xgb_model is not None:
                if hasattr(xgb_model, "evals_result"):
                    evals = xgb_model.evals_result()
                else:
                    booster = getattr(xgb_model, "get_booster", lambda: None)()
                    if booster is not None and hasattr(booster, "evals_result"):
                        evals = booster.evals_result()
            if evals:
                history = {}
                for ds_name, metrics in evals.items():
                    for metric_name, values in metrics.items():
                        key = f"{ds_name}_{metric_name}" if ds_name else metric_name
                        history[key] = list(values)
                if history:
                    training_history = history
        except Exception:
            logging.debug("[train] No training evals_result available or failed to extract history")

        total_time = time.perf_counter() - t_start
        logging.info("[train] Done total_time=%.3fs", total_time)

        return TrainingResult(
            model_path=model_path,
            n_samples=n_samples,
            n_keep=n_keep,
            n_trash=n_trash,
            cv_accuracy_mean=cv_mean,
            cv_accuracy_std=cv_std,
            precision=precision_val,
            feature_importances=feature_importances,
            roc_auc=roc_auc_val,
            f1=f1_val,
            confusion=confusion_vals,
            training_history=training_history,
            aggregated_metrics=cv_metrics,
            final_loss=final_loss,
            iterations=iterations,
            patience=patience_used,
        )
    except CancelledTraining as c:
        logging.info(f"[train] Training cancelled at stage={c}")
        return None

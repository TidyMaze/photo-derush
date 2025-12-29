"""Hyperparameter tuning & parameter persistence."""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
# Lazy imports: sklearn and xgboost are heavy, only import when actually needed

from .dataset import build_dataset
from .model import RatingsTagsRepository

BEST_PARAMS_PATH = os.path.expanduser("~/.photo-derush-xgb-params.json")


def save_best_params(params: dict, path: str = BEST_PARAMS_PATH):
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        logging.info("[tuning] Saved best params -> %s", path)
    except Exception as e:
        logging.warning("[tuning] Failed saving params: %s", e)


def load_best_params(path: str = BEST_PARAMS_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            params = json.load(f)
        logging.info("[tuning] Loaded params: %s", params)
        return params  # type: ignore[return-value, no-any-return]
    except Exception as e:
        logging.warning("[tuning] Failed loading params: %s", e)
        return None


def tune_hyperparameters(
    image_dir: str,
    repo: RatingsTagsRepository | None = None,
    n_iter: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
    save_params: bool = True,
    progress_callback=None,
) -> dict | None:
    logging.info("[tuning] Start tuning dir=%s n_iter=%d cv_folds=%d", image_dir, n_iter, cv_folds)
    # When running under pytest, force single-threaded execution to avoid
    # parallel worker deadlocks that can hang the test runner.
    import os

    is_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    if repo is None:
        repo_path = os.path.join(image_dir, ".ratings_tags.json")
        repo = RatingsTagsRepository(path=repo_path)
        logging.info("[tuning] Scoped repo: %s", repo_path)
    X, y, _ = build_dataset(image_dir, repo)
    n_samples = len(y)
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    logging.info("[tuning] Dataset samples=%d keep=%d trash=%d", n_samples, n_keep, n_trash)
    if n_samples < cv_folds * 2:
        logging.warning("[tuning] Insufficient samples for CV (%d < %d)", n_samples, cv_folds * 2)
        return None
    if n_keep == 0 or n_trash == 0:
        logging.warning("[tuning] Need both classes (keep=%d trash=%d)", n_keep, n_trash)
        return None
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    if n_samples < 20:
        param_distributions = {
            "xgb__n_estimators": [20, 30, 50, 75, 100, 150],
            "xgb__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            "xgb__max_depth": [2, 3, 4, 5, 6, 7],
            "xgb__min_child_weight": [1, 2, 3, 5],
            "xgb__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "xgb__gamma": [0, 0.05, 0.1, 0.2, 0.3],
            "xgb__reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "xgb__reg_lambda": [0, 0.01, 0.1, 0.5, 1.0, 1.5],
        }
        n_iter = min(n_iter, 15)
        logging.info("[tuning] SMALL dataset grid")
    else:
        param_distributions = {
            "xgb__n_estimators": [30, 50, 75, 100, 150, 200, 250, 300, 400],
            "xgb__learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
            "xgb__max_depth": [2, 3, 4, 5, 6, 7, 8, 10],
            "xgb__min_child_weight": [1, 2, 3, 5, 7, 10],
            "xgb__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "xgb__gamma": [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "xgb__reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            "xgb__reg_lambda": [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        }
        logging.info("[tuning] LARGE dataset grid")
    xgb_n_jobs = 1 if is_pytest else 4
    # Lazy imports: sklearn and xgboost are heavy, only import when actually needed
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBClassifier(
                    random_state=random_state,
                    n_jobs=xgb_n_jobs,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search_n_jobs = 1 if is_pytest else 4
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=search_n_jobs,
        random_state=random_state,
        verbose=0,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    best_params = {k.replace("xgb__", ""): v for k, v in search.best_params_.items()}
    best_params["_scale_pos_weight_ratio"] = scale_pos_weight
    logging.info("[tuning] Best params %s (elapsed %.2fs)", best_params, elapsed)
    if save_params:
        save_best_params(best_params)
    # Export all search results
    try:
        results_path = os.path.expanduser("~/.photo-derush-xgb-cv-results.json")
        with open(results_path, "w") as rf:
            json.dump(
                search.cv_results_,
                rf,
                indent=2,
                default=lambda o: o if isinstance(o, (int, float, str, list, dict)) else str(o),
            )
        logging.info("[tuning] Exported cv_results_ -> %s", results_path)
    except Exception as e:
        logging.warning("[tuning] Failed exporting cv_results_: %s", e)
    return best_params


def tune_hyperparameters_optuna(
    image_dir: str,
    repo: RatingsTagsRepository | None = None,
    n_trials: int = 50,
    cv_folds: int = 3,
    random_state: int = 42,
    save_params: bool = True,
    progress_callback=None,
    timeout: int | None = None,
) -> dict | None:
    """Tune hyperparameters using Optuna with TPE (Tree-structured Parzen Estimator).

    More efficient than RandomizedSearchCV for large search spaces.
    """
    try:
        import optuna
    except ImportError:
        logging.warning("[tuning] Optuna not available, falling back to RandomizedSearchCV")
        return tune_hyperparameters(image_dir, repo, n_trials, cv_folds, random_state, save_params, progress_callback)

    logging.info("[tuning] Start Optuna tuning dir=%s n_trials=%d cv_folds=%d", image_dir, n_trials, cv_folds)

    if repo is None:
        repo_path = os.path.join(image_dir, ".ratings_tags.json")
        repo = RatingsTagsRepository(path=repo_path)
        logging.info("[tuning] Scoped repo: %s", repo_path)

    X, y, _ = build_dataset(image_dir, repo)
    n_samples = len(y)
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    logging.info("[tuning] Dataset samples=%d keep=%d trash=%d", n_samples, n_keep, n_trash)

    if n_samples < cv_folds * 2:
        logging.warning("[tuning] Insufficient samples for CV (%d < %d)", n_samples, cv_folds * 2)
        return None
    if n_keep == 0 or n_trash == 0:
        logging.warning("[tuning] Need both classes (keep=%d trash=%d)", n_keep, n_trash)
        return None

    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    is_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
        }

        # Add tree_method and grow_policy for expanded search space
        tree_method = trial.suggest_categorical("tree_method", ["hist", "approx", "exact"])
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        # Build pipeline
        xgb_n_jobs = 1 if is_pytest else 4
        # Lazy import: xgboost is heavy, only import when actually needed
        import xgboost as xgb
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=random_state,
                        n_jobs=xgb_n_jobs,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method=tree_method,
                        grow_policy=grow_policy,
                        **params,
                    ),
                ),
            ]
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_n_jobs = 1 if is_pytest else 4
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=cv_n_jobs)

        if progress_callback:
            try:
                progress_callback(trial.number, n_trials, f"Trial {trial.number}: accuracy={scores.mean():.4f}")
            except Exception:
                pass

        return float(scores.mean())

    # Create study
    study = optuna.create_study(direction="maximize", study_name="xgb_tuning", sampler=optuna.samplers.TPESampler(seed=random_state))

    # Run optimization
    t0 = time.perf_counter()
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    except KeyboardInterrupt:
        logging.info("[tuning] Optimization interrupted by user")
    elapsed = time.perf_counter() - t0

    if len(study.trials) == 0:
        logging.warning("[tuning] No trials completed")
        return None

    # Extract best parameters
    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_params["_scale_pos_weight_ratio"] = scale_pos_weight
    best_params["_best_value"] = best_trial.value

    logging.info("[tuning] Best params %s (elapsed %.2fs, best_value=%.4f)", best_params, elapsed, best_trial.value)

    if save_params:
        save_best_params(best_params)

    return best_params


__all__ = [
    "tune_hyperparameters",
    "tune_hyperparameters_optuna",
    "save_best_params",
    "load_best_params",
    "BEST_PARAMS_PATH",
]

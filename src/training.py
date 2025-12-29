"""Aggregated training module (backward compatible).
Delegates responsibilities to submodules for Single Responsibility Principle:
  features.py      -> feature extraction & cache
  dataset.py       -> dataset building
  training_core.py -> model training & result dataclass
  tuning.py        -> hyperparameter tuning & param persistence
  inference.py     -> model loading & prediction
This facade preserves original import surface used by tests and callers.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Re-exported symbols
from .features import (
    FEATURE_COUNT,
    USE_FULL_FEATURES,
    batch_extract_features,
    extract_features,
    load_feature_cache,
    safe_initialize_feature_cache,
)
from .inference import load_model, predict_keep_probability
# Lazy imports: training_core imports sklearn which is heavy (~1.5s startup time)
from .tuning import BEST_PARAMS_PATH, load_best_params, save_best_params, tune_hyperparameters

# Lazy import: training_core imports sklearn which is heavy (~1.5s startup time)
# Import only when actually accessed to avoid startup delay
_training_core_cache = {}

def __getattr__(name: str):
    """Lazy import for training_core symbols."""
    if name in ("DEFAULT_MODEL_PATH", "TrainingResult", "train_keep_trash_model"):
        if name not in _training_core_cache:
            from .training_core import DEFAULT_MODEL_PATH, TrainingResult, train_keep_trash_model
            _training_core_cache["DEFAULT_MODEL_PATH"] = DEFAULT_MODEL_PATH
            _training_core_cache["TrainingResult"] = TrainingResult
            _training_core_cache["train_keep_trash_model"] = train_keep_trash_model
        return _training_core_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# For __all__ exports - these will be accessed via __getattr__ when imported
# We can't set them here without triggering the import, so they're accessed lazily

__all__ = [
    "FEATURE_COUNT",
    "USE_FULL_FEATURES",
    "extract_features",
    "batch_extract_features",
    "train_keep_trash_model",
    "predict_keep_probability",
    "safe_initialize_feature_cache",
    "load_feature_cache",
    "TrainingResult",
    "DEFAULT_MODEL_PATH",
    "tune_hyperparameters",
    "save_best_params",
    "load_best_params",
    "BEST_PARAMS_PATH",
    "load_model",
]

# ------------------------------ CLI Entrypoint ---------------------------- #


def _parse_args(argv: list[str]):  # pragma: no cover
    # Import here since this is CLI-only code (not used during app startup)
    from .training_core import DEFAULT_MODEL_PATH
    parser = argparse.ArgumentParser(description="Train keep/trash model from labeled images.")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to save model (default: %(default)s)")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning instead of training")
    parser.add_argument("--tune-iter", type=int, default=20, help="Number of tuning iterations (default: 20)")
    parser.add_argument("--tune-cv", type=int, default=3, help="Number of CV folds for tuning (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):  # pragma: no cover
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.tune:
        print(f"Starting hyperparameter tuning on {args.image_dir}...")
        print(f"  Iterations: {args.tune_iter}")
        print(f"  CV folds: {args.tune_cv}")
        best_params = tune_hyperparameters(
            args.image_dir, n_iter=args.tune_iter, cv_folds=args.tune_cv, save_params=True
        )
        if best_params is None:
            print("Tuning skipped: insufficient labeled data.")
            return 2
        print("\nTuning complete!\nBest parameters:")
        for k, v in best_params.items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        print(f"\nParameters saved to {BEST_PARAMS_PATH}")
        return 0
    else:
        # Import here since this is CLI-only code
        from .training_core import train_keep_trash_model
        result = train_keep_trash_model(args.image_dir, model_path=args.model_path)
        if result is None:
            print("Training skipped: insufficient labeled data.")
            return 2
        print(
            "Training complete:\n"
            + f"  Samples: {result.n_samples} (keep={result.n_keep}, trash={result.n_trash})\n"
            + (
                f"  CV accuracy: {result.cv_accuracy_mean:.3f} ± {result.cv_accuracy_std:.3f}\n"
                if result.cv_accuracy_mean is not None
                else "  CV accuracy: n/a (too few samples)\n"
            )
            + f"  Model saved: {result.model_path}"
        )
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

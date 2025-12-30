#!/usr/bin/env python3
"""Plot correlation of top 10 features to model predictions.

This script visualizes how the top 10 most important features correlate with
model predictions (or labels if no model available).

Usage:
  python plot_top_features_correlation.py <image_dir> [--output OUTPUT_PATH]

Example:
  python plot_top_features_correlation.py ~/Pictures --output plots/correlation.png

Features shown:
  1. Shutter Speed (5.35%)
  2. B Hist Bin 7 (4.72%)
  3. Month (3.37%)
  4. Aspect Ratio (3.17%)
  5. Std Brightness (dup) (3.01%)
  6. B Hist Bin 2 (2.90%)
  7. ISO (2.85%)
  8. Std Brightness (2.75%)
  9. Hour of Day (2.70%)
  10. Noise Level (2.56%)
"""
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Feature names and indices (from model_stats.py verification)
TOP_10_FEATURES = [
    (49, 'Shutter Speed', 5.35),
    (35, 'B Hist Bin 7', 4.72),
    (45, 'Month', 3.37),
    (2, 'Aspect Ratio', 3.17),
    (39, 'Std Brightness (dup)', 3.01),
    (30, 'B Hist Bin 2', 2.90),
    (47, 'ISO', 2.85),
    (5, 'Std Brightness', 2.75),
    (43, 'Hour of Day', 2.70),
    (42, 'Noise Level', 2.56),
]

def find_last_image_dir():
    """Find the last used image directory from app config."""
    config_path = os.path.expanduser('~/.photo_app_config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path) as f:
                data = json.load(f)
                last_dir = data.get('last_dir')
                if last_dir and os.path.isdir(last_dir):
                    return last_dir
        except Exception:
            pass
    return None

def load_dataset_and_model(image_dir: str):
    """Load dataset and trained model."""
    import joblib

    from src.dataset import build_dataset
    from src.model import RatingsTagsRepository
    from src.training_core import DEFAULT_MODEL_PATH

    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return None, None, None, None, None

    logging.info(f"Loading dataset from {image_dir}...")
    repo_path = os.path.join(image_dir, '.ratings_tags.json')

    if not os.path.exists(repo_path):
        logging.error(f"No ratings file found at {repo_path}")
        return None, None, None, None, None

    repo = RatingsTagsRepository(path=repo_path)
    X, y, filenames = build_dataset(image_dir, repo)

    if len(X) == 0:
        logging.error("No labeled samples found in dataset")
        return None, None, None, None, None

    logging.info(f"Loaded {len(X)} samples ({int(np.sum(y))} keep, {len(y) - int(np.sum(y))} trash)")

    # Load trained model via inference helper so we also get calibrator if present
    try:
        from src.inference import load_model
        from src.training_core import DEFAULT_MODEL_PATH as TC_DEFAULT_MODEL_PATH
        model_path = TC_DEFAULT_MODEL_PATH
        model_obj = None
        calibrator = None
        try:
            loaded = load_model(model_path)
            if loaded is not None:
                model_obj, meta, calibrator = loaded.model, loaded.meta, loaded.calibrator
        except Exception as e:
                logging.warning(f"Could not load model via inference helper: {e}")
                model_obj = None
    except Exception:
        # Fallback: no inference helper available
        model_obj = None
        calibrator = None

    return X, y, model_obj, filenames, calibrator

def plot_feature_correlations(X: np.ndarray, y: np.ndarray, model=None, output_path: str = None, calibrator=None):
    """Create correlation plot for top 10 features."""

    if len(X) == 0:
        logging.error("No data to plot")
        return

    # Get predictions if model exists
    if model is not None:
        try:
            logging.info(f"Model type: {type(model)}")

            # If model is a dict (from joblib.load of training result), extract the actual model
            if isinstance(model, dict):
                logging.info("Model is a dict, extracting 'model' key")
                model = model.get('model')
                if model is None:
                    logging.error("No 'model' key found in dict")
                    return

            logging.info(f"Actual model type: {type(model)}")

            # For sklearn Pipeline, need to access the final estimator
            estimator = model
            if hasattr(model, 'named_steps'):
                # It's a Pipeline, get the last step
                estimator = model.named_steps.get(list(model.named_steps.keys())[-1])
                logging.info(f"Pipeline detected, using estimator: {type(estimator)}")

            # Prefer calibrated probabilities when available
            if calibrator is not None and hasattr(calibrator, 'predict_proba'):
                try:
                    y_pred = calibrator.predict_proba(X)[:, 1].flatten()
                    logging.info(f"Got predictions via calibrator.predict_proba: shape={y_pred.shape}")
                except Exception as e:
                    logging.warning(f"calibrator.predict_proba failed: {e}, falling back to estimator/model")
                    y_pred = None
            else:
                y_pred = None

            # If no calibrated preds, try estimator or model (probabilities first)
            if y_pred is None:
                # Try estimator predict_proba
                if hasattr(estimator, 'predict_proba'):
                    try:
                        y_pred = estimator.predict_proba(X)[:, 1].flatten()
                        logging.info(f"Got predictions via estimator.predict_proba: shape={y_pred.shape}")
                    except Exception as e:
                        logging.warning(f"estimator.predict_proba failed: {e}")
                        y_pred = None

                # If still None, try model.predict_proba
                if y_pred is None and hasattr(model, 'predict_proba'):
                    try:
                        y_pred = model.predict_proba(X)[:, 1].flatten()
                        logging.info(f"Got predictions via model.predict_proba: shape={y_pred.shape}")
                    except Exception as e:
                        logging.warning(f"model.predict_proba failed: {e}")
                        y_pred = None

                # If no probability outputs, try deterministic predict() (labels)
                if y_pred is None:
                    if hasattr(estimator, 'predict'):
                        try:
                            y_pred = np.asarray(estimator.predict(X)).flatten()
                            logging.info(f"Got predictions via estimator.predict: shape={y_pred.shape}")
                        except Exception as e:
                            logging.warning(f"estimator.predict failed: {e}")
                            y_pred = None
                    elif hasattr(model, 'predict'):
                        try:
                            y_pred = np.asarray(model.predict(X)).flatten()
                            logging.info(f"Got predictions via model.predict: shape={y_pred.shape}")
                        except Exception as e:
                            logging.warning(f"model.predict failed: {e}")
                            y_pred = None
            else:
                # y_pred already set by calibrator above, do not overwrite
                pass
        except Exception as e:
            logging.warning(f"Could not get predictions: {e}")
            import traceback
            traceback.print_exc()
            y_pred = None
    else:
        # Use labels as proxy
        y_pred = y.astype(float).flatten()

    if y_pred is None:
        logging.error("Failed to get predictions from model")
        return

    # Create figure with subplots
    n_features = len(TOP_10_FEATURES)
    cols = 5
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle('Top 10 Features - Correlation to Model Score', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    correlations = []

    for idx, (feat_idx, feat_name, importance) in enumerate(TOP_10_FEATURES):
        ax = axes[idx]

        if feat_idx >= X.shape[1]:
            ax.text(0.5, 0.5, f'Index {feat_idx}\nout of range',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feat_name} (invalid)', fontsize=10)
            continue

        feature_data = X[:, feat_idx]

        # Compute stats and skip if no variation in data
        feat_min, feat_max, feat_std = float(np.min(feature_data)), float(np.max(feature_data)), float(np.std(feature_data))
        pred_min, pred_max, pred_std = float(np.min(y_pred)), float(np.max(y_pred)), float(np.std(y_pred))
        logging.debug(f"Feature {feat_idx} stats: min={feat_min:.6g} max={feat_max:.6g} std={feat_std:.6g}; pred stats: min={pred_min:.6g} max={pred_max:.6g} std={pred_std:.6g}")
        TOL = 1e-12
        if feat_std <= TOL:
            logging.warning(f"Feature {feat_idx} has no variance (std={feat_std}), skipping correlation")
            ax.text(0.5, 0.5, 'No variance in data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feat_name}\n(no variance)', fontsize=10)
            continue
        if pred_std <= TOL:
            logging.warning(f"Predictions nearly constant (std={pred_std}), falling back to labels for correlation")
            y_pred_use = y.astype(float).flatten()
            pred_std = float(np.std(y_pred_use))
        else:
            y_pred_use = y_pred
            ax.text(0.5, 0.5, 'No variance in data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feat_name}\n(no variance)', fontsize=10)
            continue

        # Calculate correlations
        try:
            pearson_r, pearson_p = pearsonr(feature_data, y_pred_use)
            spearman_r, spearman_p = spearmanr(feature_data, y_pred_use)  # type: ignore
        except Exception as e:
            logging.warning(f"Could not calculate correlation for feature {feat_idx}: {e}")
            ax.text(0.5, 0.5, f'Correlation error:\n{str(e)[:30]}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feat_name}\n(error)', fontsize=10)
            continue

        correlations.append({
            'feature': feat_name,
            'importance': importance,
            'pearson': pearson_r,
            'spearman': spearman_r
        })

        # Scatter plot with trend line
        ax.scatter(feature_data, y_pred, alpha=0.5, s=30, color='steelblue')

        # Add trend line
        z = np.polyfit(feature_data, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(feature_data.min(), feature_data.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.8)

        ax.set_xlabel('Feature Value', fontsize=9)
        ax.set_ylabel('Model Score', fontsize=9)
        title = f'{feat_name}\nImportance: {importance:.2f}%'
        ax.set_title(title, fontsize=10, fontweight='bold')

        # Add correlation info
        ax.text(0.05, 0.95, f'r={pearson_r:.3f}\nρ={spearman_r:.3f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(TOP_10_FEATURES), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = 'plots/top_features_correlation.png'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Plot saved to {output_path}")

    # Print correlation summary
    print("\n" + "="*70)
    print("FEATURE CORRELATION SUMMARY")
    print("="*70)
    print(f"{'Feature':<30} {'Importance':<12} {'Pearson r':<12} {'Spearman ρ':<12}")
    print("-"*70)
    for corr in correlations:
        print(f"{corr['feature']:<30} {corr['importance']:<12.2f}% {corr['pearson']:<12.3f} {corr['spearman']:<12.3f}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Plot correlation of top features to model score')
    parser.add_argument('image_dir', nargs='?', default=None, help='Directory containing labeled images (auto-detected if not provided)')
    parser.add_argument('--output', '-o', help='Output path for plot (default: plots/top_features_correlation.png)')
    parser.add_argument('--no-model', action='store_true', help='Skip model loading and use labels instead')

    args = parser.parse_args()

    # Auto-detect directory if not provided
    image_dir = args.image_dir
    if not image_dir:
        image_dir = find_last_image_dir()
        if not image_dir:
            logging.error("Image directory not specified and no last directory found in config")
            logging.error("Usage: python plot_top_features_correlation.py <image_dir>")
            sys.exit(1)
        logging.info(f"Auto-detected image directory: {image_dir}")

    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    X, y, model, filenames, calibrator = load_dataset_and_model(image_dir)

    if X is None:
        sys.exit(1)

    if args.no_model:
        model = None
        calibrator = None

    plot_feature_correlations(X, y, model, args.output, calibrator=calibrator)

if __name__ == '__main__':
    main()


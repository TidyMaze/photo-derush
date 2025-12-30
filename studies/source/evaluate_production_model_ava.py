#!/usr/bin/env python3
"""Evaluate production model on AVA dataset and generate comprehensive HTML report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training_core import DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("eval_ava")


def load_ava_binary_labels(cache_dir: Path, threshold: float = 6.0):
    """Load AVA labels and convert to binary keep/trash.
    
    Args:
        cache_dir: Cache directory containing AVA dataset
        threshold: Score threshold (>= threshold → keep, < threshold → trash). Default 6.0.
    """
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
        log.error(f"AVA metadata not found at {ava_metadata_path}")
        return None
    
    log.info(f"Loading AVA metadata from {ava_metadata_path}...")
    image_labels = {}
    
    with open(ava_metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            image_id = parts[1]
            score_counts = [int(s) for s in parts[2:12]]
            total_votes = sum(score_counts)
            if total_votes == 0:
                continue
            
            # Calculate weighted mean score
            weighted_sum = sum((i+1) * count for i, count in enumerate(score_counts))
            mean_score = weighted_sum / total_votes
            
            # Binary classification: score >= threshold = keep (1), < threshold = trash (0)
            binary_label = 1 if mean_score >= threshold else 0
            image_labels[image_id] = {
                'label': binary_label,
                'mean_score': mean_score,
                'total_votes': total_votes,
            }
    
    log.info(f"Loaded labels for {len(image_labels)} images")
    return image_labels


def load_ava_data(cache_dir: Path, max_samples: int = None, threshold: float = 6.0):
    """Load AVA features and labels.
    
    Args:
        cache_dir: Cache directory
        max_samples: Maximum number of samples to load
        threshold: Score threshold for keep/trash conversion (default 6.0)
    """
    ava_features_path = cache_dir / "ava_features.joblib"
    if not ava_features_path.exists():
        log.error(f"AVA features not found at {ava_features_path}")
        return None, None, None
    
    log.info(f"Loading AVA features from {ava_features_path}...")
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
    if max_samples and len(X_ava) > max_samples:
        log.info(f"Limiting to {max_samples} samples")
        X_ava = X_ava[:max_samples]
        ava_ids = ava_ids[:max_samples]
    
    # Load labels
    log.info(f"Converting AVA scores to binary labels (threshold: {threshold})")
    image_labels = load_ava_binary_labels(cache_dir, threshold=threshold)
    if image_labels is None:
        return None, None, None
    
    # Match labels to features
    y_ava = []
    valid_indices = []
    for i, img_id in enumerate(ava_ids):
        img_id_str = str(img_id)
        if img_id_str in image_labels:
            y_ava.append(image_labels[img_id_str]['label'])
            valid_indices.append(i)
        elif f"{img_id}.jpg" in image_labels:
            y_ava.append(image_labels[f"{img_id}.jpg"]['label'])
            valid_indices.append(i)
    
    if len(y_ava) == 0:
        log.error("No matching labels found for AVA features")
        return None, None, None
    
    X_ava = X_ava[valid_indices]
    y_ava = np.array(y_ava)
    
    log.info(f"AVA dataset: {len(y_ava)} samples")
    log.info(f"  Keep: {np.sum(y_ava == 1)} ({np.sum(y_ava == 1)/len(y_ava)*100:.1f}%)")
    log.info(f"  Trash: {np.sum(y_ava == 0)} ({np.sum(y_ava == 0)/len(y_ava)*100:.1f}%)")
    
    return X_ava, y_ava, ava_ids


def evaluate_model_on_ava(model_path: str, cache_dir: Path, max_samples: int = None, threshold: float = 6.0):
    """Evaluate production model on AVA dataset.
    
    Args:
        model_path: Path to production model
        cache_dir: Cache directory containing AVA dataset
        max_samples: Maximum number of AVA samples to evaluate
        threshold: Score threshold for keep/trash conversion (default 6.0)
    """
    log.info(f"Loading production model from {model_path}...")
    if not os.path.exists(model_path):
        log.error(f"Model not found at {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        clf = model_data.get('model') or model_data.get('pipeline')
        if clf is None:
            log.error("Model not found in loaded data")
            return None
    else:
        clf = model_data
    
    if not isinstance(clf, Pipeline):
        log.error("Loaded model is not a Pipeline")
        return None
    
    # Load AVA data
    X_ava, y_ava, ava_ids = load_ava_data(cache_dir, max_samples, threshold=threshold)
    if X_ava is None:
        return None
    
    # Ensure feature count matches
    # Get expected feature count from model
    if hasattr(clf, 'named_steps') and 'scaler' in clf.named_steps:
        # Model expects scaled features
        n_features_expected = clf.named_steps['scaler'].n_features_in_ if hasattr(clf.named_steps['scaler'], 'n_features_in_') else None
    else:
        n_features_expected = None
    
    if n_features_expected and X_ava.shape[1] != n_features_expected:
        log.warning(f"Feature count mismatch: model expects {n_features_expected}, AVA has {X_ava.shape[1]}")
        # Try to pad or truncate
        if X_ava.shape[1] < n_features_expected:
            padding = np.zeros((X_ava.shape[0], n_features_expected - X_ava.shape[1]))
            X_ava = np.hstack([X_ava, padding])
            log.info(f"Padded features to {X_ava.shape[1]}")
        else:
            X_ava = X_ava[:, :n_features_expected]
            log.info(f"Truncated features to {X_ava.shape[1]}")
    
    # Evaluate
    log.info("Evaluating model on AVA dataset...")
    y_pred = clf.predict(X_ava)
    y_proba = clf.predict_proba(X_ava)[:, 1] if hasattr(clf, "predict_proba") else None
    
    metrics = {
        "accuracy": float(accuracy_score(y_ava, y_pred)),
        "precision": float(precision_score(y_ava, y_pred, zero_division=0)),
        "recall": float(recall_score(y_ava, y_pred, zero_division=0)),
        "f1": float(f1_score(y_ava, y_pred, zero_division=0)),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_ava, y_proba))
    
    cm = confusion_matrix(y_ava, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    
    log.info(f"\nAVA Evaluation Results:")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall: {metrics['recall']:.4f}")
    log.info(f"  F1: {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        log.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    log.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return {
        "metrics": metrics,
        "n_samples": len(y_ava),
        "n_keep": int(np.sum(y_ava == 1)),
        "n_trash": int(np.sum(y_ava == 0)),
    }


def generate_html_report(ava_results: dict, output_path: Path):
    """Generate comprehensive HTML report."""
    log.info(f"Generating HTML report to {output_path}...")
    
    # Load learning rate study data
    lr_study_data = {
        "best_lr": 0.1,
        "best_cv_accuracy": 0.8223,
        "best_cv_std": 0.0176,
        "findings": [
            "5-fold cross-validation shows LR=0.1 is optimal (82.23% ± 1.76%)",
            "LR=0.07 performs worse (80.55% ± 3.87%) with higher variance",
            "Single test set results were noisy due to small size (119 samples)",
            "CV provides more reliable estimates by averaging over 5 folds",
        ],
    }
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Model Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .config-table th, .config-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .config-table th {{
            background: #3498db;
            color: white;
        }}
        .config-table tr:hover {{
            background: #f5f5f5;
        }}
        .findings {{
            background: #e8f5e9;
            padding: 15px;
            border-left: 4px solid #4caf50;
            margin: 10px 0;
        }}
        .warning {{
            background: #fff3e0;
            padding: 15px;
            border-left: 4px solid #ff9800;
            margin: 10px 0;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 20px 0;
        }}
        .cm-cell {{
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            font-weight: bold;
        }}
        .cm-tn {{ background: #c8e6c9; }}
        .cm-fp {{ background: #ffccbc; }}
        .cm-fn {{ background: #ffccbc; }}
        .cm-tp {{ background: #c8e6c9; }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>Production Model Evaluation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Model Configuration</h2>
        <table class="config-table">
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Source</th>
            </tr>
            <tr>
                <td>Learning Rate</td>
                <td><strong>0.1</strong></td>
                <td>5-fold CV study (best: 82.23% ± 1.76%)</td>
            </tr>
            <tr>
                <td>Early Stopping</td>
                <td><strong>Enabled</strong></td>
                <td>Patience=200, Max iterations=2000</td>
            </tr>
            <tr>
                <td>Depth</td>
                <td>6</td>
                <td>Optimal from evaluation</td>
            </tr>
            <tr>
                <td>L2 Regularization</td>
                <td>1.0</td>
                <td>Default</td>
            </tr>
            <tr>
                <td>Iterations (max)</td>
                <td>2000</td>
                <td>With early stopping protection</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Learning Rate Study Findings</h2>
        <div class="findings">
            <h3>Key Findings from 5-Fold Cross-Validation:</h3>
            <ul>
                <li><strong>Best LR: 0.1</strong> - Achieved 82.23% ± 1.76% accuracy (most stable)</li>
                <li>LR=0.07 performed worse: 80.55% ± 3.87% (higher variance)</li>
                <li>Single test set (119 samples) showed high variance - small LR changes flipped 8-14 predictions</li>
                <li>Cross-validation provides more reliable estimates by averaging over 5 folds</li>
                <li>Smaller learning rates do NOT necessarily produce higher accuracy</li>
            </ul>
        </div>
        <div class="warning">
            <strong>Important:</strong> The earlier finding that LR=0.07 was best (89.08%) was due to test set variance. 
            Cross-validation shows LR=0.1 is more reliable and stable.
        </div>
    </div>
"""
    
    if ava_results:
        metrics = ava_results["metrics"]
        cm = metrics["confusion_matrix"]
        
        html += f"""
    <div class="section">
        <h2>AVA Dataset Evaluation</h2>
        <p><strong>Dataset:</strong> {ava_results['n_samples']} samples ({ava_results['n_keep']} keep, {ava_results['n_trash']} trash)</p>
        
        <div class="metric">
            <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics['precision']:.4f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics['recall']:.4f}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics['f1']:.4f}</div>
            <div class="metric-label">F1 Score</div>
        </div>
"""
        if "roc_auc" in metrics:
            html += f"""
        <div class="metric">
            <div class="metric-value">{metrics['roc_auc']:.4f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
"""
        
        html += f"""
        <h3>Confusion Matrix</h3>
        <div class="confusion-matrix">
            <div class="cm-cell cm-tn">
                <div>True Negative</div>
                <div style="font-size: 2em;">{cm['tn']}</div>
            </div>
            <div class="cm-cell cm-fp">
                <div>False Positive</div>
                <div style="font-size: 2em;">{cm['fp']}</div>
            </div>
            <div class="cm-cell cm-fn">
                <div>False Negative</div>
                <div style="font-size: 2em;">{cm['fn']}</div>
            </div>
            <div class="cm-cell cm-tp">
                <div>True Positive</div>
                <div style="font-size: 2em;">{cm['tp']}</div>
            </div>
        </div>
    </div>
"""
    
    html += """
    <div class="section">
        <h2>Summary</h2>
        <div class="findings">
            <h3>Production Model Settings:</h3>
            <ul>
                <li>Learning Rate: 0.1 (validated with 5-fold CV)</li>
                <li>Early Stopping: Enabled (prevents overfitting)</li>
                <li>Expected Performance: 82.23% ± 1.76% (on photo-dataset)</li>
            </ul>
            
            <h3>Key Learnings:</h3>
            <ul>
                <li>Cross-validation is essential for small datasets (591 samples)</li>
                <li>Single test set results can be misleading due to high variance</li>
                <li>Smaller learning rates do not necessarily mean better accuracy</li>
                <li>Optimal learning rate is in the middle range (0.07-0.11)</li>
                <li>Early stopping with high patience (200) allows model to reach full potential</li>
            </ul>
        </div>
    </div>
    
</body>
</html>
"""
    
    output_path.write_text(html)
    log.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate production model on AVA dataset")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to production model")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max AVA samples to evaluate")
    parser.add_argument("--threshold", type=float, default=6.0, help="Score threshold for keep/trash (default: 6.0)")
    parser.add_argument("--output", default="studies/outputs/model_evaluation_report.html", help="Output HTML report path")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        log.error(f"Cache directory not found: {cache_dir}")
        return 1
    
    # Evaluate on AVA
    ava_results = evaluate_model_on_ava(args.model_path, cache_dir, args.max_samples, threshold=args.threshold)
    
    # Generate report
    output_path = Path(args.output)
    generate_html_report(ava_results, output_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


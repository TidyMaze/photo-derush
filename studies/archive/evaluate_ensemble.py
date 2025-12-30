#!/usr/bin/env python3
"""Evaluate baseline, combined models and a simple average ensemble.

Usage examples:
  poetry run python scripts/evaluate_ensemble.py ~/Pictures/photo-dataset --baseline /tmp/photo_baseline.joblib --combined /tmp/photo_combined_tuned.joblib --embeddings .cache/embeddings_resnet18_full.joblib
"""
from __future__ import annotations

import argparse
import joblib
import json
import os
import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, f1_score, confusion_matrix, brier_score_loss

from src.dataset import build_dataset
from src.model import RatingsTagsRepository


def load_model_any(path: str):
    data = joblib.load(path)
    # Model may be stored under 'model' key or be the object itself
    if isinstance(data, dict) and 'model' in data:
        return data['model']
    return data


def align_embeddings_for_filenames(filenames, emb_fnames, emb):
    emb_map = {f: i for i, f in enumerate(emb_fnames)}
    rows = []
    for fn in filenames:
        if fn in emb_map:
            rows.append(emb[emb_map[fn]])
        else:
            rows.append(np.zeros((emb.shape[1],), dtype=float))
    return np.vstack(rows)


def compute_metrics(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan')
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    brier = brier_score_loss(y_true, probs)
    return {'roc_auc': float(roc), 'precision': float(prec), 'f1': float(f1), 'confusion': (int(tn), int(fp), int(fn), int(tp)), 'brier': float(brier)}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--baseline', required=True, help='Path to baseline model joblib')
    parser.add_argument('--combined', required=True, help='Path to combined model joblib')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings joblib used by combined model')
    parser.add_argument('--pca-dim', type=int, default=None, help='If the combined model used PCA reduction before training, provide dim')
    parser.add_argument('--out', default=None, help='Optional path to save metrics JSON')
    args = parser.parse_args(argv or sys.argv[1:])

    repo = RatingsTagsRepository(path=os.path.join(args.image_dir, '.ratings_tags.json'))
    X, y, filenames = build_dataset(args.image_dir, repo)
    if X.size == 0:
        print('No labeled data')
        return 2

    emb_data = joblib.load(args.embeddings)
    emb = emb_data['embeddings']
    emb_fnames = emb_data['filenames']

    # If PCA dim provided but embeddings file is full, we assume combined model already used PCA; otherwise embeddings must match.
    if args.pca_dim is not None and emb.shape[1] != args.pca_dim:
        # Try to reduce embeddings to PCA dim here to match combined model
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.pca_dim, random_state=42)
        emb = pca.fit_transform(emb)

    emb_mat = align_embeddings_for_filenames(filenames, emb_fnames, emb)
    X_comb = np.hstack([X, emb_mat])

    # Load models
    baseline = load_model_any(args.baseline)
    combined = load_model_any(args.combined)

    # Obtain probabilities
    # Baseline may be saved with 'model' as pipeline that accepts X (features only)
    try:
        probs_baseline = baseline.predict_proba(X)[:, 1]
    except Exception as e:
        print('Baseline predict_proba failed:', e)
        probs_baseline = np.full(len(X), np.nan)

    try:
        probs_combined = combined.predict_proba(X_comb)[:, 1]
    except Exception as e:
        print('Combined predict_proba failed:', e)
        probs_combined = np.full(len(X), np.nan)

    # Ensemble: average of the two (ignoring nans)
    probs = np.nanmean(np.vstack([probs_baseline, probs_combined]), axis=0)

    metrics_baseline = compute_metrics(y, probs_baseline)
    metrics_combined = compute_metrics(y, probs_combined)
    metrics_ensemble = compute_metrics(y, probs)

    results = {'baseline': metrics_baseline, 'combined': metrics_combined, 'ensemble': metrics_ensemble}
    print('Evaluation results:')
    print(json.dumps(results, indent=2))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print('Saved metrics to', args.out)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

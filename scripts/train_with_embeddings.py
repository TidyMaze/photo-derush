#!/usr/bin/env python3
"""Train XGBoost on combined handcrafted features + CNN embeddings.

This script is a minimal experiment harness: it loads features via the project's
`build_dataset`, loads precomputed embeddings (joblib), concatenates them, and
fits the same pipeline (StandardScaler + XGBClassifier). It prints metrics for
easy comparison with the baseline training function.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.dataset import build_dataset
from src.model import RatingsTagsRepository

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


def load_embeddings(path: str):
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def align_and_concat(X_feats: np.ndarray, filenames: list[str], emb: np.ndarray, emb_fnames: list[str]):
    # Build a mapping from filename to embedding
    emb_map = {f: i for i, f in enumerate(emb_fnames)}
    rows = []
    kept_idx = []
    for i, fn in enumerate(filenames):
        if fn in emb_map:
            rows.append(emb[emb_map[fn]])
            kept_idx.append(i)
        else:
            rows.append(np.zeros((emb.shape[1],), dtype=float))
            kept_idx.append(i)
    emb_mat = np.vstack(rows)
    X_comb = np.hstack([X_feats, emb_mat])
    return X_comb


def train_and_eval(X: np.ndarray, y: np.ndarray):
    # Quick 80/20 split for eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = Pipeline([('scaler', StandardScaler()), ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return clf, {'precision': precision, 'f1': f1, 'roc_auc': roc, 'confusion': (int(tn), int(fp), int(fn), int(tp))}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings joblib')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset (fast mode)')
    parser.add_argument('--pca-dim', type=int, default=None, help='If set, reduce embeddings to this many dims using PCA')
    parser.add_argument('--model-out', default=None, help='Path to save resulting model joblib')
    args = parser.parse_args(argv or sys.argv[1:])

    # Build dataset (reuses manual-only logic)
    repo = RatingsTagsRepository(path=os.path.join(args.image_dir, '.ratings_tags.json'))
    X, y, filenames = build_dataset(args.image_dir, repo, displayed_filenames=None)
    if args.limit:
        X = X[: args.limit]
        y = y[: args.limit]
        filenames = filenames[: args.limit]

    emb, emb_fnames = load_embeddings(args.embeddings)
    # Optionally reduce embedding dimensionality
    if args.pca_dim:
        pca = PCA(n_components=args.pca_dim, random_state=42)
        # Fit PCA on embeddings available in file (not filtered by dataset)
        emb = pca.fit_transform(emb)
        print(f'Compressed embeddings to {args.pca_dim} dims')
    X_comb = align_and_concat(X, filenames, emb, emb_fnames)

    clf, metrics = train_and_eval(X_comb, y)

    print('Combined model metrics:')
    print(metrics)

    if args.model_out:
        # Build metadata similar to training_core._save_model so src.inference can validate and accept the model
        try:
            from src.model_version import create_model_metadata
            from src.features import FEATURE_COUNT, USE_FULL_FEATURES
        except Exception:
            create_model_metadata = None
            FEATURE_COUNT = X_comb.shape[1]
            USE_FULL_FEATURES = True

        metadata = None
        if create_model_metadata is not None:
            metadata = create_model_metadata(
                feature_count=FEATURE_COUNT,
                feature_mode='FULL' if USE_FULL_FEATURES else 'FAST',
                params={},
                n_samples=len(y),
            )

        data = {
            '__metadata__': metadata,
            'model': clf,
            'feature_length': int(X_comb.shape[1]),
            'n_samples': int(len(y)),
            'n_keep': int(int(np.sum(y == 1))),
            'n_trash': int(int(np.sum(y == 0))),
            'filenames': filenames,
            'precision': float(metrics.get('precision', 0.0)),
            'feature_importances': None,
        }
        # Save PCA transformer if used so inference or promotion can note it
        if args.pca_dim and 'pca' in locals() and locals().get('pca') is not None:
            data['pca'] = locals().get('pca')
            data['pca_dim'] = int(args.pca_dim)

        joblib.dump(data, args.model_out)
        print('Saved model to', args.model_out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Hyperparameter tuning for combined handcrafted+CNN-embedding XGBoost model.

Usage: scripts/tune_combined_embeddings.py IMAGE_DIR --embeddings path --fast
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from src.dataset import build_dataset
from src.model import RatingsTagsRepository


def load_embeddings(path: str):
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def align_and_concat(X_feats: np.ndarray, filenames: list[str], emb: np.ndarray, emb_fnames: list[str]):
    emb_map = {f: i for i, f in enumerate(emb_fnames)}
    rows = []
    for fn in filenames:
        if fn in emb_map:
            rows.append(emb[emb_map[fn]])
        else:
            rows.append(np.zeros((emb.shape[1],), dtype=float))
    emb_mat = np.vstack(rows)
    return np.hstack([X_feats, emb_mat])


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--out-params', default=os.path.expanduser('~/.photo_combined_best_params.json'))
    parser.add_argument('--model-out', default=os.path.expanduser('/tmp/photo_combined_tuned.joblib'))
    args = parser.parse_args(argv or sys.argv[1:])

    if args.fast:
        n_iter = 20
        cv_folds = 3
    else:
        n_iter = args.n_iter
        cv_folds = args.cv

    repo = RatingsTagsRepository(path=os.path.join(args.image_dir, '.ratings_tags.json'))
    X, y, filenames = build_dataset(args.image_dir, repo)
    if X.size == 0:
        print('No labeled data; aborting')
        return 2
    emb, emb_fnames = load_embeddings(args.embeddings)
    X_comb = align_and_concat(X, filenames, emb, emb_fnames)

    pipe = Pipeline([('scaler', StandardScaler()), ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'))])

    param_dist: dict[str, Any] = {
        'xgb__n_estimators': [50, 100, 200, 400],
        'xgb__max_depth': [3, 5, 6, 8],
        'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'xgb__subsample': [0.5, 0.7, 0.9, 1.0],
        'xgb__colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'xgb__gamma': [0, 0.1, 0.3],
        'xgb__min_child_weight': [1, 2, 4],
        'xgb__reg_lambda': [0, 0.1, 1.0],
        'xgb__reg_alpha': [0, 0.1, 1.0],
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rnd = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=2, random_state=42)

    print(f'Starting RandomizedSearchCV n_iter={n_iter} cv={cv_folds} on {X_comb.shape[0]} samples...')
    rnd.fit(X_comb, y)

    best = rnd.best_params_
    best_score = rnd.best_score_
    # Simplify keys (remove 'xgb__' prefix)
    best_simple = {k.replace('xgb__', ''): v for k, v in best.items()}
    print('Best ROC AUC (cv):', best_score)
    print('Best params:', best_simple)

    # Save params
    try:
        with open(args.out_params, 'w') as f:
            json.dump(best_simple, f, indent=2)
        print('Saved best params to', args.out_params)
    except Exception as e:
        print('Could not save params:', e)

    # Retrain final model on full data with best params
    final_params = {k.replace('xgb__', ''): v for k, v in best.items()}
    final_clf = Pipeline([('scaler', StandardScaler()), ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', **final_params))])
    final_clf.fit(X_comb, y)

    # Try build calibrator using small holdout
    try:
        if X_comb.shape[0] >= 30:
            from sklearn.model_selection import train_test_split
            X_train_small, X_calib, y_train_small, y_calib = train_test_split(X_comb, y, test_size=0.10, stratify=y, random_state=42)
            # final_clf is already fitted on full data; use cv='prefit'
            try:
                calib = CalibratedClassifierCV(final_clf, cv='prefit', method='sigmoid')
                calib.fit(X_calib, y_calib)
                calib_path = args.model_out + '.calib.joblib'
                joblib.dump(calib, calib_path)
                print('Saved calibrator to', calib_path)
            except Exception as e:
                print('Calibrator build failed:', e)
    except Exception:
        pass

    # Save final model
    try:
        joblib.dump({'model': final_clf, 'filenames': filenames}, args.model_out)
        print('Saved final model to', args.model_out)
    except Exception as e:
        print('Could not save model:', e)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

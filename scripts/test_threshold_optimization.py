#!/usr/bin/env python3
"""Test different probability thresholds to optimize F1 or accuracy."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params
import os

# Load data
image_dir = os.path.expanduser("~/Pictures/photo-dataset")
repo_path = os.path.join(image_dir, ".ratings_tags.json")
repo = RatingsTagsRepository(path=repo_path)
X, y, _ = build_dataset(image_dir, repo=repo)
X = np.array(X)
y = np.array(y)

train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

xgb_params = load_best_params() or {}
xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
n_keep, n_trash = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", xgb.XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight, objective="binary:logistic", eval_metric="logloss", **xgb_params)),
])
clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)[:, 1]

print("Testing different probability thresholds:")
best_f1 = 0
best_thresh = 0.5
best_acc = 0

for thresh in np.arange(0.3, 0.8, 0.05):
    y_pred = (y_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_acc = acc
    print(f"  Threshold {thresh:.2f}: Acc={acc:.4f}, F1={f1:.4f}")

baseline_pred = clf.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

acc_diff = best_acc - baseline_acc
print(f"\nBaseline (threshold=0.5): Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
print(f"Best F1 (threshold={best_thresh:.2f}): Acc={best_acc:.4f}, F1={best_f1:.4f}")
print(f"Accuracy change: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")


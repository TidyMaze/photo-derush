"""Focused grid search on key interacting hyperparameters.

Usage:
  python -m src.combined_search /path/to/images --params learning_rate=0.05,0.1 subsample=0.7,0.9 colsample_bytree=0.6,0.8 --n-est 100,150 --metric accuracy
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import build_dataset
from .model import RatingsTagsRepository
from .tuning import load_best_params

DEFAULT_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.8],
    "n_estimators": [100, 150, 200],
}


def _parse_list(raw: str):
    return [float(v) if "." in v else int(v) for v in raw.split(",") if v.strip()]


def parse_args(argv=None):  # pragma: no cover
    p = argparse.ArgumentParser(description="Focused multi-param grid search.")
    p.add_argument("image_dir")
    p.add_argument("--params", nargs="*", help="Param specs name=v1,v2,...")
    p.add_argument("--metric", default="accuracy")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):  # pragma: no cover
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if not os.path.isdir(args.image_dir):
        logging.error("Image dir missing")
        return 2
    repo_path = os.path.join(args.image_dir, ".ratings_tags.json")
    if not os.path.exists(repo_path):
        logging.error("Labels file missing")
        return 2
    repo = RatingsTagsRepository(path=repo_path)
    X, y, _ = build_dataset(args.image_dir, repo)
    if len(y) < args.cv_folds * 2 or len(set(y)) < 2:
        logging.error("Insufficient data for CV")
        return 2
    tuned = load_best_params() or {}
    base = {k: v for k, v in tuned.items() if not k.startswith("_")}
    grid: dict[str, list[Any]] = {}
    if args.params:
        for spec in args.params:
            if "=" not in spec:
                continue
            k, raw = spec.split("=", 1)
            grid[k.strip()] = _parse_list(raw.strip())
    else:
        grid = DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    logging.info("Total combos: %d", len(combos))
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    results = []
    for combo in combos:
        params = dict(base)
        for k, v in zip(keys, combo):
            params[k] = v
        logging.info("Testing %s", params)
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=42, n_jobs=4, objective="binary:logistic", eval_metric="logloss", **params
                    ),
                ),
            ]
        )
        scores = []
        for tr_idx, te_idx in cv.split(X, y):
            Xt, Xv = X[tr_idx], X[te_idx]
            yt, yv = y[tr_idx], y[te_idx]
            clf.fit(Xt, yt)
            preds = clf.predict(Xv)
            scores.append(accuracy_score(yv, preds))
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        results.append({"params": params, "mean": mean, "std": std})
    results.sort(key=lambda r: r["mean"], reverse=True)
    out_path = os.path.join(os.getcwd(), "sweeps", "combined_search_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved combined search results -> %s", out_path)
    if results:
        best = results[0]
        logging.info("BEST %s mean=%.4f std=%.4f", best["params"], best["mean"], best["std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

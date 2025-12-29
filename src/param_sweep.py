"""Parameter sweep utilities for exploring single hyperparameter impact.

Provides programmatic API + CLI to vary one XGBoost parameter while keeping other
(tuned or default) parameters fixed and produce a score plot.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import build_dataset
from .model import RatingsTagsRepository
from .tuning import load_best_params

SweepResult = dict[str, Any]

# ---------------------- Core Sweep Logic ---------------------- #


def sweep_hyperparameter(
    image_dir: str,
    param: str,
    values: Sequence[Any],
    cv_folds: int = 3,
    random_state: int = 42,
    metric: str = "accuracy",
    use_tuned: bool = True,
    repo: RatingsTagsRepository | None = None,
    progress_callback=None,
) -> list[SweepResult]:
    """Run a sweep varying a single XGBoost hyperparameter.

    Returns list of result dicts with keys: value, mean, std, elapsed, n_samples, n_keep, n_trash.
    If scoring cannot be computed (insufficient data) mean/std are None.
    """
    logging.info(f"[sweep] param={param} values={values} dir={image_dir}")
    if repo is None:
        repo_path = os.path.join(image_dir, ".ratings_tags.json")
        repo = RatingsTagsRepository(path=repo_path)
        logging.debug(f"[sweep] Scoped repo {repo_path}")

    X, y, _ = build_dataset(image_dir, repo, progress_callback=progress_callback)
    n_samples = len(y)
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    logging.info(f"[sweep] Dataset n={n_samples} keep={n_keep} trash={n_trash}")

    if n_samples < cv_folds * 2 or n_keep == 0 or n_trash == 0:
        logging.warning("[sweep] Insufficient data for CV; returning empty results")
        return []

    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    tuned = load_best_params() if use_tuned else None
    base_params = {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 200}
    if tuned:
        for k, v in tuned.items():
            if not k.startswith("_"):
                base_params[k] = v
        logging.info(f"[sweep] Using tuned baseline params: {base_params}")
    else:
        logging.info(f"[sweep] Using default baseline params: {base_params}")

    results: list[SweepResult] = []
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for val in values:
        # Force integer casting for discrete integer hyperparameters
        if param in {"max_depth", "n_estimators", "min_child_weight"}:
            try:
                val = int(val)
            except Exception:
                logging.warning(f"[sweep] Could not cast {param} value {val} to int; using raw")
        params = dict(base_params)
        params[param] = val
        logging.info(f"[sweep] Testing {param}={val} (type={type(val).__name__})")
        t0 = time.perf_counter()
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=random_state,
                        n_jobs=4,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **params,
                    ),
                ),
            ]
        )
        mean = std = None
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring=metric)
            mean = float(scores.mean())
            std = float(scores.std())
            logging.info(f"[sweep] {param}={val} mean={mean:.4f} std={std:.4f}")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[sweep] CV failed for {param}={val}: {e}")
        elapsed = time.perf_counter() - t0
        results.append(
            {
                "param": param,
                "value": val,
                "mean": mean,
                "std": std,
                "elapsed": elapsed,
                "n_samples": n_samples,
                "n_keep": n_keep,
                "n_trash": n_trash,
            }
        )
    return results


# ---------------------- Plotting ---------------------- #


def plot_sweep(results: Sequence[SweepResult], param: str, output_path: str | None = None):
    """Generate and save a line/bar plot for sweep results.
    Uses mean score; error bars show std.
    """
    if not results:
        logging.warning("[sweep] No results to plot")
        return None
    plotable = [r for r in results if r.get("mean") is not None]
    if not plotable:
        logging.warning("[sweep] No successful scores to plot")
        return None
    xs = [r["value"] for r in plotable]
    ys = [r["mean"] for r in plotable]
    err = [r["std"] for r in plotable]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(xs, ys, yerr=err, fmt="-o", capsize=4)
    ax.set_xlabel(param)
    ax.set_ylabel("CV accuracy")
    ax.set_title(f"Hyperparameter sweep: {param}")
    for x, y in zip(xs, ys):
        ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if output_path is None:
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, f"param_sweep_{param}.png")
    fig.savefig(output_path)
    logging.info(f"[sweep] Plot saved: {output_path}\n")
    plt.close(fig)
    return output_path


# ---------------------- CLI ---------------------- #


def _parse_cli(argv: Sequence[str]):  # pragma: no cover
    p = argparse.ArgumentParser(description="Sweep a single XGBoost hyperparameter and plot results.")
    p.add_argument("image_dir", help="Directory with labeled images + .ratings_tags.json")
    p.add_argument("--param", required=True, help="Hyperparameter name (e.g. max_depth)")
    p.add_argument(
        "--values", required=True, help="Comma separated values (e.g. 2,3,4,5) or range syntax start:end:step"
    )
    p.add_argument("--cv-folds", type=int, default=3, help="CV folds (default: 3)")
    p.add_argument("--metric", default="accuracy", help="Scoring metric (default: accuracy)")
    p.add_argument("--no-tuned", action="store_true", help="Ignore previously tuned params baseline")
    p.add_argument("--plot", action="store_true", help="Generate plot image")
    p.add_argument("--out", help="Explicit output plot path")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def _parse_values(raw: str):
    raw = raw.strip()
    if ":" in raw and "," not in raw:
        # range syntax a:b(:step optional)
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("Range syntax must be start:end or start:end:step")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        vals = []
        v = start
        # inclusive if step divides
        while v <= end + 1e-9:
            vals.append(v)
            v += step
        # Convert to ints if all are integer-like
        if all(abs(x - round(x)) < 1e-9 for x in vals):
            vals = [int(round(x)) for x in vals]
        return vals
    # comma separated
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # try int then float then raw string
        try:
            out.append(int(token))
            continue
        except ValueError:
            pass
        try:
            out.append(float(token))  # type: ignore[arg-type]
            continue
        except ValueError:
            pass
        out.append(token)  # type: ignore[arg-type]
    return out


def main(argv: Sequence[str] | None = None):  # pragma: no cover
    args = _parse_cli(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        values = _parse_values(args.values)
    except Exception as e:
        logging.error(f"Failed parsing values: {e}")
        return 2
    results = sweep_hyperparameter(
        image_dir=args.image_dir,
        param=args.param,
        values=values,
        cv_folds=args.cv_folds,
        metric=args.metric,
        use_tuned=not args.no_tuned,
    )
    if not results:
        logging.error("No results produced (insufficient data?)")
        return 1
    # Log summary
    for r in results:
        if r["mean"] is not None:
            logging.info(
                f"{args.param}={r['value']} mean={r['mean']:.4f} std={r['std']:.4f} elapsed={r['elapsed']:.2f}s"
            )
        else:
            logging.info(f"{args.param}={r['value']} score=NA elapsed={r['elapsed']:.2f}s")
    if args.plot:
        out = plot_sweep(results, args.param, output_path=args.out)
        if out:
            logging.info(f"Plot written: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

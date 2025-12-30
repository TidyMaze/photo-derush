#!/usr/bin/env python3
"""Generate comprehensive summary report with charts and metrics from benchmark results.

Usage:
    poetry run python scripts/generate_summary_report.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import BASELINE_PATH, BENCHMARK_RESULTS_PATH, load_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("summary_report")


def load_json_file(path: Path) -> dict | list | None:
    """Load JSON file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load {path}: {e}")
        return None


def generate_summary_charts(results_data: dict, output_dir: Path) -> list[Path]:
    """Generate summary charts from benchmark results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("Matplotlib not available, skipping charts")
        return []

    chart_files = []
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Algorithm Comparison Chart
    if results_data.get("algorithm_comparison"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Algorithm Comparison", fontsize=16, fontweight="bold")

        algorithms = []
        accuracy = []
        f1 = []
        roc_auc = []
        precision = []

        for result in results_data["algorithm_comparison"]:
            if "error" in result:
                continue
            algorithms.append(result.get("algorithm", "unknown"))
            metrics = result.get("metrics", {})
            accuracy.append(metrics.get("accuracy", 0.0))
            f1.append(metrics.get("f1", 0.0))
            roc_auc.append(metrics.get("roc_auc", 0.0) if isinstance(metrics.get("roc_auc"), (int, float)) else 0.0)
            precision.append(metrics.get("precision", 0.0))

        if algorithms:
            x_pos = np.arange(len(algorithms))
            width = 0.35

            # Accuracy comparison
            axes[0, 0].bar(x_pos, accuracy, width, label="Accuracy", color="#4CAF50")
            axes[0, 0].set_xlabel("Algorithm")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].set_title("Accuracy Comparison")
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(algorithms, rotation=45, ha="right")
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].grid(True, alpha=0.3)

            # F1 Score comparison
            axes[0, 1].bar(x_pos, f1, width, label="F1", color="#2196F3")
            axes[0, 1].set_xlabel("Algorithm")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].set_title("F1 Score Comparison")
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(algorithms, rotation=45, ha="right")
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)

            # ROC-AUC comparison
            axes[1, 0].bar(x_pos, roc_auc, width, label="ROC-AUC", color="#FF9800")
            axes[1, 0].set_xlabel("Algorithm")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].set_title("ROC-AUC Comparison")
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(algorithms, rotation=45, ha="right")
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)

            # Precision comparison
            axes[1, 1].bar(x_pos, precision, width, label="Precision", color="#9C27B0")
            axes[1, 1].set_xlabel("Algorithm")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_title("Precision Comparison")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(algorithms, rotation=45, ha="right")
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / "algorithm_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    # 2. Feature Ablation Chart
    if results_data.get("feature_ablation"):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Feature Ablation Study", fontsize=16, fontweight="bold")

        groups = []
        accuracies = []
        deltas = []
        baseline_acc = None

        for result in results_data["feature_ablation"]:
            group_name = result.get("group_name", "unknown")
            metrics = result.get("metrics", {})
            acc = metrics.get("accuracy", 0.0)
            groups.append(group_name.replace("no_", "").replace("_", " ").title())
            accuracies.append(acc)
            if group_name == "all":
                baseline_acc = acc
                deltas.append(0.0)
            else:
                delta = acc - baseline_acc if baseline_acc else 0.0
                deltas.append(delta)

        if groups:
            colors = ["#4CAF50" if d >= 0 else "#f44336" for d in deltas]
            bars = ax.barh(groups, accuracies, color=colors, alpha=0.7)
            ax.axvline(x=baseline_acc if baseline_acc else 0, color="red", linestyle="--", label="Baseline", linewidth=2)
            ax.set_xlabel("Accuracy")
            ax.set_title("Accuracy by Feature Group Removal")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")

            # Add delta labels
            for i, (bar, delta) in enumerate(zip(bars, deltas)):
                if delta != 0:
                    label = f"{delta:+.3f}"
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)

        plt.tight_layout()
        chart_path = output_dir / "feature_ablation.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    # 3. Ensemble Methods Comparison
    if results_data.get("ensemble_results"):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Ensemble Methods Comparison", fontsize=16, fontweight="bold")

        methods = []
        accuracies = []
        f1_scores = []

        for result in results_data["ensemble_results"]:
            method = result.get("method", "unknown")
            metrics = result.get("metrics", {})
            methods.append(method.replace("_", " ").title())
            accuracies.append(metrics.get("accuracy", 0.0))
            f1_scores.append(metrics.get("f1", 0.0))

        if methods:
            x_pos = np.arange(len(methods))
            width = 0.35

            bars1 = ax.bar(x_pos - width / 2, accuracies, width, label="Accuracy", color="#4CAF50", alpha=0.8)
            bars2 = ax.bar(x_pos + width / 2, f1_scores, width, label="F1 Score", color="#2196F3", alpha=0.8)

            ax.set_xlabel("Ensemble Method")
            ax.set_ylabel("Score")
            ax.set_title("Ensemble Methods Performance")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = output_dir / "ensemble_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    # 4. Calibration Methods Comparison
    if results_data.get("calibration_results"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Calibration Methods Comparison", fontsize=16, fontweight="bold")

        methods = []
        brier_scores = []
        eces = []
        baseline_brier = None

        for result in results_data["calibration_results"]:
            method = result.get("method", "unknown")
            brier = result.get("brier_score", 0.0)
            ece = result.get("ece", 0.0)
            methods.append(method.replace("_", " ").title())
            brier_scores.append(brier)
            eces.append(ece)
            if method == "uncalibrated":
                baseline_brier = brier

        if methods:
            x_pos = np.arange(len(methods))
            colors = ["#f44336" if m == "Uncalibrated" else "#4CAF50" for m in methods]

            axes[0].bar(x_pos, brier_scores, color=colors, alpha=0.7)
            if baseline_brier:
                axes[0].axhline(y=baseline_brier, color="red", linestyle="--", label="Baseline", linewidth=2)
            axes[0].set_xlabel("Method")
            axes[0].set_ylabel("Brier Score (lower is better)")
            axes[0].set_title("Brier Score Comparison")
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(methods, rotation=45, ha="right")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis="y")

            axes[1].bar(x_pos, eces, color=colors, alpha=0.7)
            axes[1].set_xlabel("Method")
            axes[1].set_ylabel("Expected Calibration Error (lower is better)")
            axes[1].set_title("ECE Comparison")
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(methods, rotation=45, ha="right")
            axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = output_dir / "calibration_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    # 5. Class Balance Strategies
    if results_data.get("class_balance_results"):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Class Balance Strategies Comparison", fontsize=16, fontweight="bold")

        methods = []
        accuracies = []
        precisions = []
        recalls = []

        for result in results_data["class_balance_results"]:
            method = result.get("method", "unknown")
            metrics = result.get("metrics", {})
            methods.append(method.replace("_", " ").title())
            accuracies.append(metrics.get("accuracy", 0.0))
            precisions.append(metrics.get("precision", 0.0))
            recalls.append(metrics.get("recall", 0.0))

        if methods:
            x_pos = np.arange(len(methods))
            width = 0.25

            ax.bar(x_pos - width, accuracies, width, label="Accuracy", color="#4CAF50", alpha=0.8)
            ax.bar(x_pos, precisions, width, label="Precision", color="#2196F3", alpha=0.8)
            ax.bar(x_pos + width, recalls, width, label="Recall", color="#FF9800", alpha=0.8)

            ax.set_xlabel("Method")
            ax.set_ylabel("Score")
            ax.set_title("Class Balance Strategies Performance")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = output_dir / "class_balance_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    # 6. Overall Metrics Summary (if baseline exists)
    if results_data.get("baseline"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Baseline Model Performance Summary", fontsize=16, fontweight="bold")

        baseline_metrics = results_data["baseline"].get("metrics", {})
        metric_names = ["accuracy", "precision", "recall", "f1"]
        metric_values = [baseline_metrics.get(m, 0.0) for m in metric_names]

        # Pie chart of metrics
        axes[0, 0].pie(metric_values, labels=metric_names, autopct="%1.3f", startangle=90)
        axes[0, 0].set_title("Metrics Distribution")

        # Bar chart
        axes[0, 1].bar(metric_names, metric_values, color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"], alpha=0.7)
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Metric Scores")
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Confusion matrix visualization
        cm_data = {
            "TN": baseline_metrics.get("confusion_tn", 0),
            "FP": baseline_metrics.get("confusion_fp", 0),
            "FN": baseline_metrics.get("confusion_fn", 0),
            "TP": baseline_metrics.get("confusion_tp", 0),
        }
        cm_values = [cm_data["TN"], cm_data["FP"], cm_data["FN"], cm_data["TP"]]
        cm_labels = ["TN", "FP", "FN", "TP"]
        axes[1, 0].bar(cm_labels, cm_values, color=["#4CAF50", "#f44336", "#f44336", "#4CAF50"], alpha=0.7)
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # ROC-AUC and Brier Score
        roc_auc = baseline_metrics.get("roc_auc", 0.0)
        brier = baseline_metrics.get("brier_score", 0.0)
        axes[1, 1].bar(["ROC-AUC", "Brier Score"], [roc_auc, 1 - brier], color=["#4CAF50", "#2196F3"], alpha=0.7)
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Probability Metrics")
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = output_dir / "baseline_summary.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(chart_path)

    return chart_files


def generate_text_summary(results_data: dict, output_path: Path) -> None:
    """Generate text summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL BENCHMARKING STUDY - SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Baseline
    if results_data.get("baseline"):
        baseline = results_data["baseline"]
        lines.append("BASELINE MODEL PERFORMANCE")
        lines.append("-" * 80)
        metrics = baseline.get("metrics", {})
        lines.append(f"Model: {baseline.get('model_name', 'Unknown')}")
        lines.append(f"Feature Mode: {baseline.get('feature_mode', 'Unknown')} ({baseline.get('feature_count', 0)} features)")
        lines.append(f"Train Size: {baseline.get('train_size', 0)}, Test Size: {baseline.get('test_size', 0)}")
        lines.append("")
        lines.append("Metrics:")
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]:
            if metric_name in metrics:
                value = metrics[metric_name]
                lines.append(f"  {metric_name:15s}: {value:.4f}")
        lines.append("")
        lines.append("Confusion Matrix:")
        lines.append(f"  TN={metrics.get('confusion_tn', 0)}, FP={metrics.get('confusion_fp', 0)}")
        lines.append(f"  FN={metrics.get('confusion_fn', 0)}, TP={metrics.get('confusion_tp', 0)}")
        lines.append("")

    # Algorithm Comparison
    if results_data.get("algorithm_comparison"):
        lines.append("ALGORITHM COMPARISON")
        lines.append("-" * 80)
        lines.append(f"{'Algorithm':<20} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12} {'Precision':<12}")
        lines.append("-" * 80)
        for result in results_data["algorithm_comparison"]:
            if "error" in result:
                continue
            algo = result.get("algorithm", "unknown")
            metrics = result.get("metrics", {})
            acc = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            roc_auc = metrics.get("roc_auc", "N/A")
            precision = metrics.get("precision", 0.0)
            roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
            lines.append(f"{algo:<20} {acc:<12.4f} {f1:<12.4f} {roc_str:<12} {precision:<12.4f}")
        lines.append("")

    # Feature Ablation
    if results_data.get("feature_ablation"):
        lines.append("FEATURE ABLATION STUDY")
        lines.append("-" * 80)
        baseline_acc = None
        for result in results_data["feature_ablation"]:
            group = result.get("group_name", "unknown")
            metrics = result.get("metrics", {})
            acc = metrics.get("accuracy", 0.0)
            if group == "all":
                baseline_acc = acc
            delta = acc - baseline_acc if baseline_acc and group != "all" else 0.0
            delta_str = f"{delta:+.4f}" if group != "all" else "baseline"
            lines.append(f"{group:<25} Accuracy: {acc:.4f} (Δ: {delta_str})")
        lines.append("")

    # Ensemble Methods
    if results_data.get("ensemble_results"):
        lines.append("ENSEMBLE METHODS")
        lines.append("-" * 80)
        lines.append(f"{'Method':<25} {'Accuracy':<12} {'F1':<12}")
        lines.append("-" * 80)
        for result in results_data["ensemble_results"]:
            method = result.get("method", "unknown")
            metrics = result.get("metrics", {})
            acc = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            lines.append(f"{method:<25} {acc:<12.4f} {f1:<12.4f}")
        lines.append("")

    # Calibration
    if results_data.get("calibration_results"):
        lines.append("CALIBRATION METHODS")
        lines.append("-" * 80)
        baseline_brier = None
        for result in results_data["calibration_results"]:
            method = result.get("method", "unknown")
            brier = result.get("brier_score", 0.0)
            ece = result.get("ece", 0.0)
            if method == "uncalibrated":
                baseline_brier = brier
            improvement = baseline_brier - brier if baseline_brier and method != "uncalibrated" else 0.0
            improvement_str = f"{improvement:+.4f}" if method != "uncalibrated" else "baseline"
            lines.append(f"{method:<25} Brier: {brier:.4f}, ECE: {ece:.4f} (Δ: {improvement_str})")
        lines.append("")

    # Key Findings
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)
    if results_data.get("algorithm_comparison"):
        best_algo = max(
            [r for r in results_data["algorithm_comparison"] if "error" not in r],
            key=lambda x: x.get("metrics", {}).get("accuracy", 0.0),
            default=None,
        )
        if best_algo:
            lines.append(f"Best Algorithm: {best_algo.get('algorithm')} (Accuracy: {best_algo.get('metrics', {}).get('accuracy', 0.0):.4f})")

    if results_data.get("ensemble_results"):
        best_ensemble = max(
            results_data["ensemble_results"],
            key=lambda x: x.get("metrics", {}).get("accuracy", 0.0),
            default=None,
        )
        if best_ensemble:
            lines.append(f"Best Ensemble: {best_ensemble.get('method')} (Accuracy: {best_ensemble.get('metrics', {}).get('accuracy', 0.0):.4f})")

    lines.append("")
    lines.append("=" * 80)

    output_path.write_text("\n".join(lines))
    log.info(f"Generated text summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive summary report")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)

    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = cache_dir / "summary_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading benchmark results from {cache_dir}")

    # Load all result files
    results_data = {
        "baseline": load_json_file(BASELINE_PATH),
        "benchmark_results": load_json_file(BENCHMARK_RESULTS_PATH),
        "feature_ablation": load_json_file(cache_dir / "feature_ablation_results.json"),
        "algorithm_comparison": load_json_file(cache_dir / "algorithm_comparison_results.json"),
        "ensemble_results": load_json_file(cache_dir / "ensemble_results.json"),
        "calibration_results": load_json_file(cache_dir / "calibration_results.json"),
        "class_balance_results": load_json_file(cache_dir / "class_balance_results.json"),
        "augmentation_results": load_json_file(cache_dir / "augmentation_results.json"),
        "evaluation_results": load_json_file(cache_dir / "evaluation" / "full_evaluation_results.json"),
    }

    # Check if we have any results
    has_results = any(results_data.values())
    if not has_results:
        log.warning("No benchmark results found. Run benchmark scripts first.")
        log.info("Example: poetry run python scripts/benchmark_model.py --baseline")
        return 1

    # Generate charts
    log.info("Generating summary charts...")
    chart_files = generate_summary_charts(results_data, output_dir)
    log.info(f"Generated {len(chart_files)} charts")

    # Generate text summary
    log.info("Generating text summary...")
    summary_path = output_dir / "summary_report.txt"
    generate_text_summary(results_data, summary_path)

    # Save JSON summary
    summary_json_path = output_dir / "summary_data.json"
    with open(summary_json_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    log.info(f"\nSummary report generated in: {output_dir}")
    log.info(f"  - Text report: {summary_path}")
    log.info(f"  - JSON data: {summary_json_path}")
    log.info(f"  - Charts: {len(chart_files)} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


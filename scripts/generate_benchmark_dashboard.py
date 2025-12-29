#!/usr/bin/env python3
"""Generate HTML dashboard with metrics comparison, feature importance, and error analysis.

Usage:
    poetry run python scripts/generate_benchmark_dashboard.py [--baseline PATH] [--results PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import BASELINE_PATH, BENCHMARK_RESULTS_PATH, load_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("dashboard")


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


def embed_image(path: Path) -> str:
    """Embed image as base64 data URI."""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
            ext = path.suffix[1:] or "png"
            return f"data:image/{ext};base64,{data}"
    except Exception as e:
        log.warning(f"Failed to embed image {path}: {e}")
        return ""


def generate_dashboard_html(
    baseline: dict | None,
    benchmark_results: list[dict] | None,
    feature_ablation: list[dict] | None,
    algorithm_comparison: list[dict] | None,
    ensemble_results: list[dict] | None,
    calibration_results: list[dict] | None,
    evaluation_results: dict | None,
    output_path: Path,
) -> None:
    """Generate HTML dashboard."""
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Model Benchmark Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric { font-weight: bold; color: #2196F3; }
        .improvement { color: #4CAF50; }
        .degradation { color: #f44336; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .section { margin: 30px 0; }
        .error-list { max-height: 300px; overflow-y: auto; }
        .error-item { padding: 5px; margin: 5px 0; background: #fff3cd; border-left: 3px solid #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Benchmark Dashboard</h1>
        <p>Generated: <span id="timestamp"></span></p>
"""
    ]

    # Baseline metrics
    if baseline:
        html_parts.append('<div class="section"><h2>Baseline Metrics</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        for metric_name, value in baseline.get("metrics", {}).items():
            if not metric_name.startswith("confusion_"):
                html_parts.append(f"<tr><td>{metric_name}</td><td class='metric'>{value:.4f}</td></tr>")
        html_parts.append("</table></div>")

    # Benchmark results comparison
    if benchmark_results and len(benchmark_results) > 0:
        html_parts.append('<div class="section"><h2>Benchmark Results</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Timestamp</th><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr>")
        for result in benchmark_results[-10:]:  # Last 10 results
            timestamp = result.get("timestamp", "unknown")[:19] if result.get("timestamp") else "unknown"
            model_name = result.get("model_name", "unknown")
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            roc_auc = metrics.get("roc_auc", "N/A")
            roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
            html_parts.append(
                f"<tr><td>{timestamp}</td><td>{model_name}</td><td>{accuracy:.4f}</td><td>{f1:.4f}</td><td>{roc_str}</td></tr>"
            )
        html_parts.append("</table></div>")

    # Feature ablation results
    if feature_ablation:
        html_parts.append('<div class="section"><h2>Feature Ablation Study</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature Group</th><th>Features Used</th><th>Accuracy</th><th>Delta</th><th>F1</th></tr>")
        baseline_acc = None
        for result in feature_ablation:
            group_name = result.get("group_name", "unknown")
            features_used = result.get("features_used", 0)
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            if group_name == "all":
                baseline_acc = accuracy
                delta_str = "baseline"
            else:
                delta = accuracy - baseline_acc if baseline_acc else 0.0
                delta_str = f"{delta:+.4f}" if baseline_acc else "N/A"
            html_parts.append(
                f"<tr><td>{group_name}</td><td>{features_used}</td><td>{accuracy:.4f}</td><td>{delta_str}</td><td>{f1:.4f}</td></tr>"
            )
        html_parts.append("</table></div>")

    # Algorithm comparison
    if algorithm_comparison:
        html_parts.append('<div class="section"><h2>Algorithm Comparison</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Algorithm</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Precision</th></tr>")
        for result in algorithm_comparison:
            if "error" in result:
                continue
            algo_name = result.get("algorithm", "unknown")
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            roc_auc = metrics.get("roc_auc", "N/A")
            precision = metrics.get("precision", 0.0)
            roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
            html_parts.append(
                f"<tr><td>{algo_name}</td><td>{accuracy:.4f}</td><td>{f1:.4f}</td><td>{roc_str}</td><td>{precision:.4f}</td></tr>"
            )
        html_parts.append("</table></div>")

    # Ensemble results
    if ensemble_results:
        html_parts.append('<div class="section"><h2>Ensemble Methods</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Method</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th></tr>")
        for result in ensemble_results:
            method = result.get("method", "unknown")
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            f1 = metrics.get("f1", 0.0)
            roc_auc = metrics.get("roc_auc", "N/A")
            roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
            html_parts.append(
                f"<tr><td>{method}</td><td>{accuracy:.4f}</td><td>{f1:.4f}</td><td>{roc_str}</td></tr>"
            )
        html_parts.append("</table></div>")

    # Calibration results
    if calibration_results:
        html_parts.append('<div class="section"><h2>Calibration Methods</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Method</th><th>Brier Score</th><th>ECE</th><th>Improvement</th></tr>")
        baseline_brier = None
        for result in calibration_results:
            method = result.get("method", "unknown")
            brier = result.get("brier_score", 0.0)
            ece = result.get("ece", 0.0)
            if method == "uncalibrated":
                baseline_brier = brier
                improvement_str = "baseline"
            else:
                improvement = baseline_brier - brier if baseline_brier else 0.0
                improvement_str = f"{improvement:+.4f}" if baseline_brier else "N/A"
            html_parts.append(
                f"<tr><td>{method}</td><td>{brier:.4f}</td><td>{ece:.4f}</td><td>{improvement_str}</td></tr>"
            )
        html_parts.append("</table></div>")

    # Evaluation plots
    if evaluation_results:
        html_parts.append('<div class="section"><h2>Evaluation Plots</h2>')
        plot_files = evaluation_results.get("plot_files", [])
        for plot_file in plot_files:
            plot_path = Path(plot_file)
            if plot_path.exists():
                img_data = embed_image(plot_path)
                if img_data:
                    html_parts.append(f'<div class="plot-container"><h3>{plot_path.name}</h3>')
                    html_parts.append(f'<img src="{img_data}" alt="{plot_path.name}"></div>')

        # Error analysis
        error_analysis = evaluation_results.get("error_analysis", {})
        if error_analysis:
            html_parts.append('<div class="section"><h2>Error Analysis</h2>')
            html_parts.append(f'<p>False Positives: {len(error_analysis.get("false_positives", []))}</p>')
            html_parts.append(f'<p>False Negatives: {len(error_analysis.get("false_negatives", []))}</p>')
            html_parts.append('<div class="error-list">')
            html_parts.append("<h3>Top False Positives</h3>")
            for err in error_analysis.get("false_positives", [])[:10]:
                html_parts.append(
                    f'<div class="error-item">{err.get("filename", "unknown")}: p={err.get("probability", 0.0):.4f}</div>'
                )
            html_parts.append("<h3>Top False Negatives</h3>")
            for err in error_analysis.get("false_negatives", [])[:10]:
                html_parts.append(
                    f'<div class="error-item">{err.get("filename", "unknown")}: p={err.get("probability", 0.0):.4f}</div>'
                )
            html_parts.append("</div></div>")

    # Feature importance
    if baseline and baseline.get("feature_importances"):
        html_parts.append('<div class="section"><h2>Feature Importance (Baseline)</h2>')
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature Index</th><th>Importance</th></tr>")
        for idx, importance in baseline.get("feature_importances", [])[:20]:
            html_parts.append(f"<tr><td>{idx}</td><td>{importance:.4f}</td></tr>")
        html_parts.append("</table></div>")

    html_parts.append(
        """
        <script>
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
        </script>
    </div>
</body>
</html>
"""
    )

    html_content = "\n".join(html_parts)
    output_path.write_text(html_content)
    log.info(f"Generated dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard")
    parser.add_argument("--baseline", default=None, help="Path to baseline JSON file")
    parser.add_argument("--results", default=None, help="Path to benchmark results JSON file")
    parser.add_argument("--out", default=None, help="Output HTML file path")
    args = parser.parse_args()

    # Load data files
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)

    baseline_path = Path(args.baseline) if args.baseline else BASELINE_PATH
    results_path = Path(args.results) if args.results else BENCHMARK_RESULTS_PATH

    baseline_data = load_json_file(baseline_path)
    benchmark_results = load_json_file(results_path)

    # Load additional result files
    feature_ablation = load_json_file(cache_dir / "feature_ablation_results.json")
    algorithm_comparison = load_json_file(cache_dir / "algorithm_comparison_results.json")
    ensemble_results = load_json_file(cache_dir / "ensemble_results.json")
    calibration_results = load_json_file(cache_dir / "calibration_results.json")
    evaluation_results = load_json_file(cache_dir / "evaluation" / "full_evaluation_results.json")

    # Output path
    if args.out:
        output_path = Path(args.out)
    else:
        output_path = cache_dir / "benchmark_dashboard.html"

    # Generate dashboard
    generate_dashboard_html(
        baseline_data,
        benchmark_results,
        feature_ablation,
        algorithm_comparison,
        ensemble_results,
        calibration_results,
        evaluation_results,
        output_path,
    )

    log.info(f"Dashboard generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


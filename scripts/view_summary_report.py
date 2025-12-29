#!/usr/bin/env python3
"""Open summary report in browser with all charts and metrics.

Usage:
    poetry run python scripts/view_summary_report.py
"""

import base64
import json
import logging
import sys
import webbrowser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("view_report")


def embed_image(path: Path) -> str:
    """Embed image as base64 data URI."""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
            # Return just the base64 data, not the full data URI (will be added in HTML)
            return data
    except Exception as e:
        log.warning(f"Failed to embed image {path}: {e}")
        return ""


def generate_html_viewer(summary_dir: Path) -> Path:
    """Generate HTML viewer for summary report."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Model Benchmarking Study - Summary Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 50px;
        }
        .section h2 {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card .label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .improvement {
            color: #4CAF50;
            font-weight: bold;
        }
        .degradation {
            color: #f44336;
            font-weight: bold;
        }
        .findings {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin-top: 20px;
        }
        .findings h3 {
            color: #1976D2;
            margin-bottom: 10px;
        }
        .findings ul {
            list-style-position: inside;
            line-height: 1.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Model Benchmarking Study</h1>
            <p>Comprehensive Analysis Summary Report</p>
        </div>
        <div class="content">
"""

    summary_dir = Path(summary_dir)
    summary_data_path = summary_dir / "summary_data.json"

    if summary_data_path.exists():
        with open(summary_data_path) as f:
            data = json.load(f)

        # Baseline metrics
        if data.get("baseline"):
            baseline = data["baseline"]
            metrics = baseline.get("metrics", {})
            html_content += """
            <div class="section">
                <h2>🎯 Baseline Model Performance</h2>
                <div class="metrics-grid">
"""
            for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    html_content += f"""
                    <div class="metric-card">
                        <div class="label">{metric_name.replace('_', ' ').title()}</div>
                        <div class="value">{value:.4f}</div>
                    </div>
"""
            html_content += """
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,"""
            baseline_chart = embed_image(summary_dir / "baseline_summary.png")
            html_content += baseline_chart
            html_content += '" alt="Baseline Summary"></div></div>'

        # Algorithm comparison
        if data.get("algorithm_comparison"):
            html_content += """
            <div class="section">
                <h2>🤖 Algorithm Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Accuracy</th>
                            <th>F1 Score</th>
                            <th>ROC-AUC</th>
                            <th>Precision</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for result in data["algorithm_comparison"]:
                if "error" in result:
                    continue
                algo = result.get("algorithm", "unknown")
                m = result.get("metrics", {})
                html_content += f"""
                        <tr>
                            <td><strong>{algo}</strong></td>
                            <td>{m.get('accuracy', 0.0):.4f}</td>
                            <td>{m.get('f1', 0.0):.4f}</td>
                            <td>{m.get('roc_auc', 'N/A')}</td>
                            <td>{m.get('precision', 0.0):.4f}</td>
                        </tr>
"""
            html_content += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <img src="data:image/png;base64,"""
            algo_chart = embed_image(summary_dir / "algorithm_comparison.png")
            html_content += algo_chart
            html_content += '" alt="Algorithm Comparison"></div></div>'

        # Feature ablation
        if data.get("feature_ablation"):
            html_content += """
            <div class="section">
                <h2>🔬 Feature Ablation Study</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature Group</th>
                            <th>Features Used</th>
                            <th>Accuracy</th>
                            <th>Delta</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            baseline_acc = None
            for result in data["feature_ablation"]:
                group = result.get("group_name", "unknown")
                features = result.get("features_used", 0)
                acc = result.get("metrics", {}).get("accuracy", 0.0)
                if group == "all":
                    baseline_acc = acc
                    delta_str = "baseline"
                    delta_class = ""
                else:
                    delta = acc - baseline_acc if baseline_acc else 0.0
                    delta_str = f"{delta:+.4f}"
                    delta_class = "improvement" if delta >= 0 else "degradation"
                html_content += f"""
                        <tr>
                            <td><strong>{group.replace('_', ' ').title()}</strong></td>
                            <td>{features}</td>
                            <td>{acc:.4f}</td>
                            <td class="{delta_class}">{delta_str}</td>
                        </tr>
"""
            html_content += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <img src="data:image/png;base64,"""
            ablation_chart = embed_image(summary_dir / "feature_ablation.png")
            html_content += ablation_chart
            html_content += '" alt="Feature Ablation"></div></div>'

        # Ensemble methods
        if data.get("ensemble_results"):
            html_content += """
            <div class="section">
                <h2>🎭 Ensemble Methods</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Accuracy</th>
                            <th>F1 Score</th>
                            <th>ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for result in data["ensemble_results"]:
                method = result.get("method", "unknown")
                m = result.get("metrics", {})
                html_content += f"""
                        <tr>
                            <td><strong>{method.replace('_', ' ').title()}</strong></td>
                            <td>{m.get('accuracy', 0.0):.4f}</td>
                            <td>{m.get('f1', 0.0):.4f}</td>
                            <td>{m.get('roc_auc', 'N/A')}</td>
                        </tr>
"""
            html_content += """
                    </tbody>
                </table>
                <div class="chart-container">
                    <img src="data:image/png;base64,"""
            ensemble_chart = embed_image(summary_dir / "ensemble_comparison.png")
            html_content += ensemble_chart
            html_content += '" alt="Ensemble Comparison"></div></div>'

        # Feature combinations (improvement tracking)
        if data.get("feature_combinations"):
            combo_data = data["feature_combinations"]
            html_content += """
            <div class="section">
                <h2>🎯 Feature Combination Results</h2>
                <div class="metrics-grid">
"""
            baseline_acc = combo_data.get("baseline_accuracy", 0.0)
            best_acc = combo_data.get("best_accuracy", 0.0)
            improvement = combo_data.get("improvement", 0.0)
            
            html_content += f"""
                    <div class="metric-card">
                        <div class="label">Baseline Accuracy</div>
                        <div class="value">{baseline_acc:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Best Accuracy</div>
                        <div class="value">{best_acc:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Improvement</div>
                        <div class="value">{improvement:+.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Improvement %</div>
                        <div class="value">{100*improvement/baseline_acc:+.2f}%</div>
                    </div>
                </div>
"""
            if combo_data.get("best_config"):
                best_config = combo_data["best_config"]
                html_content += f"""
                <p><strong>Best Configuration:</strong> {best_config.get('config_name', 'unknown').replace('_', ' ').title()}</p>
                <p><strong>Features Used:</strong> {best_config.get('n_features', 0)} out of 71</p>
                <p><strong>Metrics:</strong> Accuracy={best_config.get('metrics', {}).get('accuracy', 0.0):.4f}, 
                F1={best_config.get('metrics', {}).get('f1', 0.0):.4f}, 
                ROC-AUC={best_config.get('metrics', {}).get('roc_auc', 'N/A')}</p>
"""
            html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>Features</th>
                            <th>Accuracy</th>
                            <th>Delta</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for r in sorted(combo_data.get("results", []), key=lambda x: x.get("metrics", {}).get("accuracy", 0.0), reverse=True):
                config_name = r.get("config_name", "unknown").replace("_", " ").title()
                n_feat = r.get("n_features", 0)
                metrics = r.get("metrics", {})
                acc = metrics.get("accuracy", 0.0)
                f1 = metrics.get("f1", 0.0)
                delta = acc - baseline_acc
                delta_class = "improvement" if delta >= 0 else "degradation"
                html_content += f"""
                        <tr>
                            <td><strong>{config_name}</strong></td>
                            <td>{n_feat}</td>
                            <td>{acc:.4f}</td>
                            <td class="{delta_class}">{delta:+.4f}</td>
                            <td>{f1:.4f}</td>
                        </tr>
"""
            html_content += """
                    </tbody>
                </table>
            </div>
"""

        # Calibration
        if data.get("calibration_results"):
            html_content += """
            <div class="section">
                <h2>📈 Calibration Methods</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,"""
            calib_chart = embed_image(summary_dir / "calibration_comparison.png")
            html_content += calib_chart
            html_content += '" alt="Calibration Comparison"></div></div>'

        # Model improvement tracking
        if data.get("comprehensive_improvement") or data.get("model_improvement"):
            html_content += """
            <div class="section">
                <h2>🚀 Model Improvement Tracking</h2>
"""
            imp_data = data.get("comprehensive_improvement") or data.get("model_improvement", {})
            baseline_acc = imp_data.get("baseline_accuracy", 0.0)
            best_acc = imp_data.get("best_accuracy", 0.0)
            improvement = imp_data.get("improvement", 0.0)
            
            html_content += f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Baseline Accuracy</div>
                        <div class="value">{baseline_acc:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Best Accuracy</div>
                        <div class="value">{best_acc:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Improvement</div>
                        <div class="value">{improvement:+.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Improvement %</div>
                        <div class="value">{100*improvement/baseline_acc:+.2f}%</div>
                    </div>
                </div>
"""
            
            if imp_data.get("results"):
                html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Accuracy</th>
                            <th>F1 Score</th>
                            <th>ROC-AUC</th>
                            <th>Delta</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                for r in imp_data["results"]:
                    strategy = r.get("strategy", "unknown").replace("_", " ").title()
                    metrics = r.get("metrics", {})
                    acc = metrics.get("accuracy", 0.0)
                    f1 = metrics.get("f1", 0.0)
                    roc_auc = metrics.get("roc_auc", "N/A")
                    delta = acc - baseline_acc
                    delta_class = "improvement" if delta >= 0 else "degradation"
                    roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
                    html_content += f"""
                        <tr>
                            <td><strong>{strategy}</strong></td>
                            <td>{acc:.4f}</td>
                            <td>{f1:.4f}</td>
                            <td>{roc_str}</td>
                            <td class="{delta_class}">{delta:+.4f}</td>
                        </tr>
"""
                html_content += """
                    </tbody>
                </table>
"""
            html_content += "</div>"

        # Key findings
        html_content += """
            <div class="section">
                <div class="findings">
                    <h3>🔍 Key Findings</h3>
                    <ul>
"""
        if data.get("algorithm_comparison"):
            best = max(
                [r for r in data["algorithm_comparison"] if "error" not in r],
                key=lambda x: x.get("metrics", {}).get("accuracy", 0.0),
                default=None,
            )
            if best:
                html_content += f'<li><strong>Best Algorithm:</strong> {best.get("algorithm")} with {best.get("metrics", {}).get("accuracy", 0.0):.4f} accuracy</li>'

        if data.get("ensemble_results"):
            best_ens = max(
                data["ensemble_results"],
                key=lambda x: x.get("metrics", {}).get("accuracy", 0.0),
                default=None,
            )
            if best_ens:
                html_content += f'<li><strong>Best Ensemble:</strong> {best_ens.get("method").replace("_", " ").title()} with {best_ens.get("metrics", {}).get("accuracy", 0.0):.4f} accuracy</li>'

        if data.get("comprehensive_improvement") or data.get("model_improvement"):
            imp_data = data.get("comprehensive_improvement") or data.get("model_improvement", {})
            best_acc = imp_data.get("best_accuracy", 0.0)
            baseline_acc = imp_data.get("baseline_accuracy", 0.0)
            if best_acc > baseline_acc:
                html_content += f'<li><strong>Best Improvement:</strong> {best_acc:.4f} accuracy ({best_acc - baseline_acc:+.4f} improvement)</li>'

        html_content += """
                    </ul>
                </div>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    html_path = summary_dir / "summary_report.html"
    html_path.write_text(html_content)
    return html_path


if __name__ == "__main__":
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    summary_dir = cache_dir / "summary_report"

    if not summary_dir.exists():
        log.error("Summary report not found. Run generate_summary_report.py first.")
        sys.exit(1)

    html_path = generate_html_viewer(summary_dir)
    log.info(f"Generated HTML viewer: {html_path}")
    log.info("Opening in browser...")
    webbrowser.open(f"file://{html_path.absolute()}")


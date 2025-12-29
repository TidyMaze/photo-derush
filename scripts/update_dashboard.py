#!/usr/bin/env python3
"""Update HTML dashboard with latest benchmark results and improvements.

Usage:
    poetry run python scripts/update_dashboard.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.view_summary_report import generate_html_viewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("update_dashboard")


def load_latest_results(cache_dir: Path) -> dict:
    """Load all available result files."""
    results = {}
    
    # Baseline
    baseline_path = cache_dir / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results["baseline"] = json.load(f)
    
    # Benchmark results
    benchmark_path = cache_dir / "benchmark_results.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            results["benchmark_results"] = json.load(f)
    
    # Feature ablation
    ablation_path = cache_dir / "feature_ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            results["feature_ablation"] = json.load(f)
    
    # Algorithm comparison
    algo_path = cache_dir / "algorithm_comparison_results.json"
    if algo_path.exists():
        with open(algo_path) as f:
            results["algorithm_comparison"] = json.load(f)
    
    # Ensemble results
    ensemble_path = cache_dir / "ensemble_results.json"
    if ensemble_path.exists():
        with open(ensemble_path) as f:
            results["ensemble_results"] = json.load(f)
    
    # Calibration results
    calib_path = cache_dir / "calibration_results.json"
    if calib_path.exists():
        with open(calib_path) as f:
            results["calibration_results"] = json.load(f)
    
    # Class balance results
    balance_path = cache_dir / "class_balance_results.json"
    if balance_path.exists():
        with open(balance_path) as f:
            results["class_balance_results"] = json.load(f)
    
    # Feature selection results
    fs_path = cache_dir / "feature_selection_results.json"
    if fs_path.exists():
        with open(fs_path) as f:
            results["feature_selection"] = json.load(f)
    
    # Feature combination results
    combo_path = cache_dir / "feature_combination_results.json"
    if combo_path.exists():
        with open(combo_path) as f:
            results["feature_combinations"] = json.load(f)
    
    # Model improvement results
    improve_path = cache_dir / "model_improvement_results.json"
    if improve_path.exists():
        with open(improve_path) as f:
            results["model_improvement"] = json.load(f)
    
    # Evaluation results
    eval_dir = cache_dir / "evaluation"
    eval_path = eval_dir / "full_evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results["evaluation_results"] = json.load(f)
    
    return results


def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    summary_dir = cache_dir / "summary_report"
    summary_dir.mkdir(exist_ok=True)
    
    log.info("Loading latest benchmark results...")
    results = load_latest_results(cache_dir)
    
    # Save updated summary data
    summary_data_path = summary_dir / "summary_data.json"
    with open(summary_data_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"Loaded {len(results)} result categories")
    
    # Regenerate charts
    log.info("Regenerating charts...")
    from scripts.generate_summary_report import generate_summary_charts
    chart_files = generate_summary_charts(results, summary_dir)
    log.info(f"Generated {len(chart_files)} charts")
    
    # Regenerate HTML dashboard
    log.info("Updating HTML dashboard...")
    html_path = generate_html_viewer(summary_dir)
    log.info(f"Dashboard updated: {html_path}")
    
    # Print summary of improvements
    if "model_improvement" in results:
        imp = results["model_improvement"]
        baseline_acc = imp.get("baseline_accuracy", 0.0)
        best_acc = imp.get("best_accuracy", 0.0)
        improvement = imp.get("improvement", 0.0)
        
        log.info("\n" + "="*60)
        log.info("MODEL IMPROVEMENT SUMMARY")
        log.info("="*60)
        log.info(f"Baseline accuracy: {baseline_acc:.4f}")
        log.info(f"Best accuracy: {best_acc:.4f}")
        log.info(f"Improvement: {improvement:+.4f} ({100*improvement/baseline_acc:+.2f}%)")
        
        if imp.get("best_config"):
            log.info(f"Best config: {imp['best_config'].get('test_name', 'unknown')}")
    
    if "feature_combinations" in results:
        combo = results["feature_combinations"]
        log.info("\n" + "="*60)
        log.info("FEATURE COMBINATION RESULTS")
        log.info("="*60)
        log.info(f"Best accuracy: {combo.get('best_accuracy', 0.0):.4f}")
        if combo.get("best_config"):
            log.info(f"Best config: {combo['best_config'].get('config_name', 'unknown')}")
            log.info(f"  Features: {combo['best_config'].get('n_features', 0)}")
    
    log.info("\nDashboard update complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Identify images that are most likely mislabeled based on model predictions.

Usage:
    poetry run python scripts/find_mislabelled_images.py [IMAGE_DIR] [--output OUTPUT]
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("find_mislabelled")


def load_best_model(image_dir: str, repo: RatingsTagsRepository):
    """Load the best model configuration."""
    # Load feature combination results to get best feature set
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    combo_path = cache_dir / "feature_combination_results.json"
    
    feature_indices = list(range(FEATURE_COUNT))  # Default: all features
    if combo_path.exists():
        try:
            with open(combo_path) as f:
                combo_data = json.load(f)
                best_config = combo_data.get("best_config")
                if best_config:
                    feature_indices = best_config.get("feature_indices", list(range(FEATURE_COUNT)))
                    log.info(f"Using best feature set: {len(feature_indices)} features")
        except Exception as e:
            log.warning(f"Could not load feature combination: {e}")
    
    # Build dataset
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Use best feature subset
    X_subset = X[:, feature_indices]
    
    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Train model
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            **xgb_params,
        )),
    ])
    
    clf.fit(X_subset, y)
    
    return clf, X_subset, y, filenames, feature_indices


def analyze_predictions(clf, X: np.ndarray, y: np.ndarray, filenames: list[str]) -> list[dict]:
    """Analyze predictions and identify likely mislabeled images."""
    # Get predictions and probabilities
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
    
    if y_proba is None:
        log.error("Model does not support probability predictions")
        return []
    
    mismatches = []
    
    for i, (true_label, pred_label, proba, fname) in enumerate(zip(y, y_pred, y_proba, filenames)):
        # Check for disagreement
        if true_label != pred_label:
            # Calculate confidence (distance from 0.5)
            confidence = abs(proba - 0.5) * 2  # Normalize to 0-1
            
            # Calculate disagreement strength
            if true_label == 1:  # Labeled as "keep"
                disagreement = proba  # Lower proba = stronger disagreement
            else:  # Labeled as "trash"
                disagreement = 1 - proba  # Higher proba = stronger disagreement
            
            mismatches.append({
                "filename": fname,
                "true_label": "keep" if true_label == 1 else "trash",
                "predicted_label": "keep" if pred_label == 1 else "trash",
                "probability": float(proba),
                "confidence": float(confidence),
                "disagreement_strength": float(disagreement),
                "predicted_probability": float(proba),
            })
    
    # Also identify uncertain predictions (even if they match)
    uncertain = []
    for i, (true_label, proba, fname) in enumerate(zip(y, y_proba, filenames)):
        uncertainty = 1 - abs(proba - 0.5) * 2  # Higher = more uncertain
        if uncertainty > 0.3:  # Threshold for uncertainty
            uncertain.append({
                "filename": fname,
                "label": "keep" if true_label == 1 else "trash",
                "probability": float(proba),
                "uncertainty": float(uncertainty),
            })
    
    return mismatches, uncertain


def main():
    try:
        parser = argparse.ArgumentParser(description="Find likely mislabeled images")
        parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
        parser.add_argument("--output", default=None, help="Output JSON file")
        parser.add_argument("--top-n", type=int, default=20, help="Number of top mismatches to show")
        parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence for high-confidence mismatches")
        args = parser.parse_args()
        
        # Determine image directory
        if args.image_dir:
            image_dir = os.path.expanduser(args.image_dir)
        else:
            image_dir = os.path.expanduser("~/Pictures/photo-dataset")
        
        if not os.path.isdir(image_dir):
            log.error(f"Image directory does not exist: {image_dir}")
            sys.stderr.write(f"ERROR: Image directory does not exist: {image_dir}\n")
            return 1
        
        log.info(f"Loading dataset from {image_dir}")
        sys.stdout.flush()
        
        # Initialize repository
        repo_path = os.path.join(image_dir, ".ratings_tags.json")
        repo = RatingsTagsRepository(path=repo_path)
        
        # Load best model
        log.info("Loading best model configuration...")
        sys.stdout.flush()
        clf, X, y, filenames, feature_indices = load_best_model(image_dir, repo)
        
        log.info(f"Analyzing {len(y)} labeled images...")
        sys.stdout.flush()
        
        # Analyze predictions
        mismatches, uncertain = analyze_predictions(clf, X, y, filenames)
        
        log.info(f"\nFound {len(mismatches)} mismatches (model prediction != label)")
        log.info(f"Found {len(uncertain)} uncertain predictions")
        
        # Sort mismatches by disagreement strength (strongest disagreement first)
        mismatches.sort(key=lambda x: x["disagreement_strength"], reverse=True)
        
        # Categorize mismatches
        high_confidence_mismatches = [m for m in mismatches if m["confidence"] >= args.min_confidence]
        
        # Print summary
        log.info("\n" + "="*60)
        log.info("LIKELY MISLABELED IMAGES ANALYSIS")
        log.info("="*60)
        
        log.info(f"\nTotal Mismatches: {len(mismatches)}")
        log.info(f"High-Confidence Mismatches (confidence >= {args.min_confidence}): {len(high_confidence_mismatches)}")
        log.info(f"Uncertain Predictions: {len(uncertain)}")
        
        # Top mismatches
        log.info(f"\nTop {args.top_n} Most Likely Mislabeled Images:")
        log.info("-" * 60)
        log.info(f"{'Filename':<40} {'Label':<8} {'Predicted':<10} {'Prob':<8} {'Conf':<8}")
        log.info("-" * 60)
        
        for m in mismatches[:args.top_n]:
            log.info(
                f"{m['filename']:<40} {m['true_label']:<8} {m['predicted_label']:<10} "
                f"{m['probability']:<8.3f} {m['confidence']:<8.3f}"
            )
        
        # High-confidence mismatches (most likely errors)
        if high_confidence_mismatches:
            log.info(f"\nHigh-Confidence Mismatches (Most Likely Labeling Errors):")
            log.info("-" * 60)
            for m in high_confidence_mismatches[:10]:
                log.info(
                    f"{m['filename']:<40} Labeled: {m['true_label']:<8} "
                    f"Predicted: {m['predicted_label']:<8} (prob: {m['probability']:.3f})"
                )
        
        # Uncertain predictions (need review)
        if uncertain:
            uncertain.sort(key=lambda x: x["uncertainty"], reverse=True)
            log.info(f"\nMost Uncertain Predictions (Need Review):")
            log.info("-" * 60)
            for u in uncertain[:10]:
                log.info(
                    f"{u['filename']:<40} Label: {u['label']:<8} "
                    f"Prob: {u['probability']:.3f} (uncertainty: {u['uncertainty']:.3f})"
                )
        
        # Statistics by label
        keep_mismatches = [m for m in mismatches if m["true_label"] == "keep"]
        trash_mismatches = [m for m in mismatches if m["true_label"] == "trash"]
        
        log.info("\n" + "="*60)
        log.info("MISMATCH STATISTICS")
        log.info("="*60)
        log.info(f"Labeled as 'keep' but predicted 'trash': {len(keep_mismatches)}")
        log.info(f"Labeled as 'trash' but predicted 'keep': {len(trash_mismatches)}")
        
        if keep_mismatches:
            avg_prob = np.mean([m["probability"] for m in keep_mismatches])
            log.info(f"  Average probability (keep): {avg_prob:.3f}")
        
        if trash_mismatches:
            avg_prob = np.mean([m["probability"] for m in trash_mismatches])
            log.info(f"  Average probability (keep): {avg_prob:.3f}")
        
        # Save results
        output_path = args.output or (Path(__file__).resolve().parent.parent / ".cache" / "mislabelled_images.json")
        output_path.parent.mkdir(exist_ok=True)
        
        results = {
            "total_images": len(y),
            "total_mismatches": len(mismatches),
            "high_confidence_mismatches": len(high_confidence_mismatches),
            "uncertain_predictions": len(uncertain),
            "mismatches": mismatches,
            "high_confidence_mismatches_list": high_confidence_mismatches,
            "uncertain_predictions_list": uncertain,
            "statistics": {
                "keep_labeled_trash_predicted": len(keep_mismatches),
                "trash_labeled_keep_predicted": len(trash_mismatches),
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        log.info(f"\nSaved results to {output_path}")
        sys.stdout.flush()
        
        # Generate HTML report
        html_path = output_path.parent / "mislabelled_images_report.html"
        generate_html_report(results, html_path, image_dir)
        log.info(f"Generated HTML report: {html_path}")
        sys.stdout.flush()
        
        return 0
    except Exception as e:
        log.exception("Error in main")
        sys.stderr.write(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


def create_thumbnail(img_path: str, max_size: tuple[int, int] = (200, 200)) -> str:
    """Create a base64-encoded thumbnail from an image file."""
    if not os.path.exists(img_path):
        return ""
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            # Create thumbnail
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            
            # Encode as base64
            data = base64.b64encode(buffer.read()).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        log.warning(f"Failed to create thumbnail for {img_path}: {e}")
        return ""


def generate_html_report(results: dict, output_path: Path, image_dir: str):
    """Generate HTML report with image thumbnails."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Likely Mislabeled Images Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; overflow-x: hidden; }}
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; width: 100%; max-width: 100vw; }}
        .container {{ max-width: min(1400px, 100%); margin: 0 auto; background: white; padding: 20px; border-radius: 8px; overflow-x: hidden; }}
        h1 {{ color: #333; border-bottom: 3px solid #f44336; padding-bottom: 10px; word-wrap: break-word; }}
        h2 {{ color: #555; margin-top: 30px; word-wrap: break-word; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #1976D2; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(250px, 100%), 1fr)); gap: 20px; margin: 20px 0; width: 100%; max-width: 100%; }}
        .image-card {{ border: 2px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; min-width: 0; max-width: 100%; }}
        .image-card.mismatch {{ border-color: #f44336; }}
        .image-card.uncertain {{ border-color: #ff9800; }}
        .image-card img {{ width: 100%; height: auto; border-radius: 4px; max-height: 200px; object-fit: cover; }}
        .image-info {{ margin-top: 10px; font-size: 0.9em; word-wrap: break-word; overflow-wrap: break-word; }}
        .image-info strong {{ word-break: break-all; }}
        table {{ width: 100%; max-width: 100%; border-collapse: collapse; table-layout: auto; }}
        table td, table th {{ word-wrap: break-word; overflow-wrap: break-word; }}
        .thumbnail-cell {{ width: 100px; padding: 8px !important; }}
        .thumbnail-cell img {{ width: 100px; height: 100px; object-fit: cover; border-radius: 4px; border: 2px solid #ddd; }}
        .label {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; white-space: nowrap; }}
        .label.keep {{ background: #4CAF50; color: white; }}
        .label.trash {{ background: #f44336; color: white; }}
        .prob {{ margin-top: 5px; font-size: 0.85em; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Likely Mislabeled Images Report</h1>
        <p>Generated: <span id="timestamp"></span></p>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="value">{results['total_images']}</div>
            </div>
            <div class="stat-card">
                <h3>Mismatches</h3>
                <div class="value">{results['total_mismatches']}</div>
            </div>
            <div class="stat-card">
                <h3>High-Confidence Mismatches</h3>
                <div class="value">{results['high_confidence_mismatches']}</div>
            </div>
            <div class="stat-card">
                <h3>Uncertain Predictions</h3>
                <div class="value">{results['uncertain_predictions']}</div>
            </div>
        </div>
"""
    
    # High-confidence mismatches
    if results.get("high_confidence_mismatches_list"):
        html_content += """
        <h2>🚨 High-Confidence Mismatches (Most Likely Labeling Errors)</h2>
        <div class="image-grid">
"""
        for m in results["high_confidence_mismatches_list"][:30]:  # Top 30
            img_path = os.path.join(image_dir, m["filename"])
            thumbnail = create_thumbnail(img_path, max_size=(300, 300))
            
            html_content += f"""
            <div class="image-card mismatch">
                <img src="{thumbnail}" alt="{m['filename']}" onerror="this.style.display='none'">
                <div class="image-info">
                    <div><strong>{m['filename']}</strong></div>
                    <div>
                        <span class="label {m['true_label']}">Labeled: {m['true_label']}</span>
                        <span class="label {m['predicted_label']}">Predicted: {m['predicted_label']}</span>
                    </div>
                    <div class="prob">Probability: {m['probability']:.3f} | Confidence: {m['confidence']:.3f}</div>
                </div>
            </div>
"""
        html_content += "</div>"
    
    # All mismatches
    if results.get("mismatches"):
        html_content += f"""
        <h2>⚠️ All Mismatches ({len(results['mismatches'])} total)</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #f44336; color: white;">
                    <th style="padding: 10px; text-align: left;">Thumbnail</th>
                    <th style="padding: 10px; text-align: left;">Filename</th>
                    <th style="padding: 10px;">Labeled</th>
                    <th style="padding: 10px;">Predicted</th>
                    <th style="padding: 10px;">Probability</th>
                    <th style="padding: 10px;">Confidence</th>
                    <th style="padding: 10px;">Disagreement</th>
                </tr>
            </thead>
            <tbody>
"""
        for m in results["mismatches"]:
            img_path = os.path.join(image_dir, m["filename"])
            thumbnail = create_thumbnail(img_path, max_size=(100, 100))
            thumbnail_html = f'<img src="{thumbnail}" alt="{m["filename"]}">' if thumbnail else ""
            
            html_content += f"""
                <tr>
                    <td class="thumbnail-cell">{thumbnail_html}</td>
                    <td style="padding: 8px;">{m['filename']}</td>
                    <td style="padding: 8px; text-align: center;"><span class="label {m['true_label']}">{m['true_label']}</span></td>
                    <td style="padding: 8px; text-align: center;"><span class="label {m['predicted_label']}">{m['predicted_label']}</span></td>
                    <td style="padding: 8px; text-align: center;">{m['probability']:.3f}</td>
                    <td style="padding: 8px; text-align: center;">{m['confidence']:.3f}</td>
                    <td style="padding: 8px; text-align: center;">{m['disagreement_strength']:.3f}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
"""
    
    # Uncertain predictions
    if results.get("uncertain_predictions_list"):
        html_content += f"""
        <h2>❓ Uncertain Predictions ({len(results['uncertain_predictions_list'])} total)</h2>
        <p>These images have predictions close to 0.5 (uncertain) and may need review.</p>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #ff9800; color: white;">
                    <th style="padding: 10px; text-align: left;">Thumbnail</th>
                    <th style="padding: 10px; text-align: left;">Filename</th>
                    <th style="padding: 10px;">Label</th>
                    <th style="padding: 10px;">Probability</th>
                    <th style="padding: 10px;">Uncertainty</th>
                </tr>
            </thead>
            <tbody>
"""
        for u in sorted(results["uncertain_predictions_list"], key=lambda x: x["uncertainty"], reverse=True)[:30]:
            img_path = os.path.join(image_dir, u["filename"])
            thumbnail = create_thumbnail(img_path, max_size=(100, 100))
            thumbnail_html = f'<img src="{thumbnail}" alt="{u["filename"]}">' if thumbnail else ""
            
            html_content += f"""
                <tr>
                    <td class="thumbnail-cell">{thumbnail_html}</td>
                    <td style="padding: 8px;">{u['filename']}</td>
                    <td style="padding: 8px; text-align: center;"><span class="label {u['label']}">{u['label']}</span></td>
                    <td style="padding: 8px; text-align: center;">{u['probability']:.3f}</td>
                    <td style="padding: 8px; text-align: center;">{u['uncertainty']:.3f}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
"""
    
    html_content += """
        <script>
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
        </script>
    </div>
</body>
</html>
"""
    
    output_path.write_text(html_content)


if __name__ == "__main__":
    raise SystemExit(main())


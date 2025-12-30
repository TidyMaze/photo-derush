#!/usr/bin/env python3
"""Deep interpretation of PCA components using image metadata and object detection.

Usage:
    poetry run python scripts/interpret_pca_components.py [--embeddings PATH] [--pca-dim 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import RatingsTagsRepository
from src import object_detection


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def analyze_image_properties(filenames: list[str], image_dir: str):
    """Analyze properties of images to correlate with PCA components."""
    properties = {}
    
    # Load object detections
    cache = object_detection.load_object_cache()
    
    # Load ratings/tags if available
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    for fname in filenames:
        props = {
            'has_person': False,
            'object_classes': [],
            'object_count': 0,
            'rating': None,
            'tags': [],
            'is_keep': None,
        }
        
        # Check object detections
        if fname in cache:
            detections = cache[fname]
            props['object_count'] = len(detections)
            classes = [d.get('class', '').lower() for d in detections]
            props['object_classes'] = classes
            props['has_person'] = 'person' in classes
        
        # Check ratings/tags
        if repo:
            state = repo.get_state(os.path.join(image_dir, fname))
            if state:
                props['rating'] = state.get('rating')
                props['tags'] = state.get('tags', [])
                props['is_keep'] = state.get('rating') == 'keep'
        
        properties[fname] = props
    
    return properties


def correlate_component_with_properties(
    component_values: np.ndarray,
    filenames: list[str],
    properties: dict,
    component_idx: int,
):
    """Correlate PCA component values with image properties."""
    print(f"\n{'='*80}")
    print(f"COMPONENT {component_idx} - DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Split into high and low activation groups
    median = np.median(component_values)
    high_indices = np.where(component_values > median + component_values.std())[0]
    low_indices = np.where(component_values < median - component_values.std())[0]
    
    high_filenames = [filenames[i] for i in high_indices]
    low_filenames = [filenames[i] for i in low_indices]
    
    print(f"\nComponent value range: [{component_values.min():.2f}, {component_values.max():.2f}]")
    print(f"Median: {median:.2f}, Std: {component_values.std():.2f}")
    print(f"High activation group: {len(high_indices)} images (>{median + component_values.std():.2f})")
    print(f"Low activation group: {len(low_indices)} images (<{median - component_values.std():.2f})")
    
    # Analyze object detections
    print(f"\n--- OBJECT DETECTION ANALYSIS ---")
    
    high_has_person = sum(1 for f in high_filenames if properties.get(f, {}).get('has_person', False))
    low_has_person = sum(1 for f in low_filenames if properties.get(f, {}).get('has_person', False))
    
    print(f"High activation: {high_has_person}/{len(high_filenames)} have people ({high_has_person/len(high_filenames)*100:.1f}%)")
    print(f"Low activation: {low_has_person}/{len(low_filenames)} have people ({low_has_person/len(low_filenames)*100:.1f}%)")
    
    # Object classes
    high_classes = []
    low_classes = []
    for f in high_filenames:
        high_classes.extend(properties.get(f, {}).get('object_classes', []))
    for f in low_filenames:
        low_classes.extend(properties.get(f, {}).get('object_classes', []))
    
    high_class_counts = Counter(high_classes)
    low_class_counts = Counter(low_classes)
    
    print(f"\nTop object classes in HIGH activation images:")
    for cls, count in high_class_counts.most_common(5):
        pct = count / len(high_filenames) * 100
        print(f"  {cls}: {count} images ({pct:.1f}%)")
    
    print(f"\nTop object classes in LOW activation images:")
    for cls, count in low_class_counts.most_common(5):
        pct = count / len(low_filenames) * 100
        print(f"  {cls}: {count} images ({pct:.1f}%)")
    
    # Object count
    high_obj_counts = [properties.get(f, {}).get('object_count', 0) for f in high_filenames]
    low_obj_counts = [properties.get(f, {}).get('object_count', 0) for f in low_filenames]
    
    if high_obj_counts and low_obj_counts:
        print(f"\nObject count statistics:")
        print(f"  High activation: mean={np.mean(high_obj_counts):.1f}, median={np.median(high_obj_counts):.1f}")
        print(f"  Low activation: mean={np.mean(low_obj_counts):.1f}, median={np.median(low_obj_counts):.1f}")
    
    # Ratings/keep-trash
    high_keep = [properties.get(f, {}).get('is_keep') for f in high_filenames]
    low_keep = [properties.get(f, {}).get('is_keep') for f in low_filenames]
    
    high_keep_count = sum(1 for k in high_keep if k is True)
    high_trash_count = sum(1 for k in high_keep if k is False)
    low_keep_count = sum(1 for k in low_keep if k is True)
    low_trash_count = sum(1 for k in low_keep if k is False)
    
    high_keep_pct = 0.0
    low_keep_pct = 0.0
    
    if high_keep_count + high_trash_count > 0 and low_keep_count + low_trash_count > 0:
        print(f"\n--- KEEP/TRASH CORRELATION ---")
        print(f"High activation: {high_keep_count} keep, {high_trash_count} trash")
        print(f"Low activation: {low_keep_count} keep, {low_trash_count} trash")
        
        high_keep_pct = high_keep_count / (high_keep_count + high_trash_count) * 100 if (high_keep_count + high_trash_count) > 0 else 0
        low_keep_pct = low_keep_count / (low_keep_count + low_trash_count) * 100 if (low_keep_count + low_trash_count) > 0 else 0
        print(f"High activation keep rate: {high_keep_pct:.1f}%")
        print(f"Low activation keep rate: {low_keep_pct:.1f}%")
    
    # Generate interpretation
    print(f"\n--- INTERPRETATION ---")
    interpretations = []
    
    if high_has_person > low_has_person * 1.5:
        interpretations.append("Strongly associated with images containing people")
    elif low_has_person > high_has_person * 1.5:
        interpretations.append("Strongly associated with images WITHOUT people")
    
    if high_obj_counts and low_obj_counts and np.mean(high_obj_counts) > np.mean(low_obj_counts) * 1.5:
        interpretations.append("Associated with images containing more objects")
    elif high_obj_counts and low_obj_counts and np.mean(low_obj_counts) > np.mean(high_obj_counts) * 1.5:
        interpretations.append("Associated with images containing fewer objects")
    
    if high_class_counts:
        top_high_class = high_class_counts.most_common(1)[0][0]
        if top_high_class not in [c[0] for c in low_class_counts.most_common(3)]:
            interpretations.append(f"Strongly associated with '{top_high_class}' objects")
    
    if high_keep_pct > low_keep_pct + 20:
        interpretations.append("Strongly correlated with 'keep' ratings")
    elif low_keep_pct > high_keep_pct + 20:
        interpretations.append("Strongly correlated with 'trash' ratings")
    
    if interpretations:
        print("This component likely captures:")
        for i, interp in enumerate(interpretations, 1):
            print(f"  {i}. {interp}")
    else:
        print("No clear pattern identified from available metadata.")
        print("Component may capture subtle visual patterns not captured by object detection.")
    
    return {
        'component_idx': component_idx,
        'high_has_person_pct': high_has_person / len(high_filenames) * 100 if high_filenames else 0,
        'low_has_person_pct': low_has_person / len(low_filenames) * 100 if low_filenames else 0,
        'high_keep_pct': high_keep_pct,
        'low_keep_pct': low_keep_pct,
        'interpretations': interpretations,
    }


def main():
    parser = argparse.ArgumentParser(description="Interpret PCA components using image properties")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA target dimensions")
    parser.add_argument("--n-components", type=int, default=10, help="Number of components to analyze")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    # Find embeddings file
    embeddings_path = args.embeddings
    if not embeddings_path:
        cache_dir = Path(__file__).resolve().parent.parent / ".cache"
        possible = [
            cache_dir / "embeddings_resnet18_full.joblib",
            cache_dir / "embeddings_resnet18.joblib",
        ]
        for p in possible:
            if p.exists():
                embeddings_path = str(p)
                break
    
    if not embeddings_path or not os.path.exists(embeddings_path):
        print(f"Error: No embeddings file found. Tried: {possible}")
        return 1
    
    print("Loading embeddings...")
    embeddings, filenames = load_embeddings(embeddings_path)
    print(f"Loaded {len(filenames)} embeddings\n")
    
    print("Analyzing image properties...")
    properties = analyze_image_properties(filenames, image_dir)
    print(f"Analyzed {len(properties)} images\n")
    
    print("Applying PCA...")
    pca = PCA(n_components=args.pca_dim, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA complete. Explained variance: {explained_var.sum()*100:.2f}%\n")
    
    # Analyze top components
    results = []
    for comp_idx in range(min(args.n_components, embeddings_pca.shape[1])):
        component_values = embeddings_pca[:, comp_idx]
        result = correlate_component_with_properties(
            component_values,
            filenames,
            properties,
            comp_idx,
        )
        result['explained_variance'] = float(explained_var[comp_idx])
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - TOP COMPONENTS")
    print(f"{'='*80}\n")
    
    for result in results[:5]:
        print(f"Component {result['component_idx']} ({result['explained_variance']*100:.2f}% variance):")
        if result['interpretations']:
            for interp in result['interpretations']:
                print(f"  • {interp}")
        else:
            print(f"  • Pattern not clearly identified")
        print()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        output_path = Path(__file__).resolve().parent.parent / ".cache" / "pca_interpretation.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


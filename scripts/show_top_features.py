#!/usr/bin/env python3
"""Show top 10 most important features from the best model."""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cache_dir = Path(__file__).resolve().parent.parent / ".cache"

# Load best model
model_path = cache_dir / "catboost_ava_multiclass_ultimate_best.joblib"
if not model_path.exists():
    model_path = cache_dir / "catboost_ava_multiclass_final_best.joblib"
if not model_path.exists():
    model_path = cache_dir / "catboost_ava_multiclass_best.joblib"

if not model_path.exists():
    print("Model not found")
    sys.exit(1)

print("Loading model...")
model = joblib.load(model_path)

# Get CatBoost model
cat_model = model.named_steps['cat']

# Get feature importances
feature_importances = cat_model.get_feature_importance()

# Get feature names
# We need to reconstruct what features are what
# Base: 78 handcrafted + 128 embeddings = 206
# Then interactions and ratios

# Load AVA features to get base feature count
ava_features_path = cache_dir / "ava_features.joblib"
ava_data = joblib.load(ava_features_path)
n_handcrafted = ava_data['features'].shape[1]  # 78

# Embeddings are reduced to 128 via PCA
n_embeddings = 128
n_base = n_handcrafted + n_embeddings  # 206

# Typical interaction/ratio counts from training
n_interactions = 100
n_ratios = 20
n_total_expected = n_base + n_interactions + n_ratios  # ~326

print(f"\nFeature structure:")
print(f"  Handcrafted features: {n_handcrafted}")
print(f"  Embedding features (PCA): {n_embeddings}")
print(f"  Base features: {n_base}")
print(f"  Interactions: ~{n_interactions}")
print(f"  Ratios: ~{n_ratios}")
print(f"  Total features: {len(feature_importances)}")

# Get actual handcrafted feature names
from src.model_stats import _get_feature_names
handcrafted_names = _get_feature_names()

# Create feature names
feature_names = []

# Base features - handcrafted
for i in range(n_handcrafted):
    name = handcrafted_names.get(i, f"handcrafted_{i}")
    feature_names.append(name)

# Base features - embeddings
for i in range(n_embeddings):
    feature_names.append(f"embedding_pca_{i}")

# Interactions (we don't know exact pairs, so use generic names)
for i in range(min(n_interactions, len(feature_importances) - n_base)):
    feature_names.append(f"interaction_{i}")

# Ratios
for i in range(min(n_ratios, len(feature_importances) - n_base - n_interactions)):
    feature_names.append(f"ratio_{i}")

# Pad if needed
while len(feature_names) < len(feature_importances):
    feature_names.append(f"unknown_{len(feature_names)}")

# Get top 10
top_indices = np.argsort(feature_importances)[::-1][:10]

print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*80)

for rank, idx in enumerate(top_indices, 1):
    importance = feature_importances[idx]
    feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    
    # Determine feature type
    if idx < n_handcrafted:
        feature_type = "Handcrafted"
    elif idx < n_base:
        feature_type = "Embedding (PCA)"
    elif idx < n_base + n_interactions:
        feature_type = "Interaction"
    else:
        feature_type = "Ratio"
    
    print(f"{rank:2d}. {feature_name:30s} | Importance: {importance:8.2f} | Type: {feature_type}")

print("\n" + "="*80)
print("FEATURE IMPORTANCE STATISTICS")
print("="*80)
print(f"Total features: {len(feature_importances)}")
print(f"Mean importance: {np.mean(feature_importances):.2f}")
print(f"Std importance: {np.std(feature_importances):.2f}")
print(f"Max importance: {np.max(feature_importances):.2f}")
print(f"Min importance: {np.min(feature_importances):.2f}")

# Show distribution by type
handcrafted_imp = feature_importances[:n_handcrafted] if len(feature_importances) > n_handcrafted else []
embedding_imp = feature_importances[n_handcrafted:n_base] if len(feature_importances) > n_base else []
interaction_imp = feature_importances[n_base:n_base+n_interactions] if len(feature_importances) > n_base+n_interactions else []
ratio_imp = feature_importances[n_base+n_interactions:] if len(feature_importances) > n_base+n_interactions else []

print(f"\nAverage importance by type:")
if len(handcrafted_imp) > 0:
    print(f"  Handcrafted: {np.mean(handcrafted_imp):.2f}")
if len(embedding_imp) > 0:
    print(f"  Embeddings: {np.mean(embedding_imp):.2f}")
if len(interaction_imp) > 0:
    print(f"  Interactions: {np.mean(interaction_imp):.2f}")
if len(ratio_imp) > 0:
    print(f"  Ratios: {np.mean(ratio_imp):.2f}")


"""Test feature extraction and prediction on a single image."""
import os
from pathlib import Path

import numpy as np

from src.inference import load_model

# Test feature extraction
from src.training import extract_features

print("=== TESTING FEATURE EXTRACTION ===\n")

# Find first image
image_dir = Path(os.path.expanduser("~/Pictures/photo-dataset"))
if not image_dir.exists():
    print(f"Image directory not found: {image_dir}")
    exit(1)

images = list(image_dir.glob("*.jpg"))
if not images:
    print("No images found")
    exit(1)

test_image = images[0]
print(f"Testing on: {test_image.name}")

# Extract features
print("\nExtracting features...")
features = extract_features(str(test_image))

if features is None:
    print("❌ Feature extraction failed!")
    exit(1)

print(f"✓ Features extracted: {len(features)} features")
print(f"  Feature sample: {features[:5]}")

# Load model
print("\n=== TESTING MODEL LOADING ===\n")

model_path = Path.home() / ".photo-derush-keep-trash-model.joblib"
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"Loading model from: {model_path}")
loaded = load_model(str(model_path))

if loaded is None:
    print("❌ Model loading failed!")
    exit(1)
model, metadata, calibrator = loaded.model, loaded.meta, loaded.calibrator
print("✓ Model loaded successfully")
print(f"  Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")

# Test prediction
print("\n=== TESTING PREDICTION ===\n")

X = np.array([features], dtype=float)
print(f"Input shape: {X.shape}")

try:
    y_prob = model.predict_proba(X)[0]
    prob_keep = float(y_prob[1])

    print("✓ Prediction successful!")
    print(f"  Probability(trash): {y_prob[0]:.4f}")
    print(f"  Probability(keep): {y_prob[1]:.4f}")

    confidence = max(prob_keep, 1 - prob_keep)
    label = "keep" if prob_keep > 0.5 else "trash"

    print(f"\n  Predicted label: {label}")
    print(f"  Confidence: {confidence:.4f}")

    if confidence >= 0.8:
        print("  ✓ HIGH CONFIDENCE - would be included in predictions")
    else:
        print("  ✗ LOW CONFIDENCE - would be filtered out (threshold: 0.8)")
        print("\n❓ THIS IS LIKELY THE ISSUE!")
        print(f"   All {len(images)} images might have confidence < 0.8")
        print("   Try lowering the threshold or training with more diverse data")

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()


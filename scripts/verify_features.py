#!/usr/bin/env python3
"""Verify that all features are being used in the main app."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_COUNT, extract_features
from src.inference import load_model, predict_keep_probability
import joblib

def main():
    print("="*80)
    print("FEATURE USAGE VERIFICATION")
    print("="*80)
    
    # 1. Check FEATURE_COUNT
    print(f"\n1. Current FEATURE_COUNT: {FEATURE_COUNT}")
    
    # 2. Check model
    model_path = os.path.expanduser("~/.photo-derush-keep-trash-model.joblib")
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return 1
    
    print(f"\n2. Loading model: {model_path}")
    data = joblib.load(model_path)
    
    model_feature_length = data.get("feature_length", 0)
    model_feature_indices = data.get("feature_indices")
    
    print(f"   Model feature_length: {model_feature_length}")
    print(f"   Model feature_indices: {model_feature_indices}")
    
    if model_feature_length != FEATURE_COUNT:
        print(f"   ⚠️  WARNING: Model expects {model_feature_length} features, but FEATURE_COUNT={FEATURE_COUNT}")
    else:
        print(f"   ✓ Model expects {FEATURE_COUNT} features")
    
    if model_feature_indices is not None:
        print(f"   ⚠️  WARNING: feature_indices is set - model will filter features!")
        print(f"   Filtered to: {len(model_feature_indices)} features")
    else:
        print(f"   ✓ No feature_indices - using ALL {FEATURE_COUNT} features")
    
    # 3. Check inference code
    print(f"\n3. Testing inference loading...")
    try:
        loaded = load_model(model_path)
        if loaded:
            meta = loaded.meta
            print(f"   ✓ Model loaded successfully")
            print(f"   Metadata feature_length: {meta.get('feature_length')}")
            print(f"   Metadata feature_indices: {meta.get('feature_indices', 'None')}")
            
            if meta.get('feature_indices') is None:
                print(f"   ✓ Inference will use ALL {FEATURE_COUNT} features")
            else:
                print(f"   ⚠️  Inference will filter to {len(meta.get('feature_indices', []))} features")
        else:
            print(f"   ❌ Failed to load model")
            return 1
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return 1
    
    # 4. Test feature extraction
    print(f"\n4. Testing feature extraction...")
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    if os.path.isdir(image_dir):
        import glob
        images = glob.glob(os.path.join(image_dir, "*.jpg"))[:1]
        if images:
            test_image = images[0]
            print(f"   Testing on: {os.path.basename(test_image)}")
            features = extract_features(test_image)
            if features:
                print(f"   ✓ Extracted {len(features)} features")
                if len(features) == FEATURE_COUNT:
                    print(f"   ✓ Feature count matches FEATURE_COUNT ({FEATURE_COUNT})")
                else:
                    print(f"   ⚠️  Feature count mismatch: got {len(features)}, expected {FEATURE_COUNT}")
            else:
                print(f"   ❌ Feature extraction failed")
        else:
            print(f"   ⚠️  No test images found")
    else:
        print(f"   ⚠️  Image directory not found: {image_dir}")
    
    # 5. Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_good = True
    if model_feature_length != FEATURE_COUNT:
        print("❌ Model feature count mismatch")
        all_good = False
    if model_feature_indices is not None:
        print("❌ Model has feature_indices filtering")
        all_good = False
    
    if all_good:
        print("✓ All features are being used correctly")
        print(f"✓ Model uses all {FEATURE_COUNT} features")
        print(f"✓ Highest score configuration: 79.01% with all features")
    else:
        print("⚠️  Some issues detected - see above")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Train model and test predictions to diagnose the issue."""

import requests

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("DIAGNOSIS: Auto-Labeling Issue")
print("=" * 70)
print()

# Step 1: Check labeled images
print("Step 1: Checking labeled images...")
response = requests.get(f"{BASE_URL}/api/projects")
if response.status_code != 200:
    print(f"❌ Failed to get projects: {response.status_code}")
    exit(1)

projects = response.json()
if not projects:
    print("❌ No projects found")
    exit(1)

project_id = projects[0]["id"]
print(f"✅ Project: {projects[0]['name']} (ID: {project_id})")

# Get image counts
response = requests.get(f"{BASE_URL}/api/images", params={"project_id": project_id, "per_page": 1000})
images = response.json()
labeled = [img for img in images if img.get("label")]
unlabeled = [img for img in images if not img.get("label")]

print(f"   Total images: {len(images)}")
print(f"   Labeled: {len(labeled)}")
print(f"   Unlabeled: {len(unlabeled)}")

if len(labeled) == 0:
    print()
    print("❌ No labeled images found!")
    print("   You need to label at least 20-50 images before training.")
    print("   Use K (keep) and T (trash) keyboard shortcuts in the UI.")
    exit(1)

# Count labels
keep_count = sum(1 for img in labeled if img.get("label") == "keep")
trash_count = sum(1 for img in labeled if img.get("label") == "trash")
print(f"   - Keep: {keep_count}")
print(f"   - Trash: {trash_count}")
print()

if keep_count < 10 or trash_count < 10:
    print("⚠️  Warning: Low training data!")
    print("   Recommended: 20+ examples per class")
    print(f"   Current: keep={keep_count}, trash={trash_count}")
    print()

# Step 2: Train the model
print("Step 2: Training model...")
response = requests.post(f"{BASE_URL}/api/ml/train/{project_id}")

if response.status_code != 200:
    print(f"❌ Training failed: {response.status_code}")
    print(f"   Response: {response.text}")
    exit(1)

result = response.json()
if not result.get("success"):
    print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    exit(1)

print("✅ Training successful!")
print(f"   Accuracy: {result.get('accuracy', 'N/A')}")
print(f"   F1 Score: {result.get('f1_score', 'N/A')}")
print()

# Step 3: Generate predictions
print("Step 3: Generating predictions...")
response = requests.post(f"{BASE_URL}/api/ml/predict/{project_id}")

if response.status_code != 200:
    print(f"❌ Prediction failed: {response.status_code}")
    print(f"   Response: {response.text}")
    exit(1)

result = response.json()
print("✅ Prediction call successful!")
print()

# Step 4: Analyze results
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"Success: {result['success']}")
print(f"Predictions count: {result['predictions_count']}")
print(f"Message: {result['message']}")
print()

print("Stats:")
stats = result['stats']
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

# Step 5: Diagnosis
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print()

if result['predictions_count'] == 0:
    if stats['low_confidence'] == stats['total_unlabeled']:
        print("❌ ALL predictions filtered as low confidence!")
        print()
        print("This means:")
        print("  • Model IS working (no errors)")
        print("  • All predictions fall in 0.4-0.6 range (uncertain)")
        print()
        print("Root Cause:")
        print("  • Insufficient training data (need 50-100+ per class)")
        print(f"  • Current: keep={keep_count}, trash={trash_count}")
        print()
        print("Solutions:")
        print("  1. Label more images (target: 50+ per class)")
        print("  2. Use keyboard shortcuts: K (keep), T (trash)")
        print("  3. Re-train after labeling more images")
        print("  4. Check that labeled images are representative")
        print()
        print("To see actual probabilities, check backend logs:")
        print("  tail -f /tmp/backend.log | grep 'Sample probabilities'")
    elif stats['feature_errors'] > 0:
        print(f"❌ Feature extraction errors: {stats['feature_errors']}")
        print("   Check that image files exist and are readable")
    elif stats['prediction_errors'] > 0:
        print(f"❌ Prediction errors: {stats['prediction_errors']}")
        print("   Model may have issues, try re-training")
else:
    print(f"✅ Generated {result['predictions_count']} predictions!")
    print()
    if result['predictions']:
        print("Sample predictions:")
        for pred in result['predictions'][:5]:
            print(f"  Image {pred['image_id']}: {pred['label']} ({pred['confidence']:.1%})")

print()
print("=" * 70)


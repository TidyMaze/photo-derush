"""Test auto-labeling predictions with actual API call."""

import requests

print("Testing auto-labeling predictions...")
print("=" * 60)

# Get project ID first
response = requests.get("http://localhost:8000/api/projects")
if response.status_code != 200:
    print(f"❌ Failed to get projects: {response.status_code}")
    print("Is the backend running on port 8000?")
    exit(1)

projects = response.json()
if not projects:
    print("❌ No projects found")
    exit(1)

project_id = projects[0]["id"]
print(f"✅ Found project: {projects[0]['name']} (ID: {project_id})")
print()

# Call predict endpoint
print(f"Calling POST /api/ml/predict/{project_id}...")
response = requests.post(f"http://localhost:8000/api/ml/predict/{project_id}")

if response.status_code != 200:
    print(f"❌ Prediction failed: {response.status_code}")
    print(f"Response: {response.text}")
    exit(1)

result = response.json()
print("✅ Prediction successful!")
print()
print("Results:")
print(f"  Success: {result['success']}")
print(f"  Predictions count: {result['predictions_count']}")
print(f"  Message: {result['message']}")
print()
print("Stats:")
for key, value in result['stats'].items():
    print(f"  {key}: {value}")
print()

# Analyze results
stats = result['stats']
total = stats['total_unlabeled']
high_conf = stats['high_confidence']
low_conf = stats['low_confidence']

print("=" * 60)
print("ANALYSIS:")
print("=" * 60)

if high_conf == 0 and low_conf > 0:
    print("⚠️  ALL predictions were filtered as low confidence!")
    print(f"   - {low_conf} images had predictions in uncertain range (0.4-0.6)")
    print()
    print("Possible causes:")
    print("  1. Model needs more training data")
    print("  2. Images are genuinely ambiguous")
    print("  3. Feature extraction may not be discriminative enough")
    print()
    print("Recommendations:")
    print("  - Check training data: need 50-100+ labeled examples per class")
    print("  - Review sample predictions to see actual probability values")
    print("  - Consider feature engineering improvements")
elif high_conf > 0:
    print(f"✅ Generated {high_conf} high-confidence predictions")
    print(f"   - Success rate: {high_conf}/{total} = {100*high_conf/total:.1f}%")
    if result['predictions']:
        print()
        print(f"Sample predictions (showing first {len(result['predictions'])}):")
        for pred in result['predictions'][:5]:
            print(f"  - Image {pred['image_id']}: {pred['label']} (confidence: {pred['confidence']:.3f})")

print()
print("=" * 60)


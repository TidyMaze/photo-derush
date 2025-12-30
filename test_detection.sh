#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create a test directory with a few sample images for quick testing
TEST_DIR="$SCRIPT_DIR/test_images"
mkdir -p "$TEST_DIR"

# Copy first 3 images from the dataset for testing
SOURCE_DIR="${PHOTO_DATASET_DIR:-$HOME/Pictures/photo-dataset}"
if [ -d "$SOURCE_DIR" ]; then
    echo "Copying 3 sample images to test_images directory..."
    ls "$SOURCE_DIR"/*.jpg 2>/dev/null | head -3 | xargs -I {} cp {} "$TEST_DIR/" 2>/dev/null || true
    echo "Test images ready in: $TEST_DIR"
else
    echo "No source images found. Please ensure $SOURCE_DIR exists."
    echo "Or set PHOTO_DATASET_DIR environment variable."
    exit 1
fi

echo "Running object detection on test images..."
"$SCRIPT_DIR/.venv/bin/python" scripts/detect_objects.py \
  --directory "$TEST_DIR" \
  --interesting-only \
  --max-size 600 \
  --confidence 0.6 \
  --workers 1 \
  --output test_detection_results.json

echo "Test completed. Results saved to test_detection_results.json"
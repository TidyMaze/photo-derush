#!/bin/bash
cd /Users/yannrolland/work/photo-derush

# Create a test directory with a few sample images for quick testing
TEST_DIR="/Users/yannrolland/work/photo-derush/test_images"
mkdir -p "$TEST_DIR"

# Copy first 3 images from the dataset for testing
if [ -d "/Users/yannrolland/Pictures/photo-dataset" ]; then
    echo "Copying 3 sample images to test_images directory..."
    ls "/Users/yannrolland/Pictures/photo-dataset"/*.jpg | head -3 | xargs -I {} cp {} "$TEST_DIR/"
    echo "Test images ready in: $TEST_DIR"
else
    echo "No source images found. Please ensure /Users/yannrolland/Pictures/photo-dataset exists."
    exit 1
fi

echo "Running object detection on test images..."
/Users/yannrolland/work/photo-derush/.venv/bin/python scripts/detect_objects.py \
  --directory "$TEST_DIR" \
  --interesting-only \
  --max-size 600 \
  --confidence 0.6 \
  --workers 1 \
  --output test_detection_results.json

echo "Test completed. Results saved to test_detection_results.json"
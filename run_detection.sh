#!/bin/bash
cd /Users/yannrolland/work/photo-derush

# Use a test directory with just a few images if it exists, otherwise warn about the large dataset
TEST_DIR="/Users/yannrolland/work/photo-derush/test_images"
if [ -d "$TEST_DIR" ] && [ "$(ls -A $TEST_DIR/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Using test directory: $TEST_DIR ($(ls -A $TEST_DIR/*.jpg | wc -l) images)"
    TARGET_DIR="$TEST_DIR"
    OUTPUT_FILE="test_detection_results.json"
else
    echo "WARNING: This script processes ALL images in /Users/yannrolland/Pictures/photo-dataset"
    echo "This may take a very long time with hundreds of images."
    echo "Consider running 'bash test_detection.sh' first to create a test directory with sample images."
    echo ""
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
    TARGET_DIR="/Users/yannrolland/Pictures/photo-dataset"
    OUTPUT_FILE="detection_results.json"
fi

echo "Processing images in: $TARGET_DIR"
/Users/yannrolland/work/photo-derush/.venv/bin/python scripts/detect_objects.py \
  --directory "$TARGET_DIR" \
  --interesting-only \
  --max-size 600 \
  --confidence 0.6 \
  --workers 2 \
  --output "$OUTPUT_FILE"

echo "Detection completed. Results saved to $OUTPUT_FILE"

#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use a test directory with just a few images if it exists, otherwise warn about the large dataset
TEST_DIR="$SCRIPT_DIR/test_images"
if [ -d "$TEST_DIR" ] && [ "$(ls -A $TEST_DIR/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Using test directory: $TEST_DIR ($(ls -A $TEST_DIR/*.jpg 2>/dev/null | wc -l) images)"
    TARGET_DIR="$TEST_DIR"
    OUTPUT_FILE="test_detection_results.json"
else
    SOURCE_DIR="${PHOTO_DATASET_DIR:-$HOME/Pictures/photo-dataset}"
    echo "WARNING: This script processes ALL images in $SOURCE_DIR"
    echo "This may take a very long time with hundreds of images."
    echo "Consider running 'bash test_detection.sh' first to create a test directory with sample images."
    echo "Or set PHOTO_DATASET_DIR environment variable to specify a different directory."
    echo ""
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
    TARGET_DIR="$SOURCE_DIR"
    OUTPUT_FILE="detection_results.json"
fi

echo "Processing images in: $TARGET_DIR"
"$SCRIPT_DIR/.venv/bin/python" scripts/detect_objects.py \
  --directory "$TARGET_DIR" \
  --interesting-only \
  --max-size 600 \
  --confidence 0.6 \
  --workers 2 \
  --output "$OUTPUT_FILE"

echo "Detection completed. Results saved to $OUTPUT_FILE"

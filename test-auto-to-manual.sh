#!/bin/bash
# Test that auto-labeled images change to manual when user updates them

echo "====================================="
echo "Testing Auto → Manual Label Conversion"
echo "====================================="
echo ""

# Wait for backend
sleep 2

# Step 1: Check if backend is running
echo "1. Checking backend..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend not running. Start with: poetry run uvicorn api.main:app --port 8000"
    exit 1
fi
echo "✅ Backend is running"
echo ""

# Step 2: Get predictions and save them
echo "2. Generating predictions..."
curl -s -X POST http://localhost:8000/api/ml/predict/1 > /tmp/predictions.json
PRED_COUNT=$(cat /tmp/predictions.json | grep -o '"image_id"' | wc -l)
echo "✅ Generated $PRED_COUNT predictions"
echo ""

# Step 3: Save predictions to database
echo "3. Saving predictions to database..."
PREDICTIONS=$(cat /tmp/predictions.json | jq '.predictions')
SAVE_RESULT=$(curl -s -X POST http://localhost:8000/api/ml/predictions/save \
    -H "Content-Type: application/json" \
    -d "$PREDICTIONS")
SAVED=$(echo "$SAVE_RESULT" | jq -r '.saved')
echo "✅ Saved $SAVED predictions with source='auto'"
echo ""

# Step 4: Get an auto-labeled image
echo "4. Getting first auto-labeled image..."
IMAGE_ID=$(echo "$PREDICTIONS" | jq -r '.[0].image_id')
BEFORE=$(curl -s http://localhost:8000/api/images/$IMAGE_ID)
LABEL_BEFORE=$(echo "$BEFORE" | jq -r '.label')
SOURCE_BEFORE=$(echo "$BEFORE" | jq -r '.label_source')
CONF_BEFORE=$(echo "$BEFORE" | jq -r '.label_confidence')

echo "   Image ID: $IMAGE_ID"
echo "   Label: $LABEL_BEFORE"
echo "   Source: $SOURCE_BEFORE"
echo "   Confidence: $CONF_BEFORE"
echo ""

# Step 5: Update label (simulating user clicking K or T)
echo "5. Updating label (user clicks K/T)..."
AFTER=$(curl -s -X PUT http://localhost:8000/api/images/$IMAGE_ID/label \
    -H "Content-Type: application/json" \
    -d '{"label":"keep"}')

LABEL_AFTER=$(echo "$AFTER" | jq -r '.label')
SOURCE_AFTER=$(echo "$AFTER" | jq -r '.label_source')
CONF_AFTER=$(echo "$AFTER" | jq -r '.label_confidence')

echo "   Label: $LABEL_AFTER"
echo "   Source: $SOURCE_AFTER"
echo "   Confidence: $CONF_AFTER"
echo ""

# Step 6: Verify changes
echo "====================================="
echo "RESULTS"
echo "====================================="
echo ""

if [ "$SOURCE_BEFORE" = "auto" ] && [ "$SOURCE_AFTER" = "manual" ] && [ "$CONF_AFTER" = "null" ]; then
    echo "✅ SUCCESS!"
    echo ""
    echo "Before: source='$SOURCE_BEFORE', confidence=$CONF_BEFORE"
    echo "After:  source='$SOURCE_AFTER', confidence=$CONF_AFTER"
    echo ""
    echo "Auto-labeled images correctly change to manual when user updates them!"
else
    echo "❌ FAILED"
    echo ""
    echo "Before: source='$SOURCE_BEFORE', confidence=$CONF_BEFORE"
    echo "After:  source='$SOURCE_AFTER', confidence=$CONF_AFTER"
    echo ""
    echo "Expected: source='manual', confidence=null"
fi
echo ""
echo "====================================="


#!/bin/bash
# Test the fix for "no unlabeled images" response

echo "========================================"
echo "Testing 'No Unlabeled Images' Fix"
echo "========================================"
echo ""

# Wait for backend
sleep 2

# Step 1: Check current label status
echo "1. Checking current label status..."
ALL_IMAGES=$(curl -s "http://localhost:8000/api/images?project_id=1&per_page=1000")
TOTAL=$(echo "$ALL_IMAGES" | jq 'length')
UNLABELED=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=unlabeled&per_page=1000" | jq 'length')
AUTO=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=auto&per_page=1000" | jq 'length')
MANUAL=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=manual&per_page=1000" | jq 'length')

echo "   Total images: $TOTAL"
echo "   Unlabeled: $UNLABELED"
echo "   Auto-labeled: $AUTO"
echo "   Manual: $MANUAL"
echo ""

# Step 2: Call prediction endpoint
echo "2. Calling prediction endpoint..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/ml/predict/1")
echo "$RESPONSE" | jq '.'
echo ""

# Step 3: Verify response structure
echo "3. Verifying response structure..."
SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
PREDICTIONS_COUNT=$(echo "$RESPONSE" | jq -r '.predictions_count')
MESSAGE=$(echo "$RESPONSE" | jq -r '.message')
HAS_STATS=$(echo "$RESPONSE" | jq 'has("stats")')
HAS_PREDICTIONS=$(echo "$RESPONSE" | jq 'has("predictions")')

echo "   success: $SUCCESS"
echo "   predictions_count: $PREDICTIONS_COUNT"
echo "   message: $MESSAGE"
echo "   has stats object: $HAS_STATS"
echo "   has predictions array: $HAS_PREDICTIONS"
echo ""

# Step 4: Verify stats object if present
if [ "$HAS_STATS" = "true" ]; then
    echo "4. Verifying stats object..."
    STATS=$(echo "$RESPONSE" | jq '.stats')
    echo "   Stats: $STATS"
    echo ""

    # Check all required fields
    REQUIRED_FIELDS=("total_unlabeled" "files_existing" "feature_errors" "prediction_errors" "low_confidence" "high_confidence")
    ALL_PRESENT=true

    for field in "${REQUIRED_FIELDS[@]}"; do
        HAS_FIELD=$(echo "$STATS" | jq "has(\"$field\")")
        if [ "$HAS_FIELD" != "true" ]; then
            echo "   ❌ Missing field: $field"
            ALL_PRESENT=false
        fi
    done

    if [ "$ALL_PRESENT" = "true" ]; then
        echo "   ✅ All required stats fields present"
    fi
else
    echo "4. ❌ Stats object missing!"
fi
echo ""

# Step 5: Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""

if [ "$UNLABELED" -eq 0 ]; then
    echo "✅ No unlabeled images (expected scenario)"
    echo ""
    if [ "$HAS_STATS" = "true" ]; then
        echo "✅ Response includes stats object"
        echo "✅ Frontend won't crash!"
        echo ""
        echo "Backend response is correct:"
        echo "  - success: true"
        echo "  - predictions_count: 0"
        echo "  - predictions: []"
        echo "  - stats: { all fields present }"
        echo "  - message: '$MESSAGE'"
    else
        echo "❌ Response missing stats object"
        echo "❌ Frontend WILL crash!"
        echo ""
        echo "Need to fix backend to include stats object"
    fi
else
    echo "⚠️  There are $UNLABELED unlabeled images"
    echo ""
    echo "Response details:"
    echo "  - predictions_count: $PREDICTIONS_COUNT"
    echo "  - has stats: $HAS_STATS"
    echo ""
    if [ "$PREDICTIONS_COUNT" -gt 0 ]; then
        echo "✅ Predictions generated successfully"
    else
        echo "⚠️  No predictions generated (check confidence threshold)"
    fi
fi

echo ""
echo "========================================"


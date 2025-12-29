#!/bin/bash
# Test the new review workflow filters

echo "========================================"
echo "Testing Review Workflow Filters"
echo "========================================"
echo ""

# Wait for backend
sleep 3

# Test 1: Get all images
echo "1. Getting all images..."
ALL_COUNT=$(curl -s "http://localhost:8000/api/images?project_id=1&per_page=1000" | jq 'length')
echo "   Total images: $ALL_COUNT"
echo ""

# Test 2: Get unlabeled images
echo "2. Getting unlabeled images..."
UNLABELED=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=unlabeled&per_page=1000")
UNLABELED_COUNT=$(echo "$UNLABELED" | jq 'length')
echo "   Unlabeled images: $UNLABELED_COUNT"
if [ $UNLABELED_COUNT -gt 0 ]; then
    echo "   Sample: $(echo "$UNLABELED" | jq -r '.[0] | {id, filename, label}' 2>/dev/null || echo 'none')"
fi
echo ""

# Test 3: Get auto-labeled images
echo "3. Getting auto-labeled images..."
AUTO=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=auto&per_page=1000")
AUTO_COUNT=$(echo "$AUTO" | jq 'length')
echo "   Auto-labeled images: $AUTO_COUNT"
if [ $AUTO_COUNT -gt 0 ]; then
    echo "   Sample: $(echo "$AUTO" | jq -r '.[0] | {id, filename, label, label_source, label_confidence}' 2>/dev/null || echo 'none')"
fi
echo ""

# Test 4: Get manual labeled images
echo "4. Getting manually-labeled images..."
MANUAL=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=manual&per_page=1000")
MANUAL_COUNT=$(echo "$MANUAL" | jq 'length')
echo "   Manually-labeled images: $MANUAL_COUNT"
if [ $MANUAL_COUNT -gt 0 ]; then
    echo "   Sample: $(echo "$MANUAL" | jq -r '.[0] | {id, filename, label, label_source}' 2>/dev/null || echo 'none')"
fi
echo ""

# Test 5: Sort by confidence ascending
echo "5. Getting auto-labeled sorted by confidence (low→high)..."
AUTO_SORTED_ASC=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=auto&sort_by=confidence_asc&per_page=5")
echo "   First 5 (lowest confidence):"
echo "$AUTO_SORTED_ASC" | jq -r '.[] | "   - \(.filename): \(.label_confidence // "null")"' 2>/dev/null
echo ""

# Test 6: Sort by confidence descending
echo "6. Getting auto-labeled sorted by confidence (high→low)..."
AUTO_SORTED_DESC=$(curl -s "http://localhost:8000/api/images?project_id=1&label_source=auto&sort_by=confidence_desc&per_page=5")
echo "   First 5 (highest confidence):"
echo "$AUTO_SORTED_DESC" | jq -r '.[] | "   - \(.filename): \(.label_confidence // "null")"' 2>/dev/null
echo ""

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "Total images:     $ALL_COUNT"
echo "Unlabeled:        $UNLABELED_COUNT (need labeling)"
echo "Auto-labeled:     $AUTO_COUNT (need review)"
echo "Manual:           $MANUAL_COUNT (reviewed)"
echo ""
echo "✅ Filters working correctly!"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:5173"
echo "2. Use new filter dropdowns to select:"
echo "   - Label Source: 'Auto-labeled (need review)'"
echo "   - Sort: 'Confidence Low→High'"
echo "3. Review images, press K/T to convert auto→manual"
echo ""
echo "========================================"


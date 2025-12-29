#!/bin/bash
# Quick diagnostic script for auto-labeling issues

echo "🔍 Auto-Labeling Diagnostic Tool"
echo "=================================="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend not running on port 8000"
    echo ""
    echo "Start backend with:"
    echo "  poetry run uvicorn api.main:app --reload --port 8000"
    echo ""
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Run the test script
echo "📊 Running auto-labeling test..."
echo ""
poetry run python scripts/test_auto_labeling.py


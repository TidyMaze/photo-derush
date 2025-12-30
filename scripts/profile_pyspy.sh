#!/bin/bash
# Profile app with py-spy (requires sudo for macOS)
# Usage: ./scripts/profile_pyspy.sh [duration_seconds]

set -e

DURATION="${1:-60}"
APP_PID=""

cleanup() {
    if [ -n "$APP_PID" ]; then
        echo "Stopping app (PID: $APP_PID)..."
        kill "$APP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting app in background..."
cd "$(dirname "$0")/.."
PROFILING=1 PHOTO_DERUSH_AUTOQUIT_MS=$((DURATION * 1000)) poetry run python app.py > /tmp/app_profile.log 2>&1 &
APP_PID=$!

echo "App started with PID: $APP_PID"
echo "Waiting 5 seconds for app to initialize..."
sleep 5

echo ""
echo "Starting py-spy profiler (duration: ${DURATION}s, requires sudo)..."
echo "Profiling: CPU (all threads/processes/subprocesses)"
sudo poetry run py-spy record \
    -o /tmp/app_profile_pyspy.json \
    --pid "$APP_PID" \
    --subprocesses \
    --threads \
    --rate 100 \
    --duration "$DURATION" 2>&1 | tee /tmp/pyspy_output.log

echo ""
echo "Profile saved to /tmp/app_profile_pyspy.json"
echo "Converting to pstats format..."
poetry run py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof 2>&1 || sudo poetry run py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof 2>&1

echo ""
echo "Analyzing profile (all threads/processes)..."
python3 tools/analyze_profile.py 2>&1 | head -100

echo ""
echo "View with: py-spy top --input /tmp/app_profile_pyspy.json"
echo "Or: py-spy flamegraph --input /tmp/app_profile_pyspy.json --output /tmp/flamegraph.svg"
echo "Or: python3 tools/analyze_profile.py"


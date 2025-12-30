#!/bin/bash
# Non-intrusive profiling script for photo-derush app
# Usage: ./scripts/profile_app.sh [py-spy|scalene] [duration_seconds]

set -e

PROFILER="${1:-py-spy}"
DURATION="${2:-60}"
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
PROFILING=1 poetry run python app.py > /tmp/app_profile.log 2>&1 &
APP_PID=$!

echo "App started with PID: $APP_PID"
echo "Waiting 3 seconds for app to initialize..."
sleep 3

case "$PROFILER" in
    py-spy)
        echo "Starting py-spy profiler (duration: ${DURATION}s)..."
        echo "Profiling: CPU (all threads/processes/subprocesses)"
        echo "Note: py-spy is CPU-only. Memory profiling uses tracemalloc (enabled with PROFILING=1)"
        poetry run py-spy record \
            -o /tmp/app_profile_pyspy.json \
            --pid "$APP_PID" \
            --subprocesses \
            --threads \
            --rate 100 \
            --duration "$DURATION" || {
            echo "py-spy failed, trying without poetry..."
            py-spy record \
                -o /tmp/app_profile_pyspy.json \
                --pid "$APP_PID" \
                --subprocesses \
                --threads \
                --rate 100 \
                --duration "$DURATION"
        }
        echo "Profile saved to /tmp/app_profile_pyspy.json"
        echo "View with: py-spy top --input /tmp/app_profile_pyspy.json"
        echo "Or: py-spy flamegraph --input /tmp/app_profile_pyspy.json --output /tmp/flamegraph.svg"
        echo "Or: py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof && python3 tools/analyze_profile.py"
        ;;
    scalene)
        echo "Starting scalene profiler (duration: ${DURATION}s)..."
        scalene --profile-all \
            --outfile /tmp/app_profile_scalene.html \
            --pid "$APP_PID" \
            --duration "$DURATION" || {
            echo "scalene not found. Install with: pip install scalene"
            exit 1
        }
        echo "Profile saved to /tmp/app_profile_scalene.html"
        echo "Open in browser: open /tmp/app_profile_scalene.html"
        ;;
    *)
        echo "Unknown profiler: $PROFILER"
        echo "Usage: $0 [py-spy|scalene] [duration_seconds]"
        exit 1
        ;;
esac

echo "Profiling complete. App will continue running."
echo "To stop app: kill $APP_PID"


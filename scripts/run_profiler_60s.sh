#!/bin/bash
# Run app with profiler for 60 seconds and collect results

set -e

cd "$(dirname "$0")/.."

echo "Starting app with profiler for 60 seconds..."
echo "Press Ctrl+C to stop early"
echo ""

# Clean old profiles
rm -f /tmp/app_profile_*.prof /tmp/app_memory_*.txt /tmp/app_memory_*.pkl

# Start app in background
PROFILING=1 poetry run python app.py > /tmp/app_output.log 2>&1 &
APP_PID=$!

echo "App started (PID: $APP_PID)"
echo "Waiting 60 seconds..."

# Wait 60 seconds
sleep 60

echo "Stopping app..."
kill $APP_PID 2>/dev/null || true
wait $APP_PID 2>/dev/null || true

echo ""
echo "Profile collection complete"
echo "Profiles saved to /tmp/app_profile_*.prof"
echo ""
echo "To analyze: poetry run python tools/analyze_profile.py /tmp/app_profile_aggregated.prof"


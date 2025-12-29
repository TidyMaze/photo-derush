#!/usr/bin/env bash
# Small helper script to run the Qt desktop app with Poetry. Use this from the repository root.
set -euo pipefail

# Install dependencies if needed
poetry install --no-interaction

# Ensure logs dir
mkdir -p logs
LOGFILE="logs/qt-run-$(date +%Y%m%d-%H%M%S).log"

echo "Starting Photo Derush desktop app. Writing logs to ${LOGFILE}"
poetry run python app.py --log-file "${LOGFILE}"

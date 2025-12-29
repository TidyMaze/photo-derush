#!/usr/bin/env bash
set -euxo pipefail
OUT=logs/lldb-yolov8-backtrace.txt
mkdir -p logs
# Build an lldb commands file to run the app and dump bt all on crash, then quit
cat > /tmp/lldb_cmds.txt <<'LLC'
settings set target.run-args DETECTION_DEVICE=cpu -- app.py --log-file logs/qt-run-lldb-yolov8.log
run
bt all
thread info
quit
LLC
# Launch lldb non-interactively and redirect session to file
lldb -- ./.venv/bin/python3 < /tmp/lldb_cmds.txt &> "$OUT" || true

#!/bin/bash
# run_with_caffeinate.sh - Keep Mac awake while running main.py
# Prevents screen-lock from stopping signal generation

echo "[CAFFEINATE] Starting caffeinate to prevent Mac sleep..."
caffeinate -dims python3 /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py &

# Capture PID of caffeinate process
CAFFEINATE_PID=$!
echo "[CAFFEINATE] Process PID: $CAFFEINATE_PID"

# Monitor subprocess
wait $CAFFEINATE_PID
EXITCODE=$?
echo "[CAFFEINATE] Process exited with code $EXITCODE"
exit $EXITCODE

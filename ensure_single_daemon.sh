#!/bin/bash
# CRITICAL: Enforce SINGLE DAEMON RULE
# Must run BEFORE starting main.py

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
cd "$WORKSPACE"

echo "🔐 SINGLE DAEMON ENFORCEMENT CHECK"
echo "================================="

# Count running daemon instances
RUNNING=$(pgrep -f "smart-filter-v14-main/main.py" | wc -l)
echo "Running daemon instances: $RUNNING"

if [ "$RUNNING" -gt 1 ]; then
    echo "❌ ERROR: $RUNNING daemons running! Only 1 allowed!"
    echo "Killing all instances..."
    pkill -f "smart-filter-v14-main/main.py"
    pkill -f "main.py"
    sleep 3
    echo "✅ All instances killed"
fi

if [ "$RUNNING" -eq 1 ]; then
    echo "⚠️  WARNING: Daemon already running! Not starting new one."
    PID=$(pgrep -f "smart-filter-v14-main/main.py")
    echo "Existing PID: $PID"
    exit 1
fi

# Safe to start
echo ""
echo "✅ Safe to start: No daemon running"
echo "Starting single daemon instance..."
cd "$WORKSPACE/smart-filter-v14-main"
nohup python3 main.py > ../main_daemon.log 2>&1 &

sleep 2
NEW_PID=$(pgrep -f "smart-filter-v14-main/main.py")
echo "✅ New daemon started: PID $NEW_PID"
echo ""
echo "RULE ENFORCED: Only 1 daemon instance running"

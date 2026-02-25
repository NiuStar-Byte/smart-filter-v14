#!/bin/bash

# SmartFilter Quick Restart Script
# Use: ./restart.sh
# Changes take effect in ~2 seconds

echo "🔄 Restarting SmartFilter..."
echo ""

# Kill existing process
echo "1️⃣  Killing existing SmartFilter process..."
pkill -9 -f "python.*main.py" 2>/dev/null
sleep 1

# Start fresh
echo "2️⃣  Starting SmartFilter (fresh)..."
cd "$(dirname "$0")"
python3 main.py > smartfilter_run.log 2>&1 &
PID=$!

sleep 1

# Verify it started
if ps -p $PID > /dev/null; then
    echo "✅ SmartFilter started successfully (PID: $PID)"
    echo ""
    echo "📊 Watch signals:"
    echo "   tail -f smartfilter_run.log | grep '^\[SIGNAL\]\|^\[✅ SENT\]'"
    echo ""
    echo "📝 Full logs:"
    echo "   tail -f smartfilter_run.log"
else
    echo "❌ Failed to start SmartFilter"
    echo ""
    echo "Check errors:"
    echo "   tail -20 smartfilter_run.log"
    exit 1
fi

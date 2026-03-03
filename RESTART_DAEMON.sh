#!/bin/bash
# RESTART_DAEMON.sh - Kill old daemon, clear logs, start fresh with fixed code

cd ~/.openclaw/workspace/smart-filter-v14-main/

echo "🛑 Killing old daemon..."
pkill -f "python3 main.py" 2>/dev/null
sleep 2

echo "🧹 Clearing old logs..."
> main_daemon.log
> main_test.log

echo "📊 Pulling latest code from GitHub..."
git pull origin main 2>/dev/null || echo "   (Already up to date)"

echo ""
echo "🚀 Starting fresh daemon with FIXED code..."
nohup python3 main.py > main_daemon.log 2>&1 &
DAEMON_PID=$!

echo "✅ Daemon started (PID: $DAEMON_PID)"
echo ""
echo "📋 Next steps:"
echo "   1. Wait 10 seconds for signals to start flowing"
echo "   2. Watch live logs:"
echo "      tail -f main_daemon.log | grep PHASE3B-SCORE"
echo ""
echo "   3. View Phase 3B report (auto-refresh):"
echo "      python3 track_phase3b_simple.py"
echo ""
echo "   4. Expected results:"
echo "      ✅ TREND_CONT with score 37-62 (not 0)"
echo "      ❌ AMBIGUOUS/NONE with score 0 (disabled)"
echo ""

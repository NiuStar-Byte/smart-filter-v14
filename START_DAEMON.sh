#!/bin/bash
# START_DAEMON.sh - Start SmartFilter daemon with Phase 2-FIXED + Phase 3B

echo "🚀 Starting SmartFilter daemon..."
echo "   Working directory: $(pwd)"
echo "   Timestamp: $(date '+%Y-%m-%d %H:%M:%S GMT+7')"
echo ""

# Clear old logs (start fresh with fixed code)
echo "🔄 Clearing old logs..."
> main_daemon.log
> main_test.log

# Start daemon in background
echo "📊 Starting main.py daemon..."
nohup python3 main.py > main_daemon.log 2>&1 &
DAEMON_PID=$!

echo "✅ Daemon started (PID: $DAEMON_PID)"
echo ""
echo "📋 Log location: $(pwd)/main_daemon.log"
echo ""
echo "🔍 To watch Phase 3B logs:"
echo "   tail -f main_daemon.log | grep -E 'PHASE3B-(RQ|FALLBACK|SCORE)'"
echo ""
echo "📊 To view Phase 3B report:"
echo "   python3 track_phase3b_simple.py"
echo ""
echo "⏸️  To stop daemon:"
echo "   kill $DAEMON_PID"

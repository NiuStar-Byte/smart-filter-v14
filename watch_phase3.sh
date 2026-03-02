#!/bin/bash
# Watch PHASE 3 Performance (Refreshes every 5 seconds)
# Usage: ./watch_phase3.sh

cd "$(dirname "$0")"

echo "🚀 PHASE 3 Live Monitor (refreshing every 5s)"
echo "📍 Press Ctrl+C to stop"
echo ""

python3 PHASE3_TRACKER.py

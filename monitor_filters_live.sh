#!/bin/bash

# monitor_filters_live.sh
# Live monitoring dashboard: refreshes both filter analyzers every 30 seconds
# Shows bash + python output side-by-side with timestamps

INTERVAL=30

echo "Starting live filter monitoring dashboard..."
echo "Updates every $INTERVAL seconds. Press Ctrl+C to stop."
sleep 2

while true; do
    clear
    
    echo "╔════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    LIVE FILTER EFFECTIVENESS MONITORING DASHBOARD                                   ║"
    echo "║                        Updated: $(date '+%Y-%m-%d %H:%M:%S %Z')                                          ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    echo "┌─── BASH INSTRUMENTATION TRACKER ─────────────────────────────────────────────────────────────────────────┐"
    bash /Users/geniustarigan/.openclaw/workspace/track_instrumentation_real.sh 2>/dev/null
    echo "└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    
    echo ""
    echo "┌─── PYTHON DETAILED ANALYZER ──────────────────────────────────────────────────────────────────────────┐"
    python3 /Users/geniustarigan/.openclaw/workspace/filter_effectiveness_analyzer_detailed.py 2>/dev/null
    echo "└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    
    echo ""
    echo "Next refresh in $INTERVAL seconds... (Press Ctrl+C to stop)"
    sleep $INTERVAL
done

#!/bin/bash
# Hourly PEC Reporter Daemon
# Run this in background to capture reports every hour
# Usage: bash hourly_reporter_daemon.sh &
# Or: nohup bash hourly_reporter_daemon.sh > hourly_reporter_daemon.log 2>&1 &

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
cd "$WORKSPACE"

echo "🚀 Hourly PEC Reporter Daemon started at $(date)"
echo "Will capture reports every hour at :00 minutes"

while true; do
    # Get current time
    CURRENT_MINUTE=$(date +%M)
    
    # If we're at :00 or :01, run the reporter (avoid race conditions)
    if [[ "$CURRENT_MINUTE" == "00" || "$CURRENT_MINUTE" == "01" ]]; then
        echo ""
        echo "⏰ $(date '+%Y-%m-%d %H:%M:%S') - Running hourly reporter..."
        python3 hourly_reporter.py
        echo "✅ Reporter completed at $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Sleep until next hour to avoid duplicate runs
        sleep 120
    fi
    
    # Check every 30 seconds
    sleep 30
done

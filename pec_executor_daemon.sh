#!/bin/bash
#
# PEC EXECUTOR DAEMON - Auto-execute signals every 5 minutes
# 
# Monitors SIGNALS_MASTER.jsonl for OPEN signals
# Checks current prices via KuCoin API
# Updates signal status: OPEN → TP_HIT/SL_HIT/TIMEOUT
# Writes exit prices and P&L back to SIGNALS_MASTER.jsonl
#
# Usage:
#   bash pec_executor_daemon.sh &          # Start in background
#   pkill -f pec_executor_daemon           # Stop the daemon
#
# Logs: pec_executor_daemon.log

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
DAEMON_LOG="$WORKSPACE/pec_executor_daemon.log"
PID_FILE="$WORKSPACE/pec_executor_daemon.pid"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DAEMON_LOG"
}

# Start message
log "🚀 PEC EXECUTOR DAEMON started"
log "   Workspace: $WORKSPACE"
log "   Log: $DAEMON_LOG"
log "   Update interval: 5 minutes"

# Save PID for later stopping
echo $$ > "$PID_FILE"

# Infinite loop - run update every 5 minutes
while true; do
    cd "$WORKSPACE"
    
    # Run the executor
    python3 pec_executor.py 2>&1 | tee -a "$DAEMON_LOG"
    
    # Wait 5 minutes before next run
    sleep 300
done

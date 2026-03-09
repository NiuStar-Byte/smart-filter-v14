#!/bin/bash

# CRITICAL PROCESS WATCHDOG - Ensures daemon and executor ALWAYS running
# Run this in background: nohup bash monitor_critical_processes.sh &

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
DAEMON_DIR="$WORKSPACE/smart-filter-v14-main"
MAIN_LOG="$WORKSPACE/main_daemon.log"
EXECUTOR_LOG="$WORKSPACE/pec_executor_daemon.log"

# Check interval (60 seconds)
CHECK_INTERVAL=60

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WORKSPACE/watchdog.log"
}

check_and_restart_daemon() {
    if ! pgrep -f "python3 $DAEMON_DIR/main.py" > /dev/null 2>&1; then
        log_message "⚠️  DAEMON NOT RUNNING - Restarting..."
        cd "$DAEMON_DIR"
        nohup python3 main.py > "$MAIN_LOG" 2>&1 &
        sleep 3
        if pgrep -f "python3 $DAEMON_DIR/main.py" > /dev/null 2>&1; then
            log_message "✅ Daemon restarted successfully"
        else
            log_message "❌ Failed to restart daemon!"
        fi
    fi
}

check_and_restart_executor() {
    if ! pgrep -f "python3 $DAEMON_DIR/pec_executor.py" > /dev/null 2>&1; then
        log_message "⚠️  EXECUTOR NOT RUNNING - Restarting..."
        cd "$DAEMON_DIR"
        nohup python3 pec_executor.py > "$EXECUTOR_LOG" 2>&1 &
        sleep 3
        if pgrep -f "python3 $DAEMON_DIR/pec_executor.py" > /dev/null 2>&1; then
            log_message "✅ Executor restarted successfully"
        else
            log_message "❌ Failed to restart executor!"
        fi
    fi
}

log_message "🔄 Watchdog started - monitoring critical processes every ${CHECK_INTERVAL}s"

# Main loop
while true; do
    check_and_restart_daemon
    check_and_restart_executor
    sleep $CHECK_INTERVAL
done

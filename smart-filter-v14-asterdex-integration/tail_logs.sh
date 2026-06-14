#!/bin/bash
#
# Quick Log Viewer for Asterdex Entry Poster
# Use this to view live logs of the running instance
#
# Usage:
#   ./tail_logs.sh          # Tail last 50 lines + live updates
#   ./tail_logs.sh last     # Show last 100 lines (no tail)
#   ./tail_logs.sh search TERM  # Search for specific term
#

LOG_FILE="/tmp/asterdex_entry_poster_live.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No log file found at $LOG_FILE"
    echo "Run: ./run_with_logs.sh"
    exit 1
fi

case "$1" in
    "last")
        echo "=== Last 100 lines of logs ==="
        tail -100 "$LOG_FILE"
        ;;
    "search")
        if [ -z "$2" ]; then
            echo "Usage: ./tail_logs.sh search TERM"
            exit 1
        fi
        echo "=== Search results for: $2 ==="
        grep "$2" "$LOG_FILE" | tail -50
        ;;
    *)
        echo "=== Live Asterdex Entry Poster Logs (Press Ctrl+C to stop) ==="
        echo ""
        tail -50 "$LOG_FILE"
        echo ""
        echo "--- Live updates below ---"
        tail -f "$LOG_FILE"
        ;;
esac

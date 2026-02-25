#!/bin/bash
# PEC Daemon Control: Start/Stop/Restart/Monitor PEC Auto-Update

DAEMON_PID_FILE="/tmp/pec_daemon.pid"
DAEMON_LOG="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/pec_daemon.log"
DAEMON_SCRIPT="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/pec_daemon.py"
WORK_DIR="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"

case "$1" in
    start)
        if pgrep -f "pec_daemon.py" > /dev/null; then
            echo "❌ PEC Daemon already running (PID: $(pgrep -f "pec_daemon.py"))"
            exit 1
        fi
        cd "$WORK_DIR"
        nohup python3 pec_daemon.py > "$DAEMON_LOG" 2>&1 &
        PID=$!
        echo "$PID" > "$DAEMON_PID_FILE"
        echo "✅ PEC Daemon started (PID: $PID)"
        echo "📊 Monitor: tail -f $DAEMON_LOG"
        ;;
    
    stop)
        if pkill -f "pec_daemon.py"; then
            echo "✅ PEC Daemon stopped"
            rm -f "$DAEMON_PID_FILE"
        else
            echo "❌ PEC Daemon not running"
            exit 1
        fi
        ;;
    
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    
    status)
        if pgrep -f "pec_daemon.py" > /dev/null; then
            PID=$(pgrep -f "pec_daemon.py")
            echo "✅ PEC Daemon running (PID: $PID)"
            echo "📊 Log: $DAEMON_LOG"
            echo "📝 Last 10 lines:"
            tail -10 "$DAEMON_LOG" 2>/dev/null || echo "(log not found)"
        else
            echo "❌ PEC Daemon not running"
            exit 1
        fi
        ;;
    
    watch)
        tail -f "$DAEMON_LOG"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|watch}"
        echo ""
        echo "Examples:"
        echo "  $0 start       # Start PEC daemon (5-min updates)"
        echo "  $0 status      # Check if running"
        echo "  $0 watch       # Monitor live updates"
        echo "  $0 restart     # Restart daemon"
        echo "  $0 stop        # Stop daemon"
        exit 1
        ;;
esac

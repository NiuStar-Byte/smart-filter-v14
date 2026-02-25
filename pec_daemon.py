#!/usr/bin/env python3
"""
PEC Daemon: Runs in background, auto-updates signal status every 5 minutes
Start with: python3 pec_daemon.py &
Kill with: pkill -f pec_daemon.py
Monitor with: tail -f pec_daemon.log
"""

import time
import sys
from datetime import datetime
from pec_executor import PECExecutor

def main():
    """Run PEC updates every 5 minutes (300 seconds)"""
    executor = PECExecutor()
    interval = 300  # 5 minutes
    
    print(f"[DAEMON] PEC Auto-Update started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[DAEMON] Updates every {interval//60} minutes", flush=True)
    print(f"[DAEMON] Stop with: pkill -f pec_daemon.py\n", flush=True)
    
    try:
        while True:
            # Run update
            summary = executor.update_signals()
            
            # Print announcement ONLY if something changed
            executor.print_announcement(summary)
            
            # Sleep until next update
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n[DAEMON] Stopped by user", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[DAEMON] Error: {e}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

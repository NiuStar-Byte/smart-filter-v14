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
    sys.stdout.flush()
    
    cycle = 0
    while True:
        try:
            cycle += 1
            # Run update with timeout protection
            try:
                summary = executor.update_signals()
                executor.print_announcement(summary)
            except Exception as e:
                print(f"[DAEMON ERROR] Cycle {cycle} failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
            
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Sleep until next update
            time.sleep(interval)
        
        except KeyboardInterrupt:
            print(f"\n[DAEMON] Stopped by user", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"[DAEMON CRITICAL] Outer loop error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(10)  # Wait before retry instead of crashing


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
PEC Watchdog - Auto-restart PEC Executor if it crashes.
Monitors the process every 30 seconds and restarts if dead.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
import signal

# Configuration
PEC_SCRIPT = "pec_executor.py"
CHECK_INTERVAL = 30  # seconds
LOG_FILE = "pec_watchdog.log"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
GRACEFUL_SHUTDOWN = False

def log(msg):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(os.path.join(WORK_DIR, LOG_FILE), "a") as f:
        f.write(line + "\n")

def signal_handler(sig, frame):
    """Graceful shutdown on SIGTERM/SIGINT."""
    global GRACEFUL_SHUTDOWN
    GRACEFUL_SHUTDOWN = True
    log("⚠️  Watchdog received shutdown signal. Exiting gracefully...")
    sys.exit(0)

def is_pec_running():
    """Check if pec_executor.py is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", PEC_SCRIPT],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        log(f"❌ Error checking process: {e}")
        return False

def start_pec():
    """Start PEC executor in background."""
    try:
        os.chdir(WORK_DIR)
        proc = subprocess.Popen(
            ["python3", PEC_SCRIPT],
            stdout=open(os.devnull, 'w'),
            stderr=open(os.devnull, 'w'),
            start_new_session=True  # Detach from parent
        )
        time.sleep(2)  # Give it time to start
        if is_pec_running():
            log(f"✅ PEC Executor started (PID will auto-continue)")
            return True
        else:
            log(f"❌ PEC Executor failed to start")
            return False
    except Exception as e:
        log(f"❌ Error starting PEC: {e}")
        return False

def main():
    """Main watchdog loop."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    log("🦇 PEC Watchdog started")
    log(f"📁 Working directory: {WORK_DIR}")
    log(f"⏱️  Check interval: {CHECK_INTERVAL}s")
    
    # Initial startup check
    if not is_pec_running():
        log("⏳ PEC Executor not running. Starting...")
        start_pec()
    else:
        log("✅ PEC Executor already running")
    
    crash_count = 0
    startup_failures = 0
    
    # Watch loop
    while not GRACEFUL_SHUTDOWN:
        try:
            time.sleep(CHECK_INTERVAL)
            
            if not is_pec_running():
                crash_count += 1
                log(f"⚠️  PEC Executor died (crash #{crash_count}). Restarting...")
                
                if start_pec():
                    startup_failures = 0
                else:
                    startup_failures += 1
                    if startup_failures >= 3:
                        log("❌ 3 consecutive startup failures. Watchdog giving up.")
                        sys.exit(1)
            else:
                # Still running
                if crash_count > 0:
                    log(f"✅ PEC Executor recovered after {crash_count} crash(es)")
                    crash_count = 0
        
        except KeyboardInterrupt:
            log("⚠️  Watchdog interrupted by user")
            break
        except Exception as e:
            log(f"❌ Watchdog error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

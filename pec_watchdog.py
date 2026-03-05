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
WORK_DIR = os.path.dirname(os.path.abspath(__file__))  # Workspace root
PEC_SCRIPT_PATH = os.path.join(WORK_DIR, "smart-filter-v14-main", "pec_executor.py")  # Absolute path to executor
PEC_SCRIPT = "pec_executor.py"  # Process name for pgrep
CHECK_INTERVAL = 30  # seconds
LOG_FILE = "pec_watchdog.log"
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
        # Verify script exists
        if not os.path.exists(PEC_SCRIPT_PATH):
            error_msg = f"PEC script not found at {PEC_SCRIPT_PATH}"
            log(f"❌ {error_msg}")
            log_to_health_monitor("WATCHDOG", "SCRIPT_MISSING", error_msg)
            return False
        
        # Use absolute path to executor in submodule
        proc = subprocess.Popen(
            ["python3", PEC_SCRIPT_PATH],
            stdout=open(os.devnull, 'w'),
            stderr=open(os.devnull, 'w'),
            start_new_session=True,  # Detach from parent
            cwd=WORK_DIR  # Run from workspace root
        )
        time.sleep(2)  # Give it time to start
        if is_pec_running():
            log(f"✅ PEC Executor started (PID will auto-continue)")
            return True
        else:
            error_msg = "PEC Executor process exited immediately after start"
            log(f"❌ {error_msg}")
            log_to_health_monitor("WATCHDOG", "START_FAILED", error_msg)
            return False
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log(f"❌ Error starting PEC: {error_msg}")
        log_to_health_monitor("WATCHDOG", "EXCEPTION", error_msg)
        return False

def log_to_health_monitor(component, error_type, message):
    """Log errors to health monitor log"""
    try:
        health_log = os.path.join(WORK_DIR, 'pec_system_health.log')
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+7")
        line = f"[{ts}] 🔴 {component:<15} | {error_type:<20} | {message}\n"
        with open(health_log, 'a') as f:
            f.write(line)
    except:
        pass  # If health log fails, at least we tried in main log

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

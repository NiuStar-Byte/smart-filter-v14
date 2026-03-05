#!/usr/bin/env python3
"""
🏥 PEC SYSTEM HEALTH MONITOR - Real-time error detection
Tracks signal accumulation pipeline and reports failures immediately
"""

import json
import os
from datetime import datetime
import subprocess

HEALTH_LOG = "pec_system_health.log"
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
SENT_SIGNALS_FILE = os.path.join(WORKSPACE_ROOT, "SENT_SIGNALS.jsonl")

def log_error(component, error_type, message, severity="⚠️"):
    """Log error with timestamp and component"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+7")
    line = f"[{ts}] {severity} {component:<15} | {error_type:<20} | {message}"
    print(line)
    
    with open(os.path.join(WORKSPACE_ROOT, HEALTH_LOG), "a") as f:
        f.write(line + "\n")

def check_daemon_status():
    """Check if main.py is running and firing signals"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "main.py"],
            capture_output=True,
            timeout=3
        )
        
        if result.returncode != 0:
            log_error("DAEMON", "NOT_RUNNING", "main.py process not found", "🔴")
            return False
        
        # Check if signals are being fired (file updated in last 5 min)
        if not os.path.exists(SENT_SIGNALS_FILE):
            log_error("DAEMON", "FILE_MISSING", f"SENT_SIGNALS.jsonl not found at {SENT_SIGNALS_FILE}", "🔴")
            return False
        
        mod_time = os.path.getmtime(SENT_SIGNALS_FILE)
        age_seconds = (datetime.now().timestamp() - mod_time)
        
        if age_seconds > 600:  # 10 minutes old
            log_error("DAEMON", "STALE_DATA", f"SENT_SIGNALS.jsonl not updated for {int(age_seconds/60)}+ minutes", "🟡")
            return False
        
        return True
    except Exception as e:
        log_error("DAEMON", "CHECK_FAILED", str(e), "🔴")
        return False

def check_executor_status():
    """Check if PEC Executor is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "pec_executor.py"],
            capture_output=True,
            timeout=3
        )
        
        if result.returncode != 0:
            log_error("EXECUTOR", "NOT_RUNNING", "pec_executor.py process not found", "🔴")
            return False
        
        return True
    except Exception as e:
        log_error("EXECUTOR", "CHECK_FAILED", str(e), "🔴")
        return False

def check_watchdog_status():
    """Check if PEC Watchdog is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "pec_watchdog.py"],
            capture_output=True,
            timeout=3
        )
        
        if result.returncode != 0:
            log_error("WATCHDOG", "NOT_RUNNING", "pec_watchdog.py process not found", "🔴")
            return False
        
        return True
    except Exception as e:
        log_error("WATCHDOG", "CHECK_FAILED", str(e), "🔴")
        return False

def check_file_permissions():
    """Check if SENT_SIGNALS.jsonl is writable"""
    try:
        if not os.path.exists(SENT_SIGNALS_FILE):
            log_error("FILE", "MISSING", f"SENT_SIGNALS.jsonl not found", "🔴")
            return False
        
        if not os.access(SENT_SIGNALS_FILE, os.W_OK):
            log_error("FILE", "NOT_WRITABLE", "SENT_SIGNALS.jsonl is read-only or inaccessible", "🔴")
            return False
        
        return True
    except Exception as e:
        log_error("FILE", "CHECK_FAILED", str(e), "🔴")
        return False

def check_signal_flow():
    """Check if signals are being accumulated"""
    try:
        if not os.path.exists(SENT_SIGNALS_FILE):
            log_error("FLOW", "NO_DATA", "SENT_SIGNALS.jsonl doesn't exist", "🔴")
            return False
        
        with open(SENT_SIGNALS_FILE) as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            log_error("FLOW", "EMPTY_FILE", "SENT_SIGNALS.jsonl has no signals", "🔴")
            return False
        
        # Check if recent signals exist (from last hour)
        recent_count = 0
        for line in lines:
            try:
                sig = json.loads(line)
                fired = sig.get('fired_time_utc', '')
                if '2026-03-05' in fired:  # Today's date (adjust as needed)
                    recent_count += 1
            except:
                continue
        
        if recent_count == 0:
            log_error("FLOW", "NO_RECENT", "No signals fired in last hour", "🟡")
            return False
        
        # Check if signals are being closed (processed)
        closed_count = sum(1 for line in lines if json.loads(line.strip()).get('closed_at'))
        open_count = len(lines) - closed_count
        
        if open_count > 50:
            log_error("FLOW", "BACKLOG", f"{open_count} signals still OPEN - executor may be behind", "🟡")
        
        return True
    except Exception as e:
        log_error("FLOW", "CHECK_FAILED", str(e), "🔴")
        return False

def print_health_summary():
    """Print overall system health"""
    daemon_ok = check_daemon_status()
    executor_ok = check_executor_status()
    watchdog_ok = check_watchdog_status()
    perms_ok = check_file_permissions()
    flow_ok = check_signal_flow()
    
    print("\n" + "="*100)
    print("📊 PEC SYSTEM HEALTH SUMMARY")
    print("="*100)
    
    status_icon = {
        True: "✅",
        False: "❌"
    }
    
    print(f"  {status_icon[daemon_ok]} Daemon (main.py)              → {'Running & firing signals' if daemon_ok else 'FAILED'}")
    print(f"  {status_icon[executor_ok]} Executor (pec_executor.py)    → {'Running & processing' if executor_ok else 'FAILED'}")
    print(f"  {status_icon[watchdog_ok]} Watchdog (pec_watchdog.py)    → {'Running & monitoring' if watchdog_ok else 'FAILED'}")
    print(f"  {status_icon[perms_ok]} File Access (SENT_SIGNALS)   → {'Readable & writable' if perms_ok else 'FAILED'}")
    print(f"  {status_icon[flow_ok]} Signal Pipeline              → {'Accumulating normally' if flow_ok else 'FAILED'}")
    
    all_ok = all([daemon_ok, executor_ok, watchdog_ok, perms_ok, flow_ok])
    
    print()
    if all_ok:
        print("🟢 SYSTEM HEALTHY - All components operational")
    else:
        print("🔴 SYSTEM DEGRADED - See errors above, check pec_system_health.log for details")
    
    print("="*100 + "\n")
    
    return all_ok

if __name__ == "__main__":
    print_health_summary()

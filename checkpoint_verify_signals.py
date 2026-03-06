#!/usr/bin/env python3
"""
CHECKPOINT VERIFICATION - Compare SIGNALS_MASTER.jsonl vs SIGNALS_INDEPENDENT_AUDIT.txt

Verifies that both files have matching record counts. Any discrepancy is:
1. Logged to SIGNALS_CHECKPOINT.txt (immutable verification history)
2. Reported to stdout
3. Triggers alert if mismatch detected

Usage:
  python3 checkpoint_verify_signals.py          # Single check
  python3 checkpoint_verify_signals.py --watch  # Continuous (hourly)
"""

import os
import json
from datetime import datetime, timezone, timedelta
import sys
import time

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
audit_file = os.path.join(workspace, "SIGNALS_INDEPENDENT_AUDIT.txt")
checkpoint_file = os.path.join(workspace, "SIGNALS_CHECKPOINT.txt")

def get_jakarta_time():
    """Get current time in Jakarta (GMT+7)"""
    utc_now = datetime.now(timezone.utc)
    jakarta_tz = timezone(timedelta(hours=7))
    return utc_now.astimezone(jakarta_tz)

def count_signals(filepath):
    """Count JSON lines in a file"""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return None

def verify_checkpoint():
    """Verify signal counts in both files"""
    now = get_jakarta_time()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S GMT+7")
    
    # Count signals in both files
    master_count = count_signals(master_file)
    audit_count = count_signals(audit_file)
    
    # Determine status
    if master_count is None or audit_count is None:
        status = "❌ ERROR - Could not read files"
        discrepancy = "Unknown"
        action = "Manual investigation required"
    elif master_count == audit_count:
        status = "✅ SYNCED"
        discrepancy = 0
        action = "No action needed"
    else:
        diff = audit_count - master_count
        if diff > 0:
            status = f"⚠️ DISCREPANCY - Audit trail ahead by {diff}"
            action = f"Audit trail has {diff} more signal(s). SIGNALS_MASTER may be missing data."
        else:
            status = f"⚠️ DISCREPANCY - SIGNALS_MASTER ahead by {abs(diff)}"
            action = f"SIGNALS_MASTER has {abs(diff)} extra signal(s). Audit trail may be missing data."
        discrepancy = abs(diff)
    
    # Format output
    output = {
        "timestamp": timestamp,
        "timestamp_iso": now.isoformat(),
        "master_count": master_count,
        "audit_count": audit_count,
        "discrepancy": discrepancy,
        "status": status,
        "action": action
    }
    
    return output

def log_checkpoint(result):
    """Log checkpoint result to SIGNALS_CHECKPOINT.txt (append-only)"""
    try:
        checkpoint_entry = {
            "timestamp": result["timestamp"],
            "timestamp_iso": result["timestamp_iso"],
            "master_count": result["master_count"],
            "audit_count": result["audit_count"],
            "discrepancy": result["discrepancy"],
            "status": result["status"],
            "action": result["action"]
        }
        
        with open(checkpoint_file, 'a') as f:
            f.write(json.dumps(checkpoint_entry) + '\n')
    except Exception as e:
        print(f"[ERROR] Could not write checkpoint: {e}")

def print_result(result):
    """Pretty-print checkpoint result"""
    print("\n" + "="*80)
    print(f"SIGNALS CHECKPOINT VERIFICATION")
    print("="*80)
    print(f"\nTimestamp: {result['timestamp']}")
    print(f"\n  SIGNALS_MASTER.jsonl:           {result['master_count']:,} signals")
    print(f"  SIGNALS_INDEPENDENT_AUDIT.txt: {result['audit_count']:,} signals")
    print(f"\n  {result['status']}")
    
    if result['discrepancy'] > 0:
        print(f"\n  ⚠️ ALERT: {result['action']}")
    else:
        print(f"\n  Status: {result['action']}")
    
    print("\n" + "="*80 + "\n")

def main():
    """Run checkpoint verification"""
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        print("[INFO] Starting continuous checkpoint monitoring...")
        print("[INFO] Will verify every hour at :00 minute")
        
        while True:
            now = get_jakarta_time()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()
            
            # Run checkpoint at start of each hour
            if now.minute == 0:
                result = verify_checkpoint()
                log_checkpoint(result)
                print_result(result)
            
            # Wait until next hour
            time.sleep(60)
    else:
        # Single check
        result = verify_checkpoint()
        log_checkpoint(result)
        print_result(result)
        
        # Exit with status code if discrepancy
        if result['discrepancy'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()

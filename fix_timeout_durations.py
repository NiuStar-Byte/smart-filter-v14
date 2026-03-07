#!/usr/bin/env python3
"""
Fix SIGNALS_MASTER.jsonl TIMEOUT durations
Recalculates closed_at as ACTUAL timeout time (not check time)

ISSUE: Historical TIMEOUT signals have closed_at = current_check_time
FIX: Recalculate as fired_time + (max_bars × tf_minutes)

This preserves all other fields but corrects the duration inflation.
"""

import json
from datetime import datetime, timedelta, timezone
import os
import shutil

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
backup_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl.backup_before_duration_fix")

MAX_BARS = {
    "15min": 15,   # 15 bars × 15min = 225 min = 3.75 hours
    "30min": 10,   # 10 bars × 30min = 300 min = 5 hours
    "1h": 5        # 5 bars × 60min = 300 min = 5 hours
}

TF_MINUTES = {
    "15min": 15,
    "30min": 30,
    "1h": 60
}

print("[INFO] Fixing TIMEOUT signal durations in SIGNALS_MASTER.jsonl")
print(f"[INFO] Backup: {backup_file}")

# Backup current file
shutil.copy2(master_file, backup_file)
print(f"[INFO] Backup created")

try:
    records = []
    fixed_count = 0
    stale_count = 0
    
    # Read all records
    with open(master_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append(record)
    
    # Fix TIMEOUT/STALE_TIMEOUT signals
    for record in records:
        status = record.get('status')
        
        if status in ['TIMEOUT', 'STALE_TIMEOUT']:
            fired_time_str = record.get('fired_time_utc')
            timeframe = record.get('timeframe')
            
            if fired_time_str and timeframe and timeframe in MAX_BARS:
                try:
                    # Calculate ACTUAL timeout time
                    fired_time = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
                    max_bars = MAX_BARS[timeframe]
                    tf_minutes = TF_MINUTES[timeframe]
                    
                    actual_timeout_time = fired_time + timedelta(minutes=max_bars * tf_minutes)
                    
                    # Update closed_at with actual timeout time
                    old_closed_at = record.get('closed_at')
                    record['closed_at'] = actual_timeout_time.isoformat()
                    
                    if status == 'TIMEOUT':
                        fixed_count += 1
                    else:
                        stale_count += 1
                    
                    # Debug: Show a few examples
                    if fixed_count <= 3 or stale_count <= 1:
                        old_duration_str = ""
                        if old_closed_at:
                            try:
                                old_closed = datetime.fromisoformat(old_closed_at.replace('Z', '+00:00'))
                                old_duration_sec = int((old_closed - fired_time).total_seconds())
                                old_duration_hours = old_duration_sec / 3600
                                old_duration_str = f" (was {old_duration_hours:.2f}h)"
                            except:
                                pass
                        
                        new_duration_sec = int((actual_timeout_time - fired_time).total_seconds())
                        new_duration_hours = new_duration_sec / 3600
                        
                        print(f"  [{status}] {record.get('symbol')} {timeframe}: {new_duration_hours:.2f}h{old_duration_str}")
                
                except Exception as e:
                    print(f"[WARN] Error fixing {record.get('symbol')}: {e}")
    
    # Write back corrected records
    with open(master_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\n[SUCCESS] Fixed {fixed_count} TIMEOUT + {stale_count} STALE_TIMEOUT signals")
    print(f"[INFO] Durations are now ACTUAL timeout times, not check times")
    print(f"[INFO] Run: python3 analyze_timeout_buckets.py (again)")

except Exception as e:
    print(f"[ERROR] Fix failed: {e}")
    print(f"[RESTORE] Restoring from backup...")
    shutil.copy2(backup_file, master_file)
    print(f"[INFO] Restored from backup")

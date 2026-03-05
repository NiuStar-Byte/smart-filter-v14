#!/usr/bin/env python3
"""
HOURLY SIGNAL LEDGER - Immutable, locked records per hour
Creates a permanent record: once an hour is locked, the count never changes
Prevents "immutable" hours from showing new signals later
"""

import json
import os
from datetime import datetime, timezone, timedelta

LEDGER_FILE = "/Users/geniustarigan/.openclaw/workspace/HOURLY_SIGNAL_LEDGER.jsonl"
SENT_SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"

def get_hour_key(hour_dt):
    """Get unique key for an hour: YYYY-MM-DD_HH"""
    return hour_dt.strftime("%Y-%m-%d_%H")

def count_signals_in_hour(hour_start, hour_end):
    """Count signals fired within [hour_start, hour_end)"""
    count = 0
    
    if not os.path.exists(SENT_SIGNALS_FILE):
        return count
    
    try:
        with open(SENT_SIGNALS_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    signal = json.loads(line.strip())
                    fired_str = signal.get('fired_time_utc', '')
                    
                    if not fired_str:
                        continue
                    
                    fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                    if fired.tzinfo is None:
                        fired = fired.replace(tzinfo=timezone.utc)
                    
                    if hour_start <= fired < hour_end:
                        count += 1
                except:
                    pass
    except:
        pass
    
    return count

def load_ledger():
    """Load all locked hours from ledger"""
    ledger = {}
    
    if not os.path.exists(LEDGER_FILE):
        return ledger
    
    try:
        with open(LEDGER_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    hour_key = entry.get('hour_key')
                    if hour_key:
                        ledger[hour_key] = entry
                except:
                    pass
    except:
        pass
    
    return ledger

def lock_hour(hour_dt):
    """
    Lock an hour: count signals and record as immutable
    Returns: (hour_key, signal_count, is_new_record)
    """
    hour_start, hour_end = hour_dt, hour_dt + timedelta(hours=1)
    hour_key = get_hour_key(hour_dt)
    signal_count = count_signals_in_hour(hour_start, hour_end)
    
    # Load existing ledger
    ledger = load_ledger()
    
    # Check if already locked
    if hour_key in ledger:
        prev_count = ledger[hour_key].get('signal_count', 0)
        is_new = False
        
        # Warn if count changed
        if signal_count != prev_count:
            print(f"[WARN] Hour {hour_key} already locked with {prev_count} signals, "
                  f"but now has {signal_count} signals! Data integrity issue!")
        
        return (hour_key, signal_count, is_new)
    
    # New hour - lock it
    entry = {
        'hour_key': hour_key,
        'hour_utc': hour_dt.isoformat(),
        'signal_count': signal_count,
        'locked_at_utc': datetime.utcnow().isoformat(),
        'status': 'IMMUTABLE'
    }
    
    # Append to ledger
    try:
        with open(LEDGER_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"[ERROR] Failed to write ledger: {e}")
    
    print(f"[LEDGER] Locked hour {hour_key}: {signal_count} signals")
    
    return (hour_key, signal_count, True)

def get_hour_status(hour_dt):
    """
    Get status of an hour: locked/immutable count or current live count
    Returns: (status, signal_count, is_locked)
    """
    hour_key = get_hour_key(hour_dt)
    now = datetime.now(timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    ledger = load_ledger()
    
    if hour_key in ledger:
        # Hour is locked - use ledger count
        locked_count = ledger[hour_key].get('signal_count', 0)
        return ('✓ IMMUTABLE', locked_count, True)
    elif hour_dt < current_hour:
        # Past hour, not yet locked - lock it now
        _, count, _ = lock_hour(hour_dt)
        return ('✓ IMMUTABLE', count, True)
    else:
        # Current or future hour - show live count
        hour_start, hour_end = hour_dt, hour_dt + timedelta(hours=1)
        count = count_signals_in_hour(hour_start, hour_end)
        
        if hour_dt == current_hour:
            return ('🔄 still accumulating', count, False)
        else:
            return ('🔄 not yet started', count, False)

def show_hourly_breakdown():
    """Show 24-hour breakdown with proper immutable/accumulating status (GMT+7)"""
    print("[HOURLY BREAKDOWN - WITH LEDGER INTEGRITY CHECK (GMT+7)]")
    print("=" * 100)
    
    now = datetime.now(timezone.utc)
    current_hour_utc = now.replace(minute=0, second=0, microsecond=0)
    
    for hours_back in range(24, -1, -1):
        hour_utc = current_hour_utc - timedelta(hours=hours_back)
        status, count, is_locked = get_hour_status(hour_utc)
        
        # Convert UTC to GMT+7 for display
        hour_gmt7 = hour_utc + timedelta(hours=7)
        hour_display = f"{hour_gmt7.strftime('%H')}:00-{(hour_gmt7 + timedelta(hours=1)).strftime('%H')}:00"
        
        if count > 0 or hours_back <= 1:  # Show hours with signals + last 2 hours
            print(f"  {hour_display}: {count:3d} fired | {status}")
    
    print("=" * 100)
    print("\n✓ IMMUTABLE = Locked in ledger, count will never change")
    print("🔄 still accumulating = Current hour, more signals may arrive")

if __name__ == '__main__':
    show_hourly_breakdown()

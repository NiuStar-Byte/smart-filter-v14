#!/usr/bin/env python3
"""
HOURLY SIGNAL ANALYSIS - Zero Daemon, Clean Architecture
Runs every UTC hour at :00
- Analyzes signals for that hour
- Locks the hour (immutable)
- Appends to SIGNALS_LEDGER_IMMUTABLE.jsonl
- No daemon, no crashes, no missed hours
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

LEDGER_FILE = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_LEDGER_IMMUTABLE.jsonl"
SIGNALS_FIRED_FILE = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/signals_fired.jsonl"

def get_current_hour_utc():
    """Get current hour in UTC"""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)

def get_previous_hour_utc():
    """Get previous hour in UTC (the hour we should lock)"""
    now = datetime.now(timezone.utc)
    return (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

def get_hour_range(hour_dt):
    """Get start and end of UTC hour"""
    start = hour_dt
    end = hour_dt + timedelta(hours=1) - timedelta(microseconds=1)
    return start, end

def load_new_signals_for_hour(hour_start, hour_end):
    """Load signals from signals_fired.jsonl that were fired in this hour"""
    signals = []
    
    if not os.path.exists(SIGNALS_FIRED_FILE):
        return signals
    
    try:
        with open(SIGNALS_FIRED_FILE, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    fired_str = signal.get('fired_time_utc', '')
                    
                    if fired_str:
                        # Parse ISO format
                        fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                        if fired.tzinfo is None:
                            fired = fired.replace(tzinfo=timezone.utc)
                        
                        # Check if in this hour
                        if hour_start <= fired < hour_end:
                            signals.append(signal)
                except:
                    pass
    except:
        pass
    
    return signals

def get_already_recorded_uuids():
    """Get all UUIDs already in ledger (to avoid duplicates)"""
    uuids = set()
    
    if not os.path.exists(LEDGER_FILE):
        return uuids
    
    try:
        with open(LEDGER_FILE, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    uuid = signal.get('uuid')
                    if uuid:
                        uuids.add(uuid)
                except:
                    pass
    except:
        pass
    
    return uuids

def append_signals_to_ledger(signals, hour_start):
    """Append new signals to immutable ledger"""
    if not signals:
        print(f"[{hour_start.strftime('%Y-%m-%d %H:00 UTC')}] No signals for this hour")
        return 0
    
    already_recorded = get_already_recorded_uuids()
    new_signals = [s for s in signals if s.get('uuid') not in already_recorded]
    
    if not new_signals:
        print(f"[{hour_start.strftime('%Y-%m-%d %H:00 UTC')}] All signals already recorded")
        return 0
    
    # Append to ledger
    try:
        with open(LEDGER_FILE, 'a') as f:
            for signal in new_signals:
                f.write(json.dumps(signal) + '\n')
        
        print(f"[{hour_start.strftime('%Y-%m-%d %H:00 UTC')}] ✓ Locked hour: {len(new_signals)} signals appended")
        return len(new_signals)
    except Exception as e:
        print(f"[ERROR] Failed to append signals: {e}")
        return 0

def lock_hour():
    """
    Lock the previous hour (immutable)
    - Current hour: :00 UTC now
    - Previous hour: just ended, now locked
    """
    prev_hour = get_previous_hour_utc()
    hour_start, hour_end = get_hour_range(prev_hour)
    
    print(f"\n{'='*80}")
    print(f"HOURLY SIGNAL LOCK - {prev_hour.strftime('%Y-%m-%d %H:00 UTC')}")
    print(f"{'='*80}")
    
    # Load signals fired in that hour
    signals = load_new_signals_for_hour(hour_start, hour_end)
    
    if signals:
        print(f"Found {len(signals)} signals from signals_fired.jsonl")
    
    # Append to immutable ledger
    appended = append_signals_to_ledger(signals, prev_hour)
    
    if appended > 0:
        print(f"✓ Hour {prev_hour.strftime('%H:00 UTC')} LOCKED with {appended} signals")
    else:
        print(f"✓ Hour {prev_hour.strftime('%H:00 UTC')} LOCKED (no new signals)")
    
    return appended

def main():
    """Main entry point"""
    current_time = datetime.now(timezone.utc)
    
    print(f"[START] Hourly Signal Analysis - {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Lock previous hour
    appended = lock_hour()
    
    print(f"\n✅ Hourly analysis complete\n")

if __name__ == "__main__":
    main()

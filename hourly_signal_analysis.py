#!/usr/bin/env python3
"""
HOURLY SIGNAL ANALYSIS - Handle Live, Constantly-Updating SENT_SIGNALS.jsonl
Runs every hour at UTC :00
- Reads LIVE SENT_SIGNALS.jsonl (signals keep arriving, file changes)
- Locks past hours as IMMUTABLE (hour closed, no more signals for it)
- Current hour marked as "still accumulating" (more signals may arrive)
- Reports clean hourly breakdown without breaking on file updates
"""

import json
import os
from datetime import datetime, timezone, timedelta

SENT_SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"

def get_current_hour_utc():
    """Get current hour in UTC (top of hour)"""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)

def get_hour_start_end(hour_dt):
    """Get start and end timestamps for a UTC hour"""
    start = hour_dt
    end = hour_dt + timedelta(hours=1)
    return start, end

def load_signals_for_hour(hour_start, hour_end):
    """
    Load signals from SENT_SIGNALS.jsonl fired within [hour_start, hour_end)
    Returns list of signals (dict objects)
    """
    signals_in_hour = []
    
    if not os.path.exists(SENT_SIGNALS_FILE):
        print(f"[WARN] {SENT_SIGNALS_FILE} not found")
        return signals_in_hour
    
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
                    
                    # Parse ISO format timestamp
                    try:
                        fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                        if fired.tzinfo is None:
                            fired = fired.replace(tzinfo=timezone.utc)
                    except:
                        continue
                    
                    # Check if fired within this hour window
                    if hour_start <= fired < hour_end:
                        signals_in_hour.append(signal)
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Reading SENT_SIGNALS.jsonl: {e}")
    
    return signals_in_hour

def analyze_hour(hour_start, hour_end, is_current_hour=False):
    """
    Analyze signals for a specific hour
    Returns: {
        'hour': 'HH:00-HH:59',
        'hour_utc': hour_start,
        'is_current': is_current_hour,
        'status': 'IMMUTABLE' or '🔄 still accumulating',
        'total_fired': int,
        'first_fired': timestamp or None,
        'last_fired': timestamp or None,
        'by_status': {'OPEN': 0, 'TP_HIT': 0, 'SL_HIT': 0, 'TIMEOUT': 0, 'STALE_TIMEOUT': 0}
    }
    """
    signals = load_signals_for_hour(hour_start, hour_end)
    
    # Determine status
    if is_current_hour:
        status = "🔄 still accumulating"
    else:
        status = "✓ IMMUTABLE"
    
    # Get timestamps
    fired_times = []
    for sig in signals:
        try:
            fired_str = sig.get('fired_time_utc', '')
            if fired_str:
                fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                if fired.tzinfo is None:
                    fired = fired.replace(tzinfo=timezone.utc)
                fired_times.append(fired)
        except:
            pass
    
    fired_times.sort()
    first_fired = fired_times[0].strftime("%H:%M:%S") if fired_times else None
    last_fired = fired_times[-1].strftime("%H:%M:%S") if fired_times else None
    
    # Status breakdown
    status_breakdown = {
        'OPEN': sum(1 for s in signals if s.get('status') == 'OPEN'),
        'TP_HIT': sum(1 for s in signals if s.get('status') == 'TP_HIT'),
        'SL_HIT': sum(1 for s in signals if s.get('status') == 'SL_HIT'),
        'TIMEOUT': sum(1 for s in signals if s.get('status') == 'TIMEOUT'),
        'STALE_TIMEOUT': sum(1 for s in signals if s.get('status') == 'STALE_TIMEOUT')
    }
    
    hour_display = f"{hour_start.strftime('%H')}:00-{hour_end.strftime('%H')}:00"
    
    return {
        'hour': hour_display,
        'hour_utc': hour_start.isoformat(),
        'is_current': is_current_hour,
        'status': status,
        'total_fired': len(signals),
        'first_fired': first_fired,
        'last_fired': last_fired,
        'by_status': status_breakdown
    }

def run_hourly_analysis():
    """
    Main function: Analyze current hour and show hourly breakdown
    Runs at top of every hour (UTC :00)
    """
    now = datetime.now(timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    print("[START] Hourly Signal Analysis - 2026-03-05 UTC")
    print("=" * 80)
    
    # Analyze current hour
    curr_start, curr_end = get_hour_start_end(current_hour)
    curr_result = analyze_hour(curr_start, curr_end, is_current_hour=True)
    
    print(f"\n📍 CURRENT HOUR ({curr_result['hour']} UTC):")
    print(f"  Status: {curr_result['status']}")
    print(f"  Signals fired: {curr_result['total_fired']}")
    if curr_result['first_fired']:
        print(f"  First signal: {curr_result['first_fired']}")
        print(f"  Last signal: {curr_result['last_fired']}")
    print(f"  Breakdown: OPEN={curr_result['by_status']['OPEN']} | "
          f"TP={curr_result['by_status']['TP_HIT']} | "
          f"SL={curr_result['by_status']['SL_HIT']} | "
          f"TIMEOUT={curr_result['by_status']['TIMEOUT']}")
    
    # Analyze previous hours (locked)
    print(f"\n📋 HOURLY BREAKDOWN (All past hours IMMUTABLE):")
    print("-" * 80)
    
    for hours_back in range(24, 0, -1):
        past_hour = current_hour - timedelta(hours=hours_back)
        past_start, past_end = get_hour_start_end(past_hour)
        
        past_result = analyze_hour(past_start, past_end, is_current_hour=False)
        
        if past_result['total_fired'] > 0:
            print(f"  {past_result['hour']}: {past_result['total_fired']} fired | {past_result['status']} | "
                  f"TP={past_result['by_status']['TP_HIT']} SL={past_result['by_status']['SL_HIT']} "
                  f"TIMEOUT={past_result['by_status']['TIMEOUT']}")
    
    print("\n" + "=" * 80)
    print("✅ Hourly analysis complete")
    print("=" * 80)

if __name__ == '__main__':
    run_hourly_analysis()

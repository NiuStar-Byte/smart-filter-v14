"""
STRICT DEDUP - Catches exact duplicates (same symbol + timeframe + fired_time)
ADDITION ONLY - Does not modify existing dedup logic
Used to prevent signals like BTC-USDT 1h firing twice in same millisecond
"""

import json
from pathlib import Path
from datetime import datetime

SIGNALS_FILE = Path("/Users/geniustarigan/.openclaw/workspace/COMPLETE_SIGNALS.jsonl")

def is_exact_duplicate(new_signal: dict) -> bool:
    """
    Check if exact duplicate already exists in COMPLETE_SIGNALS.jsonl
    
    Exact match: symbol + timeframe + fired_time_utc
    (ignores slight price differences, only catches true duplicates)
    
    Args:
        new_signal: Signal dict with symbol, timeframe, fired_time_utc
    
    Returns:
        True if duplicate found, False if unique
    """
    if not SIGNALS_FILE.exists():
        return False
    
    try:
        new_symbol = new_signal.get('symbol')
        new_tf = new_signal.get('timeframe')
        new_fired = new_signal.get('fired_time_utc')
        
        if not (new_symbol and new_tf and new_fired):
            return False
        
        # Quick check: scan recent lines (last 1000) for performance
        with open(SIGNALS_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-1000:]:  # Check last 1000 signals only
                try:
                    existing = json.loads(line)
                    if (existing.get('symbol') == new_symbol and
                        existing.get('timeframe') == new_tf and
                        existing.get('fired_time_utc') == new_fired):
                        return True  # Exact duplicate found
                except:
                    pass
        
        return False
    except Exception as e:
        print(f"[STRICT_DEDUP] Error checking duplicate: {e}", flush=True)
        return False

def log_dedup_stats():
    """Log how many exact duplicates were caught today"""
    try:
        with open(SIGNALS_FILE, 'r') as f:
            signals = [json.loads(line) for line in f if line.strip()]
        
        # Count by (symbol, timeframe, fired_time)
        seen = {}
        duplicates = 0
        for sig in signals:
            key = (sig.get('symbol'), sig.get('timeframe'), sig.get('fired_time_utc'))
            if key in seen:
                duplicates += 1
            else:
                seen[key] = True
        
        if duplicates > 0:
            print(f"[STRICT_DEDUP] Found {duplicates} exact duplicate signals today", flush=True)
        return duplicates
    except Exception as e:
        print(f"[STRICT_DEDUP] Error getting stats: {e}", flush=True)
        return 0

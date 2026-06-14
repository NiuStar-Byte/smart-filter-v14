#!/usr/bin/env python3
"""
ASTERDEX Entry Logger - Isolated logging for performance tracking
Logs posted entries to ASTERDEX_POSTED_ENTRIES.jsonl for later correlation.
ZERO impact on posting logic.
"""

import json
from pathlib import Path
from datetime import datetime

POSTED_ENTRIES_FILE = Path(__file__).parent / "ASTERDEX_POSTED_ENTRIES.jsonl"


def log_posted_entry(signal_uuid, symbol, side, entry_price, quantity, 
                    timeframe, tier, mtf_alignment_band, route, confidence_level,
                    tp_price=None, sl_price=None, entry_order_id=None, tp_order_id=None, sl_order_id=None):
    """
    Log a posted entry to tracking file.
    
    Called AFTER successful posting to Asterdex.
    This is fire-and-forget - exceptions don't impact trading.
    
    Tracking starts from Jun 7, 2026 onwards (ignore entries before that).
    
    CRITICAL: Stores ORDER IDs from Asterdex - these are the unique identifiers
    for matching against trade history (like UUID for signals).
    """
    try:
        # Filter: only log entries from Jun 7 2026 onwards
        from datetime import datetime
        now = datetime.utcnow()
        tracking_start = datetime(2026, 6, 7, 0, 0, 0)
        
        if now < tracking_start:
            return True  # Silently skip pre-tracking entries
        
        entry = {
            "signal_uuid": signal_uuid,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "timeframe": timeframe,
            "tier": tier,
            "mtf_alignment_band": mtf_alignment_band,
            "route": route,
            "confidence_level": confidence_level,
            "tp_price": tp_price,
            "sl_price": sl_price,
            # ORDER IDs - Critical for matching against trade history
            "entry_order_id": entry_order_id,
            "tp_order_id": tp_order_id,
            "sl_order_id": sl_order_id,
            "posted_timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "POSTED"
        }
        
        with open(POSTED_ENTRIES_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        return True
    except Exception as e:
        # Silent fail - don't impact posting
        print(f"[WARN] Failed to log posted entry {signal_uuid}: {e}")
        return False

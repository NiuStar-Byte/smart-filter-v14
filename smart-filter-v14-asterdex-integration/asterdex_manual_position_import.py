#!/usr/bin/env python3
"""
ASTERDEX MANUAL POSITION IMPORT
Import closed positions from Asterdex UI export and update LIVE tracker

Usage:
  python3 asterdex_manual_position_import.py <json_file>
  
Where <json_file> is positions exported from Asterdex (array of position objects)
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

LIVE_FILE = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
CUTOFF = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)

def load_existing():
    """Load existing positions"""
    existing = {}
    if LIVE_FILE.exists():
        with open(LIVE_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    pos = json.loads(line)
                    pos_id = pos.get('position_id')
                    if pos_id:
                        existing[pos_id] = pos
    return existing

def import_positions(json_file):
    """Import positions from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except:
        print(f"❌ Could not read {json_file}")
        return 0
    
    # Handle both array and object
    if isinstance(data, dict):
        positions = [data]
    else:
        positions = data
    
    existing = load_existing()
    added = 0
    
    for pos in positions:
        # Normalize position data
        normalized = {
            "position_id": pos.get('position_id') or f"{pos.get('symbol')}_{pos.get('entry_order_id')}_{pos.get('exit_order_id')}",
            "symbol": pos.get('symbol'),
            "side": pos.get('side'),
            "entry_price": float(pos.get('entry_price', 0)),
            "exit_price": float(pos.get('exit_price', 0)),
            "quantity": float(pos.get('quantity', 0)),
            "entry_order_id": pos.get('entry_order_id'),
            "exit_order_id": pos.get('exit_order_id'),
            "opened": pos.get('opened'),
            "closed": pos.get('closed'),
            "pnl_usd": float(pos.get('pnl_usd', 0)),
            "pnl_pct": float(pos.get('pnl_pct', 0)),
            "leverage": pos.get('leverage', '10x')
        }
        
        # Check if Jun 7+ by closed date
        try:
            closed_str = normalized.get('closed', '')
            if 'T' in closed_str:
                closed_dt = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
                if closed_dt < CUTOFF:
                    continue  # Skip positions closed before Jun 7
        except:
            pass
        
        # Add if new
        if normalized['position_id'] not in existing:
            with open(LIVE_FILE, 'a') as f:
                f.write(json.dumps(normalized) + '\n')
            added += 1
            print(f"  ✅ Added {normalized['symbol']} ({normalized['side']}) P&L ${normalized['pnl_usd']:+.2f}")
    
    return added

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 asterdex_manual_position_import.py <json_file>")
        print("")
        print("Step 1: Export closed positions from Asterdex UI as JSON")
        print("Step 2: Run this script with the JSON file")
        print("Step 3: Tracker updates automatically with new positions")
        sys.exit(1)
    
    json_file = sys.argv[1]
    print(f"📥 Importing positions from {json_file}...")
    added = import_positions(json_file)
    
    if added > 0:
        print(f"")
        print(f"✅ SUCCESS: Added {added} new positions")
        print(f"📊 Run tracker to see updated metrics:")
        print(f"   python3 asterdex_tracker_final_jun7.py")
    else:
        print(f"ℹ️  No new positions to import")

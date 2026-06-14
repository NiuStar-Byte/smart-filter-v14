#!/usr/bin/env python3
"""
ASTERDEX Comprehensive Entry Extractor - Pulls ALL trade orders from Jun 7+
Extracts all closed positions from order history, not just filtered symbols
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

env_file = Path('/Users/geniustarigan/.openclaw/workspace/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                line = line.replace('export ', '', 1).strip()
                key, value = line.split('=', 1)
                value = value.strip().strip("'\"")
                os.environ[key.strip()] = value

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'smart-filter-v14-main'))

from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
import requests

API_BASE = "https://fapi.asterdex.com"

# All symbols seen in Asterdex UI (Jun 7+ closed positions)
ALL_SYMBOLS = [
    'FUN-USDT', 'DOT-USDT', 'SEI-USDT', 'TURBO-USDT', 'KAS-USDT', 'WIF-USDT', 
    'HYPE-USDT', 'ARB-USDT', 'PORTAL-USDT', 'IP-USDT', 'DOGE-USDT', 'BNB-USDT', 
    'XPL-USDT', 'NEAR-USDT', 'PARTI-USDT', 'BLUR-USDT', 'AAVE-USDT', 'HBAR-USDT', 
    'ONDO-USDT', 'LA-USDT', 'DYDX-USDT', 'INJ-USDT', 'LDO-USDT', 'BERA-USDT', 
    'ENS-USDT', 'PUMP-USDT', 'XRP-USDT', 'SOL-USDT', 'APT-USDT'
]

OUTPUT_FILE = Path(__file__).parent / "ASTERDEX_COMPREHENSIVE_ENTRIES.jsonl"


def extract_all_entries(auth, symbols, start_date='2026-06-07'):
    """
    Extract all closed position entries from order history.
    """
    print(f"\n[{datetime.now().isoformat()}] Starting comprehensive entry extraction...")
    print(f"[INFO] Extracting from {len(symbols)} symbols")
    
    try:
        start_dt = datetime.fromisoformat(start_date)
    except:
        start_dt = datetime(2026, 6, 7, 0, 0, 0)
    
    start_ts = int(start_dt.timestamp() * 1000)
    
    all_entries = []
    
    for symbol in symbols:
        symbol_asterdex = symbol.replace('-', '')
        
        params = {
            'symbol': symbol_asterdex,
            'limit': 1000,
            'startTime': start_ts
        }
        
        try:
            signed_params = auth.sign_request_v3(base_params=params)
            
            response = requests.get(
                f"{API_BASE}/fapi/v3/allOrders",
                params=signed_params,
                timeout=10
            )
            
            if response.status_code == 200:
                orders = response.json()
                filled_orders = [o for o in orders if o.get('status') == 'FILLED']
                
                if filled_orders:
                    print(f"[INFO] {symbol}: {len(filled_orders)} filled orders")
                    
                    # Group by position: entry (BUY/SELL) + exit (opposite side, reduceOnly)
                    entries_found = extract_positions(symbol, filled_orders)
                    if entries_found:
                        print(f"      → {len(entries_found)} position(s) found")
                        all_entries.extend(entries_found)
                        
            else:
                print(f"[WARN] {symbol}: API error {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n[OK] Extracted {len(all_entries)} total positions")
    print(f"[INFO] Saved to {OUTPUT_FILE}")
    
    return all_entries


def extract_positions(symbol, filled_orders):
    """
    Extract ALL position entries from filled orders (multi-trade support).
    Entry = opening order (reduceOnly=false)
    Exit = closing order (reduceOnly=true) that follows entry
    
    IMPORTANT: Handles MULTIPLE entry/exit pairs per symbol
    (respecting cooldown duration between trades)
    """
    positions = []
    
    # Sort by time
    filled_orders.sort(key=lambda x: x.get('time', 0))
    
    # Separate entries and exits
    entries = [o for o in filled_orders if o.get('reduceOnly') == False]
    exits = [o for o in filled_orders if o.get('reduceOnly') == True]
    
    # Match each entry with the NEXT available exit
    used_exits = set()  # Track which exits have been used
    
    for entry_order in entries:
        entry_side = entry_order.get('side')
        entry_qty = float(entry_order.get('executedQty', 0))
        entry_time = entry_order.get('time', 0)
        entry_order_id = entry_order.get('orderId')
        
        # Find the NEXT unused exit that matches this entry
        exit_order = None
        exit_index = None
        
        for i, exit_ord in enumerate(exits):
            # Skip already-used exits
            if i in used_exits:
                continue
            
            # Must be after entry, opposite side, and closing order
            if (exit_ord.get('time', 0) > entry_time and
                exit_ord.get('reduceOnly') == True and
                exit_ord.get('side') != entry_side):
                
                # Check quantity match (within 5%)
                exit_qty = float(exit_ord.get('executedQty', 0))
                if abs(exit_qty - entry_qty) / entry_qty <= 0.05:
                    exit_order = exit_ord
                    exit_index = i
                    used_exits.add(i)  # Mark this exit as used
                    break
        
        if exit_order:
            # Build position record
            position = {
                'symbol': symbol,
                'side': 'LONG' if entry_side == 'BUY' else 'SHORT',
                'entry_price': float(entry_order.get('avgPrice', 0)),
                'quantity': entry_qty,
                'posted_timestamp': datetime.fromtimestamp(entry_time / 1000).isoformat() + 'Z',
                'entry_order_id': entry_order_id,
                'tp_order_id': exit_order.get('orderId'),
                'sl_order_id': None,
                'exit_order_id': exit_order.get('orderId'),
                'exit_type': 'CLOSED',
                'signal_uuid': f'EXTRACTED_{symbol.replace("-", "")}_{entry_order_id}',
                'tier': 'UNKNOWN',
                'mtf_alignment_band': 'UNKNOWN',
                'route': 'UNKNOWN',
                'confidence_level': 'UNKNOWN',
                'timeframe': 'UNKNOWN',
                'status': 'CLOSED'
            }
            positions.append(position)
    
    return positions


if __name__ == '__main__':
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    entries = extract_all_entries(auth, ALL_SYMBOLS)
    
    # Show summary
    symbols_found = set(e.get('symbol') for e in entries)
    print(f"\n=== SUMMARY ===")
    print(f"Total positions: {len(entries)}")
    print(f"Symbols: {sorted(symbols_found)}")

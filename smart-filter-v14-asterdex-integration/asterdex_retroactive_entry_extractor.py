#!/usr/bin/env python3
"""
ASTERDEX Retroactive Entry Extractor - Extract posted entries from trade history
Since entries were already posted (Jun 1-8) but not logged, we extract them backwards
from the order/trade history on Asterdex.

Logic:
- Fetch ALL orders from Jun 1 onwards
- Find FILLED orders
- Match entry orders (BUY/SELL) with exit orders (opposite side)
- Reconstruct the posted entry records with ORDER IDs
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load environment
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
import requests

API_BASE = "https://fapi.asterdex.com"


def fetch_all_orders(auth, symbol, start_date='2026-06-01', end_date='2026-06-08'):
    """
    Fetch ALL orders from Asterdex for a symbol in a date range.
    
    This includes FILLED, CANCELLED, PARTIALLY_FILLED, etc.
    We'll filter for FILLED only.
    """
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        # Convert symbol
        symbol_asterdex = symbol.replace('-', '')
        
        params = {
            'symbol': symbol_asterdex,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000,
        }
        
        signed_params = auth.sign_request_v3(base_params=params)
        
        response = requests.get(
            f"{API_BASE}/fapi/v3/allOrders",
            params=signed_params,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[WARN] API error for {symbol}: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"[WARN] Failed to fetch orders for {symbol}: {e}")
        return []


def extract_entries_from_orders(orders, symbol):
    """
    Extract posted entry pairs from orders.
    
    A "posted entry" consists of:
    - Entry order: BUY or SELL (FILLED)
    - TP order: TAKE_PROFIT_MARKET (optionally FILLED)
    - SL order: STOP_LOSS_MARKET (optionally FILLED)
    
    Returns list of reconstructed entry records.
    """
    entries = []
    
    # Group by clientOrderId to find entry + TP + SL groups
    # Or match by price + side + time
    
    filled_orders = [o for o in orders if o.get('status') == 'FILLED']
    
    if not filled_orders:
        return entries
    
    # Convert symbol back to our format
    symbol_formatted = symbol.replace('USDT', '-USDT') if symbol.endswith('USDT') else symbol
    
    # For each entry order, look for corresponding TP/SL
    for order in filled_orders:
        order_type = order.get('type')
        
        # Only process LIMIT orders as entries (TP/SL are special types)
        if order_type != 'LIMIT':
            continue
        
        side = order.get('side')
        entry_price = float(order.get('avgPrice', 0))
        qty = float(order.get('executedQty', 0))
        order_id = order.get('orderId')
        order_time = order.get('updateTime')
        
        if not entry_price or not qty:
            continue
        
        # Try to find corresponding TP and SL orders
        # TP: opposite side, TAKE_PROFIT_MARKET type, after entry_time
        # SL: opposite side, STOP_LOSS_MARKET type, after entry_time
        
        tp_price = None
        sl_price = None
        tp_order_id = None
        sl_order_id = None
        exit_order_id = None
        exit_type = None
        
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        
        for other_order in filled_orders:
            if other_order.get('updateTime', 0) <= order_time:
                continue  # Must be after entry
            
            if other_order.get('side') != opposite_side:
                continue
            
            other_type = other_order.get('type')
            other_price = float(other_order.get('avgPrice', 0))
            other_qty = float(other_order.get('executedQty', 0))
            other_id = other_order.get('orderId')
            
            if other_qty != qty:
                continue  # Must match quantity
            
            if other_type == 'TAKE_PROFIT_MARKET':
                tp_price = other_price
                tp_order_id = other_id
                exit_order_id = other_id
                exit_type = 'TP_HIT'
            elif other_type == 'STOP_LOSS_MARKET':
                sl_price = other_price
                sl_order_id = other_id
                # Only set as exit if TP not found
                if not exit_order_id:
                    exit_order_id = other_id
                    exit_type = 'SL_HIT'
        
        # Reconstruct entry record
        # Note: We don't have signal_uuid, tier, mtf_alignment, etc. for historical entries
        # That's OK - we can still track WR and P&L
        
        entry = {
            'symbol': symbol_formatted,
            'side': 'LONG' if side == 'BUY' else 'SHORT',
            'entry_price': entry_price,
            'quantity': qty,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'posted_timestamp': datetime.fromtimestamp(order_time / 1000).isoformat() + 'Z',
            'entry_order_id': order_id,
            'tp_order_id': tp_order_id,
            'sl_order_id': sl_order_id,
            'exit_order_id': exit_order_id,
            'exit_type': exit_type,
            # These will be unknown for historical entries
            'signal_uuid': f"HISTORICAL_{symbol}_{order_id}",  # Placeholder
            'tier': 'UNKNOWN',
            'mtf_alignment_band': 'UNKNOWN',
            'route': 'UNKNOWN',
            'confidence_level': 'UNKNOWN',
            'timeframe': 'UNKNOWN',
            'status': 'CLOSED' if exit_order_id else 'OPEN',
        }
        
        entries.append(entry)
    
    return entries


def extract_all_retroactive_entries(symbols, start_date='2026-06-01', end_date='2026-06-08'):
    """
    Extract all retroactive entries from all symbols' order history.
    """
    try:
        auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize auth: {e}")
        return []
    
    all_entries = []
    
    for symbol in symbols:
        print(f"[INFO] Extracting entries for {symbol}...")
        
        # Convert to Asterdex format
        symbol_asterdex = symbol.replace('-', '')
        
        # Fetch all orders for this symbol
        orders = fetch_all_orders(auth, symbol, start_date, end_date)
        
        if not orders:
            continue
        
        # Extract entries from orders
        entries = extract_entries_from_orders(orders, symbol_asterdex)
        
        if entries:
            print(f"  ✅ Found {len(entries)} entry pairs")
            all_entries.extend(entries)
        else:
            print(f"  (no entries)")
    
    return all_entries


def save_retroactive_entries(entries):
    """
    Save extracted entries to ASTERDEX_POSTED_ENTRIES.jsonl
    """
    output_file = Path(__file__).parent / "ASTERDEX_POSTED_ENTRIES.jsonl"
    
    with open(output_file, 'a') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n✅ Saved {len(entries)} retroactive entries to ASTERDEX_POSTED_ENTRIES.jsonl")
    return len(entries)


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("ASTERDEX RETROACTIVE ENTRY EXTRACTOR")
    print("Extracting posted entries from order history (Jun 1-8 2026)")
    print("="*60 + "\n")
    
    # Symbols to extract
    symbols = [
        'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'ADA-USDT',
        'BNB-USDT', 'AVAX-USDT', 'LINK-USDT', 'DOT-USDT', 'NEAR-USDT',
        'ARB-USDT', 'OP-USDT', 'SEI-USDT', 'BLUR-USDT', 'WIF-USDT',
        'DYDX-USDT', 'RAY-USDT',
    ]
    
    # Extract entries
    print(f"[INFO] Extracting from {len(symbols)} symbols...")
    entries = extract_all_retroactive_entries(symbols)
    
    print(f"\n[INFO] Total entries extracted: {len(entries)}")
    
    if entries:
        # Save to file
        saved = save_retroactive_entries(entries)
        
        print("\n✅ EXTRACTION COMPLETE")
        print(f"   Saved: {saved} entries")
        print(f"   File: ASTERDEX_POSTED_ENTRIES.jsonl")
        print("\nNow run: python3 asterdex_performance_system.py")
    else:
        print("\n⚠️  No entries found in order history")
        print("   Either no entries were posted, or API access restricted")


if __name__ == '__main__':
    main()

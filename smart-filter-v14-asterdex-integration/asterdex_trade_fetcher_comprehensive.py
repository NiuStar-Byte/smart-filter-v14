#!/usr/bin/env python3
"""
ASTERDEX Comprehensive Trade Fetcher - Fetch orders for ALL symbols from Jun 7+
"""

import json
import os
import sys
from datetime import datetime
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

from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
import requests

API_BASE = "https://fapi.asterdex.com"
TRADES_FILE = Path(__file__).parent / "ASTERDEX_TRADES_COMPREHENSIVE.jsonl"

# All symbols seen in extracted entries
COMPREHENSIVE_SYMBOLS = [
    'AAVE-USDT', 'APT-USDT', 'BERA-USDT', 'BLUR-USDT', 'BNB-USDT', 'DOGE-USDT', 'DOT-USDT', 
    'DYDX-USDT', 'ENS-USDT', 'HBAR-USDT', 'INJ-USDT', 'IP-USDT', 'KAS-USDT', 
    'LA-USDT', 'LDO-USDT', 'NEAR-USDT', 'ONDO-USDT', 'PARTI-USDT', 'PUMP-USDT', 
    'SEI-USDT', 'TURBO-USDT', 'WIF-USDT', 'XPL-USDT', 'XRP-USDT'
]


def fetch_all_orders(auth, symbols, start_date='2026-06-07T00:00:01+07:00'):
    """Fetch all orders for comprehensive symbols."""
    
    try:
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    except:
        start_dt = datetime(2026, 6, 7, 0, 0, 1)
    
    start_timestamp = start_dt.timestamp()
    
    print(f"[{datetime.now().isoformat()}] Fetching orders for {len(symbols)} symbols...")
    
    all_orders = []
    
    for symbol in symbols:
        symbol_asterdex = symbol.replace('-', '')
        
        params = {
            'symbol': symbol_asterdex,
            'limit': 1000,
            'startTime': int(start_timestamp * 1000)
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
                    all_orders.extend(filled_orders)
            else:
                print(f"[WARN] {symbol}: API error {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
    
    # Save orders
    with open(TRADES_FILE, 'w') as f:
        for order in all_orders:
            # Convert symbol format and add side
            symbol = order.get('symbol', '')
            if symbol and '-' not in symbol and symbol.endswith('USDT'):
                symbol = symbol[:-4] + '-USDT'
                order['symbol'] = symbol
            f.write(json.dumps(order) + '\n')
    
    print(f"\n[OK] Fetched and saved {len(all_orders)} total orders")
    print(f"[INFO] File: {TRADES_FILE}")
    
    return all_orders


if __name__ == '__main__':
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    orders = fetch_all_orders(auth, COMPREHENSIVE_SYMBOLS)
    
    # Summary
    symbols_found = set(o.get('symbol') for o in orders)
    print(f"\n=== SUMMARY ===")
    print(f"Total orders: {len(orders)}")
    print(f"Symbols: {sorted(symbols_found)}")

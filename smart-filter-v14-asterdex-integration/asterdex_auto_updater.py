#!/usr/bin/env python3
"""
ASTERDEX AUTO-UPDATER - Runs on cron, fetches new closed positions, updates tracker
Run every 30 minutes to catch new position closures
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
sys.path.insert(0, str(Path(__file__).parent.parent / 'smart-filter-v14-main'))

from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
import requests

API_BASE = "https://fapi.asterdex.com"
TRACKER_FILE = Path(__file__).parent / "ASTERDEX_POSITIONS_LIVE.jsonl"
LOG_FILE = Path(__file__).parent / "asterdex_auto_update.log"

# All 29 symbols that have had activity
SYMBOLS = [
    'APT-USDT', 'FUN-USDT', 'DOT-USDT', 'SEI-USDT', 'TURBO-USDT', 'KAS-USDT', 'WIF-USDT', 
    'HYPE-USDT', 'ARB-USDT', 'PORTAL-USDT', 'IP-USDT', 'DOGE-USDT', 'BNB-USDT', 
    'XPL-USDT', 'NEAR-USDT', 'PARTI-USDT', 'BLUR-USDT', 'AAVE-USDT', 'HBAR-USDT', 
    'ONDO-USDT', 'LA-USDT', 'DYDX-USDT', 'INJ-USDT', 'LDO-USDT', 'BERA-USDT', 
    'ENS-USDT', 'PUMP-USDT', 'XRP-USDT', 'SOL-USDT'
]


def log(msg):
    """Log message to file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def load_existing_positions():
    """Load existing tracked positions"""
    positions = {}
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            for line in f:
                if line.strip():
                    pos = json.loads(line.strip())
                    pos_key = f"{pos['symbol']}_{pos['entry_price']}_{pos['opened']}"
                    positions[pos_key] = pos
    return positions


def fetch_new_positions(auth):
    """Fetch new closed positions from API"""
    new_positions = []
    
    for symbol in SYMBOLS:
        symbol_api = symbol.replace('-', '')
        
        params = {
            'symbol': symbol_api,
            'limit': 1000,
            'startTime': int(datetime(2026, 6, 1).timestamp() * 1000)
        }
        
        try:
            signed_params = auth.sign_request_v3(base_params=params)
            response = requests.get(
                f"{API_BASE}/fapi/v3/allOrders",
                params=signed_params,
                timeout=10
            )
            
            if response.status_code != 200:
                continue
            
            orders = response.json()
            filled_orders = [o for o in orders if o.get('status') == 'FILLED']
            
            # Extract entry/exit pairs
            entries = [o for o in filled_orders if o.get('reduceOnly') == False]
            exits = [o for o in filled_orders if o.get('reduceOnly') == True]
            
            used_exits = set()
            
            for entry in entries:
                entry_qty = float(entry.get('executedQty', 0))
                entry_time = entry.get('time', 0)
                entry_side = entry.get('side')
                entry_price = float(entry.get('avgPrice', 0))
                entry_id = entry.get('orderId')
                
                if entry_qty == 0:
                    continue
                
                # Find matching exit (reduce-only or reversal)
                best_exit = None
                best_exit_idx = None
                min_time_diff = float('inf')
                
                for i, exit_order in enumerate(exits):
                    if i in used_exits:
                        continue
                    
                    exit_time = exit_order.get('time', 0)
                    if exit_time <= entry_time:
                        continue
                    
                    if exit_order.get('side') == entry_side:
                        continue
                    
                    exit_qty = float(exit_order.get('executedQty', 0))
                    
                    if entry_qty > 0 and abs(exit_qty - entry_qty) / entry_qty > 0.05:
                        continue
                    
                    time_diff = exit_time - entry_time
                    if time_diff > 0 and time_diff < min_time_diff:
                        best_exit = exit_order
                        best_exit_idx = i
                        min_time_diff = time_diff
                
                if best_exit:
                    used_exits.add(best_exit_idx)
                    
                    exit_price = float(best_exit.get('avgPrice', 0))
                    exit_time = best_exit.get('time', 0)
                    exit_id = best_exit.get('orderId')
                    
                    # Calculate P&L
                    if entry_side == 'BUY':
                        pnl_usd = (exit_price - entry_price) * entry_qty
                        pnl_pct = ((exit_price - entry_price) / entry_price * 100)
                    else:
                        pnl_usd = (entry_price - exit_price) * entry_qty
                        pnl_pct = ((entry_price - exit_price) / entry_price * 100)
                    
                    position = {
                        'position_id': f"{symbol}_{entry_id}_{exit_id}",
                        'symbol': symbol,
                        'side': 'LONG' if entry_side == 'BUY' else 'SHORT',
                        'entry_price': round(entry_price, 8),
                        'exit_price': round(exit_price, 8),
                        'quantity': round(entry_qty, 8),
                        'entry_order_id': entry_id,
                        'exit_order_id': exit_id,
                        'opened': datetime.fromtimestamp(entry_time / 1000).isoformat() + 'Z',
                        'closed': datetime.fromtimestamp(exit_time / 1000).isoformat() + 'Z',
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'leverage': '10x',
                    }
                    
                    new_positions.append(position)
        
        except Exception as e:
            log(f"[WARN] {symbol}: {e}")
    
    return new_positions


def run_update():
    """Main update logic"""
    log("=== ASTERDEX AUTO-UPDATER STARTED ===")
    
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    
    # Load existing
    existing = load_existing_positions()
    log(f"Loaded {len(existing)} existing positions")
    
    # Fetch new
    new = fetch_new_positions(auth)
    log(f"Fetched {len(new)} total positions from API")
    
    # Merge (avoid duplicates)
    count_added = 0
    for pos in new:
        pos_key = f"{pos['symbol']}_{pos['entry_price']}_{pos['opened']}"
        if pos_key not in existing:
            existing[pos_key] = pos
            count_added += 1
    
    log(f"Added {count_added} new position(s)")
    
    # Save all
    all_positions = list(existing.values())
    with open(TRACKER_FILE, 'w') as f:
        for pos in sorted(all_positions, key=lambda x: x.get('opened', '')):
            f.write(json.dumps(pos) + '\n')
    
    log(f"Saved {len(all_positions)} total positions")
    log("=== ASTERDEX AUTO-UPDATER COMPLETED ===\n")


if __name__ == '__main__':
    run_update()

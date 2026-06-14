#!/usr/bin/env python3
"""
ASTERDEX COMPLETE TRACKER - Paginated order fetching for ALL positions
Handles all 38+ closed positions from Jun 7+ with pagination support
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import time

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

ALL_SYMBOLS = [
    'APT-USDT', 'FUN-USDT', 'DOT-USDT', 'SEI-USDT', 'TURBO-USDT', 'KAS-USDT', 'WIF-USDT', 
    'HYPE-USDT', 'ARB-USDT', 'PORTAL-USDT', 'IP-USDT', 'DOGE-USDT', 'BNB-USDT', 
    'XPL-USDT', 'NEAR-USDT', 'PARTI-USDT', 'BLUR-USDT', 'AAVE-USDT', 'HBAR-USDT', 
    'ONDO-USDT', 'LA-USDT', 'DYDX-USDT', 'INJ-USDT', 'LDO-USDT', 'BERA-USDT', 
    'ENS-USDT', 'PUMP-USDT', 'XRP-USDT', 'SOL-USDT'
]

TRACKER_FILE = Path(__file__).parent / "ASTERDEX_POSITIONS_TRACKED.jsonl"


class AsterdexCompleteTracker:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.start_date = datetime(2026, 6, 7, 0, 0, 0)
        self.start_ts = int(self.start_date.timestamp() * 1000)
    
    def fetch_all_orders_paginated(self, symbol):
        """
        Fetch ALL orders for a symbol with pagination support.
        Uses limit=1000 per page and paginate by timestamp.
        """
        symbol_api = symbol.replace('-', '')
        all_orders = []
        current_end_time = None
        
        while True:
            params = {
                'symbol': symbol_api,
                'limit': 1000,
                'startTime': self.start_ts
            }
            
            # If we've already fetched some, use endTime to paginate
            if current_end_time:
                params['endTime'] = current_end_time - 1
            
            try:
                signed_params = self.auth.sign_request_v3(base_params=params)
                response = requests.get(
                    f"{API_BASE}/fapi/v3/allOrders",
                    params=signed_params,
                    timeout=10
                )
                
                if response.status_code != 200:
                    break
                
                orders = response.json()
                if not orders:
                    break
                
                filled_orders = [o for o in orders if o.get('status') == 'FILLED']
                all_orders.extend(filled_orders)
                
                # Pagination: get the oldest timestamp and fetch earlier
                if len(filled_orders) < 1000:
                    break
                
                current_end_time = min(o.get('time', 0) for o in filled_orders)
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                break
        
        return all_orders
    
    def extract_all_positions(self, symbol, all_orders):
        """
        Extract ALL entry/exit pairs from complete order history.
        Handles multiple positions per symbol.
        """
        positions = []
        
        if not all_orders:
            return positions
        
        # Sort by time ascending
        all_orders.sort(key=lambda x: x.get('time', 0))
        
        # Build timeline of entries and exits
        entries = [o for o in all_orders if o.get('reduceOnly') == False]
        exits = [o for o in all_orders if o.get('reduceOnly') == True]
        
        # Match entries to exits using time-based proximity + quantity
        used_exits = set()
        
        for entry in entries:
            entry_qty = float(entry.get('executedQty', 0))
            if entry_qty == 0:
                continue
            
            entry_time = entry.get('time', 0)
            entry_side = entry.get('side')
            entry_price = float(entry.get('avgPrice', 0))
            entry_id = entry.get('orderId')
            
            # Find NEXT exit that matches
            best_exit = None
            best_exit_idx = None
            min_time_diff = float('inf')
            
            for i, exit_order in enumerate(exits):
                if i in used_exits:
                    continue
                
                exit_time = exit_order.get('time', 0)
                if exit_time <= entry_time:
                    continue
                
                # Must be opposite side
                if exit_order.get('side') == entry_side:
                    continue
                
                exit_qty = float(exit_order.get('executedQty', 0))
                
                # Check quantity match (allow 5% variance)
                if entry_qty > 0 and abs(exit_qty - entry_qty) / entry_qty > 0.05:
                    continue
                
                # Prefer exit closest in time (but after entry)
                time_diff = exit_time - entry_time
                if time_diff > 0 and time_diff < min_time_diff:
                    best_exit = exit_order
                    best_exit_idx = i
                    min_time_diff = time_diff
            
            # Create position if we found an exit
            if best_exit:
                used_exits.add(best_exit_idx)
                
                exit_price = float(best_exit.get('avgPrice', 0))
                exit_time = best_exit.get('time', 0)
                exit_id = best_exit.get('orderId')
                
                # Calculate P&L
                if entry_side == 'BUY':  # LONG
                    pnl_usd = (exit_price - entry_price) * entry_qty
                    pnl_pct = ((exit_price - entry_price) / entry_price * 100)
                else:  # SHORT
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
                    'opened_timestamp': datetime.fromtimestamp(entry_time / 1000).isoformat() + 'Z',
                    'closed_timestamp': datetime.fromtimestamp(exit_time / 1000).isoformat() + 'Z',
                    'pnl_usd': round(pnl_usd, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'status': 'CLOSED',
                    'win': 1 if pnl_usd > 0 else (0 if pnl_usd < 0 else 0.5)
                }
                
                positions.append(position)
        
        return positions
    
    def run(self):
        """Main tracker execution"""
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}] ASTERDEX COMPLETE TRACKER (PAGINATED)")
        print(f"{'='*80}")
        print(f"[INFO] Fetching all closed positions from {len(ALL_SYMBOLS)} symbols")
        print(f"[INFO] Baseline: >= {self.start_date.isoformat()} GMT+7")
        print(f"[INFO] Using paginated fetching for completeness\n")
        
        all_positions = []
        
        for i, symbol in enumerate(ALL_SYMBOLS, 1):
            print(f"[{i:2d}/{len(ALL_SYMBOLS)}] {symbol:12s}", end=' ')
            sys.stdout.flush()
            
            orders = self.fetch_all_orders_paginated(symbol)
            positions = self.extract_all_positions(symbol, orders)
            
            if positions:
                print(f"→ {len(positions):2d} closed position(s)")
                all_positions.extend(positions)
            else:
                print(f"→ no closed positions")
        
        # Save to file
        with open(TRACKER_FILE, 'w') as f:
            for pos in sorted(all_positions, key=lambda x: x.get('opened_timestamp', '')):
                f.write(json.dumps(pos) + '\n')
        
        print(f"\n{'='*80}")
        print(f"[OK] Total positions tracked: {len(all_positions)}")
        print(f"[OK] Saved to {TRACKER_FILE}")
        print(f"{'='*80}\n")
        
        # Report
        self.report(all_positions)
        
        return all_positions
    
    def report(self, positions):
        """Generate performance report"""
        if not positions:
            return
        
        wins = len([p for p in positions if p.get('win') == 1])
        losses = len([p for p in positions if p.get('win') == 0])
        breakeven = len([p for p in positions if p.get('win') == 0.5])
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / len(positions) if positions else 0
        
        print(f"PERFORMANCE")
        print(f"  Total: {len(positions)} | Wins: {wins} ({wins/len(positions)*100:.1f}%) | Losses: {losses} ({losses/len(positions)*100:.1f}%) | Breakeven: {breakeven}")
        print(f"  P&L: ${total_pnl:.2f} USD total | ${avg_pnl:.2f} USD avg\n")
        
        sorted_pnl = sorted(positions, key=lambda x: x.get('pnl_usd', 0), reverse=True)
        print(f"Top 5 Winners:")
        for pos in sorted_pnl[:5]:
            print(f"  {pos['symbol']:10s} {pos['side']:5s} ${pos['pnl_usd']:7.2f}")
        
        print(f"\nTop 5 Losers:")
        for pos in sorted_pnl[-5:]:
            print(f"  {pos['symbol']:10s} {pos['side']:5s} ${pos['pnl_usd']:7.2f}")


if __name__ == '__main__':
    tracker = AsterdexCompleteTracker()
    positions = tracker.run()

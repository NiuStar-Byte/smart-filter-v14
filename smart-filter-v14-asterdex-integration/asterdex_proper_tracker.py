#!/usr/bin/env python3
"""
ASTERDEX PROPER TRACKER - Complete Position Tracking System
Extracts ALL closed positions from Jun 7+ entries
Tracks all closed positions continuously and adds new ones as they close
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

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
    'APT-USDT', 'FUN-USDT', 'DOT-USDT', 'SEI-USDT', 'TURBO-USDT', 'KAS-USDT', 'WIF-USDT', 
    'HYPE-USDT', 'ARB-USDT', 'PORTAL-USDT', 'IP-USDT', 'DOGE-USDT', 'BNB-USDT', 
    'XPL-USDT', 'NEAR-USDT', 'PARTI-USDT', 'BLUR-USDT', 'AAVE-USDT', 'HBAR-USDT', 
    'ONDO-USDT', 'LA-USDT', 'DYDX-USDT', 'INJ-USDT', 'LDO-USDT', 'BERA-USDT', 
    'ENS-USDT', 'PUMP-USDT', 'XRP-USDT', 'SOL-USDT'
]

TRACKER_FILE = Path(__file__).parent / "ASTERDEX_POSITIONS_TRACKED.jsonl"
STATE_FILE = Path(__file__).parent / ".asterdex_tracker_state.json"


class AsterdexProperTracker:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.start_date = datetime(2026, 6, 7, 0, 0, 0)
        self.start_ts = int(self.start_date.timestamp() * 1000)
        self.tracked_positions = self.load_tracked()
    
    def load_tracked(self):
        """Load previously tracked positions"""
        positions = {}
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE) as f:
                for line in f:
                    if line.strip():
                        pos = json.loads(line.strip())
                        positions[pos.get('position_id')] = pos
        return positions
    
    def fetch_all_orders(self, symbol):
        """Fetch all orders for a symbol from Jun 7+"""
        symbol_asterdex = symbol.replace('-', '')
        params = {
            'symbol': symbol_asterdex,
            'limit': 1000,
            'startTime': self.start_ts
        }
        
        try:
            signed_params = self.auth.sign_request_v3(base_params=params)
            response = requests.get(
                f"{API_BASE}/fapi/v3/allOrders",
                params=signed_params,
                timeout=10
            )
            
            if response.status_code == 200:
                orders = response.json()
                return [o for o in orders if o.get('status') == 'FILLED']
            else:
                print(f"[WARN] {symbol}: API error {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            return []
    
    def extract_positions_from_orders(self, symbol, filled_orders):
        """
        Extract ALL position entries from filled orders.
        Entry = opening order (reduceOnly=false)
        Exit = closing order (reduceOnly=true) that follows entry
        """
        positions = []
        
        if not filled_orders:
            return positions
        
        # Sort by time
        filled_orders.sort(key=lambda x: x.get('time', 0))
        
        # Separate entries and exits
        entries = [o for o in filled_orders if o.get('reduceOnly') == False]
        exits = [o for o in filled_orders if o.get('reduceOnly') == True]
        
        # Match each entry with the NEXT available exit
        used_exits = set()
        
        for entry_order in entries:
            entry_side = entry_order.get('side')
            entry_qty = float(entry_order.get('executedQty', 0))
            entry_time = entry_order.get('time', 0)
            entry_price = float(entry_order.get('avgPrice', 0))
            entry_order_id = entry_order.get('orderId')
            
            # Find the NEXT unused exit
            exit_order = None
            exit_index = None
            
            for i, exit_ord in enumerate(exits):
                if i in used_exits:
                    continue
                
                exit_time = exit_ord.get('time', 0)
                exit_side = exit_ord.get('side')
                exit_qty = float(exit_ord.get('executedQty', 0))
                
                # Must be after entry, opposite side, reduce-only
                if (exit_time > entry_time and
                    exit_side != entry_side and
                    exit_ord.get('reduceOnly') == True):
                    
                    # Check quantity match (within 5%)
                    if entry_qty > 0 and abs(exit_qty - entry_qty) / entry_qty <= 0.05:
                        exit_order = exit_ord
                        exit_index = i
                        used_exits.add(i)
                        break
            
            if exit_order:
                exit_price = float(exit_order.get('avgPrice', 0))
                exit_time = exit_order.get('time', 0)
                exit_order_id = exit_order.get('orderId')
                
                # Calculate P&L
                if entry_side == 'BUY':  # LONG
                    pnl_usd = (exit_price - entry_price) * entry_qty
                else:  # SHORT
                    pnl_usd = (entry_price - exit_price) * entry_qty
                
                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_side == 'BUY' else ((entry_price - exit_price) / entry_price * 100)
                
                position_id = f"{symbol}_{entry_order_id}_{exit_order_id}"
                
                position = {
                    'position_id': position_id,
                    'symbol': symbol,
                    'side': 'LONG' if entry_side == 'BUY' else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': entry_qty,
                    'entry_order_id': entry_order_id,
                    'exit_order_id': exit_order_id,
                    'opened_timestamp': datetime.fromtimestamp(entry_time / 1000).isoformat() + 'Z',
                    'closed_timestamp': datetime.fromtimestamp(exit_time / 1000).isoformat() + 'Z',
                    'pnl_usd': round(pnl_usd, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'status': 'CLOSED',
                    'win': 1 if pnl_usd > 0 else (0 if pnl_usd < 0 else 0.5)  # 1 for win, 0 for loss, 0.5 for breakeven
                }
                
                positions.append(position)
        
        return positions
    
    def run(self):
        """Main tracking loop"""
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}] ASTERDEX PROPER TRACKER")
        print(f"{'='*80}")
        print(f"[INFO] Extracting all closed positions from {len(ALL_SYMBOLS)} symbols")
        print(f"[INFO] Baseline: >= {self.start_date.isoformat()} GMT+7")
        
        all_positions = []
        new_positions = 0
        
        for i, symbol in enumerate(ALL_SYMBOLS, 1):
            symbol_asterdex = symbol.replace('-', '')
            print(f"\r[{i:2d}/{len(ALL_SYMBOLS)}] {symbol:12s}", end='', flush=True)
            
            orders = self.fetch_all_orders(symbol)
            positions = self.extract_positions_from_orders(symbol, orders)
            
            for pos in positions:
                pos_id = pos.get('position_id')
                if pos_id not in self.tracked_positions:
                    new_positions += 1
                    self.tracked_positions[pos_id] = pos
                
                all_positions.append(pos)
        
        print(f"\n\n[INFO] Total extracted: {len(all_positions)} closed positions")
        print(f"[INFO] New positions found: {new_positions}")
        
        # Save all to file
        with open(TRACKER_FILE, 'w') as f:
            for pos in sorted(all_positions, key=lambda x: x.get('opened_timestamp', '')):
                f.write(json.dumps(pos) + '\n')
        
        print(f"[OK] Saved to {TRACKER_FILE}")
        
        # Generate report
        self.generate_report(all_positions)
        
        return all_positions
    
    def generate_report(self, positions):
        """Generate performance report"""
        if not positions:
            print("[WARN] No positions to report")
            return
        
        wins = [p for p in positions if p.get('win') == 1]
        losses = [p for p in positions if p.get('win') == 0]
        breakeven = [p for p in positions if p.get('win') == 0.5]
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / len(positions) if positions else 0
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*80}")
        print(f"Total Positions:  {len(positions)}")
        print(f"  Wins:           {len(wins)} ({len(wins)/len(positions)*100:.1f}%)")
        print(f"  Losses:         {len(losses)} ({len(losses)/len(positions)*100:.1f}%)")
        print(f"  Breakeven:      {len(breakeven)}")
        print(f"\nP&L Summary:")
        print(f"  Total P&L:      ${total_pnl:.2f} USD")
        print(f"  Avg P&L:        ${avg_pnl:.2f} USD per trade")
        
        # Top performers
        sorted_by_pnl = sorted(positions, key=lambda x: x.get('pnl_usd', 0), reverse=True)
        print(f"\nTop 5 Winners:")
        for i, pos in enumerate(sorted_by_pnl[:5], 1):
            print(f"  {i}. {pos.get('symbol'):10s} {pos.get('side'):5s} ${pos.get('pnl_usd'):7.2f} ({pos.get('pnl_pct'):+6.2f}%)")
        
        print(f"\nTop 5 Losers:")
        for i, pos in enumerate(sorted_by_pnl[-5:], 1):
            print(f"  {i}. {pos.get('symbol'):10s} {pos.get('side'):5s} ${pos.get('pnl_usd'):7.2f} ({pos.get('pnl_pct'):+6.2f}%)")
        
        # By symbol
        print(f"\nBy Symbol:")
        by_symbol = defaultdict(list)
        for pos in positions:
            by_symbol[pos.get('symbol')].append(pos)
        
        for symbol in sorted(by_symbol.keys()):
            sym_positions = by_symbol[symbol]
            sym_pnl = sum(p.get('pnl_usd', 0) for p in sym_positions)
            sym_wins = len([p for p in sym_positions if p.get('win') == 1])
            print(f"  {symbol:10s} {len(sym_positions):2d} trades | ${sym_pnl:7.2f} | WR: {sym_wins/len(sym_positions)*100:5.1f}%")
        
        print(f"{'='*80}\n")


def main():
    tracker = AsterdexProperTracker()
    positions = tracker.run()
    

if __name__ == '__main__':
    main()

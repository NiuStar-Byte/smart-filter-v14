#!/usr/bin/env python3
"""
ASTERDEX FINAL TRACKER - Complete position extraction including reversals
Handles BOTH traditional closes (reduceOnly=True) AND reversals (opposite direction)
Extracts ALL 38+ closed positions from Jun 7+ entries
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


class AsterdexFinalTracker:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.baseline_date = datetime(2026, 6, 7, 0, 0, 0)
        self.baseline_ts = int(self.baseline_date.timestamp() * 1000)
    
    def fetch_all_orders(self, symbol):
        """Fetch ALL orders for symbol from Jun 1 onwards"""
        symbol_api = symbol.replace('-', '')
        
        # Start from Jun 1 to capture all orders
        start_ts = int(datetime(2026, 6, 1, 0, 0, 0).timestamp() * 1000)
        
        params = {
            'symbol': symbol_api,
            'limit': 1000,
            'startTime': start_ts
        }
        
        try:
            signed_params = self.auth.sign_request_v3(base_params=params)
            response = requests.get(
                f"{API_BASE}/fapi/v3/allOrders",
                params=signed_params,
                timeout=10
            )
            
            if response.status_code == 200:
                # Filter to FILLED orders from Jun 7 onwards (baseline)
                orders = response.json()
                return [o for o in orders if o.get('status') == 'FILLED' and o.get('time', 0) >= self.baseline_ts]
            
        except Exception as e:
            pass
        
        return []
    
    def extract_positions(self, symbol, all_orders):
        """
        Extract BOTH traditional closes AND reversals.
        
        Traditional: ENTRY (reduceOnly=False) → EXIT (reduceOnly=True)
        Reversal:    ENTRY (reduceOnly=False) → OPPOSITE DIRECTION (reduceOnly=False)
        """
        positions = []
        
        if not all_orders:
            return positions
        
        # Sort by time
        all_orders.sort(key=lambda x: x.get('time', 0))
        
        used_orders = set()
        
        for i, order in enumerate(all_orders):
            if i in used_orders or order.get('reduceOnly') == True:
                continue
            
            entry_side = order.get('side')
            entry_qty = float(order.get('executedQty', 0))
            entry_time = order.get('time', 0)
            entry_price = float(order.get('avgPrice', 0))
            entry_id = order.get('orderId')
            
            if entry_qty == 0:
                continue
            
            exit_order = None
            exit_idx = None
            
            # Method 1: Look for reduceOnly=True exit
            for j in range(i + 1, len(all_orders)):
                if j in used_orders:
                    continue
                
                candidate = all_orders[j]
                
                # Must be opposite side, reduce-only, matching quantity
                if (candidate.get('reduceOnly') == True and
                    candidate.get('side') != entry_side and
                    candidate.get('time', 0) > entry_time):
                    
                    exit_qty = float(candidate.get('executedQty', 0))
                    if entry_qty > 0 and abs(exit_qty - entry_qty) / entry_qty <= 0.05:
                        exit_order = candidate
                        exit_idx = j
                        break
            
            # Method 2: If no reduce-only exit found, look for reversal (opposite direction entry)
            if not exit_order:
                for j in range(i + 1, len(all_orders)):
                    if j in used_orders:
                        continue
                    
                    candidate = all_orders[j]
                    
                    # Must be opposite side, NOT reduce-only, matching quantity, close in time
                    if (candidate.get('reduceOnly') == False and
                        candidate.get('side') != entry_side and
                        candidate.get('time', 0) > entry_time):
                        
                        exit_qty = float(candidate.get('executedQty', 0))
                        if entry_qty > 0 and abs(exit_qty - entry_qty) / entry_qty <= 0.05:
                            exit_order = candidate
                            exit_idx = j
                            break
            
            # Create position if we found an exit
            if exit_order:
                used_orders.add(i)
                used_orders.add(exit_idx)
                
                exit_side = exit_order.get('side')
                exit_qty = float(exit_order.get('executedQty', 0))
                exit_time = exit_order.get('time', 0)
                exit_price = float(exit_order.get('avgPrice', 0))
                exit_id = exit_order.get('orderId')
                
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
                    'win': 1 if pnl_usd > 0 else (0 if pnl_usd < 0 else 0.5),
                    'close_type': 'REDUCE_ONLY' if exit_order.get('reduceOnly') == True else 'REVERSAL'
                }
                
                positions.append(position)
        
        return positions
    
    def run(self):
        """Execute complete tracker"""
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}] ASTERDEX FINAL TRACKER")
        print(f"{'='*80}")
        print(f"[INFO] Extracting all closed positions (Traditional + Reversal)")
        print(f"[INFO] Symbols: {len(ALL_SYMBOLS)}")
        print(f"[INFO] Baseline: >= {self.baseline_date.isoformat()} GMT+7\n")
        
        all_positions = []
        close_type_count = {'REDUCE_ONLY': 0, 'REVERSAL': 0}
        
        for i, symbol in enumerate(ALL_SYMBOLS, 1):
            print(f"[{i:2d}/{len(ALL_SYMBOLS)}] {symbol:12s}", end=' ')
            sys.stdout.flush()
            
            # Fetch through Jun 9 to capture all closes
            orders = self.fetch_all_orders(symbol)
            positions = self.extract_positions(symbol, orders)
            
            if positions:
                print(f"→ {len(positions):2d} position(s)", end='')
                for ct in ['REDUCE_ONLY', 'REVERSAL']:
                    count = len([p for p in positions if p.get('close_type') == ct])
                    if count > 0:
                        close_type_count[ct] += count
                        print(f" ({ct}: {count})", end='')
                print()
                all_positions.extend(positions)
            else:
                print(f"→ none")
        
        # Save to file
        with open(TRACKER_FILE, 'w') as f:
            for pos in sorted(all_positions, key=lambda x: x.get('opened_timestamp', '')):
                f.write(json.dumps(pos) + '\n')
        
        print(f"\n{'='*80}")
        print(f"[OK] Total positions: {len(all_positions)}")
        print(f"  Traditional closes: {close_type_count['REDUCE_ONLY']}")
        print(f"  Reversals: {close_type_count['REVERSAL']}")
        print(f"[OK] Saved to {TRACKER_FILE}")
        print(f"{'='*80}\n")
        
        self.report(all_positions)
        
        return all_positions
    
    def report(self, positions):
        """Performance report"""
        if not positions:
            return
        
        wins = len([p for p in positions if p.get('win') == 1])
        losses = len([p for p in positions if p.get('win') == 0])
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / len(positions) if positions else 0
        
        print(f"PERFORMANCE SUMMARY")
        print(f"  Total: {len(positions)} | Wins: {wins} ({wins/len(positions)*100:.1f}%) | Losses: {losses} ({losses/len(positions)*100:.1f}%)")
        print(f"  P&L: ${total_pnl:.2f} USD total | ${avg_pnl:.2f} USD avg\n")
        
        sorted_pnl = sorted(positions, key=lambda x: x.get('pnl_usd', 0), reverse=True)
        print(f"Top 5:")
        for pos in sorted_pnl[:5]:
            print(f"  {pos['symbol']:10s} {pos['side']:5s} ${pos['pnl_usd']:7.2f} ({pos['pnl_pct']:+6.2f}%)")
        
        print(f"\nBottom 5:")
        for pos in sorted_pnl[-5:]:
            print(f"  {pos['symbol']:10s} {pos['side']:5s} ${pos['pnl_usd']:7.2f} ({pos['pnl_pct']:+6.2f}%)")


if __name__ == '__main__':
    tracker = AsterdexFinalTracker()
    positions = tracker.run()

#!/usr/bin/env python3
"""
ASTERDEX LIVE MONITOR - Real-time position tracking in terminal
Fetches and displays new positions every 5 minutes
Shows live logs of position updates
"""

import json
import os
import sys
import time
from datetime import datetime
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
TRACKER_FILE = Path(__file__).parent / "ASTERDEX_POSITIONS_LIVE.jsonl"

SYMBOLS = [
    'APT-USDT', 'FUN-USDT', 'DOT-USDT', 'SEI-USDT', 'TURBO-USDT', 'KAS-USDT', 'WIF-USDT', 
    'HYPE-USDT', 'ARB-USDT', 'PORTAL-USDT', 'IP-USDT', 'DOGE-USDT', 'BNB-USDT', 
    'XPL-USDT', 'NEAR-USDT', 'PARTI-USDT', 'BLUR-USDT', 'AAVE-USDT', 'HBAR-USDT', 
    'ONDO-USDT', 'LA-USDT', 'DYDX-USDT', 'INJ-USDT', 'LDO-USDT', 'BERA-USDT', 
    'ENS-USDT', 'PUMP-USDT', 'XRP-USDT', 'SOL-USDT'
]


class AsterdexLiveMonitor:
    """Live position monitor with terminal updates every 5 minutes"""
    
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.positions = self.load_positions()
        self.refresh_interval = 300  # 5 minutes
        self.last_update = datetime.now()
    
    def log(self, msg, level="INFO"):
        """Log message with timestamp"""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')
        print(f"[{ts}] [{level}] {msg}")
    
    def load_positions(self):
        """Load existing positions from file"""
        positions = {}
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE) as f:
                for line in f:
                    if line.strip():
                        pos = json.loads(line.strip())
                        positions[pos.get('position_id')] = pos
        return positions
    
    def save_positions(self):
        """Save all positions to file"""
        with open(TRACKER_FILE, 'w') as f:
            for pos in sorted(self.positions.values(), key=lambda x: x.get('opened', '')):
                f.write(json.dumps(pos) + '\n')
    
    def fetch_new_positions(self):
        """Fetch new closed positions from API"""
        new_count = 0
        
        for symbol in SYMBOLS:
            symbol_api = symbol.replace('-', '')
            
            params = {
                'symbol': symbol_api,
                'limit': 1000,
                'startTime': int(datetime(2026, 6, 1).timestamp() * 1000)
            }
            
            try:
                signed_params = self.auth.sign_request_v3(base_params=params)
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
                    
                    # Find matching exit
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
                        
                        position_id = f"{symbol}_{entry_id}_{exit_id}"
                        
                        if position_id not in self.positions:
                            position = {
                                'position_id': position_id,
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
                            
                            self.positions[position_id] = position
                            new_count += 1
                            
                            # Log new position
                            self.log(f"✅ NEW: {symbol} {position['side']:5s} ${pnl_usd:7.2f} ({pnl_pct:+6.2f}%)", "NEW")
            
            except Exception as e:
                pass
        
        return new_count
    
    def calculate_duration(self, opened_str, closed_str):
        """Calculate duration in hours"""
        try:
            opened = datetime.fromisoformat(opened_str.replace('Z', '+00:00'))
            closed = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
            duration = (closed - opened).total_seconds() / 3600
            return round(duration, 2)
        except:
            return None
    
    def display_summary(self):
        """Display current summary statistics"""
        if not self.positions:
            return
        
        positions = list(self.positions.values())
        
        wins = [p for p in positions if p.get('pnl_usd', 0) > 0]
        losses = [p for p in positions if p.get('pnl_usd', 0) < 0]
        breakeven = [p for p in positions if p.get('pnl_usd', 0) == 0]
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / len(positions) if positions else 0
        
        win_loss_count = len(wins) + len(losses)
        wr_pct = (len(wins) / win_loss_count * 100) if win_loss_count > 0 else 0
        
        durations_win = []
        durations_loss = []
        
        for p in wins:
            dur = self.calculate_duration(p.get('opened', ''), p.get('closed', ''))
            if dur is not None:
                durations_win.append(dur)
        
        for p in losses:
            dur = self.calculate_duration(p.get('opened', ''), p.get('closed', ''))
            if dur is not None:
                durations_loss.append(dur)
        
        avg_tp_duration = sum(durations_win) / len(durations_win) if durations_win else 0
        avg_sl_duration = sum(durations_loss) / len(durations_loss) if durations_loss else 0
        
        rr_values = []
        for p in positions:
            if p.get('pnl_usd', 0) != 0:
                margin_per_trade = 2.0
                lev = int(p.get('leverage', '10x').rstrip('x'))
                notional = margin_per_trade * lev
                
                if notional > 0:
                    rr = abs(p.get('pnl_usd', 0)) / notional
                    rr_values.append(rr)
        
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
        
        # Clear screen and display header
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"\n{'='*120}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}] ASTERDEX LIVE POSITION MONITOR (Updates every 5 minutes)")
        print(f"{'='*120}\n")
        
        # Display positions table (last 10)
        recent = sorted(self.positions.values(), key=lambda x: x.get('opened', ''), reverse=True)[:10]
        
        print(f"{'#':3s} {'Symbol':12s} {'Side':5s} {'Entry':12s} {'Exit':12s} {'Qty':12s} {'P&L':10s} {'%':8s} {'Duration':10s}")
        print(f"{'-'*120}")
        
        for i, pos in enumerate(recent, 1):
            duration = self.calculate_duration(pos.get('opened', ''), pos.get('closed', ''))
            duration_str = f"{duration}h" if duration else "N/A"
            
            print(f"{i:3d} {pos['symbol']:12s} {pos['side']:5s} {pos['entry_price']:12.8f} {pos['exit_price']:12.8f} {pos['quantity']:12.2f} ${pos['pnl_usd']:8.2f} {pos['pnl_pct']:7.2f}% {duration_str:10s}")
        
        # Display summary
        print(f"\n{'='*120}")
        print(f"SUMMARY (Total: {len(positions)} positions)")
        print(f"{'='*120}")
        print(f"  Total Positions:     {len(positions):4d}  |  Wins: {len(wins):2d} ({len(wins)/len(positions)*100:5.1f}%)  |  Losses: {len(losses):2d} ({len(losses)/len(positions)*100:5.1f}%)  |  Breakeven: {len(breakeven)}")
        print(f"  Overall WR:          {len(wins)}/{win_loss_count} = {wr_pct:.1f}%")
        print(f"  Total P&L:           ${total_pnl:9.2f} USD")
        print(f"  Avg P&L/trade:       ${avg_pnl:9.2f} USD")
        print(f"  Avg Risk:Reward:     {avg_rr:8.2f}:1.0")
        print(f"  Avg TP Duration:     {avg_tp_duration:7.2f} hours")
        print(f"  Avg SL Duration:     {avg_sl_duration:7.2f} hours")
        print(f"{'='*120}\n")
    
    def run(self):
        """Main monitoring loop"""
        self.log("🚀 ASTERDEX LIVE MONITOR STARTED", "START")
        self.log(f"Refresh interval: {self.refresh_interval} seconds (5 minutes)", "INFO")
        self.log(f"Current positions: {len(self.positions)}", "INFO")
        self.log("Watching for new closed positions...", "INFO")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                self.log(f"--- CYCLE {cycle} ---", "CYCLE")
                
                # Fetch new positions
                new_count = self.fetch_new_positions()
                
                if new_count > 0:
                    self.log(f"Found {new_count} new position(s)", "UPDATE")
                    self.save_positions()
                else:
                    self.log(f"No new positions (Total: {len(self.positions)})", "CHECK")
                
                # Display summary
                self.display_summary()
                
                # Wait for next update
                self.log(f"Next update in {self.refresh_interval} seconds...", "WAIT")
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            self.log("Monitor stopped by user", "STOP")
            sys.exit(0)


if __name__ == '__main__':
    monitor = AsterdexLiveMonitor()
    monitor.run()

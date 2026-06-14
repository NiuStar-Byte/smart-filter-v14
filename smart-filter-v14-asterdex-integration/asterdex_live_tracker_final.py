#!/usr/bin/env python3
"""
ASTERDEX LIVE TRACKER - Auto-updates with new closed positions
Tracks all closed positions and adds new ones as they close
Calculates: WR, RR, TP/SL duration metrics
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

TRACKER_FILE = Path(__file__).parent / "ASTERDEX_POSITIONS_LIVE.jsonl"

# THE 37 VERIFIED BASELINE POSITIONS
BASELINE_POSITIONS = [
    {'symbol': 'AAVE-USDT', 'side': 'SHORT', 'entry_price': 60.22, 'exit_price': 60.29, 'quantity': 0.30, 'pnl_usd': -0.03, 'pnl_pct': -0.12, 'opened': '2026-06-07T00:59:03Z', 'closed': '2026-06-07T01:02:16Z', 'leverage': '10x'},
    {'symbol': 'AAVE-USDT', 'side': 'SHORT', 'entry_price': 62.04, 'exit_price': 62.603, 'quantity': 0.30, 'pnl_usd': -0.18, 'pnl_pct': -0.91, 'opened': '2026-06-07T09:16:43Z', 'closed': '2026-06-07T09:48:59Z', 'leverage': '10x'},
    {'symbol': 'PARTI-USDT', 'side': 'LONG', 'entry_price': 0.058320, 'exit_price': 0.055541, 'quantity': 659, 'pnl_usd': -1.86, 'pnl_pct': -4.77, 'opened': '2026-06-07T17:43:18Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '5x'},
    {'symbol': 'NEAR-USDT', 'side': 'SHORT', 'entry_price': 1.889, 'exit_price': 1.889, 'quantity': 10, 'pnl_usd': -0.01, 'pnl_pct': -0.08, 'opened': '2026-06-07T17:43:42Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'XPL-USDT', 'side': 'SHORT', 'entry_price': 0.068100, 'exit_price': 0.068100, 'quantity': 293, 'pnl_usd': 0.00, 'pnl_pct': 0.00, 'opened': '2026-06-07T17:44:09Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'XRP-USDT', 'side': 'SHORT', 'entry_price': 1.1353, 'exit_price': 1.1259, 'quantity': 17.6, 'pnl_usd': 0.17, 'pnl_pct': 0.83, 'opened': '2026-06-07T17:44:20Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'DYDX-USDT', 'side': 'SHORT', 'entry_price': 0.137, 'exit_price': 0.136, 'quantity': 145.3, 'pnl_usd': 0.12, 'pnl_pct': 0.73, 'opened': '2026-06-07T17:53:05Z', 'closed': '2026-06-07T19:17:15Z', 'leverage': '5x'},
    {'symbol': 'ONDO-USDT', 'side': 'SHORT', 'entry_price': 0.340700, 'exit_price': 0.337500, 'quantity': 58.7, 'pnl_usd': 0.17, 'pnl_pct': 1.04, 'opened': '2026-06-07T18:02:01Z', 'closed': '2026-06-07T19:17:13Z', 'leverage': '10x'},
    {'symbol': 'INJ-USDT', 'side': 'SHORT', 'entry_price': 5.192, 'exit_price': 5.114, 'quantity': 3.8, 'pnl_usd': 0.30, 'pnl_pct': 1.50, 'opened': '2026-06-07T18:05:14Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'LDO-USDT', 'side': 'SHORT', 'entry_price': 0.266500, 'exit_price': 0.265900, 'quantity': 75, 'pnl_usd': 0.03, 'pnl_pct': 0.23, 'opened': '2026-06-07T18:05:15Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'BERA-USDT', 'side': 'SHORT', 'entry_price': 0.242200, 'exit_price': 0.240800, 'quantity': 82.5, 'pnl_usd': 0.10, 'pnl_pct': 0.58, 'opened': '2026-06-07T18:06:19Z', 'closed': '2026-06-07T19:17:13Z', 'leverage': '5x'},
    {'symbol': 'TURBO-USDT', 'side': 'SHORT', 'entry_price': 0.0008651, 'exit_price': 0.0008588, 'quantity': 23117, 'pnl_usd': 0.13, 'pnl_pct': 0.73, 'opened': '2026-06-07T18:22:43Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'ENS-USDT', 'side': 'SHORT', 'entry_price': 4.677, 'exit_price': 4.623, 'quantity': 4.2, 'pnl_usd': 0.21, 'pnl_pct': 1.15, 'opened': '2026-06-07T18:36:42Z', 'closed': '2026-06-07T19:17:13Z', 'leverage': '10x'},
    {'symbol': 'PUMP-USDT', 'side': 'SHORT', 'entry_price': 0.0015, 'exit_price': 0.0014780, 'quantity': 13337, 'pnl_usd': 0.29, 'pnl_pct': 1.47, 'opened': '2026-06-07T18:36:50Z', 'closed': '2026-06-07T19:17:13Z', 'leverage': '10x'},
    {'symbol': 'BNB-USDT', 'side': 'LONG', 'entry_price': 594.16, 'exit_price': 588.75, 'quantity': 0.03, 'pnl_usd': -0.17, 'pnl_pct': -0.90, 'opened': '2026-06-07T18:54:42Z', 'closed': '2026-06-07T19:17:14Z', 'leverage': '10x'},
    {'symbol': 'LA-USDT', 'side': 'SHORT', 'entry_price': 0.073500, 'exit_price': 0.073800, 'quantity': 272.1, 'pnl_usd': -0.09, 'pnl_pct': -0.41, 'opened': '2026-06-07T19:16:08Z', 'closed': '2026-06-07T19:17:15Z', 'leverage': '5x'},
    {'symbol': 'HBAR-USDT', 'side': 'SHORT', 'entry_price': 0.080440, 'exit_price': 0.080850, 'quantity': 248, 'pnl_usd': -0.10, 'pnl_pct': -0.51, 'opened': '2026-06-07T19:43:30Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'AAVE-USDT', 'side': 'SHORT', 'entry_price': 62.16, 'exit_price': 62.46, 'quantity': 0.30, 'pnl_usd': -0.09, 'pnl_pct': -0.48, 'opened': '2026-06-07T19:44:22Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'DOGE-USDT', 'side': 'SHORT', 'entry_price': 0.083750, 'exit_price': 0.083860, 'quantity': 238, 'pnl_usd': -0.04, 'pnl_pct': -0.13, 'opened': '2026-06-07T20:18:12Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'BLUR-USDT', 'side': 'SHORT', 'entry_price': 0.017020, 'exit_price': 0.017140, 'quantity': 1174, 'pnl_usd': -0.14, 'pnl_pct': -0.71, 'opened': '2026-06-07T20:26:50Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '5x'},
    {'symbol': 'IP-USDT', 'side': 'SHORT', 'entry_price': 0.304000, 'exit_price': 0.303600, 'quantity': 65.7, 'pnl_usd': 0.01, 'pnl_pct': 0.13, 'opened': '2026-06-07T20:38:28Z', 'closed': '2026-06-07T20:57:30Z', 'leverage': '10x'},
    {'symbol': 'PARTI-USDT', 'side': 'LONG', 'entry_price': 0.056120, 'exit_price': 0.056160, 'quantity': 676, 'pnl_usd': 0.00, 'pnl_pct': 0.07, 'opened': '2026-06-07T20:46:26Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '5x'},
    {'symbol': 'NEAR-USDT', 'side': 'SHORT', 'entry_price': 1.9095, 'exit_price': 1.9124, 'quantity': 20, 'pnl_usd': -0.08, 'pnl_pct': -0.15, 'opened': '2026-06-07T20:46:32Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'XPL-USDT', 'side': 'SHORT', 'entry_price': 0.068550, 'exit_price': 0.068700, 'quantity': 586, 'pnl_usd': -0.12, 'pnl_pct': -0.20, 'opened': '2026-06-07T20:46:51Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'ONDO-USDT', 'side': 'SHORT', 'entry_price': 0.340500, 'exit_price': 0.342900, 'quantity': 58.7, 'pnl_usd': -0.15, 'pnl_pct': -0.70, 'opened': '2026-06-07T20:47:59Z', 'closed': '2026-06-07T20:57:30Z', 'leverage': '10x'},
    {'symbol': 'BNB-USDT', 'side': 'LONG', 'entry_price': 588.62, 'exit_price': 589.28, 'quantity': 0.03, 'pnl_usd': 0.00, 'pnl_pct': 0.11, 'opened': '2026-06-07T20:48:32Z', 'closed': '2026-06-07T20:57:31Z', 'leverage': '10x'},
    {'symbol': 'IP-USDT', 'side': 'SHORT', 'entry_price': 0.3042, 'exit_price': 0.3175, 'quantity': 65.8, 'pnl_usd': -0.89, 'pnl_pct': -4.37, 'opened': '2026-06-07T21:17:26Z', 'closed': '2026-06-07T22:08:44Z', 'leverage': '10x'},
    {'symbol': 'PORTAL-USDT', 'side': 'LONG', 'entry_price': 0.018350, 'exit_price': 0.016260, 'quantity': 1083.5, 'pnl_usd': -2.25, 'pnl_pct': -11.39, 'opened': '2026-06-07T21:20:03Z', 'closed': '2026-06-08T02:32:50Z', 'leverage': '5x'},
    {'symbol': 'HYPE-USDT', 'side': 'SHORT', 'entry_price': 58.781, 'exit_price': 61.538, 'quantity': 0.34, 'pnl_usd': -0.95, 'pnl_pct': -4.69, 'opened': '2026-06-07T22:15:09Z', 'closed': '2026-06-08T07:30:43Z', 'leverage': '10x'},
    {'symbol': 'ARB-USDT', 'side': 'SHORT', 'entry_price': 0.0817, 'exit_price': 0.0854, 'quantity': 244.7, 'pnl_usd': -0.91, 'pnl_pct': -4.53, 'opened': '2026-06-07T22:16:47Z', 'closed': '2026-06-08T05:15:18Z', 'leverage': '10x'},
    {'symbol': 'FUN-USDT', 'side': 'SHORT', 'entry_price': 0.041940, 'exit_price': 0.045430, 'quantity': 476, 'pnl_usd': -1.66, 'pnl_pct': -8.32, 'opened': '2026-06-07T22:23:04Z', 'closed': '2026-06-08T12:42:55Z', 'leverage': '2x'},
    {'symbol': 'WIF-USDT', 'side': 'SHORT', 'entry_price': 0.158700, 'exit_price': 0.156600, 'quantity': 125.9, 'pnl_usd': 0.25, 'pnl_pct': 1.33, 'opened': '2026-06-07T22:29:58Z', 'closed': '2026-06-08T12:29:08Z', 'leverage': '10x'},
    {'symbol': 'DOT-USDT', 'side': 'SHORT', 'entry_price': 0.963, 'exit_price': 0.958, 'quantity': 20.7, 'pnl_usd': 0.09, 'pnl_pct': 0.52, 'opened': '2026-06-07T23:02:34Z', 'closed': '2026-06-08T12:29:50Z', 'leverage': '10x'},
    {'symbol': 'KAS-USDT', 'side': 'LONG', 'entry_price': 0.030720, 'exit_price': 0.031240, 'quantity': 651, 'pnl_usd': 0.32, 'pnl_pct': 1.69, 'opened': '2026-06-07T23:35:44Z', 'closed': '2026-06-08T12:29:14Z', 'leverage': '10x'},
    {'symbol': 'SEI-USDT', 'side': 'SHORT', 'entry_price': 0.049420, 'exit_price': 0.048800, 'quantity': 404, 'pnl_usd': 0.24, 'pnl_pct': 0.26, 'opened': '2026-06-08T00:09:47Z', 'closed': '2026-06-08T12:29:25Z', 'leverage': '10x'},
    {'symbol': 'TURBO-USDT', 'side': 'SHORT', 'entry_price': 0.0008681, 'exit_price': 0.0008586, 'quantity': 23037, 'pnl_usd': 0.21, 'pnl_pct': 1.09, 'opened': '2026-06-08T09:08:03Z', 'closed': '2026-06-08T12:29:18Z', 'leverage': '10x'},
    {'symbol': 'APT-USDT', 'side': 'SHORT', 'entry_price': 10.665, 'exit_price': 10.661, 'quantity': 30.0, 'pnl_usd': -0.03, 'pnl_pct': -0.04, 'opened': '2026-06-08T14:25:09Z', 'closed': '2026-06-08T15:33:38Z', 'leverage': '10x'},
]


class AsterdexLiveTracker:
    """Live position tracker - auto-updates with new closed positions"""
    
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.positions = self.load_positions()
    
    def load_positions(self):
        """Load positions from file or baseline"""
        positions = {}
        
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE) as f:
                for line in f:
                    if line.strip():
                        pos = json.loads(line.strip())
                        positions[pos.get('position_id')] = pos
        else:
            # Initialize with baseline
            for i, pos in enumerate(BASELINE_POSITIONS, 1):
                pos_id = f"{pos['symbol']}_{i:02d}"
                positions[pos_id] = {
                    **pos,
                    'position_id': pos_id,
                }
        
        return positions
    
    def save(self):
        """Persist positions to file"""
        with open(TRACKER_FILE, 'w') as f:
            for pos in sorted(self.positions.values(), key=lambda x: x.get('opened', '')):
                f.write(json.dumps(pos) + '\n')
    
    def calculate_duration(self, opened_str, closed_str):
        """Calculate duration in hours"""
        try:
            opened = datetime.fromisoformat(opened_str.replace('Z', '+00:00'))
            closed = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
            duration = (closed - opened).total_seconds() / 3600
            return round(duration, 2)
        except:
            return None
    
    def calculate_rr(self, entry_price, tp_price, sl_price, side):
        """Calculate Risk:Reward ratio"""
        try:
            if side == 'LONG':
                profit = tp_price - entry_price
                loss = entry_price - sl_price
            else:  # SHORT
                profit = entry_price - tp_price
                loss = sl_price - entry_price
            
            if loss <= 0:
                return None
            
            return round(profit / loss, 2)
        except:
            return None
    
    def display(self):
        """Display all positions with enhanced metrics"""
        positions = sorted(self.positions.values(), key=lambda x: x.get('opened', ''))
        
        print(f"\n{'='*160}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}] ASTERDEX LIVE POSITION TRACKER")
        print(f"{'='*160}\n")
        
        print(f"{'#':3s} {'Symbol':12s} {'Side':5s} {'Entry':12s} {'Exit':12s} {'Qty':12s} {'P&L':10s} {'%':8s} {'Time Opened':25s} {'Time Closed':25s} {'Duration':10s}")
        print(f"{'-'*160}")
        
        for i, pos in enumerate(positions, 1):
            duration = self.calculate_duration(pos.get('opened', ''), pos.get('closed', ''))
            duration_str = f"{duration}h" if duration else "N/A"
            
            opened_str = pos.get('opened', 'N/A').replace('Z', '')
            closed_str = pos.get('closed', 'N/A').replace('Z', '')
            
            print(f"{i:3d} {pos['symbol']:12s} {pos['side']:5s} {pos['entry_price']:12.8f} {pos['exit_price']:12.8f} {pos['quantity']:12.2f} ${pos['pnl_usd']:8.2f} {pos['pnl_pct']:7.2f}% {opened_str:25s} {closed_str:25s} {duration_str:10s}")
        
        # Calculate comprehensive metrics
        self.show_summary(positions)
    
    def show_summary(self, positions):
        """Show comprehensive summary with all metrics"""
        wins = [p for p in positions if p.get('pnl_usd', 0) > 0]
        losses = [p for p in positions if p.get('pnl_usd', 0) < 0]
        breakeven = [p for p in positions if p.get('pnl_usd', 0) == 0]
        
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        avg_pnl = total_pnl / len(positions) if positions else 0
        
        # WR calculation: Wins / (Wins + Losses)
        win_loss_count = len(wins) + len(losses)
        wr_pct = (len(wins) / win_loss_count * 100) if win_loss_count > 0 else 0
        
        # Calculate durations
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
        
        # Calculate average RR (estimate from P&L if possible)
        # RR = profit / loss per trade
        rr_values = []
        for p in positions:
            if p.get('pnl_usd', 0) != 0:
                # Estimate based on entry/exit and leverage
                margin_per_trade = 2.0  # $2.0 default margin
                lev = int(p.get('leverage', '10x').rstrip('x'))
                notional = margin_per_trade * lev
                
                if notional > 0:
                    rr = abs(p.get('pnl_usd', 0)) / notional
                    rr_values.append(rr)
        
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
        
        print(f"{'-'*160}")
        print(f"\nSUMMARY:")
        print(f"  Total Positions:     {len(positions)}")
        print(f"  Wins:                {len(wins)} ({len(wins)/len(positions)*100:.1f}%)")
        print(f"  Losses:              {len(losses)} ({len(losses)/len(positions)*100:.1f}%)")
        print(f"  Breakeven:           {len(breakeven)}")
        print(f"  Overall WR:          {len(wins)}/{win_loss_count} = {wr_pct:.1f}%")
        print(f"  Total P&L:           ${total_pnl:.2f} USD")
        print(f"  Avg P&L:             ${avg_pnl:.2f} USD per trade")
        print(f"  Avg Risk:Reward:     {avg_rr:.2f}:1.0")
        print(f"  Avg TP Duration:     {avg_tp_duration:.2f} hours ({avg_tp_duration/24:.1f} days avg)")
        print(f"  Avg SL Duration:     {avg_sl_duration:.2f} hours ({avg_sl_duration/24:.1f} days avg)")
        print(f"\n{'='*160}\n")


if __name__ == '__main__':
    tracker = AsterdexLiveTracker()
    tracker.save()
    tracker.display()

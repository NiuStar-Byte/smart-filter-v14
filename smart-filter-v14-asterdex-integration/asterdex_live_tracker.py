#!/usr/bin/env python3
"""
ASTERDEX Live Position Tracker
Shows all closed positions from Jun 7+ GMT+7 with live updates
Run: python3 asterdex_live_tracker.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

def load_trades():
    """Load all correlated trades from Jun 7+ GMT+7."""
    trades = []
    
    corr_file = Path(__file__).parent / "ASTERDEX_PERFORMANCE_CORRELATED.jsonl"
    if not corr_file.exists():
        return []
    
    with open(corr_file) as f:
        for line in f:
            if line.strip():
                try:
                    trade = json.loads(line)
                    
                    # Filter to Jun 7+ GMT+7
                    try:
                        dt = datetime.fromisoformat(trade.get("posted_timestamp", "").replace("Z", "+00:00")).replace(tzinfo=None)
                        if dt >= datetime(2026, 6, 7, 0, 0, 1):
                            trades.append(trade)
                    except:
                        pass
                except:
                    pass
    
    return sorted(trades, key=lambda x: x.get("posted_timestamp", ""))

def display_tracking():
    """Display live tracking table."""
    trades = load_trades()
    
    # Header
    print(f"\n{'='*160}")
    print(f"ASTERDEX LIVE POSITION TRACKING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"{'='*160}\n")
    
    print(f"{'#':<3} {'SYMBOL':<12} {'SIDE':<6} {'ENTRY':<12} {'EXIT':<12} {'QTY':<10} {'P&L $':<10} {'WIN':<5} {'OPENED (GMT+7)':<21} {'CLOSED (GMT+7)':<21}")
    print(f"{'-'*160}\n")
    
    if not trades:
        print("No closed positions yet. Waiting for trades from Jun 7+ GMT+7...\n")
    else:
        for i, trade in enumerate(trades, 1):
            sym = trade.get("symbol", "N/A")
            side = trade.get("side", "N/A")
            entry = float(trade.get("entry_price", 0))
            exit_p = float(trade.get("exit_price", 0))
            qty = float(trade.get("quantity", 0))
            pnl = float(trade.get("realized_pnl_usd", 0))
            win = "✅" if trade.get("win") else "❌"
            
            try:
                opened = datetime.fromisoformat(trade.get("posted_timestamp", "").replace("Z", "+00:00"))
                opened_str = opened.strftime("%Y-%m-%d %H:%M:%S")
            except:
                opened_str = "N/A"
            
            try:
                closed = datetime.fromtimestamp(float(trade.get("exit_timestamp", 0)) / 1000)
                closed_str = closed.strftime("%Y-%m-%d %H:%M:%S")
            except:
                closed_str = "N/A"
            
            print(f"{i:<3} {sym:<12} {side:<6} {entry:<12.8f} {exit_p:<12.8f} {qty:<10.2f} {pnl:<10.2f} {win:<5} {opened_str:<21} {closed_str:<21}")
    
    print(f"\n{'-'*160}\n")
    
    # Summary
    if trades:
        wins = sum(1 for t in trades if t.get("win"))
        losses = len(trades) - wins
        total_pnl = sum(t.get("realized_pnl_usd", 0) for t in trades)
        wr = wins / len(trades) * 100
        avg_pnl = total_pnl / len(trades)
        
        print(f"📊 SUMMARY: {len(trades)} positions | ✅ {wins} WINS ({wr:.1f}%) | ❌ {losses} LOSSES | 💰 P&L: ${total_pnl:+.2f} USD | Avg: ${avg_pnl:+.3f}/trade\n")
    else:
        print(f"⏳ No positions tracked yet\n")
    
    print(f"{'='*160}\n")
    print("📋 VERIFICATION: Compare each row against Asterdex UI Trade History")
    print("🔄 Auto-refresh every 10 seconds (Press Ctrl+C to stop)\n")

if __name__ == '__main__':
    try:
        while True:
            display_tracking()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n✋ Tracking stopped.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

#!/usr/bin/env python3
"""
FILTER EFFECTIVENESS: BASELINE vs LIVE PERFORMANCE
Compares:
1. BASELINE SIGNALS (Before 2026-03-05) - Win Rate per filter
2. LIVE SIGNALS (From 2026-03-05+) - Fire count per filter

Shows which filters improved vs degraded after enhancement.
"""

import json
from datetime import datetime
from collections import defaultdict

SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"
ENHANCEMENT_START = "2026-03-05T00:00:00"

class FilterEffectivenessComparison:
    def __init__(self):
        self.baseline_signals = []  # Signals before enhancement
        self.live_signals = []      # Signals after enhancement
        
    def load_and_split_signals(self):
        """Load signals and split by enhancement date."""
        cutoff_dt = datetime.fromisoformat(ENHANCEMENT_START.replace('Z', '+00:00'))
        
        with open(SIGNALS_MASTER_PATH, 'r') as f:
            for line in f:
                try:
                    s = json.loads(line.strip())
                    if not s.get('sent_time_utc'):
                        continue
                    
                    signal_time = datetime.fromisoformat(s['sent_time_utc'].replace('Z', '+00:00'))
                    if signal_time < cutoff_dt:
                        self.baseline_signals.append(s)
                    else:
                        self.live_signals.append(s)
                except:
                    pass
        
        return len(self.baseline_signals), len(self.live_signals)
    
    def calculate_metrics(self):
        """Calculate effectiveness metrics for both periods."""
        
        # BASELINE metrics
        baseline_closed = sum(1 for s in self.baseline_signals if s.get('closed_at'))
        baseline_wins = sum(1 for s in self.baseline_signals if s.get('status') == 'TP_HIT')
        baseline_losses = sum(1 for s in self.baseline_signals if s.get('status') == 'SL_HIT')
        baseline_wr = (baseline_wins / baseline_closed * 100) if baseline_closed > 0 else 0
        baseline_pnl = sum(s.get('pnl_usd') or 0 for s in self.baseline_signals)
        
        # LIVE metrics
        live_closed = sum(1 for s in self.live_signals if s.get('closed_at'))
        live_wins = sum(1 for s in self.live_signals if s.get('status') == 'TP_HIT')
        live_losses = sum(1 for s in self.live_signals if s.get('status') == 'SL_HIT')
        live_wr = (live_wins / live_closed * 100) if live_closed > 0 else 0
        live_pnl = sum(s.get('pnl_usd') or 0 for s in self.live_signals)
        
        return {
            "baseline": {
                "total": len(self.baseline_signals),
                "closed": baseline_closed,
                "wins": baseline_wins,
                "losses": baseline_losses,
                "wr": baseline_wr,
                "pnl": baseline_pnl,
                "avg_trade_pnl": baseline_pnl / baseline_closed if baseline_closed > 0 else 0
            },
            "live": {
                "total": len(self.live_signals),
                "closed": live_closed,
                "wins": live_wins,
                "losses": live_losses,
                "wr": live_wr,
                "pnl": live_pnl,
                "avg_trade_pnl": live_pnl / live_closed if live_closed > 0 else 0
            }
        }
    
    def print_comparison(self, metrics):
        """Print detailed comparison report."""
        
        baseline = metrics["baseline"]
        live = metrics["live"]
        
        print("\n" + "="*140)
        print("FILTER EFFECTIVENESS: BASELINE vs LIVE PERFORMANCE")
        print("="*140)
        
        print("\n📊 BASELINE (Immutable Reference - BEFORE 2026-03-05)")
        print("-"*140)
        print(f"   Total signals fired: {baseline['total']}")
        print(f"   Closed: {baseline['closed']} | Wins: {baseline['wins']} | Losses: {baseline['losses']}")
        print(f"   Win Rate: {baseline['wr']:.1f}%")
        print(f"   Total P&L: ${baseline['pnl']:,.2f}")
        print(f"   Avg P&L per trade: ${baseline['avg_trade_pnl']:,.2f}")
        
        print("\n📊 LIVE (Post-Enhancement - FROM 2026-03-05 onwards)")
        print("-"*140)
        print(f"   Total signals fired: {live['total']}")
        print(f"   Closed: {live['closed']} | Wins: {live['wins']} | Losses: {live['losses']}")
        print(f"   Win Rate: {live['wr']:.1f}%")
        print(f"   Total P&L: ${live['pnl']:,.2f}")
        print(f"   Avg P&L per trade: ${live['avg_trade_pnl']:,.2f}")
        
        print("\n📈 COMPARISON & IMPACT")
        print("-"*140)
        
        signal_delta = live['total'] - baseline['total']
        signal_delta_pct = (signal_delta / baseline['total'] * 100) if baseline['total'] > 0 else 0
        
        wr_delta = live['wr'] - baseline['wr']
        
        pnl_delta = live['pnl'] - baseline['pnl']
        pnl_improvement = (pnl_delta / abs(baseline['pnl']) * 100) if baseline['pnl'] != 0 else 0
        
        print(f"{'Metric':<30} {'Baseline':<20} {'Live':<20} {'Delta':<20}")
        print("-"*140)
        print(f"{'Signals Fired':<30} {baseline['total']:<20} {live['total']:<20} {signal_delta:+.0f} ({signal_delta_pct:+.1f}%)")
        print(f"{'Closed Signals':<30} {baseline['closed']:<20} {live['closed']:<20} {live['closed']-baseline['closed']:+.0f}")
        print(f"{'Win Rate':<30} {baseline['wr']:<19.1f}% {live['wr']:<19.1f}% {wr_delta:+.1f}pp")
        print(f"{'Total P&L':<30} ${baseline['pnl']:<19,.2f} ${live['pnl']:<19,.2f} ${pnl_delta:+,.2f}")
        print(f"{'Avg P&L per Trade':<30} ${baseline['avg_trade_pnl']:<19,.2f} ${live['avg_trade_pnl']:<19,.2f} ${live['avg_trade_pnl']-baseline['avg_trade_pnl']:+,.2f}")
        
        print("\n💡 INTERPRETATION")
        print("-"*140)
        
        if signal_delta < 0:
            print(f"   • Fewer signals fired: {signal_delta_pct:.1f}% reduction")
            print(f"     → Enhancements are FILTERING (higher selectivity)")
        else:
            print(f"   • More signals fired: {signal_delta_pct:.1f}% increase")
            print(f"     → Enhancements are OPENING UP (lower gates)")
        
        if wr_delta > 2:
            print(f"   • Win rate IMPROVED: +{wr_delta:.1f}pp")
            print(f"     → Better signal quality ✅")
        elif wr_delta < -2:
            print(f"   • Win rate DEGRADED: {wr_delta:.1f}pp")
            print(f"     → Signal quality declined ❌")
        else:
            print(f"   • Win rate STABLE: {wr_delta:+.1f}pp")
            print(f"     → Enhancements not affecting quality")
        
        if pnl_delta > 0:
            print(f"   • P&L IMPROVED: +${pnl_delta:,.2f} ({pnl_improvement:+.1f}%)")
            print(f"     → Enhancements profitable ✅")
        else:
            print(f"   • P&L DEGRADED: ${pnl_delta:,.2f} ({pnl_improvement:+.1f}%)")
            print(f"     → Enhancements losing money ❌")
        
        print("\n" + "="*140 + "\n")

if __name__ == "__main__":
    comparator = FilterEffectivenessComparison()
    baseline_count, live_count = comparator.load_and_split_signals()
    print(f"[INFO] Loaded {baseline_count} baseline + {live_count} live signals")
    
    metrics = comparator.calculate_metrics()
    comparator.print_comparison(metrics)

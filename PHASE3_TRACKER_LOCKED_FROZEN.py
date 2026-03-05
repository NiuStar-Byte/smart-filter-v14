#!/usr/bin/env python3
"""
🔒 PHASE 3B TRACKER - Reversal Quality Gates (Live Monitoring)

Phase 3B: 4-Gate Reversal Quality Check
├─ RQ1: Detector Consensus
├─ RQ2: Momentum Alignment  
├─ RQ3: Trend Strength
└─ RQ4: Direction-Regime Match

Status: ✅ LIVE & COLLECTING (Parallel with Phase 2-FIXED)
Deployment: 2026-03-03 19:31 GMT+7
Timeline: Monitor 7 days (Mar 3-10) → Decision Mar 10 14:30 GMT+7
"""

import json
from collections import defaultdict
from datetime import datetime, timezone

# 🔒 FOUNDATION BASELINE LOCKED (2026-03-04 01:10 GMT+7)
FOUNDATION = {
    "total_signals": 853,
    "closed_trades": 830,
    "win_rate": 25.7,
    "long_wr": 29.6,
    "short_wr": 46.2,
    "pnl": -5498.59
}

PHASE3B_START = datetime(2026, 3, 3, 19, 31, 0, tzinfo=timezone.utc)  # Mar 3 02:31 GMT+7 = 19:31 UTC (when Phase 3B deployed)
SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"

class Phase3BTracker:
    def __init__(self):
        self.phase3b_signals = []
        self.load_signals()
    
    def load_signals(self):
        """Load Phase 3B signals (from 19:31 UTC Mar 3 onwards) - REVERSAL routes only"""
        try:
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line.strip())
                        if not sig:
                            continue
                        
                        # Skip OPEN signals
                        if sig.get('status') == 'OPEN':
                            continue
                        
                        fired_str = sig.get('fired_time_utc') or sig.get('fired_at', '')
                        if not fired_str:
                            continue
                        
                        # Parse as naive datetime and make UTC
                        fired = datetime.fromisoformat(fired_str.split('+')[0]).replace(tzinfo=timezone.utc)
                        
                        # Phase 3B: From 19:31 UTC Mar 3 onwards
                        if fired >= PHASE3B_START:
                            # Focus on REVERSAL signals (Phase 3B specialty)
                            if sig.get('route') == 'REVERSAL':
                                self.phase3b_signals.append(sig)
                    except Exception as e:
                        continue
        except FileNotFoundError:
            print(f"❌ {SIGNALS_FILE} not found")
    
    def calculate_metrics(self):
        """Calculate Phase 3B metrics"""
        if not self.phase3b_signals:
            return None
        
        # Filter: closed trades
        closed = []
        for s in self.phase3b_signals:
            if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                # Exclude stale timeouts
                flag = s.get('data_quality_flag', '')
                if flag and 'STALE_TIMEOUT' in flag:
                    continue
                closed.append(s)
        
        if not closed:
            return None
        
        # Win rate calculation
        tp_hits = len([s for s in closed if s.get('status') == 'TP_HIT'])
        sl_hits = len([s for s in closed if s.get('status') == 'SL_HIT'])
        timeout_trades = [s for s in closed if s.get('status') == 'TIMEOUT']
        timeout_wins = len([s for s in timeout_trades if float(s.get('pnl_usd', 0)) > 0])
        
        win_rate = ((tp_hits + timeout_wins) / len(closed) * 100) if closed else 0
        
        # P&L
        total_pnl = sum(float(s.get('pnl_usd', 0) or 0) for s in closed)
        
        # Direction breakdown
        long_trades = [s for s in closed if s.get('signal_type') == 'LONG']
        long_wins = len([s for s in long_trades if s.get('status') == 'TP_HIT'])
        long_timeout_wins = len([s for s in long_trades if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0])
        long_wr = ((long_wins + long_timeout_wins) / len(long_trades) * 100) if long_trades else 0
        
        short_trades = [s for s in closed if s.get('signal_type') == 'SHORT']
        short_wins = len([s for s in short_trades if s.get('status') == 'TP_HIT'])
        short_timeout_wins = len([s for s in short_trades if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0])
        short_wr = ((short_wins + short_timeout_wins) / len(short_trades) * 100) if short_trades else 0
        
        # Regime breakdown
        regimes = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
        for s in closed:
            regime = s.get('regime', 'UNKNOWN')
            regimes[regime]['trades'] += 1
            regimes[regime]['pnl'] += float(s.get('pnl_usd', 0) or 0)
            
            if s.get('status') == 'TP_HIT':
                regimes[regime]['wins'] += 1
            elif s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0:
                regimes[regime]['wins'] += 1
        
        return {
            'total_signals': len(self.phase3b_signals),
            'closed_trades': len(closed),
            'win_rate': win_rate,
            'long_wr': long_wr,
            'short_wr': short_wr,
            'total_pnl': total_pnl,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'timeout_wins': timeout_wins,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'regimes': dict(regimes),
        }
    
    def print_report(self):
        """Generate Phase 3B tracking report"""
        print("\n" + "="*110)
        print("🔒 PHASE 3B TRACKER - REVERSAL QUALITY GATES (Live Monitoring)")
        print("="*110)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        print(f"📌 Deployment: 2026-03-03 19:31 GMT+7 (parallel with Phase 2-FIXED)")
        print(f"📌 Focus: REVERSAL signals only (4-gate quality validation)")
        print("="*110)
        print()
        
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("⏳ PHASE 3B: No REVERSAL signals collected yet")
            print(f"   Total Phase 3B REVERSAL signals: {len(self.phase3b_signals)}")
            print("   Closed trades: 0")
            print()
            print("="*110)
            return
        
        # Comparison with Foundation
        print("METRIC                  │ FOUNDATION (Baseline) │ PHASE 3B (REVERSAL)  │ STATUS")
        print("─"*110)
        
        # Total signals
        delta_signals = metrics['total_signals'] - FOUNDATION['total_signals']
        print(f"Total Signals           │        {FOUNDATION['total_signals']:6d}          │      {metrics['total_signals']:6d}       │ 📊 {delta_signals:+d}")
        
        # Closed trades
        delta_closed = metrics['closed_trades'] - FOUNDATION['closed_trades']
        print(f"Closed Trades           │        {FOUNDATION['closed_trades']:6d}          │      {metrics['closed_trades']:6d}       │ ✓ {delta_closed:+d}")
        
        print("─"*110)
        
        # Win Rate
        delta_wr = metrics['win_rate'] - FOUNDATION['win_rate']
        if delta_wr > 2:
            status = "✅ IMPROVING"
        elif delta_wr > 0:
            status = "⚠️ MARGINAL"
        else:
            status = "❌ DECLINING"
        
        print(f"Overall Win Rate        │        {FOUNDATION['win_rate']:6.1f}%         │      {metrics['win_rate']:6.1f}%      │ {status}")
        
        # LONG/SHORT
        delta_long = metrics['long_wr'] - FOUNDATION['long_wr']
        delta_short = metrics['short_wr'] - FOUNDATION['short_wr']
        
        print(f"  ├─ LONG WR            │        {FOUNDATION['long_wr']:6.1f}%         │      {metrics['long_wr']:6.1f}%      │ {'✅' if delta_long > 0 else '⚠️'} {delta_long:+.1f}%")
        print(f"  └─ SHORT WR           │        {FOUNDATION['short_wr']:6.1f}%         │      {metrics['short_wr']:6.1f}%      │ {'✅' if delta_short > 0 else '⚠️'} {delta_short:+.1f}%")
        
        print("─"*110)
        
        # P&L
        delta_pnl = metrics['total_pnl'] - FOUNDATION['pnl']
        if delta_pnl > 500:
            status = "✅ PROFITABLE"
        elif delta_pnl > 0:
            status = "⚠️ IMPROVING"
        else:
            status = "❌ WORSE"
        
        print(f"Total P&L               │      ${FOUNDATION['pnl']:9.2f}      │    ${metrics['total_pnl']:9.2f}    │ {status}")
        
        print()
        print("="*110)
        print("📊 DETAILED BREAKDOWN")
        print("="*110)
        print()
        
        # Direction summary
        print(f"Direction Breakdown:")
        print(f"  • LONG:  {metrics['long_trades']:4d} closed trades | {metrics['long_wr']:5.1f}% WR")
        print(f"  • SHORT: {metrics['short_trades']:4d} closed trades | {metrics['short_wr']:5.1f}% WR")
        print()
        
        # Regime breakdown
        print(f"Regime Performance (REVERSAL signals):")
        print(f"  Regime      │ Closed │ Wins │ WR     │ P&L")
        print(f"  ────────────┼────────┼──────┼────────┼─────────")
        for regime, stats in sorted(metrics['regimes'].items()):
            trades = stats['trades']
            wins = stats['wins']
            wr = (wins / trades * 100) if trades > 0 else 0
            pnl = stats['pnl']
            print(f"  {regime:<11} │ {trades:6d} │ {wins:4d} │ {wr:6.1f}% │ ${pnl:8.2f}")
        
        print()
        print("="*110)
        print("🎯 PHASE 3B PERFORMANCE SUMMARY")
        print("="*110)
        print()
        
        if metrics['closed_trades'] >= 20:
            if metrics['win_rate'] >= FOUNDATION['win_rate']:
                print(f"✅ PHASE 3B IMPROVING: WR {metrics['win_rate']:.1f}% ≥ Baseline {FOUNDATION['win_rate']}%")
                improvement = metrics['win_rate'] - FOUNDATION['win_rate']
                print(f"   Impact: +{improvement:.1f}% WR improvement on REVERSAL signals")
                print(f"   Status: Phase 3B quality gates are working!")
            else:
                print(f"⚠️ PHASE 3B BELOW BASELINE: WR {metrics['win_rate']:.1f}% < {FOUNDATION['win_rate']}%")
                decline = FOUNDATION['win_rate'] - metrics['win_rate']
                print(f"   Impact: -{decline:.1f}% WR decline on REVERSAL signals")
                print(f"   Status: Monitor for improvement or consider refinement")
        else:
            trades_needed = 20 - metrics['closed_trades']
            print(f"⏳ PHASE 3B IN PROGRESS")
            print(f"   Closed trades: {metrics['closed_trades']}/20 (need {trades_needed} more)")
            print(f"   Current WR: {metrics['win_rate']:.1f}% (preliminary)")
            print(f"   Continue monitoring daily...")
        
        print()
        print(f"📌 Decision Threshold: Phase 3B WR ≥ {FOUNDATION['win_rate']}% by Mar 10 14:30 GMT+7")
        print()
        print("="*110 + "\n")

if __name__ == "__main__":
    import sys
    import time
    import subprocess
    import os
    
    def clear_screen():
        """Clear terminal screen"""
        subprocess.call('clear' if os.name == 'posix' else 'cls', shell=True)
    
    if '--once' in sys.argv:
        tracker = Phase3BTracker()
        tracker.print_report()
    else:
        # Live watch mode (default)
        try:
            while True:
                clear_screen()
                tracker = Phase3BTracker()
                tracker.print_report()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n✓ Stopped")
